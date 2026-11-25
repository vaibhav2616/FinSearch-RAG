"""
FinSearch-Pro Retriever Module.

Architecture:
1. Dense Vector Retrieval via ParentDocumentRetriever:
   - Preserves complete financial Markdown tables in parent chunks (1,500 chars).
   - Indexes precise child chunks (250 chars) in Chroma vector store.
2. Sparse Lexical Retrieval via BM25Retriever:
   - Indexes exact financial tickers (AAPL, MSFT, GOOGL), dates, and accounting terms.
3. Hybrid Retrieval via EnsembleRetriever with Reciprocal Rank Fusion (RRF):
   - Weights: 0.4 BM25 + 0.6 Dense Vector Retriever.
4. Cross-Encoder Reranker:
   - Uses HuggingFaceCrossEncoder ('BAAI/bge-reranker-base') when model weights/network are available.
   - Robust offline token cross-attention fallback when offline or model cache unavailable.
"""

import os
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from data.sec_data import SEC_10K_DOCUMENTS


class FallbackCrossEncoderReranker:
    """
    Deterministic cross-attention / token interaction reranker fallback.
    Used when HuggingFace bge-reranker model download is unavailable or offline.
    Scores document-query interaction considering:
    - Exact ticker & fiscal year match
    - Query term frequency in tabular context
    - Position and coverage of key financial numbers
    """
    def __init__(self, model_name: str = "token-cross-attention-fallback"):
        self.model_name = model_name

    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        scores = []
        for query, text in pairs:
            q_lower = query.lower()
            t_lower = text.lower()
            
            # Tokenize words and numbers
            q_tokens = re.findall(r"\b[\w\$\%\.]+\b", q_lower)
            t_tokens = re.findall(r"\b[\w\$\%\.]+\b", t_lower)
            
            if not q_tokens or not t_tokens:
                scores.append(0.0)
                continue
            
            t_counter = Counter(t_tokens)
            
            # Exact match rewards for financial tickers & numbers
            score = 0.0
            for token in q_tokens:
                if token in ("aapl", "apple", "msft", "microsoft", "googl", "google", "alphabet"):
                    if token in t_lower:
                        score += 3.0
                elif any(char.isdigit() for char in token):
                    if token in t_lower:
                        score += 2.5
                elif token in ("capex", "eps", "revenue", "sales", "cloud", "margin", "margins", "fcf"):
                    if token in t_lower:
                        score += 2.0
                elif token in t_counter:
                    # BM25-like sublinear term frequency
                    tf = t_counter[token]
                    score += math.log(1.0 + tf) * 1.0

            # Normalize to approximately [0, 1] range using sigmoid-like scaling
            normalized_score = 1.0 / (1.0 + math.exp(-0.3 * (score - 4.0)))
            scores.append(round(normalized_score, 4))
        return scores


class FinSearchRetrieverEngine:
    """
    Comprehensive FinSearch-Pro Retrieval Engine.
    Exposes:
    - naive_vector_retrieve(query, top_k)
    - hybrid_retrieve(query, top_k)
    - rerank(query, documents, top_k)
    - retrieve_and_rerank(query, top_k)
    """
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        use_disk_cache: bool = False
    ):
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory

        # Initialize Embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception:
            # Fallback to standard all-MiniLM if bge has revision issues
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

        # Parent Document Store & Text Splitters
        self.parent_store = InMemoryStore()
        
        # 1,500-char parent chunk to preserve multi-column Markdown tables intact
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n# ", "\n\n## ", "\n\n| ", "\n\n", "\n", " "]
        )
        
        # 250-char child chunk for precise vector matching
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            separators=["\n| ", "\n", ". ", " "]
        )

        # Chroma vector store for child chunks
        self.vectorstore = Chroma(
            collection_name="finsearch_child_chunks",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory if use_disk_cache else None
        )

        # ParentDocumentRetriever
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.parent_store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter
        )

        # In-memory document storage for BM25 and fallback
        self.raw_documents: List[Document] = []
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None

        # Cross-Encoder Reranker
        self._init_cross_encoder()

        # Ingest SEC corpus
        self._ingest_sec_corpus()

    def _init_cross_encoder(self):
        """Initializes HuggingFace Cross-Encoder with fallback."""
        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            self.cross_encoder = HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-base",
                model_kwargs={"device": "cpu"}
            )
            # Verify scoring capability
            _ = self.cross_encoder.score([("test", "test")])
            self.is_offline_reranker = False
        except Exception:
            self.cross_encoder = FallbackCrossEncoderReranker()
            self.is_offline_reranker = True

    def _ingest_sec_corpus(self):
        """Ingests SEC 10-K documents into ParentDocumentRetriever and BM25."""
        docs = []
        for item in SEC_10K_DOCUMENTS:
            doc = Document(
                page_content=item["content"],
                metadata={
                    "ticker": item["ticker"],
                    "company": item["company"],
                    "fiscal_year": item["fiscal_year"],
                    "section": item["section"],
                    "title": item["title"]
                }
            )
            docs.append(doc)
        
        self.raw_documents = docs
        
        # Ingest into ParentDocumentRetriever (populates vectorstore with child chunks and parent_store with parent chunks)
        self.parent_retriever.add_documents(docs)

        # Create BM25 retriever on parent-chunked or raw documents
        parent_docs = self.parent_splitter.split_documents(docs)
        self.bm25_retriever = BM25Retriever.from_documents(parent_docs)
        self.bm25_retriever.k = 5

        # Create EnsembleRetriever with Reciprocal Rank Fusion (weights: 0.4 BM25, 0.6 Dense)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.parent_retriever],
            weights=[0.4, 0.6]
        )

    def naive_vector_retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Performs baseline naive dense vector retrieval (direct similarity on parent docs)."""
        # Retrieve parent documents directly from parent retriever
        docs = self.parent_retriever.invoke(query)
        return docs[:top_k]

    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Performs hybrid retrieval using BM25 + ParentDocumentRetriever via RRF."""
        if self.ensemble_retriever is None:
            return self.naive_vector_retrieve(query, top_k=top_k)
        docs = self.ensemble_retriever.invoke(query)
        return docs[:top_k]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        Reranks retrieved documents using the Cross-Encoder.
        Returns a list of (Document, score) sorted in descending order of relevance.
        """
        if not documents:
            return []
        
        pairs = [(query, doc.page_content) for doc in documents]
        try:
            scores = self.cross_encoder.score(pairs)
        except Exception:
            fallback = FallbackCrossEncoderReranker()
            scores = fallback.score(pairs)

        doc_score_pairs = list(zip(documents, [float(s) for s in scores]))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return doc_score_pairs[:top_k]

    def retrieve_and_rerank(
        self,
        query: str,
        top_k_retrieve: int = 6,
        top_k_rerank: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        Complete retrieval pipeline:
        1. Hybrid Search (BM25 + ParentDocumentRetriever with RRF)
        2. Cross-Encoder Reranker
        """
        initial_docs = self.hybrid_retrieve(query, top_k=top_k_retrieve)
        reranked = self.rerank(query, initial_docs, top_k=top_k_rerank)
        return reranked


# Global singleton instance
_GLOBAL_ENGINE: Optional[FinSearchRetrieverEngine] = None

def get_finsearch_retriever() -> FinSearchRetrieverEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = FinSearchRetrieverEngine()
    return _GLOBAL_ENGINE
