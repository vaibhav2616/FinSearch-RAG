"""
Test suite for Level 2: FinSearch-Pro.

Verifies:
1. Multi-column financial Markdown table preservation in parent chunks.
2. Small child chunks (250 chars) precision & parent document store retrieval.
3. Hybrid search (BM25 + Dense vector with RRF).
4. Cross-Encoder reranker scoring & fallback behavior.
5. Multi-provider chain orchestration & answer factual accuracy on IBM FinQA benchmarks.
6. Financial negative guardrails (rejection of speculative advice and ungrounded tickers).
7. Evaluation dataset pipeline with Faithfulness, Response Relevancy, and Context Precision.
"""

import os
import sys
import pytest

# Ensure level-2-finsearch root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retriever import (
    FinSearchRetrieverEngine,
    FallbackCrossEncoderReranker,
    get_finsearch_retriever
)
from chain import FinSearchChain, DeterministicOfflineLLM
from evaluate_ragas import run_evaluation, compute_deterministic_sample_metrics
from data.sec_data import SEC_10K_DOCUMENTS, GOLD_BENCHMARK_QA


@pytest.fixture(scope="module")
def retriever():
    """Provides the global FinSearch retriever engine."""
    return get_finsearch_retriever()


@pytest.fixture(scope="module")
def offline_chain(retriever):
    """Provides an offline deterministic chain instance."""
    return FinSearchChain(provider="offline", retriever_engine=retriever)


def test_sec_corpus_integrity():
    """Verifies that all 3 tickers and cross-company comparisons are represented with tables."""
    assert len(SEC_10K_DOCUMENTS) >= 12
    tickers = {doc["ticker"] for doc in SEC_10K_DOCUMENTS}
    assert {"AAPL", "MSFT", "GOOGL", "CROSS"}.issubset(tickers)

    # Verify tables exist and have valid Markdown syntax
    for doc in SEC_10K_DOCUMENTS:
        if "Statements" in doc["section"] or "Category" in doc["section"] or "Segment" in doc["section"]:
            assert "| :---" in doc["content"] or "| ---" in doc["content"]
            assert "$" in doc["content"]


def test_parent_document_retriever_preservation(retriever):
    """Verifies that retrieved parent chunks preserve intact multi-column Markdown tables."""
    results = retriever.naive_vector_retrieve("Apple FY2023 Total net sales diluted EPS", top_k=2)
    assert len(results) > 0
    top_doc = results[0]
    
    # Check that parent chunk preserved table borders and financial headers
    assert "|" in top_doc.page_content
    assert top_doc.metadata.get("ticker") in ["AAPL", "CROSS"]


def test_child_and_parent_chunk_sizes(retriever):
    """Verifies that child chunk size is ~250 chars and parent chunk size is ~1,500 chars."""
    assert retriever.child_splitter._chunk_size == 250
    assert retriever.parent_splitter._chunk_size == 1500


def test_bm25_exact_ticker_and_metric_retrieval(retriever):
    """Verifies that BM25 retrieves exact financial terms (CapEx, Intelligent Cloud, YouTube ads)."""
    assert retriever.bm25_retriever is not None
    
    msft_docs = retriever.bm25_retriever.invoke("Microsoft Intelligent Cloud segment revenue")
    assert any("Intelligent Cloud" in d.page_content for d in msft_docs)

    googl_docs = retriever.bm25_retriever.invoke("YouTube ads revenue Google Search")
    assert any("YouTube ads" in d.page_content for d in googl_docs)


def test_hybrid_rrf_ensemble(retriever):
    """Verifies EnsembleRetriever combines Dense and BM25 with expected weights."""
    assert retriever.ensemble_retriever is not None
    assert retriever.ensemble_retriever.weights == [0.4, 0.6]

    hybrid_results = retriever.hybrid_retrieve("Apple Free Cash Flow CapEx payments", top_k=3)
    assert len(hybrid_results) > 0
    assert any("Free Cash Flow" in d.page_content for d in hybrid_results)


def test_cross_encoder_reranking_order(retriever):
    """Verifies that reranker scores documents and ranks the most relevant on top."""
    query = "Microsoft capital expenditures CapEx FY2023"
    reranked = retriever.retrieve_and_rerank(query, top_k_retrieve=5, top_k_rerank=2)
    
    assert len(reranked) == 2
    top_doc, top_score = reranked[0]
    assert "CapEx" in top_doc.page_content or "Property and equipment" in top_doc.page_content
    assert top_score >= reranked[1][1]


def test_fallback_cross_encoder_deterministic():
    """Verifies the token cross-attention fallback scoring algorithm."""
    fallback = FallbackCrossEncoderReranker()
    scores = fallback.score([
        ("Apple FY23 revenue", "Apple FY23 Total net sales was $383,285 million"),
        ("Apple FY23 revenue", "Global weather forecasts and ocean currents")
    ])
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_chain_financial_ground_truth_answers(offline_chain):
    """Verifies factual accuracy of answers against IBM FinQA gold calculations."""
    # Test 1: Apple Net Sales
    res_aapl = offline_chain.answer_query("What was Apple's total net sales in FY2023?")
    assert "383,285" in res_aapl["answer"]
    assert "394,328" in res_aapl["answer"]

    # Test 2: Microsoft Intelligent Cloud
    res_msft = offline_chain.answer_query("What was Microsoft Intelligent Cloud revenue in FY2023?")
    assert "87,907" in res_msft["answer"]

    # Test 3: Google Search vs YouTube ads
    res_googl = offline_chain.answer_query("What was the revenue of Google Search vs YouTube ads in FY2023?")
    assert "175,033" in res_googl["answer"]
    assert "31,510" in res_googl["answer"]

    # Test 4: Highest Operating Margin
    res_margin = offline_chain.answer_query("Which company had the highest operating margin in FY2023?")
    assert "41.77%" in res_margin["answer"]
    assert "Microsoft" in res_margin["answer"]


def test_negative_financial_guardrails(offline_chain):
    """Verifies that the chain refuses speculation and handles ungrounded companies."""
    res_tsla = offline_chain.answer_query("Should I buy Tesla stock and what is TSLA FY2025 revenue prediction?")
    ans_lower = res_tsla["answer"].lower()
    assert "do not contain this information" in ans_lower or "covers apple" in ans_lower


def test_evaluation_pipeline_metrics():
    """Verifies the automated evaluation pipeline returns comparative metrics for all 3 strategies."""
    results = run_evaluation(num_samples=3, provider="offline")
    
    assert "naive" in results
    assert "hybrid" in results
    assert "hybrid_rerank" in results

    # Hybrid + Reranker should achieve superior faithfulness and context precision
    assert results["hybrid_rerank"]["avg_context_precision"] >= results["naive"]["avg_context_precision"]
    assert results["hybrid_rerank"]["avg_faithfulness"] >= results["naive"]["avg_faithfulness"]


def test_deterministic_metric_calculator():
    """Unit test for compute_deterministic_sample_metrics calculation logic."""
    metrics = compute_deterministic_sample_metrics(
        query="Apple Free Cash Flow FY23",
        answer="Apple Free Cash Flow was $99,584 million calculated from operating cash flow of $110,543 million minus CapEx of $10,959 million.",
        context=["Free Cash Flow was $99,584 million. Operating cash flow $110,543 million, CapEx $10,959 million."],
        ground_truth="Apple generated $99,584 million in Free Cash Flow.",
        expected_topics=["99,584", "Free Cash Flow"]
    )
    assert metrics["faithfulness"] == 1.0
    assert metrics["response_relevancy"] >= 0.90
    assert metrics["context_precision"] >= 0.90
