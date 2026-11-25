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
from chain import FinSearchChain, DeterministicOfflineLLM, detect_provider_from_key
from evaluate_ragas import run_evaluation, compute_deterministic_sample_metrics
from data.sec_data import SEC_10K_DOCUMENTS, GOLD_BENCHMARK_QA
from langchain_core.documents import Document


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
    assert "out of scope" in ans_lower or "do not contain" in ans_lower

    res_buy_aapl = offline_chain.answer_query("Should I buy Apple stock right now?")
    assert "out of scope" in res_buy_aapl["answer"].lower()


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


def test_dynamic_ingestion_and_zero_hardcoding(retriever, offline_chain):
    """
    Verifies live document ingestion (add_documents & ingest_uploaded_file)
    with a fictional company report and queries it dynamically to prove zero hardcoding.
    """
    custom_md = (
        "# NovaTech Corp FY2024 Annual Financial Summary\n\n"
        "| Financial Metric | FY2024 | FY2023 |\n"
        "| :--- | :--- | :--- |\n"
        "| **Total Revenue** | $ 42,500 | $ 35,000 |\n"
        "| **Operating Income** | $ 12,750 | $ 9,800 |\n"
        "| **Free Cash Flow** | $ 10,200 | $ 8,100 |\n\n"
        "NovaTech Corp achieved 21.43% YoY revenue growth driven by quantum cloud adoption."
    ).encode("utf-8")

    added = retriever.ingest_uploaded_file(custom_md, "novatech_fy24.md", ticker="NVTC")
    assert added == 1
    assert "NVTC" in retriever.get_available_tickers()

    # CSV ingestion test
    csv_bytes = b"Segment,FY2024,FY2023\nQuantum Cloud,25000,19000\nEdge Hardware,17500,16000\n"
    added_csv = retriever.ingest_uploaded_file(csv_bytes, "novatech_segments.csv", ticker="NVTC")
    assert added_csv == 1

    # Query dynamically with ticker_filter="NVTC"
    res = offline_chain.answer_query(
        "What was NovaTech's total revenue and operating income in FY2024?",
        ticker_filter="NVTC",
    )
    assert "42,500" in res["answer"]
    assert "12,750" in res["answer"]
    assert all(doc.metadata.get("ticker") == "NVTC" for doc, _ in res["scored_documents"])


def test_universal_api_key_auto_detection():
    """Verifies Universal AI API key prefix auto-detection."""
    assert detect_provider_from_key("") == "offline"
    assert detect_provider_from_key("AIzaSyDummyKey123") == "gemini"
    assert detect_provider_from_key("gsk_dummyKey123") == "groq"
    assert detect_provider_from_key("sk-or-v1-dummyKey123") == "openrouter"
    assert detect_provider_from_key("sk-proj-dummyKey123") == "openai"


def test_app_ui_smoke_light_dark_and_chat():
    """AppTest UI smoke test covering Light/Dark theme toggle and chat query."""
    from streamlit.testing.v1 import AppTest

    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30).run()
    assert not at.exception

    # Toggle Light theme
    at.segmented_control(key="theme_mode").set_value("Light").run()
    assert not at.exception

    # Toggle back to Dark theme
    at.segmented_control(key="theme_mode").set_value("Dark").run()
    assert not at.exception

    # Submit a chat query
    at.chat_input(key="finsearch_chat_input").set_value("What was Apple's total net sales in FY2023?").run(timeout=30)
    assert not at.exception
    rendered_text = " ".join(m.value for m in at.markdown)
    assert "383,285" in rendered_text


def _make_minimal_pdf(text: str) -> bytes:
    """Builds a minimal valid single-page PDF binary containing `text`."""
    import io
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe_text}) Tj ET".encode("latin-1")
    objs = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
        f"4 0 obj << /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream\nendobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = []
    for obj in objs:
        offsets.append(buf.tell())
        buf.write(obj)
    xref_pos = buf.tell()
    buf.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets:
        buf.write(f"{off:010d} 00000 n \n".encode("latin-1"))
    buf.write(f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1"))
    return buf.getvalue()


def test_pdf_binary_ingestion_and_empty_csv(retriever, offline_chain):
    """Verifies real binary .pdf ingestion via pypdf and graceful handling of empty CSVs."""
    pdf_bytes = _make_minimal_pdf("HeliosBio FY2024 Revenue was $18,400 million and R&D was $3,200 million.")
    added_pdf = retriever.ingest_uploaded_file(pdf_bytes, "helios_fy24.pdf", ticker="HLIO")
    assert added_pdf == 1
    assert "HLIO" in retriever.get_available_tickers()

    res = offline_chain.answer_query("What was HeliosBio revenue in FY2024?", ticker_filter="HLIO")
    assert "18,400" in res["answer"]

    # Empty CSV should return 0 without raising
    assert retriever.ingest_uploaded_file(b"", "empty.csv", ticker="HLIO") == 0
    assert retriever.ingest_uploaded_file(b"   \n", "blank.csv", ticker="HLIO") == 0


def test_offline_synthesizer_edge_cases(offline_chain):
    """Regression tests for imperative verbs, out-of-scope years, and chronological YoY math."""
    # 1. Imperative capitalized verb 'Summarize' must NOT trigger false out-of-scope refusal
    res_sum = offline_chain.answer_query("Summarize Apple's total net sales in FY2023")
    assert "383,285" in res_sum["answer"]

    # 2. Out-of-scope year FY2025 must be refused even when '25' appears in 25.31%
    llm = DeterministicOfflineLLM()
    ctx_25 = (
        "--- [Document 1 | Ticker: AAPL | Section: Consolidated | FY: FY2023 | Reranker Relevance Score: 0.9500] ---\n"
        "Title: Apple FY23\n"
        "| Metric | FY2023 | FY2022 |\n"
        "| :--- | :--- | :--- |\n"
        "| **Net Profit Margin** | 25.31% | 25.31% |\n"
    )
    ans_2025 = llm.invoke(f"Context:\n{ctx_25}\n\nQuestion:\nWhat was Apple net profit margin in FY2025?\n\nStructured Financial Answer:")
    assert "out of scope" in ans_2025.lower()

    # 3. Chronological table (FY2022 before FY2023) must compute FY2023 - FY2022 (+50.00, +50.00%)
    ctx_chrono = (
        "--- [Document 1 | Ticker: NVTC | Section: CSV | FY: Uploaded | Reranker Relevance Score: 0.9500] ---\n"
        "Title: NovaTech\n"
        "| Metric | FY2022 | FY2023 |\n"
        "| :--- | :--- | :--- |\n"
        "| **Revenue** | 100 | 150 |\n"
    )
    ans_chrono = llm.invoke(f"Context:\n{ctx_chrono}\n\nQuestion:\nWhat was Revenue growth in FY2023 vs FY2022?\n\nStructured Financial Answer:")
    assert "+50.00" in ans_chrono
    assert "+50.00%" in ans_chrono
