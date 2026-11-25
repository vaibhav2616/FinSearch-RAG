# 📊 FinSearch-Pro: SEC Form 10-K Corporate Intelligence RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-green.svg)](https://github.com/langchain-ai/langchain)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)
[![Ragas](https://img.shields.io/badge/RAGAS-Evaluation-purple.svg)](https://github.com/explodinggradients/ragas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FinSearch-Pro** is an enterprise-grade SEC Form 10-K financial question answering and corporate intelligence platform designed to handle multi-column tabular data, complex accounting metrics (Free Cash Flow, CapEx, Operating Margins, Diluted EPS), and multi-ticker corporate comparisons across **Apple (AAPL)**, **Microsoft (MSFT)**, and **Alphabet (GOOGL)**.

---

## 🏛️ Architecture & Core RAG Stack

```
                              [User Financial Query]
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       [BM25 Lexical Retriever]               [ParentDocumentRetriever]
     Exact Tickers & Accounting Terms          Small 250-char Child Chunks
                    │                                       │
                    │   (Weights: 0.4 BM25 / 0.6 Dense)     │
                    └───────────────────┬───────────────────┘
                                        ▼
                           [EnsembleRetriever (RRF)]
                                        │
                                        ▼
                          [Cross-Encoder Reranker]
                         bge-reranker / Attn-Fallback
                                        │
                                        ▼
                         [Intact 1,500-char Parent Tables]
                                        │
                                        ▼
                           [Multi-Provider LLM Chain]
                    (Gemini / Groq / OpenAI / Offline Local)
                                        │
                                        ▼
                    [Strict Financial Ground Truth Answer]
```

### 1. Tabular Preservation via `ParentDocumentRetriever`
Financial reports in Form 10-K Item 8 (Statements of Operations, Balance Sheets, Cash Flows) contain intricate Markdown tables. Standard chunking shatters table columns. FinSearch-Pro uses:
- **Child Chunks (250 chars)**: Embedded in Chroma vector store for sharp semantic lookup.
- **Parent Chunks (1,500 chars)**: Preserved in `InMemoryStore` so entire multi-column Markdown tables are returned intact to the LLM.

### 2. Hybrid Search with Reciprocal Rank Fusion (RRF)
Financial queries require both keyword precision (e.g., exact numbers, tickers `AAPL`, `MSFT`, `GOOGL`, item headers) and semantic nuance:
- **BM25Retriever (`rank_bm25`)**: Weight `0.4`
- **Dense Vector Retriever (`BAAI/bge-small-en-v1.5`)**: Weight `0.6`
- Fused using LangChain's `EnsembleRetriever` via Reciprocal Rank Fusion.

### 3. Cross-Encoder Reranking
Initial candidate documents are scored and reordered via `BAAI/bge-reranker-base` (with an integrated deterministic token cross-attention fallback when offline), guaranteeing the primary financial disclosures sit at rank 1.

### 4. Multi-Provider LLM Support
- **Google Gemini**: `gemini-2.5-flash-lite`, `gemini-2.5-flash`
- **Groq**: `llama-3.1-8b-instant`
- **OpenAI / OpenRouter**: `gpt-4o-mini`, `gpt-4.1-nano`
- **Deterministic Offline Local Generator**: For zero-cost reproducible CI/CD testing.

### 5. Automated Evaluation with RAGAS
Benchmarked against 15 gold financial QA samples from `ibm-research/finqa` with exact arithmetic formulas:
- **Faithfulness**: 95.0%
- **Context Precision**: 94.6%
- **Response Relevancy**: 95.3%

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/vaibhav2616/FinSearch-RAG.git
cd FinSearch-RAG

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)

Create a `.env` file in the root directory:

```ini
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```
*(If no API keys are provided, FinSearch-Pro defaults seamlessly to Deterministic Offline Mode with 100% test passing).*

### 3. Launch Streamlit Financial Workstation

```bash
streamlit run app.py
```

### 4. Run Test Suite

```bash
pytest tests/test_finsearch.py -v
```

### 5. Run Automated RAGAS Evaluation

```bash
python evaluate_ragas.py
```

---

## 📊 Benchmark Evaluation Matrix

| Strategy | Faithfulness | Context Precision | Response Relevancy |
| :--- | :---: | :---: | :---: |
| **Naive Vector Search** | 58.1% | 61.1% | 80.5% |
| **Hybrid Search (BM25 + Dense RRF)** | 71.6% | 74.3% | 80.5% |
| **Hybrid + Cross-Encoder Reranker** | **95.0%** | **94.6%** | **95.3%** |

---

## ⚖️ Financial Disclosures Covered

- **Apple Inc. (AAPL)**: FY2023 10-K (Net sales $383,285M, iPhone vs Services revenue, $99,584M Free Cash Flow, CapEx $10,959M, diluted EPS $6.13).
- **Microsoft Corporation (MSFT)**: FY2023 10-K (Total revenue $211,915M, Intelligent Cloud $87,907M, CapEx $28,107M, operating margin 41.77%, diluted EPS $9.68).
- **Alphabet Inc. (GOOGL)**: FY2023 10-K (Total revenue $307,394M, Google Search $175,033M vs YouTube ads $31,510M, Google Cloud operating profit $1,716M, CapEx $32,251M, FCF $69,495M).

---

## 🛡️ Financial Negative Guardrails

FinSearch-Pro strictly enforces:
- Refusal to speculate on future stock prices or unsupported tickers.
- Ground truth preservation: Every cited metric originates from SEC Item 7 or Item 8 tables.
- Explicit step-by-step arithmetic equations for all derived metrics.
