"""
FinSearch-Pro: SEC Form 10-K & Custom Financial Filing Intelligence Workstation.
Features:
- Conversational st.chat_input + st.chat_message history + st.status thinking timer.
- Auto-executing quick financial st.pills.
- Universal AI API Key auto-detection (Gemini, Groq, OpenRouter, OpenAI, Offline).
- Live document uploader (PDF, TXT, MD, CSV) with dynamic ParentDocument + BM25 re-indexing.
- High-contrast Light & Dark editorial financial theme toggle.
- Dynamic RAGAS evaluation deltas.
"""

import os
import sys
import time
import pandas as pd
import streamlit as st

# Add current folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import get_finsearch_retriever
from chain import FinSearchChain, detect_provider_from_key
from evaluate_ragas import run_evaluation

# Page configuration
st.set_page_config(
    page_title="FinSearch-Pro | Corporate Intelligence",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _build_theme_css(mode: str) -> str:
    """Returns high-contrast editorial financial CSS for Light or Dark mode."""
    if mode == "Light":
        bg_main = "#FAF8F5"
        bg_sidebar = "#F2EFE9"
        bg_surface = "#FFFFFF"
        bg_elevated = "#F5F2EB"
        text_primary = "#18181B"
        text_muted = "#52525B"
        border_col = "#D6D3CD"
        accent = "#0F766E"
        accent_text = "#FFFFFF"
        pill_bg = "#EFECE4"
        pill_active_bg = "#0F766E"
        pill_active_text = "#FFFFFF"
        alert_bg = "#F0FDF4"
    else:
        bg_main = "#0B1120"
        bg_sidebar = "#111827"
        bg_surface = "#1E293B"
        bg_elevated = "#0F172A"
        text_primary = "#F8FAFC"
        text_muted = "#94A3B8"
        border_col = "#334155"
        accent = "#38BDF8"
        accent_text = "#0B1120"
        pill_bg = "#1E293B"
        pill_active_bg = "#38BDF8"
        pill_active_text = "#0B1120"
        alert_bg = "#1E293B"

    return f"""
    <style>
        .stApp, [data-testid="stAppViewContainer"] {{
            background-color: {bg_main};
            color: {text_primary};
        }}
        [data-testid="stHeader"] {{
            background-color: {bg_main};
        }}
        [data-testid="stSidebar"] {{
            background-color: {bg_sidebar};
            border-right: 1px solid {border_col};
        }}
        [data-testid="stSidebar"] *, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"] {{
            color: {text_primary};
        }}
        .sub-header, .doc-meta, [data-testid="stCaptionContainer"] {{
            color: {text_muted} !important;
        }}
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stTextInputField"] {{
            background-color: {bg_surface} !important;
            color: {text_primary} !important;
            border: 1px solid {border_col} !important;
        }}
        [data-testid="stTextInputField"] input {{
            color: {text_primary} !important;
        }}
        [data-testid="stSelectboxVirtualDropdown"] {{
            background-color: {bg_surface} !important;
            color: {text_primary} !important;
            border: 1px solid {border_col} !important;
        }}
        [data-testid="stSelectboxVirtualDropdown"] li {{
            color: {text_primary} !important;
        }}
        [data-testid="stSelectboxVirtualDropdown"] li:hover {{
            background-color: {bg_elevated} !important;
        }}
        [data-testid="stButtonGroup"] {{
            gap: 0.35rem;
        }}
        [data-testid="stBaseButton-pills"] {{
            background-color: {pill_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {border_col} !important;
        }}
        [data-testid="stBaseButton-pills"][aria-pressed="true"],
        [data-testid="stBaseButton-pills"][aria-checked="true"] {{
            background-color: {pill_active_bg} !important;
            color: {pill_active_text} !important;
            border-color: {pill_active_bg} !important;
        }}
        [data-testid="stChatInputSubmitButton"] {{
            background-color: {accent} !important;
            color: {accent_text} !important;
        }}
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div {{
            background-color: {bg_surface} !important;
            border-color: {border_col} !important;
        }}
        [data-testid="stChatInputTextArea"],
        [data-testid="stChatInput"] textarea {{
            background-color: {bg_surface} !important;
            color: {text_primary} !important;
        }}
        [data-testid="stChatInputTextArea"]::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {text_muted} !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {bg_surface} !important;
            color: {text_primary} !important;
            border: 1px dashed {border_col} !important;
        }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            color: {text_primary} !important;
        }}
        [data-baseweb="tab"] {{
            color: {text_muted} !important;
        }}
        [data-baseweb="tab"][aria-selected="true"] {{
            color: {accent} !important;
        }}
        [data-testid="stAlertContainer"] {{
            background-color: {alert_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {border_col} !important;
        }}
        [data-testid="stChatMessage"] {{
            background-color: {bg_surface};
            border: 1px solid {border_col};
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }}
        [data-testid="stExpander"] {{
            background-color: {bg_elevated};
            border: 1px solid {border_col};
            border-radius: 8px;
        }}
        .ticker-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 5px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 6px;
            border: 1px solid {accent};
            color: {accent};
            background-color: {bg_elevated};
        }}
    </style>
    """


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()
if "last_executed_pill" not in st.session_state:
    st.session_state.last_executed_pill = None

# Load shared retriever engine
engine = get_finsearch_retriever()

# Sidebar Configuration
with st.sidebar:
    st.markdown("### FinSearch-Pro")
    st.caption("Corporate Filing & SEC 10-K Intelligence")

    theme_mode = st.segmented_control(
        "Appearance",
        options=["Light", "Dark"],
        default="Dark",
        key="theme_mode",
    ) or "Dark"

    st.divider()

    st.markdown("#### Universal AI API Key")
    api_key_input = st.text_input(
        "API Key (auto-detects provider)",
        type="password",
        key="universal_api_key",
        placeholder="AIza... / gsk_... / sk-or-... / sk-...",
    )
    detected_provider = detect_provider_from_key(api_key_input)
    provider_label_map = {
        "offline": "Offline Dynamic Synthesizer (Zero-Cost)",
        "gemini": "Google Gemini (Auto-Detected)",
        "groq": "Groq (Auto-Detected)",
        "openrouter": "OpenRouter (Auto-Detected)",
        "openai": "OpenAI (Auto-Detected)",
    }
    st.caption(f"Active Engine: **{provider_label_map.get(detected_provider, detected_provider)}**")

    selected_model_opt = st.selectbox(
        "Model",
        options=[
            "Auto (Lightweight Default)",
            "gemini-2.5-flash-lite",
            "llama-3.1-8b-instant",
            "gpt-4.1-nano",
            "gpt-4o-mini",
            "gemini-2.5-flash",
        ],
        index=0,
        key="model_choice",
    )
    selected_model = None if selected_model_opt == "Auto (Lightweight Default)" else selected_model_opt

    st.divider()

    st.markdown("#### Live Document Uploader")
    upload_ticker = st.text_input("Ticker Tag for Uploads", value="CUSTOM", key="upload_ticker")
    uploaded_files = st.file_uploader(
        "Upload Financial Filings (PDF, TXT, MD, CSV)",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        key="doc_uploader",
    )
    if uploaded_files:
        for uf in uploaded_files:
            file_sig = (uf.name, uf.size, upload_ticker.strip().upper())
            if file_sig not in st.session_state.ingested_files:
                try:
                    added_count = engine.ingest_uploaded_file(
                        uf.getvalue(), uf.name, ticker=upload_ticker
                    )
                    st.session_state.ingested_files.add(file_sig)
                    st.success(f"Indexed {uf.name} ({added_count} doc(s)) under [{upload_ticker.upper()}].")
                except Exception as exc:
                    st.error(f"Failed to ingest {uf.name}: {exc}")

    available_tickers = engine.get_available_tickers()
    st.metric("Indexed Documents", len(engine.raw_documents), f"{len(available_tickers)} Tickers")

    st.divider()

    st.markdown("#### Retrieval & Filtering")
    strategy_label = st.radio(
        "RAG Pipeline Strategy",
        options=[
            "Hybrid + Cross-Encoder Reranker",
            "Hybrid Search (BM25 + Dense RRF)",
            "Naive Vector Search",
        ],
        index=0,
        key="retrieval_strategy_label",
    )
    strat_map = {
        "Hybrid + Cross-Encoder Reranker": "hybrid_rerank",
        "Hybrid Search (BM25 + Dense RRF)": "hybrid",
        "Naive Vector Search": "naive",
    }
    retrieval_strategy = strat_map[strategy_label]
    top_k_slider = st.slider("Max Parent Documents", min_value=1, max_value=5, value=3, key="top_k_slider")

    ticker_filter = st.selectbox(
        "Filter by Ticker",
        options=["ALL"] + available_tickers,
        index=0,
        key="ticker_filter",
    )

# Inject Theme CSS
st.html(_build_theme_css(theme_mode))

# Main Header
st.title("FinSearch-Pro: Enterprise Filing Intelligence")
st.caption(
    "ParentDocumentRetriever (1,500-char tables) | BM25 + Dense RRF | "
    "Cross-Encoder Reranker | Dynamic Context Synthesizer"
)

# Tabs
tab_query, tab_compare, tab_eval, tab_corpus = st.tabs([
    "Intelligence Chat",
    "Big Tech Comparison",
    "RAGAS Evaluation",
    "Indexed Corpus",
])

# ----------------------------------------------------
# TAB 1: Conversational Intelligence Chat
# ----------------------------------------------------
with tab_query:
    QUICK_PILLS = {
        "Apple FY23 Net Sales": "What was Apple's total net sales in FY2023, and how did it change compared to FY2022?",
        "Microsoft Cloud & CapEx": "What was Microsoft Intelligent Cloud segment revenue and CapEx in FY2023?",
        "Google Search vs YouTube": "What was the revenue of Google Search vs YouTube ads in FY2023?",
        "Compare Operating Margins": "Which company had the highest operating margin in FY2023 among Apple, Microsoft, and Alphabet?",
    }

    pill_col, clear_col = st.columns([5, 1], vertical_alignment="center")
    with pill_col:
        selected_pill = st.pills(
            "Quick Financial Queries",
            options=list(QUICK_PILLS.keys()),
            key="quick_pills",
            label_visibility="collapsed",
        )
    with clear_col:
        if st.session_state.messages and st.button("Clear Chat", key="clear_chat_btn", width="stretch"):
            st.session_state.messages = []
            st.session_state.last_executed_pill = None
            st.rerun()

    # Render existing conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("thinking_label"):
                with st.expander(msg["thinking_label"], expanded=False):
                    st.markdown(msg.get("thinking_details", ""))
            st.markdown(msg["content"])
            for idx, src in enumerate(msg.get("sources", []), 1):
                with st.expander(
                    f"Source {idx}: [{src['ticker']}] {src['title']} — Score: {src['score']:.4f}",
                    expanded=False,
                ):
                    st.markdown(src["content"])

    chat_prompt = st.chat_input(
        "Ask a financial question across SEC 10-Ks or uploaded filings...",
        key="finsearch_chat_input",
    )

    active_prompt = None
    if chat_prompt and chat_prompt.strip():
        active_prompt = chat_prompt.strip()
    elif selected_pill and selected_pill != st.session_state.last_executed_pill:
        st.session_state.last_executed_pill = selected_pill
        active_prompt = QUICK_PILLS[selected_pill]

    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            t0 = time.perf_counter()
            with st.status(":shimmer[Searching filings & synthesizing...]", type="compact") as status:
                chain = FinSearchChain(
                    provider="auto" if api_key_input else "offline",
                    model_name=selected_model,
                    api_key=api_key_input,
                    retriever_engine=engine,
                )
                result = chain.answer_query(
                    query=active_prompt,
                    retrieval_strategy=retrieval_strategy,
                    top_k=top_k_slider,
                    ticker_filter=ticker_filter,
                )
                elapsed = time.perf_counter() - t0
                thinking_details = (
                    f"- **Strategy:** `{retrieval_strategy}`\n"
                    f"- **Ticker Filter:** `{ticker_filter}`\n"
                    f"- **Engine:** `{result.get('provider', 'offline')}`\n"
                    f"- **Retrieved Parent Chunks:** `{len(result['scored_documents'])}`"
                )
                st.markdown(thinking_details)
                thinking_label = f"Thought for {elapsed:.2f}s"
                status.update(label=thinking_label, state="complete")

            st.markdown(result["answer"])

            sources_payload = []
            for idx, (doc, score) in enumerate(result["scored_documents"], 1):
                src_item = {
                    "ticker": doc.metadata.get("ticker", "N/A"),
                    "title": doc.metadata.get("title", "Financial Filing"),
                    "score": float(score),
                    "content": doc.page_content,
                }
                sources_payload.append(src_item)
                with st.expander(
                    f"Source {idx}: [{src_item['ticker']}] {src_item['title']} — Score: {src_item['score']:.4f}",
                    expanded=(idx == 1),
                ):
                    st.markdown(src_item["content"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "thinking_label": thinking_label,
            "thinking_details": thinking_details,
            "sources": sources_payload,
        })


# ----------------------------------------------------
# TAB 2: Big Tech Comparison
# ----------------------------------------------------
with tab_compare:
    st.markdown("### FY2023 Big Tech Financial Benchmark")
    st.markdown("Standardized metrics extracted directly from SEC Form 10-K Consolidated Statements:")

    comparison_data = [
        {"Metric": "Total Revenue / Net Sales", "Apple (AAPL)": "$383,285M", "Microsoft (MSFT)": "$211,915M", "Alphabet (GOOGL)": "$307,394M", "Leader": "Apple ($383,285M)"},
        {"Metric": "Gross Margin (%)", "Apple (AAPL)": "44.13%", "Microsoft (MSFT)": "68.92%", "Alphabet (GOOGL)": "56.62%", "Leader": "Microsoft (68.92%)"},
        {"Metric": "Operating Income", "Apple (AAPL)": "$114,301M", "Microsoft (MSFT)": "$88,523M", "Alphabet (GOOGL)": "$84,278M", "Leader": "Apple ($114,301M)"},
        {"Metric": "Operating Margin (%)", "Apple (AAPL)": "29.82%", "Microsoft (MSFT)": "41.77%", "Alphabet (GOOGL)": "27.42%", "Leader": "Microsoft (41.77%)"},
        {"Metric": "Net Income", "Apple (AAPL)": "$96,995M", "Microsoft (MSFT)": "$72,361M", "Alphabet (GOOGL)": "$73,796M", "Leader": "Apple ($96,995M)"},
        {"Metric": "Net Profit Margin (%)", "Apple (AAPL)": "25.31%", "Microsoft (MSFT)": "34.15%", "Alphabet (GOOGL)": "24.01%", "Leader": "Microsoft (34.15%)"},
        {"Metric": "Operating Cash Flow", "Apple (AAPL)": "$110,543M", "Microsoft (MSFT)": "$87,582M", "Alphabet (GOOGL)": "$101,746M", "Leader": "Apple ($110,543M)"},
        {"Metric": "Capital Expenditures (CapEx)", "Apple (AAPL)": "$10,959M", "Microsoft (MSFT)": "$28,107M", "Alphabet (GOOGL)": "$32,251M", "Leader": "Alphabet ($32,251M)"},
        {"Metric": "Free Cash Flow (FCF)", "Apple (AAPL)": "$99,584M", "Microsoft (MSFT)": "$59,475M", "Alphabet (GOOGL)": "$69,495M", "Leader": "Apple ($99,584M)"},
        {"Metric": "Diluted EPS", "Apple (AAPL)": "$6.13", "Microsoft (MSFT)": "$9.68", "Alphabet (GOOGL)": "$5.80", "Leader": "Microsoft ($9.68)"},
    ]

    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, width="stretch", hide_index=True)

    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Highest Operating Margin", "Microsoft", "41.77%")
    col_c2.metric("Highest Free Cash Flow", "Apple", "$99,584 Million")
    col_c3.metric("Highest CapEx Investment", "Alphabet", "$32,251 Million")


# ----------------------------------------------------
# TAB 3: RAGAS Automated Evaluation
# ----------------------------------------------------
with tab_eval:
    st.markdown("### RAGAS Quality Benchmark Matrix")
    st.markdown("Quantitative comparison of **Naive Vector Search** vs **Hybrid RRF** vs **Hybrid + Cross-Encoder Reranker** on Gold IBM FinQA samples.")

    if st.button("Run Automated Benchmark Evaluation", key="run_eval_btn"):
        with st.spinner("Evaluating retrieval strategies across benchmark queries..."):
            st.session_state.eval_results = run_evaluation(num_samples=5, provider="offline")

    if st.session_state.eval_results is not None:
        eval_data = st.session_state.eval_results

        chart_rows = [
            {
                "Strategy": "Naive Vector RAG",
                "Faithfulness": eval_data["naive"]["avg_faithfulness"],
                "Response Relevancy": eval_data["naive"]["avg_response_relevancy"],
                "Context Precision": eval_data["naive"]["avg_context_precision"],
            },
            {
                "Strategy": "Hybrid (BM25 + Dense RRF)",
                "Faithfulness": eval_data["hybrid"]["avg_faithfulness"],
                "Response Relevancy": eval_data["hybrid"]["avg_response_relevancy"],
                "Context Precision": eval_data["hybrid"]["avg_context_precision"],
            },
            {
                "Strategy": "Hybrid + Cross-Encoder Reranker",
                "Faithfulness": eval_data["hybrid_rerank"]["avg_faithfulness"],
                "Response Relevancy": eval_data["hybrid_rerank"]["avg_response_relevancy"],
                "Context Precision": eval_data["hybrid_rerank"]["avg_context_precision"],
            },
        ]
        df_eval = pd.DataFrame(chart_rows).set_index("Strategy")

        st.dataframe(df_eval, width="stretch")
        st.bar_chart(df_eval)

        f_delta = eval_data["hybrid_rerank"]["avg_faithfulness"] - eval_data["naive"]["avg_faithfulness"]
        cp_delta = eval_data["hybrid_rerank"]["avg_context_precision"] - eval_data["naive"]["avg_context_precision"]
        r_delta = eval_data["hybrid_rerank"]["avg_response_relevancy"] - eval_data["naive"]["avg_response_relevancy"]

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Reranker Faithfulness", f"{eval_data['hybrid_rerank']['avg_faithfulness']:.2%}", f"{f_delta:+.1%} vs Naive")
        col_m2.metric("Reranker Context Precision", f"{eval_data['hybrid_rerank']['avg_context_precision']:.2%}", f"{cp_delta:+.1%} vs Naive")
        col_m3.metric("Reranker Relevancy", f"{eval_data['hybrid_rerank']['avg_response_relevancy']:.2%}", f"{r_delta:+.1%} vs Naive")
    else:
        st.info("Click the button above to run the automated benchmark evaluation across retrieval strategies.")


# ----------------------------------------------------
# TAB 4: Indexed Financial Corpus
# ----------------------------------------------------
with tab_corpus:
    st.markdown("### Indexed Financial Filings")
    st.markdown(f"Total indexed documents: **{len(engine.raw_documents)}** across **{len(available_tickers)}** tickers (`{', '.join(available_tickers)}`).")

    for idx, doc in enumerate(engine.raw_documents, 1):
        meta = doc.metadata
        ticker = meta.get("ticker", "N/A")
        title = meta.get("title", f"Document {idx}")
        company = meta.get("company", "N/A")
        section = meta.get("section", "N/A")
        fy = meta.get("fiscal_year", "N/A")
        with st.expander(f"[{ticker}] {title}"):
            st.markdown(
                f'<span class="ticker-badge">{ticker}</span> **Company:** {company} | **Section:** {section} | **Fiscal Year:** {fy}',
                unsafe_allow_html=True,
            )
            st.markdown(doc.page_content)
