"""
FinSearch-Pro Generation & Multi-Provider LLM Chain.

Supports:
- Auto-detection via Universal AI API Key (`AIza...` -> Gemini, `gsk_...` -> Groq, `sk-or-...` -> OpenRouter, `sk-...` -> OpenAI)
- Gemini: `gemini-2.5-flash-lite`, `gemini-2.5-flash`
- Groq: `llama-3.1-8b-instant`
- OpenAI / OpenRouter: `gpt-4.1-nano`, `gpt-4o-mini`
- Dynamic Context-Driven Offline Synthesizer (zero-hardcoding, parses retrieved tables & prose)

Features:
- Enforces strict retrieved-context ground truth preservation.
- Negative financial guardrails: concise out-of-scope refusal when evidence is absent or speculative.
- Formats dynamic arithmetic/comparisons from extracted tabular figures.
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from retriever import FinSearchRetrieverEngine, get_finsearch_retriever


OUT_OF_SCOPE_REFUSAL = (
    "Out of scope: The indexed financial filings do not contain evidence to answer this question."
)

FINANCIAL_SYSTEM_PROMPT = """You are FinSearch-Pro, an elite SEC Form 10-K Corporate Intelligence and Financial Analyst assistant.
Analyze the provided financial filing excerpts with extreme precision and adhere to strict financial reporting principles:

CRITICAL INSTRUCTIONS:
1. ONLY use figures, metrics, calculations, and disclosures present in the provided Context.
2. PRESERVE multi-column table numbers with exact dollar amounts (e.g., $383,285 million).
3. If an explicit calculation is requested (such as Free Cash Flow, CapEx change, Operating Margin, or EPS):
   - State the exact formula used (e.g., Operating Cash Flow - CapEx = Free Cash Flow).
   - Show the step-by-step arithmetic with source figures from the tables.
   - Express percentages to two decimal places (e.g., 29.82%).
4. FINANCIAL NEGATIVE GUARDRAILS:
   - If the requested data, ticker, or fiscal period is NOT present in the provided context, clearly declare: "Out of scope: The indexed financial filings do not contain evidence to answer this question."
   - Do NOT speculate on future stock prices, make investment buy/sell recommendations, or extrapolate non-reported numbers.

Context:
{context}

Question:
{question}

Structured Financial Answer (include Source Citation & Arithmetic if applicable):"""


def detect_provider_from_key(api_key: Optional[str]) -> str:
    """Detects the LLM provider from an API key prefix."""
    if not api_key or not api_key.strip():
        return "offline"
    key = api_key.strip()
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("sk-or-"):
        return "openrouter"
    if key.startswith("sk-"):
        return "openai"
    return "openai"


def _resolve_auto_provider(api_key: Optional[str]) -> Tuple[str, Optional[str]]:
    """Resolves provider and key when provider='auto'."""
    if api_key and api_key.strip():
        return detect_provider_from_key(api_key), api_key.strip()
    for env_name, prov in [
        ("GEMINI_API_KEY", "gemini"),
        ("GOOGLE_API_KEY", "gemini"),
        ("GROQ_API_KEY", "groq"),
        ("OPENROUTER_API_KEY", "openrouter"),
        ("OPENAI_API_KEY", "openai"),
    ]:
        val = os.getenv(env_name)
        if val and val.strip():
            return prov, val.strip()
    return "offline", None


class DeterministicOfflineLLM:
    """
    Dynamic, context-driven offline financial answer synthesizer.
    Parses retrieved Markdown tables and prose chunks from the prompt context
    without any hardcoded company or metric answers.
    """

    _STOPWORDS = {
        "what", "was", "were", "the", "in", "for", "of", "and", "or", "how",
        "did", "it", "to", "a", "an", "is", "are", "with", "from", "by", "on",
        "at", "which", "company", "had", "between", "produce", "produced",
        "generate", "generated", "much", "compare", "total", "year", "fiscal",
        "among", "does", "its", "their", "this", "that", "these", "those",
        "can", "you", "tell", "me", "give", "show", "list", "explain", "about",
        "highest", "lowest", "most", "least", "achieved", "achieve", "over",
        "percentage", "breakdown", "amount", "used", "formula", "step",
    }

    _IGNORED_PROPER_NOUNS = {
        "What", "Which", "How", "When", "Where", "Why", "Who", "Compare",
        "Show", "List", "Calculate", "Tell", "Give", "Explain", "Does", "Did",
        "Was", "Were", "Are", "Is", "The", "In", "For", "Of", "And", "Or",
        "Among", "Between", "Should", "Could", "Would", "Can", "Will",
        "Summarize", "Analyze", "Analyse", "Evaluate", "Report", "Check",
        "Find", "Provide", "Compute", "Determine", "Highlight", "Outline",
        "Detail", "Overview", "Please", "Break", "Describe", "State",
        "Revenue", "Revenues", "Sales", "Net", "Income", "Operating", "Margin",
        "Margins", "Gross", "Profit", "Cash", "Flow", "Free", "Capital",
        "Expenditures", "Expenditure", "CapEx", "EPS", "Diluted", "Basic",
        "Total", "Segment", "Cloud", "Search", "Services", "Products",
        "Product", "Assets", "Liabilities", "Equity", "Balance", "Sheet",
        "Research", "Development", "Expenses", "Growth", "Debt", "Tax",
        "Taxes", "Shares", "Dividend", "Dividends", "Quarter", "Annual",
        "Filing", "Filings", "Statement", "Statements", "Consolidated",
        "Segments", "Data", "Metrics", "Numbers", "Figures", "Results",
        "Performance", "Breakdown", "Summary",
        "FY", "FY2021", "FY2022", "FY2023", "FY2024", "FY2025", "FY21", "FY22",
        "FY23", "FY24", "FY25", "Q1", "Q2", "Q3", "Q4", "USD", "GAAP", "SEC",
        "10-K", "MD&A", "Item", "YoY", "AI", "Stock", "Price", "Prediction",
    }

    _SYNONYMS = {
        "capex": ["capex", "capital expenditure", "capital expenditures", "property, plant and equipment", "property and equipment"],
        "fcf": ["fcf", "free cash flow"],
        "eps": ["eps", "earnings per share", "diluted"],
        "revenue": ["revenue", "revenues", "net sales", "sales"],
        "revenues": ["revenue", "revenues", "net sales", "sales"],
        "sales": ["sales", "net sales", "revenue", "revenues"],
        "margin": ["margin", "margins", "operating margin", "gross margin"],
        "margins": ["margin", "margins", "operating margin", "gross margin"],
        "ad": ["ad", "ads", "advertising"],
        "ads": ["ad", "ads", "advertising"],
        "advertising": ["advertising", "ads", "ad"],
        "growth": ["growth", "increase", "increased", "change", "vs"],
    }

    _SPECULATIVE_PHRASES = (
        "should i buy",
        "should i sell",
        "buy or sell",
        "stock price prediction",
        "price target",
        "will the stock",
        "investment advice",
        "recommend buying",
        "predict",
    )

    def __init__(self, model_name: str = "offline-deterministic-finsearch"):
        self.model_name = model_name

    def _expand_query_terms(self, question: str) -> List[str]:
        raw_tokens = re.findall(r"[a-zA-Z0-9\$\.\%\-]+", question.lower())
        terms: List[str] = []
        for tok in raw_tokens:
            clean = tok.strip(".,?!'\"()")
            if not clean or clean in self._STOPWORDS:
                continue
            if len(clean) == 1 and not clean.isdigit():
                continue
            if clean not in terms:
                terms.append(clean)
            for syn in self._SYNONYMS.get(clean, []):
                if syn not in terms:
                    terms.append(syn)
        return terms

    def _check_out_of_scope(self, question: str, context_text: str) -> bool:
        if not context_text or not context_text.strip():
            return True

        q_lower = question.lower()
        c_lower = context_text.lower()

        # Refuse speculative / investment advice
        if any(phrase in q_lower for phrase in self._SPECULATIVE_PHRASES):
            return True

        # Check if any explicit proper noun / ticker in the question is absent from context
        # Strip possessive 's
        cleaned_q = re.sub(r"'s\b", "", question)
        candidate_entities = re.findall(r"\b[A-Z][a-zA-Z0-9&\-]{1,}\b", cleaned_q)
        for ent in candidate_entities:
            if ent in self._IGNORED_PROPER_NOUNS or re.match(r"^FY\d+$", ent, re.IGNORECASE):
                continue
            if ent.lower() not in c_lower:
                return True

        # Check fiscal years explicitly mentioned in the question (e.g., FY2025, 2025)
        q_years = re.findall(r"\b(?:FY\s*)?(20\d{2})\b", question, re.IGNORECASE)
        for yr in q_years:
            if yr not in c_lower and not re.search(rf"\bfy\s*{yr[-2:]}\b", c_lower):
                return True

        # Ensure at least one meaningful query term appears in context
        terms = self._expand_query_terms(question)
        core_terms = [t for t in terms if not re.match(r"^(fy)?20\d{2}$", t)]
        if core_terms and not any(t in c_lower for t in core_terms):
            return True

        return False

    @staticmethod
    def _parse_doc_blocks(context_text: str) -> List[Dict[str, Any]]:
        parts = re.split(r"(?=--- \[Document \d+)", context_text)
        blocks: List[Dict[str, Any]] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            header_match = re.match(
                r"--- \[Document \d+ \| Ticker: (.*?) \| Section: (.*?) \| FY: (.*?) \| Reranker Relevance Score: (.*?)\] ---\s*(?:Title:\s*(.*?)\n)?(.*)",
                part,
                re.DOTALL,
            )
            if header_match:
                ticker, section, fy, score_str, title, body = header_match.groups()
                try:
                    score = float(score_str)
                except ValueError:
                    score = 0.0
                blocks.append({
                    "ticker": ticker.strip(),
                    "section": section.strip(),
                    "fiscal_year": fy.strip(),
                    "score": score,
                    "title": (title or section).strip(),
                    "body": body.strip(),
                })
            else:
                blocks.append({
                    "ticker": "DOC",
                    "section": "Retrieved Context",
                    "fiscal_year": "N/A",
                    "score": 0.5,
                    "title": "Retrieved Financial Context",
                    "body": part,
                })
        return blocks

    @staticmethod
    def _extract_tables_and_prose(body: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        lines = body.splitlines()
        tables: List[Dict[str, Any]] = []
        prose_lines: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*[:\-]+", lines[i + 1].strip()):
                header_cells = [c.strip() for c in line.strip("|").split("|")]
                sep_line = lines[i + 1].strip()
                rows: List[List[str]] = []
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    row_cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    rows.append(row_cells)
                    j += 1
                tables.append({
                    "headers": header_cells,
                    "separator": sep_line,
                    "rows": rows,
                })
                i = j
                continue
            elif line and not line.startswith("#") and not line.startswith("---"):
                prose_lines.append(line)
            i += 1

        # Split prose into sentences
        prose_text = " ".join(prose_lines)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose_text) if s.strip()]
        return tables, sentences

    @staticmethod
    def _parse_number(cell: str) -> Optional[float]:
        if not cell:
            return None
        cleaned = cell.replace("$", "").replace(",", "").replace("%", "").strip()
        is_neg = False
        if cleaned.startswith("(") and cleaned.endswith(")"):
            is_neg = True
            cleaned = cleaned[1:-1].strip()
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if not m:
            return None
        val = float(m.group(0))
        return -val if is_neg else val

    def _score_text(self, text: str, terms: List[str]) -> float:
        t_lower = text.lower()
        score = 0.0
        for term in terms:
            if term in t_lower:
                score += 2.0 if len(term) > 3 else 1.0
        if re.search(r"\d", text):
            score += 0.5
        return score

    def _compute_dynamic_insights(
        self,
        question: str,
        matched_tables: List[Tuple[Dict[str, Any], List[List[str]]]]
    ) -> List[str]:
        q_lower = question.lower()
        insights: List[str] = []

        is_compare_superlative = any(w in q_lower for w in ("highest", "lowest", "largest", "most", "compare", "which company"))
        is_yoy = any(w in q_lower for w in ("change", "growth", "vs", "compared to", "increase", "decrease"))

        for tbl, rows in matched_tables:
            headers = tbl["headers"]
            # Check YoY / multi-column numeric difference on top matched row
            is_time_table = any(
                re.search(r"(?:20\d{2}|\bfy\b|\bq[1-4]\b|year|period)", h, re.IGNORECASE)
                for h in headers[1:]
            )
            if is_yoy and is_time_table and len(headers) >= 3 and rows:
                top_row = rows[0]
                nums = [(h, self._parse_number(c)) for h, c in zip(headers[1:], top_row[1:])]
                valid_nums = [(h, n) for h, n in nums if n is not None]
                if len(valid_nums) >= 2:
                    h1, v1 = valid_nums[0]
                    h2, v2 = valid_nums[1]
                    y1_m = re.search(r"(20\d{2})", h1)
                    y2_m = re.search(r"(20\d{2})", h2)
                    # Ensure newer period minus older period regardless of column order
                    if y1_m and y2_m and int(y1_m.group(1)) < int(y2_m.group(1)):
                        new_h, new_v, old_h, old_v = h2, v2, h1, v1
                    else:
                        new_h, new_v, old_h, old_v = h1, v1, h2, v2
                    if old_v != 0:
                        diff = new_v - old_v
                        pct = (diff / abs(old_v)) * 100.0
                        metric_name = top_row[0].replace("**", "")
                        insights.append(
                            f"**Period Change ({metric_name}):** {new_h} ({new_v:,.2f}) vs {old_h} ({old_v:,.2f}) -> "
                            f"Difference: `{diff:+,.2f}` (`{pct:+.2f}%`)"
                        )

            # Check cross-column superlative (e.g. Company columns in comparison table)
            if is_compare_superlative and len(headers) >= 3 and rows:
                top_row = rows[0]
                # Skip a trailing "Leader" / "Winner" summary column if present
                col_pairs = [
                    (h, self._parse_number(c))
                    for h, c in zip(headers[1:], top_row[1:])
                    if "leader" not in h.lower() and "winner" not in h.lower()
                ]
                valid_cols = [(h, n) for h, n in col_pairs if n is not None]
                if len(valid_cols) >= 2:
                    want_min = "lowest" in q_lower or "least" in q_lower
                    best_h, best_v = (
                        min(valid_cols, key=lambda x: x[1])
                        if want_min
                        else max(valid_cols, key=lambda x: x[1])
                    )
                    metric_name = top_row[0].replace("**", "")
                    suffix = "%" if "%" in str(top_row[1]) else ""
                    insights.append(
                        f"**{'Lowest' if want_min else 'Highest'} {metric_name}:** **{best_h}** (`{best_v:,.2f}{suffix}`)"
                    )

        return insights

    def invoke(self, prompt: str) -> str:
        question_match = re.search(
            r"Question:\s*(.*?)(?:\n\s*Structured Financial Answer|\Z)",
            prompt,
            re.DOTALL | re.IGNORECASE,
        )
        question_text = question_match.group(1).strip() if question_match else prompt

        context_match = re.search(
            r"Context:\s*(.*?)(?:\n\s*Question:|\Z)",
            prompt,
            re.DOTALL | re.IGNORECASE,
        )
        context_text = context_match.group(1).strip() if context_match else ""

        if self._check_out_of_scope(question_text, context_text):
            return OUT_OF_SCOPE_REFUSAL

        terms = self._expand_query_terms(question_text)
        blocks = self._parse_doc_blocks(context_text)

        # Rank document blocks by relevance to query terms + reranker score
        ranked_blocks = []
        for blk in blocks:
            title_meta = f"{blk['ticker']} {blk['title']} {blk['section']}"
            overlap = self._score_text(title_meta + " " + blk["body"], terms)
            combined_rank = overlap + blk["score"] * 2.0
            ranked_blocks.append((combined_rank, blk))
        ranked_blocks.sort(key=lambda x: x[0], reverse=True)

        output_sections: List[str] = []
        matched_tables_for_calc: List[Tuple[Dict[str, Any], List[List[str]]]] = []
        citations: List[str] = []

        for _, blk in ranked_blocks[:2]:
            tables, sentences = self._extract_tables_and_prose(blk["body"])
            citations.append(f"[{blk['ticker']}] {blk['title']} ({blk['section']})")

            block_parts: List[str] = []
            for tbl in tables:
                scored_rows: List[Tuple[float, List[str]]] = []
                for row in tbl["rows"]:
                    row_str = " ".join(row)
                    r_score = self._score_text(row_str, terms)
                    scored_rows.append((r_score, row))

                # Sort rows by score descending, preserving original order on ties
                scored_rows.sort(key=lambda x: x[0], reverse=True)
                # Keep matching rows (or top 5 if table is small)
                matching_rows = [r for s, r in scored_rows if s > 0.5][:6]
                if not matching_rows:
                    matching_rows = tbl["rows"][:5]

                if matching_rows:
                    matched_tables_for_calc.append((tbl, matching_rows))
                    header_line = "| " + " | ".join(tbl["headers"]) + " |"
                    sep_line = tbl["separator"]
                    row_lines = ["| " + " | ".join(r) + " |" for r in matching_rows]
                    block_parts.append("\n".join([header_line, sep_line] + row_lines))

            # Top matching prose sentences
            scored_sents = [(self._score_text(s, terms), s) for s in sentences]
            scored_sents.sort(key=lambda x: x[0], reverse=True)
            top_sents = [s for s_score, s in scored_sents if s_score > 0.5][:3]
            if not top_sents and sentences and not block_parts:
                top_sents = sentences[:2]
            if top_sents:
                block_parts.append("\n".join(f"- {s}" for s in top_sents))

            if block_parts:
                output_sections.append(
                    f"### [{blk['ticker']}] {blk['title']}\n" + "\n\n".join(block_parts)
                )

        if not output_sections:
            return OUT_OF_SCOPE_REFUSAL

        insights = self._compute_dynamic_insights(question_text, matched_tables_for_calc)
        if insights:
            output_sections.append("### Dynamic Calculation & Comparison\n" + "\n".join(f"- {ins}" for ins in insights))

        output_sections.append(f"**Source:** {'; '.join(citations)}")
        return "\n\n".join(output_sections)


class FinSearchChain:
    """
    Unified FinSearch-Pro Chain orchestrating:
    - Multi-provider LLMs (Auto, Gemini, Groq, OpenAI/OpenRouter, Offline)
    - Prompt assembly with multi-column tabular context
    - Document citation and reranker score reporting
    """

    def __init__(
        self,
        provider: str = "offline",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        retriever_engine: Optional[FinSearchRetrieverEngine] = None
    ):
        self.requested_provider = provider.lower()
        if self.requested_provider == "auto":
            resolved_provider, resolved_key = _resolve_auto_provider(api_key)
            self.provider = resolved_provider
            api_key = resolved_key
        else:
            self.provider = self.requested_provider

        self.retriever = retriever_engine or get_finsearch_retriever()
        self.prompt = PromptTemplate(
            template=FINANCIAL_SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )
        self.llm = self._init_llm(provider=self.provider, model_name=model_name, api_key=api_key)

    def _init_llm(self, provider: str, model_name: Optional[str], api_key: Optional[str]):
        """Initializes the requested LLM provider with graceful fallbacks."""
        if provider == "gemini":
            selected_model = model_name or "gemini-2.5-flash-lite"
            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                return DeterministicOfflineLLM()
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=selected_model,
                    google_api_key=key,
                    temperature=0.0
                )
            except Exception as e:
                print(f"Failed to load ChatGoogleGenerativeAI ({e}). Falling back to Offline LLM.")
                return DeterministicOfflineLLM()

        elif provider == "groq":
            selected_model = model_name or "llama-3.1-8b-instant"
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                return DeterministicOfflineLLM()
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model=selected_model,
                    groq_api_key=key,
                    temperature=0.0
                )
            except Exception as e:
                print(f"Failed to load ChatGroq ({e}). Falling back to Offline LLM.")
                return DeterministicOfflineLLM()

        elif provider in ("openai", "openrouter"):
            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if not key:
                return DeterministicOfflineLLM()
            is_openrouter = provider == "openrouter" or key.strip().startswith("sk-or-")
            selected_model = model_name or ("openai/gpt-4.1-nano" if is_openrouter else "gpt-4.1-nano")
            base_url = "https://openrouter.ai/api/v1" if is_openrouter else None
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=selected_model,
                    api_key=key,
                    base_url=base_url,
                    temperature=0.0
                )
            except Exception as e:
                print(f"Failed to load ChatOpenAI ({e}). Falling back to Offline LLM.")
                return DeterministicOfflineLLM()

        return DeterministicOfflineLLM()

    def format_context(self, scored_docs: List[Tuple[Document, float]]) -> str:
        """Formats intact parent chunks with metadata headers for the prompt."""
        formatted_sections = []
        for i, (doc, score) in enumerate(scored_docs, 1):
            title = doc.metadata.get("title", "SEC 10-K Disclosure")
            ticker = doc.metadata.get("ticker", "N/A")
            sec_section = doc.metadata.get("section", "N/A")
            fy = doc.metadata.get("fiscal_year", "N/A")

            section_header = (
                f"--- [Document {i} | Ticker: {ticker} | Section: {sec_section} | "
                f"FY: {fy} | Reranker Relevance Score: {score:.4f}] ---\nTitle: {title}\n"
            )
            formatted_sections.append(section_header + doc.page_content)
        return "\n\n".join(formatted_sections)

    def answer_query(
        self,
        query: str,
        retrieval_strategy: str = "hybrid_rerank",
        top_k: int = 3,
        ticker_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes query answering given retrieval strategy:
        - 'naive': Naive Dense Vector Retrieval
        - 'hybrid': BM25 + Dense RRF Ensemble
        - 'hybrid_rerank': Hybrid + Cross-Encoder Rerank
        """
        if retrieval_strategy == "naive":
            raw_docs = self.retriever.naive_vector_retrieve(
                query, top_k=top_k, ticker_filter=ticker_filter
            )
            scored_docs = [(d, 0.5) for d in raw_docs]
        elif retrieval_strategy == "hybrid":
            raw_docs = self.retriever.hybrid_retrieve(
                query, top_k=top_k, ticker_filter=ticker_filter
            )
            scored_docs = [(d, 0.75) for d in raw_docs]
        else:  # hybrid_rerank (default)
            scored_docs = self.retriever.retrieve_and_rerank(
                query, top_k_retrieve=6, top_k_rerank=top_k, ticker_filter=ticker_filter
            )

        context_text = self.format_context(scored_docs)
        formatted_prompt = self.prompt.format(context=context_text, question=query)

        if isinstance(self.llm, DeterministicOfflineLLM):
            response_text = self.llm.invoke(formatted_prompt)
        else:
            try:
                res = self.llm.invoke(formatted_prompt)
                response_text = res.content if hasattr(res, "content") else str(res)
            except Exception as e:
                print(f"Error during LLM invoke ({e}). Falling back to Offline LLM.")
                response_text = DeterministicOfflineLLM().invoke(formatted_prompt)

        return {
            "query": query,
            "answer": response_text,
            "retrieval_strategy": retrieval_strategy,
            "ticker_filter": ticker_filter or "ALL",
            "provider": self.provider,
            "scored_documents": scored_docs,
            "context_used": context_text
        }
