"""
Gap Analysis Agent.

Two-stage retrieval:
  1. Orientation pass — agent generates queries targeting specific data gaps
  2. Targeted retrieval — surfaces what's present so gaps can be identified
  3. Analysis — what's missing, why it matters, confidence assessment

Identifies gaps WITHIN ingested documents (distinct from the missing
document audit which identifies gaps in document COVERAGE).
"""
from models.router import call_llm_with_context, call_llm_fast
from retrieval.hybrid import HybridRetriever

# ── Universal gap queries — always included ───────────────────────────────────
UNIVERSAL_QUERIES = [
    "capital expenditure budget breakdown cost estimate detail",
    "related party transactions insider dealings conflicts",
    "management compensation incentive alignment equity",
    "off balance sheet commitments contingent liabilities",
    "reserve certification third party independent engineer",
]

ORIENTATION_SYSTEM = """You are a forensic due-diligence analyst identifying 
information gaps in corporate disclosures. You specialise in finding what 
companies reference but do not explain, and what they should disclose but don't."""

ORIENTATION_PROMPT = """You are analyzing {company} ({ticker}), a {sector} company,
for information gaps and unanswered analytical questions.

Here is a sample of their public filings:

{sample_text}

Based on what you see, generate exactly 8 retrieval queries that would surface 
content related to the most important analytical gaps for this company.

Think about: quantitative data that's mentioned but not provided, 
future commitments without baselines, related-party relationships, 
valuation inputs that are absent, and operating metrics that would 
be standard for this sector but aren't disclosed.

Return exactly 8 queries, one per line, no numbering, no explanation."""

GAP_SYSTEM = """You are a forensic due-diligence analyst reviewing corporate 
documents for information gaps and unanswered questions.

Find what the analyst CANNOT determine from available documents, and why it matters.

Focus on:
1. Quantitative gaps — numbers referenced but not provided
2. Narrative gaps — topics mentioned but not explained
3. Comparative gaps — claims made without benchmarks
4. Future commitment gaps — plans without baselines or budgets
5. Related-party gaps — transactions not fully disclosed

For each gap:
- What is missing specifically?
- Where is it referenced (if anywhere)? [Source N or "Not mentioned"]
- Why does it matter to the investment thesis?
- Criticality: CRITICAL / HIGH / MEDIUM
- How to obtain it

Be specific. "Financial statements" is not a gap.
"FY2025 capex breakdown by project" is a gap."""

GAP_PROMPT = """Review the following documents for {company} ({ticker}),
a {sector} company, and identify critical information gaps.

For each gap:
1. **Gap**: What specific information is missing?
2. **Reference**: Where is it touched on? [Source N or "Not mentioned"]
3. **Investment relevance**: Why does this matter to the thesis?
4. **Criticality**: CRITICAL / HIGH / MEDIUM
5. **How to fill**: Where would an analyst typically find this?

Also state:
- What the analyst CAN confidently conclude from available documents
- What conclusions would be speculation given current coverage

End with a **Confidence Assessment**: HIGH / MEDIUM / LOW — and why.{audit_context}"""


def _generate_queries(
    ticker: str,
    company_config: dict,
    retriever: HybridRetriever,
) -> list[str]:
    """Generate gap-focused retrieval queries from corpus orientation."""
    name   = company_config.get("name", ticker)
    sector = company_config.get("sector", "")

    orientation_chunks = retriever.retrieve(
        "financial data metrics operating statistics valuation assumptions",
        top_k=20,
    )

    if not orientation_chunks:
        return UNIVERSAL_QUERIES

    sample_parts = []
    for chunk in orientation_chunks[:8]:
        label = f"[{chunk.get('doc_type','')} | {chunk.get('date','')[:7]}]"
        sample_parts.append(f"{label}\n{chunk['text'][:300]}")

    sample_text = "\n\n---\n\n".join(sample_parts)

    prompt = ORIENTATION_PROMPT.format(
        company     = name,
        ticker      = ticker,
        sector      = sector,
        sample_text = sample_text,
    )

    response = call_llm_fast(
        prompt     = prompt,
        system     = ORIENTATION_SYSTEM,
        max_tokens = 300,
    )

    generated = [
        q.strip().lstrip("-•·123456789. ")
        for q in response.strip().split("\n")
        if q.strip() and len(q.strip()) > 10
    ][:8]

    print(f"  Agent generated {len(generated)} gap analysis queries")

    all_queries = generated + [
        q for q in UNIVERSAL_QUERIES
        if not any(word in " ".join(generated).lower()
                   for word in q.split()[:2])
    ]

    return all_queries


def run_gap_analysis(
    ticker: str,
    retriever: HybridRetriever,
    audit_result: dict = None,
    company_config: dict = None,
) -> dict:
    """
    Run gap analysis for a company.

    Stage 1: Agent generates gap-focused retrieval queries.
    Stage 2: Targeted retrieval.
    Stage 3: Claude identifies what's missing and why it matters.
    """
    company_config = company_config or {}
    name   = company_config.get("name", ticker)
    sector = company_config.get("sector", "")

    print(f"  Running gap analysis for {ticker}...")

    if not retriever.chunks:
        return {
            "ticker":      ticker,
            "analysis":    "[UNCERTAIN — No source documents available]",
            "chunks_used": 0,
            "queries_used": [],
        }

    # ── Stage 1: Generate queries ─────────────────────────────────────────────
    print(f"  Generating gap analysis queries from corpus orientation...")
    queries = _generate_queries(ticker, company_config, retriever)

    # ── Stage 2: Targeted retrieval ───────────────────────────────────────────
    chunks = retriever.retrieve_multi(queries, top_k_per_query=4)

    if not chunks:
        return {
            "ticker":       ticker,
            "analysis":     "[UNCERTAIN — No relevant chunks retrieved]",
            "chunks_used":  0,
            "queries_used": queries,
        }

    # ── Add audit context if available ───────────────────────────────────────
    audit_context = ""
    if audit_result:
        missing = audit_result.get("missing", [])
        if missing:
            missing_str = ", ".join(
                f"{m['type']} ({m['criticality']})" for m in missing[:8]
            )
            audit_context = (
                f"\n\nNote: The following document types were NOT successfully "
                f"ingested: {missing_str}. Factor this into your gap assessment."
            )

    # ── Stage 3: Analysis ─────────────────────────────────────────────────────
    prompt = GAP_PROMPT.format(
        company       = name,
        ticker        = ticker,
        sector        = sector,
        audit_context = audit_context,
    )

    analysis = call_llm_with_context(
        query          = prompt,
        context_chunks = chunks,
        system         = GAP_SYSTEM,
        max_tokens     = 3000,
    )

    return {
        "ticker":       ticker,
        "analysis":     analysis,
        "chunks_used":  len(chunks),
        "queries_used": queries,
        "source_docs":  list({c.get("doc_type") for c in chunks}),
    }
