"""
Bear Case Agent.

Purpose: Stress-test the investment thesis. Explicitly designed to find
reasons NOT to invest, not to summarize management's bull case.
All claims must cite source chunks.
"""
from models.router import call_llm_with_context
from retrieval.hybrid import HybridRetriever
from config import COMPANIES

SYSTEM_PROMPT = """You are a skeptical, adversarial investment analyst. Your job is to 
stress-test the investment thesis and identify every reason why this investment could fail.

You are NOT summarizing management's view. You are explicitly looking for:
- Risks that management mentions briefly or downplays
- Assumptions in the bull case that could prove wrong
- Operational, financial, regulatory, or competitive risks
- Red flags in the financials or disclosures
- What bears would say about this company

Rules:
1. Every claim must be followed by [Source N] citing the retrieved document.
2. If you cannot find supporting evidence, write [UNCERTAIN — not grounded in documents].
3. Do NOT fabricate risks not evidenced in the documents.
4. Be specific — vague risks like "macroeconomic headwinds" are useless unless they appear in the documents.
5. Prioritize non-consensus risks — things the market may not be pricing in."""

BEAR_QUERIES = {
    "SOC": [
        "regulatory risk California offshore oil permit Santa Barbara SYU restart",
        "plug and abandonment liability cost overruns Sable Offshore",
        "debt financing liquidity risk credit facility covenant",
        "operational risk pipeline restart timeline delays",
        "litigation legal proceedings environmental liability",
        "management compensation insider selling stock ownership",
        "production decline rates reserve depletion",
        "political risk California oil offshore opposition",
        "cash burn rate going concern working capital",
        "contractor risk construction execution restart",
    ],
    "AKSO": [
        "order backlog risk cancellation customer concentration",
        "energy transition risk oilfield services demand decline",
        "Norwegian labor cost inflation offshore services margins",
        "working capital cash flow project execution risk",
        "debt leverage net debt financing risk",
        "renewables offshore wind competition margin pressure",
        "currency risk NOK USD exposure",
        "key contract loss customer concentration Equinor",
        "litigation regulatory compliance risk",
        "cost inflation supply chain materials escalation",
    ],
}

BEAR_PROMPT_TEMPLATE = """Analyze the following retrieved documents and identify the TOP bear case risks for {company}.

Focus specifically on:
1. **Existential or near-term risks** — what could cause permanent capital loss or force a wind-down?
2. **Timeline/execution risks** — what specific milestones could slip and what is the evidence?
3. **Financial risks** — leverage, liquidity, cash burn, covenant risk
4. **Regulatory/legal risks** — specific proceedings, environmental liabilities, permit dependencies
5. **Management credibility risks** — has management been overly optimistic? Any guidance misses?
6. **Non-consensus risks** — risks the market may be underpricing

For each risk:
- State the risk clearly in one sentence
- Provide specific evidence from the source documents (quote key numbers or language)
- Assess severity: HIGH / MEDIUM / LOW
- Cite the source with [Source N]

If a risk cannot be grounded in the provided documents, flag it as [UNCERTAIN].

Company: {company} ({ticker})
Exchange: {exchange}
Sector: {sector}"""


def run_bear_case(ticker: str, retriever: HybridRetriever) -> dict:
    """
    Run the bear case agent for a given company.
    Returns structured dict with risks and analysis text.
    """
    company_info = COMPANIES.get(ticker, {})
    queries = BEAR_QUERIES.get(ticker, BEAR_QUERIES["SOC"])

    print(f"  Running bear case analysis for {ticker}...")

    # Retrieve relevant chunks across all bear case queries
    chunks = retriever.retrieve_multi(queries, top_k_per_query=5)

    if not chunks:
        return {
            "ticker": ticker,
            "analysis": "[UNCERTAIN — No source documents available for bear case analysis]",
            "chunks_used": 0,
            "error": "No chunks retrieved",
        }

    prompt = BEAR_PROMPT_TEMPLATE.format(
        company=company_info.get("name", ticker),
        ticker=ticker,
        exchange=company_info.get("exchange", ""),
        sector=company_info.get("sector", ""),
    )

    analysis = call_llm_with_context(
        query=prompt,
        context_chunks=chunks,
        system=SYSTEM_PROMPT,
        max_tokens=3000,
    )

    return {
        "ticker": ticker,
        "analysis": analysis,
        "chunks_used": len(chunks),
        "source_docs": list({c.get("doc_type") for c in chunks}),
    }
