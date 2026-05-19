"""
Gap Analysis Agent.

Identifies information gaps beyond the missing document audit:
  - Topics mentioned but not explained
  - Numbers referenced but not substantiated
  - Future commitments with no baseline
  - Related-party relationships not fully disclosed
"""
from models.router import call_llm_with_context
from retrieval.hybrid import HybridRetriever
from config import COMPANIES

SYSTEM_PROMPT = """You are a forensic due-diligence analyst reviewing corporate documents 
for information gaps and unanswered questions.

Your job is to identify what the analyst CANNOT answer from the available documents,
and why it matters.

Focus on:
1. Quantitative gaps — numbers referenced but not provided
2. Narrative gaps — topics mentioned but not explained
3. Comparative gaps — claims made without benchmarks or context
4. Future commitment gaps — plans announced without baselines
5. Related-party gaps — transactions or relationships not fully disclosed

For each gap:
- What is missing?
- Where was it referenced (if anywhere)? [Source N]
- Why does it matter to the investment thesis?
- Criticality: CRITICAL / HIGH / MEDIUM

Be specific. "Financial statements" is not a gap — "FY2024 capex breakdown by project" is."""

GAP_QUERIES = {
    "SOC": [
        "capital expenditure budget breakdown project cost estimate",
        "production target volumes guidance barrels per day",
        "reserve report third party certification engineer",
        "offtake agreement sales contracts counterparty",
        "insurance coverage liability protection",
        "environmental remediation cost estimate accrual",
        "related party transactions insider dealings",
        "credit facility covenant terms conditions",
        "regulatory permit application status timeline",
        "management incentive compensation alignment",
    ],
    "AKSO": [
        "order backlog breakdown segment customer concentration",
        "EBITDA margin bridge cost structure breakdown",
        "capital employed return ROCE by segment",
        "pension liability defined benefit obligation",
        "related party Aker group transactions",
        "joint venture minority interest exposure",
        "working capital cycle days outstanding",
        "energy transition revenue percentage target",
        "executive compensation long-term incentive plan",
        "off-balance sheet commitments operating leases",
    ],
}

GAP_PROMPT = """Review the following documents for {company} ({ticker}) and identify 
critical information gaps.

For each gap you identify, provide:
1. **Gap**: What specific information is missing?
2. **Reference**: Where (if anywhere) is this topic touched on? [Source N or "Not mentioned"]
3. **Investment Relevance**: Why does this gap matter to the investment thesis?
4. **Criticality**: CRITICAL / HIGH / MEDIUM
5. **How to Fill**: Where would an analyst typically find this? (e.g., "Bloomberg terminal", 
   "direct company contact", "third-party data provider")

Also address:
- What can the analyst confidently conclude from the available documents?
- What conclusions would be speculation given current document coverage?

End with a **Confidence Assessment**: Given the available documents, what is the overall 
confidence level in any investment analysis? HIGH / MEDIUM / LOW — and why."""


def run_gap_analysis(ticker: str, retriever: HybridRetriever, audit_result: dict = None) -> dict:
    """
    Run gap analysis for a company.
    Optionally incorporates the document audit result for context.
    """
    company_info = COMPANIES.get(ticker, {})
    queries = GAP_QUERIES.get(ticker, GAP_QUERIES["SOC"])

    print(f"  Running gap analysis for {ticker}...")

    chunks = retriever.retrieve_multi(queries, top_k_per_query=4)

    # Add audit context to prompt if available
    audit_context = ""
    if audit_result:
        missing = audit_result.get("missing", [])
        if missing:
            missing_str = ", ".join(
                f"{m['type']} ({m['criticality']})" for m in missing[:8]
            )
            audit_context = f"\n\nNote: The following document types were NOT successfully ingested: {missing_str}. Factor this into your gap assessment."

    prompt = GAP_PROMPT.format(
        company=company_info.get("name", ticker),
        ticker=ticker,
    ) + audit_context

    if not chunks:
        return {
            "ticker": ticker,
            "analysis": "[UNCERTAIN — No source documents available for gap analysis]",
            "chunks_used": 0,
        }

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
