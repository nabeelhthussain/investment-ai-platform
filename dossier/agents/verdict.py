"""
Verdict Agent.

Produces the executive verdict: PROCEED / STOP / NEEDS_MORE_INFO.
Explicitly adversarial — defaults toward caution.
"""
from models.router import call_llm_with_context
from retrieval.hybrid import HybridRetriever
from config import COMPANIES

SYSTEM_PROMPT = """You are a senior investment analyst producing an executive verdict 
for a research committee.

Your default position is skepticism. Your job is to find reasons NOT to proceed 
with research, not to summarize the bull case.

The verdict must be one of:
- PROCEED: Sufficient evidence to justify deeper research. Risk/reward appears 
  compelling enough to merit analyst time. Key risks are identified and manageable.
- STOP: Risk/reward is unfavorable, thesis has critical flaws, or risks are 
  unquantifiable. Do not invest further research time.
- NEEDS_MORE_INFO: Thesis is plausible but critical information gaps prevent 
  a confident verdict. Specify exactly what information is needed.

You must support every element of your verdict with specific citations [Source N].
If evidence is insufficient, say so — do not fabricate a verdict."""

VERDICT_QUERIES = {
    "SOC": [
        "business model revenue production oil Santa Ynez restart",
        "financial position cash debt equity capital structure",
        "valuation NAV reserves barrels oil equivalent",
        "management team CEO CFO track record",
        "competitive advantage moat differentiator",
        "catalyst timeline upcoming events milestones",
        "risk reward investment case thesis",
        "shareholder returns dividends buybacks equity",
        "total addressable market opportunity",
        "going concern audit opinion liquidity",
    ],
    "AKSO": [
        "revenue order intake backlog growth trajectory",
        "EBITDA margin profitability improvement",
        "return on equity capital employed ROCE",
        "dividend policy shareholder returns",
        "balance sheet net debt leverage ratio",
        "competitive position market share oilfield services",
        "strategy energy transition electrification",
        "management credibility delivery on targets",
        "valuation EV EBITDA peer comparison",
        "order book visibility revenue predictability",
    ],
}

VERDICT_PROMPT = """Based on the following documents, produce an investment research verdict for 
{company} ({ticker}).

Structure your verdict as follows:

## Executive Verdict
**Decision: [PROCEED / STOP / NEEDS_MORE_INFO]**
**Confidence: [HIGH / MEDIUM / LOW]**

One paragraph summarizing the core rationale. Be direct and specific.

## Bull Case (what would have to be true)
List 3-4 conditions that would make this a compelling investment.
Each must be grounded in evidence from documents [Source N].

## Bear Case (why this could fail)
List 3-4 specific failure modes.
Each must cite evidence [Source N].

## What Would Change This Verdict
List 3-4 specific, observable conditions that would flip the verdict.
These should be concrete and measurable, not vague.

## Key Information Gaps
What critical information is missing that would materially affect this verdict?
Rate each gap: CRITICAL / HIGH / MEDIUM.

---
Remember: your default is skepticism. Burden of proof is on the bull case."""


def run_verdict(ticker: str, retriever: HybridRetriever) -> dict:
    """
    Run the verdict agent for a company.
    """
    company_info = COMPANIES.get(ticker, {})
    queries = VERDICT_QUERIES.get(ticker, VERDICT_QUERIES["SOC"])

    print(f"  Running verdict analysis for {ticker}...")

    chunks = retriever.retrieve_multi(queries, top_k_per_query=5)

    if not chunks:
        return {
            "ticker": ticker,
            "verdict": "NEEDS_MORE_INFO",
            "confidence": "LOW",
            "analysis": "[UNCERTAIN — Insufficient documents available for verdict]",
            "chunks_used": 0,
        }

    prompt = VERDICT_PROMPT.format(
        company=company_info.get("name", ticker),
        ticker=ticker,
    )

    analysis = call_llm_with_context(
        query=prompt,
        context_chunks=chunks,
        system=SYSTEM_PROMPT,
        max_tokens=3000,
    )

    # Extract verdict from response
    verdict = _parse_verdict(analysis)

    return {
        "ticker": ticker,
        "verdict": verdict["decision"],
        "confidence": verdict["confidence"],
        "analysis": analysis,
        "chunks_used": len(chunks),
        "source_docs": list({c.get("doc_type") for c in chunks}),
    }


def _parse_verdict(text: str) -> dict:
    """Extract structured verdict from LLM response."""
    import re
    decision = "NEEDS_MORE_INFO"
    confidence = "LOW"

    for line in text.split("\n"):
        if "Decision:" in line or "**Decision:" in line:
            if "PROCEED" in line.upper():
                decision = "PROCEED"
            elif "STOP" in line.upper():
                decision = "STOP"
            elif "NEEDS_MORE_INFO" in line.upper() or "NEEDS MORE" in line.upper():
                decision = "NEEDS_MORE_INFO"

        if "Confidence:" in line or "**Confidence:" in line:
            if "HIGH" in line.upper():
                confidence = "HIGH"
            elif "MEDIUM" in line.upper():
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

    return {"decision": decision, "confidence": confidence}
