"""
Contradiction Detector Agent — enhanced with FinBERT quantitative tone scoring.
"""
from models.router import call_llm_with_context, call_llm
from retrieval.hybrid import HybridRetriever
from config import COMPANIES

try:
    from ingestion.tone_analyzer import analyze_tone_trends, format_tone_section
    TONE_AVAILABLE = True
except ImportError:
    TONE_AVAILABLE = False

SYSTEM_PROMPT = """You are a forensic analyst specializing in detecting management 
credibility issues in corporate disclosures.

Your task is to find:
1. Direct contradictions — statements that logically conflict across time periods
2. Guidance misses — where actual results diverged significantly from prior guidance
3. Tone shifts — changes from optimistic to cautious (or vice versa) on key topics
4. Framing changes — same facts presented very differently across periods
5. Omissions — topics prominently discussed in one period that quietly disappear later

Rules:
- Be specific. Quote or closely paraphrase the relevant language from documents.
- Always cite sources with [Source N].
- Distinguish between: CONTRADICTION (factual conflict), TONE_SHIFT (sentiment change), 
  GUIDANCE_MISS (forecast vs actual divergence), OMISSION (topic disappears).
- Do NOT flag normal updates or expected revisions as contradictions.
- If you cannot find evidence of contradictions, say so clearly — do not fabricate."""

CONTRADICTION_QUERIES = {
    "SOC": [
        "restart timeline schedule date production first oil Santa Ynez",
        "capital expenditure budget cost estimate project",
        "regulatory approval permit status timeline",
        "cash liquidity working capital financial position",
        "production target guidance forecast",
        "management guidance outlook forward looking",
        "risk factors material risks changed updated",
        "litigation legal proceedings settlement",
    ],
    "AKSO": [
        "order intake backlog guidance forecast outlook",
        "revenue margin EBITDA target guidance",
        "renewable energy offshore wind strategy commitment",
        "capital allocation dividend buyback policy",
        "headcount workforce restructuring",
        "debt leverage financial position target",
        "project execution delivery milestones",
        "strategic priorities focus areas",
    ],
}

CONTRADICTION_PROMPT = """Analyze the following documents from {company} ({ticker}) 
spanning multiple time periods.

Identify any of the following:

**1. Direct Contradictions**
Find statements where management says one thing in one document and something 
materially different in another. Quote or closely paraphrase both sides.

**2. Guidance Misses**  
Find cases where management gave a forecast or target in an earlier document 
that was subsequently missed. What was the original guidance? What was the outcome?

**3. Tone Shifts**
Find topics where the tone changed significantly — e.g., from "strong demand environment" 
to "challenging market conditions" — without a clear external event explaining the shift.

**4. Strategic Omissions**
Find major strategic themes from earlier documents that quietly disappeared 
from later communications.

For each finding, provide:
- Type: CONTRADICTION / GUIDANCE_MISS / TONE_SHIFT / OMISSION
- Earlier statement (with [Source N])
- Later statement or absence (with [Source N])  
- Significance: what should an analyst read into this?

If you find NO credible contradictions or shifts, state that clearly."""


def run_contradiction_detection(ticker: str, retriever: HybridRetriever) -> dict:
    """
    Run contradiction detection for a company.
    Returns analysis dict.
    """
    company_info = COMPANIES.get(ticker, {})
    queries = CONTRADICTION_QUERIES.get(ticker, CONTRADICTION_QUERIES["SOC"])

    print(f"  Running contradiction detection for {ticker}...")
    chunks = retriever.retrieve_multi(queries, top_k_per_query=4)

    if not chunks:
        return {"ticker": ticker, "chunks_used": 0, "tone_section": "",
                "analysis": "[UNCERTAIN — Insufficient documents available]"}

    # FinBERT quantitative layer
    tone_result = None
    tone_section = ""
    shift_summary = ""
    if TONE_AVAILABLE:
        print("  Running FinBERT tone analysis...")
        all_chunks = retriever.retrieve_multi(queries, top_k_per_query=8)
        tone_result = analyze_tone_trends(all_chunks)
        tone_section = format_tone_section(tone_result)
        shifts = tone_result.get("significant_shifts", [])
        if shifts:
            shift_summary = "\n\nFinBERT pre-screening detected these tone shifts to investigate:\n"
            for s in shifts[:5]:
                shift_summary += (f"- {s['topic'].replace('_',' ').title()}: became "
                                  f"{'MORE NEGATIVE' if s['neg_delta'] > 0 else 'MORE POSITIVE'} "
                                  f"from {s['from_period']} to {s['to_period']} "
                                  f"(Δneg={s['neg_delta']:+.3f}, {s['magnitude']})\n")
    else:
        print("  Skipping FinBERT (install with: pip install transformers torch)")

    chunks_sorted = sorted(chunks, key=lambda x: x.get("date", ""))
    prompt = CONTRADICTION_PROMPT.format(
        company=company_info.get("name", ticker), ticker=ticker,
    ) + shift_summary

    analysis = call_llm_with_context(
        query=prompt, context_chunks=chunks_sorted,
        system=SYSTEM_PROMPT, max_tokens=3000,
    )

    return {"ticker": ticker, "analysis": analysis, "tone_section": tone_section,
            "tone_result": tone_result, "chunks_used": len(chunks),
            "date_range": _get_date_range(chunks),
            "source_docs": list({c.get("doc_type") for c in chunks})}


def _get_date_range(chunks: list[dict]) -> str:
    dates = sorted([c.get("date", "") for c in chunks if c.get("date")])
    if not dates:
        return "unknown"
    return f"{dates[0]} to {dates[-1]}"
