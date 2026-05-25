"""
Verdict Agent.

Reads the outputs of the bear case, contradiction, and gap analysis agents
and produces an informed executive verdict. Does NOT do independent retrieval —
it reasons over what the other three agents already found.

This mirrors how a research committee chair operates: reads three independent
memos, then makes a final call.
"""
from models.router import call_llm
from config import PRIMARY_MODEL

SYSTEM_PROMPT = """You are a senior portfolio manager making a final research verdict.

You have received three independent memos from your analysts:
  1. Bear case memo — adversarial risk analysis
  2. Contradiction memo — management credibility assessment  
  3. Gap analysis memo — information gaps and blind spots

Your job is to synthesize these into a final verdict. You are NOT doing new research.
You are making a judgment based on what your analysts found.

Default to skepticism. The burden of proof is on the bull case.
Be direct. Do not hedge unnecessarily."""

VERDICT_PROMPT = """Based on the three analyst memos below, produce an executive verdict
for {company} ({ticker}).

---
## Memo 1: Bear Case Analysis
{bear_analysis}

---
## Memo 2: Contradiction & Tone Analysis
{contradiction_analysis}

---
## Memo 3: Information Gap Analysis
{gap_analysis}

---

Now produce your verdict in this exact structure:

## Executive Verdict
**Decision: [PROCEED / STOP / NEEDS_MORE_INFO]**
**Confidence: [HIGH / MEDIUM / LOW]**

One paragraph. Be direct. Reference specific findings from the memos above.

## Why This Decision
3-5 bullet points. Each must reference a specific finding from one of the memos.
Do not introduce new claims not in the memos.

## Bull Case (what would have to be true)
3 conditions that would make this a compelling investment.
Ground each in evidence or absence of evidence from the memos.

## What Would Change This Verdict
3 specific, observable, measurable conditions that would flip the decision.
These must be concrete — not "if risks improve" but "if X specific thing happens".

## Key Risks Not Fully Resolved
The top 2-3 risks from the bear case that remain open even if the verdict is PROCEED.

---
Decision criteria:
- PROCEED: risk/reward compelling enough to justify deeper research. Key risks identified and manageable.
- STOP: risk/reward unfavorable, thesis has critical flaws, or risks unquantifiable.
- NEEDS_MORE_INFO: thesis plausible but critical gaps prevent a confident verdict."""


def run_verdict(
    ticker: str,
    company_config: dict,
    bear_result: dict,
    contradiction_result: dict,
    gap_result: dict,
) -> dict:
    """
    Run the verdict agent.
    Reads bear, contradiction, and gap outputs — no independent retrieval.
    """
    company_config = company_config or {}
    name = company_config.get("name", ticker)

    print(f"  Running verdict analysis for {ticker} (reading agent memos)...")

    # Truncate each memo if very long to stay within context
    def _truncate(text: str, max_chars: int = 3000) -> str:
        if not text:
            return "[No analysis available]"
        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    bear_text         = _truncate(bear_result.get("analysis", ""))
    contradiction_text = _truncate(contradiction_result.get("analysis", ""))
    gap_text          = _truncate(gap_result.get("analysis", ""))

    prompt = VERDICT_PROMPT.format(
        company               = name,
        ticker                = ticker,
        bear_analysis         = bear_text,
        contradiction_analysis = contradiction_text,
        gap_analysis          = gap_text,
    )

    analysis = call_llm(
        prompt      = prompt,
        system      = SYSTEM_PROMPT,
        max_tokens  = 2000,
        temperature = 0.1,
    )

    verdict    = _parse_verdict(analysis)
    confidence = _parse_confidence(analysis)

    return {
        "ticker":      ticker,
        "verdict":     verdict,
        "confidence":  confidence,
        "analysis":    analysis,
        "inputs_used": {
            "bear_chunks":         bear_result.get("chunks_used", 0),
            "contradiction_chunks": contradiction_result.get("chunks_used", 0),
            "gap_chunks":          gap_result.get("chunks_used", 0),
        },
    }


def _parse_verdict(text: str) -> str:
    for line in text.split("\n"):
        upper = line.upper()
        if "DECISION:" in upper or "**DECISION:" in upper:
            if "PROCEED" in upper:
                return "PROCEED"
            if "STOP" in upper:
                return "STOP"
            if "NEEDS_MORE_INFO" in upper or "NEEDS MORE" in upper:
                return "NEEDS_MORE_INFO"
    return "NEEDS_MORE_INFO"


def _parse_confidence(text: str) -> str:
    for line in text.split("\n"):
        upper = line.upper()
        if "CONFIDENCE:" in upper or "**CONFIDENCE:" in upper:
            if "HIGH" in upper:
                return "HIGH"
            if "MEDIUM" in upper:
                return "MEDIUM"
            return "LOW"
    return "LOW"
