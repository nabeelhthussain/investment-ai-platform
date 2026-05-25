"""
Phase 3: Analyst Brief Generator.

Produces a concise one-page brief synthesizing phases 1 and 2.
Uses the fast model since this is compression, not analysis.
"""
from datetime import datetime
from models.router import call_llm
from config import FAST_MODEL, COMPANIES


BRIEF_SYSTEM = """You are producing a one-page investment research brief for a senior analyst.

Rules:
- Be direct and specific. No filler language.
- Every claim must trace to the source analysis provided.
- Flag uncertainty explicitly — do not smooth over gaps.
- The brief should take 2-3 minutes to read.
- Prioritize actionable information over background."""

BRIEF_PROMPT = """Based on the Phase 1 ingestion report and Phase 2 dossier below, 
produce a concise one-page analyst brief for {company} ({ticker}).

## Required Sections (stay within these word limits):

**1. Company Snapshot** (~50 words)
What the company does, market cap/exchange, primary business driver.

**2. Document Coverage** (~40 words)
What was available, what was missing, confidence level in the analysis.

**3. Verdict** (~30 words)
The one-line verdict (PROCEED / STOP / NEEDS_MORE_INFO) and core rationale.

**4. Top 3 Bear Risks** (~120 words total)
The three most material downside risks. One sentence each + one sentence of evidence.
Each must include a source citation.

**5. Key Contradictions or Concerns** (~80 words)
Any management credibility issues, tone shifts, or concerning patterns found.
If none found, state that clearly.

**6. What Would Change This Verdict** (~60 words)
2-3 specific, observable conditions that would flip the verdict.

**7. Analyst Priority Actions** (~50 words)  
What should the analyst do first? Specific next steps.

---

## Phase 1 Ingestion Summary:
{ingestion_summary}

## Phase 2 Dossier:
{dossier_text}
"""


def generate_analyst_brief(
    ticker: str,
    audit_result: dict,
    dossier_result: dict,
    company_config: dict = None,
) -> str:
    """
    Generate the one-page analyst brief.
    """
    company = company_config or COMPANIES.get(ticker, {})
    now = datetime.now().strftime("%B %d, %Y")

    # Build condensed ingestion summary
    ingestion_summary = _build_ingestion_summary(ticker, audit_result)

    # Truncate dossier if very long (keep most important parts)
    dossier_text = dossier_result.get("dossier_text", "")
    if len(dossier_text) > 12000:
        # Keep verdict, bear case, contradictions — trim gaps section
        dossier_text = dossier_text[:12000] + "\n\n[...dossier truncated for brief generation...]"

    prompt = BRIEF_PROMPT.format(
        company=company.get("name", ticker),
        ticker=ticker,
        ingestion_summary=ingestion_summary,
        dossier_text=dossier_text,
    )

    brief_body = call_llm(
        prompt=prompt,
        system=BRIEF_SYSTEM,
        model=FAST_MODEL,
        max_tokens=1500,
        temperature=0.1,
    )

    # Wrap with header/footer
    verdict = dossier_result.get("verdict", "NEEDS_MORE_INFO")
    confidence = dossier_result.get("confidence", "LOW")

    VERDICT_COLORS = {
        "PROCEED": "🟢",
        "STOP": "🔴",
        "NEEDS_MORE_INFO": "🟡",
    }
    emoji = VERDICT_COLORS.get(verdict, "🟡")

    header = f"""# Analyst Brief: {company.get('name', ticker)}
**{ticker}** | {company.get('exchange', '')} | {company.get('sector', '')} | {now}

{emoji} **{verdict}** — {confidence} confidence

---

"""
    footer = f"""
---
*One-page brief generated from Phase 1 + Phase 2 analysis. All source citations in the full dossier.*  
*Generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | Investment AI Platform v1.0*
"""

    return header + brief_body + footer


def _build_ingestion_summary(ticker: str, audit_result: dict) -> str:
    present = [p["type"] for p in audit_result.get("present", [])]
    missing_critical = [m["type"] for m in audit_result.get("missing", []) if m["criticality"] == "CRITICAL"]
    missing_high = [m["type"] for m in audit_result.get("missing", []) if m["criticality"] == "HIGH"]

    parts = [
        f"Coverage: {audit_result.get('coverage_pct', 0)}% of expected document types.",
        f"Available: {', '.join(present) if present else 'none'}.",
    ]
    if missing_critical:
        parts.append(f"CRITICAL missing: {', '.join(missing_critical)}.")
    if missing_high:
        parts.append(f"High-priority missing: {', '.join(missing_high)}.")

    return " ".join(parts)
