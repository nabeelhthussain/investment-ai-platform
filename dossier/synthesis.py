"""
Dossier Synthesis.

Takes all agent outputs and produces the final formatted
Deep Research Dossier as a markdown document.
"""
from datetime import datetime
from config import COMPANIES


VERDICT_EMOJI = {
    "PROCEED": "🟢",
    "STOP": "🔴",
    "NEEDS_MORE_INFO": "🟡",
}

CONFIDENCE_LABEL = {
    "HIGH": "High confidence",
    "MEDIUM": "Medium confidence",
    "LOW": "Low confidence — significant information gaps",
}


def synthesize_dossier(
    ticker: str,
    audit_result: dict,
    bear_result: dict,
    contradiction_result: dict,
    verdict_result: dict,
    gap_result: dict,
) -> str:
    """
    Assemble the full Deep Research Dossier as markdown.
    All agent analysis is included verbatim — no secondary summarization
    that could introduce additional hallucination risk.
    """
    company = COMPANIES.get(ticker, {})
    verdict = verdict_result.get("verdict", "NEEDS_MORE_INFO")
    confidence = verdict_result.get("confidence", "LOW")

    lines = []

    # ── Header ──────────────────────────────────────────────────────────
    lines.append(f"# Deep Research Dossier: {company.get('name', ticker)}")
    lines.append(f"**Ticker:** {ticker} | **Exchange:** {company.get('exchange', '')} | "
                 f"**Sector:** {company.get('sector', '')}")
    lines.append(f"**Report date:** {datetime.now().strftime('%B %d, %Y')}")
    lines.append(f"**Reporting standard:** {company.get('reporting_standard', 'N/A')} | "
                 f"**Currency:** {company.get('currency', 'N/A')}")
    lines.append("")
    lines.append("> **Purpose:** This dossier is designed to stress-test the investment thesis, "
                 "not summarize management's view. All claims are grounded in ingested source "
                 "documents. Ungrounded inferences are explicitly flagged as "
                 "`[UNCERTAIN — not grounded in documents]`.")
    lines.append("")

    # ── Executive Verdict Banner ─────────────────────────────────────────
    emoji = VERDICT_EMOJI.get(verdict, "🟡")
    lines.append("---")
    lines.append("")
    lines.append(f"## {emoji} Executive Verdict: {verdict}")
    lines.append(f"*{CONFIDENCE_LABEL.get(confidence, confidence)}*")
    lines.append("")

    # ── Document Coverage Summary ────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Document Coverage")
    lines.append("")
    coverage = audit_result.get("coverage_pct", 0)
    total_words = audit_result.get("total_words", 0)
    total_docs = audit_result.get("total_documents_ingested", 0)
    lines.append(f"- **Documents ingested:** {total_docs}")
    lines.append(f"- **Total words processed:** {total_words:,}")
    lines.append(f"- **Document type coverage:** {coverage}% of expected types")
    lines.append(f"- **Critical gaps:** {audit_result.get('critical_missing_count', 0)}")
    lines.append(f"- **High-priority gaps:** {audit_result.get('high_missing_count', 0)}")
    lines.append("")
    lines.append(f"*{audit_result.get('analyst_summary', '')}*")
    lines.append("")

    # ── Verdict Detail ────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Verdict Analysis")
    lines.append("")
    lines.append(verdict_result.get("analysis", "[No verdict analysis available]"))
    lines.append("")

    # ── Bear Case ─────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Bear Case — Adversarial Risk Analysis")
    lines.append("")
    lines.append(f"*Based on {bear_result.get('chunks_used', 0)} retrieved source passages "
                 f"from: {', '.join(bear_result.get('source_docs', []))}*")
    lines.append("")
    lines.append(bear_result.get("analysis", "[No bear case analysis available]"))
    lines.append("")

    # ── Contradictions ────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Management Contradictions & Tone Shifts")
    lines.append("")
    date_range = contradiction_result.get("date_range", "")
    if date_range:
        lines.append(f"*Analysis covers documents from: {date_range}*")
        lines.append("")
    lines.append(contradiction_result.get("analysis", "[No contradiction analysis available]"))
    lines.append("")

    # ── Gap Analysis ──────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Information Gaps & Analytical Limitations")
    lines.append("")
    lines.append(gap_result.get("analysis", "[No gap analysis available]"))
    lines.append("")

    # ── Missing Documents ─────────────────────────────────────────────────
    missing = audit_result.get("missing", [])
    if missing:
        lines.append("---")
        lines.append("")
        lines.append("## Missing Documents Detail")
        lines.append("")
        lines.append("| Document Type | Criticality | Status | Analyst Impact |")
        lines.append("|---|---|---|---|")
        for m in missing:
            impact = m.get("analyst_impact", m.get("reason", ""))[:120]
            if len(m.get("analyst_impact", "")) > 120:
                impact += "..."
            lines.append(
                f"| {m['type']} | **{m['criticality']}** | {m['status']} | {impact} |"
            )
        lines.append("")

    # ── Present Documents ─────────────────────────────────────────────────
    present = audit_result.get("present", [])
    if present:
        lines.append("---")
        lines.append("")
        lines.append("## Successfully Ingested Documents")
        lines.append("")
        lines.append("| Document Type | Count | Total Words | Most Recent |")
        lines.append("|---|---|---|---|")
        for p in present:
            most_recent = p.get("dates", ["?"])[0]
            lines.append(
                f"| {p['type']} | {p['count']} | {p.get('total_words', 0):,} | {most_recent} |"
            )
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("- **Retrieval:** Hybrid BM25 + TF-IDF with Reciprocal Rank Fusion (RRF)")
    lines.append("- **LLM:** Anthropic Claude (claude-sonnet-4-5 for analysis, claude-haiku-4-5-20251001 for extraction)")
    lines.append("- **Citation policy:** All factual claims cite source documents. Ungrounded inferences flagged `[UNCERTAIN]`")
    lines.append("- **Adversarial framing:** Agents are explicitly instructed to stress-test, not summarize")
    lines.append("")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} by Investment AI Platform v1.0*")

    return "\n".join(lines)
