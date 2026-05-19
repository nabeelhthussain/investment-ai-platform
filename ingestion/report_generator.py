"""
Phase 1 output: Ingestion Report.

Produces a human-readable markdown document covering:
- What was ingested and its quality
- Missing document analysis
- What the analyst can expect to find in the documents
- AI-generated document summaries per doc type
"""
from datetime import datetime
from models.router import call_llm_with_context, call_llm_fast
from config import COMPANIES


def _summarize_doc_type(ticker: str, doc_type: str, chunks: list[dict]) -> str:
    """Generate a brief summary of what's in a given doc type's chunks."""
    if not chunks:
        return "[No content available]"

    # Use a small sample of chunks for the summary
    sample = chunks[:6]
    prompt = (
        f"Briefly summarize what an analyst would find in these {doc_type} documents "
        f"for {ticker}. Focus on: key financial figures mentioned, major topics covered, "
        f"time period covered, and quality/completeness of information. "
        f"Keep it to 3-4 sentences. Be specific about numbers and facts you see."
    )
    return call_llm_with_context(
        query=prompt,
        context_chunks=sample,
        system="You are summarizing document content for an investment analyst. Be concise and factual.",
        max_tokens=400,
    )


def generate_ingestion_report(
    ticker: str,
    documents: list[dict],
    chunks: list[dict],
    audit_result: dict,
) -> str:
    """
    Generate the full Phase 1 Ingestion Report as markdown.
    """
    company = COMPANIES.get(ticker, {})
    now = datetime.now().strftime("%B %d, %Y")

    lines = []

    # ── Header ──────────────────────────────────────────────────────────
    lines.append(f"# Phase 1 Ingestion Report: {company.get('name', ticker)}")
    lines.append(f"**Ticker:** {ticker} | **Exchange:** {company.get('exchange', '')} | "
                 f"**Generated:** {now}")
    lines.append("")
    lines.append("> This report describes what was ingested, the quality of available materials, "
                 "and what an analyst can and cannot learn from the current document corpus.")
    lines.append("")

    # ── Executive Summary ────────────────────────────────────────────────
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(audit_result.get("analyst_summary", ""))
    lines.append("")

    total_docs = audit_result.get("total_documents_ingested", 0)
    total_words = audit_result.get("total_words", 0)
    total_chunks = len(chunks)
    coverage = audit_result.get("coverage_pct", 0)

    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Documents ingested | {total_docs} |")
    lines.append(f"| Total words | {total_words:,} |")
    lines.append(f"| Retrieval chunks | {total_chunks:,} |")
    lines.append(f"| Document type coverage | {coverage}% |")
    lines.append(f"| Critical document gaps | {audit_result.get('critical_missing_count', 0)} |")
    lines.append(f"| High-priority gaps | {audit_result.get('high_missing_count', 0)} |")
    lines.append("")

    # ── What the analyst can find ────────────────────────────────────────
    lines.append("## What an Analyst Will Find in These Documents")
    lines.append("")

    # Group chunks by doc_type for summaries
    by_type: dict[str, list[dict]] = {}
    for chunk in chunks:
        dt = chunk.get("doc_type", "unknown")
        by_type.setdefault(dt, []).append(chunk)

    present = audit_result.get("present", [])
    if present:
        for doc_info in present:
            dt = doc_info["type"]
            dt_chunks = by_type.get(dt, [])
            count = doc_info.get("count", 0)
            words = doc_info.get("total_words", 0)
            dates = doc_info.get("dates", [])
            date_str = f"{dates[-1]} to {dates[0]}" if len(dates) > 1 else (dates[0] if dates else "unknown")

            lines.append(f"### {dt}")
            lines.append(f"*{count} document(s) | {words:,} words | Period: {date_str}*")
            lines.append("")

            summary = _summarize_doc_type(ticker, dt, dt_chunks)
            lines.append(summary)
            lines.append("")
    else:
        lines.append("*No documents were successfully ingested.*")
        lines.append("")

    # ── Missing Document Report ──────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Missing Context Report")
    lines.append("")
    lines.append("The following expected document types were not available in accessible public sources. "
                 "This section explains what each missing document would have contained and "
                 "the analytical risk of proceeding without it.")
    lines.append("")

    missing = audit_result.get("missing", [])
    if missing:
        for m in missing:
            crit = m["criticality"]
            status = m["status"]
            impact = m.get("analyst_impact", "No specific impact assessment available.")

            crit_badge = f"**[{crit}]**"
            lines.append(f"### {m['type']} — {crit_badge}")
            lines.append(f"*{m['description']} | Status: {status}*")
            lines.append("")
            lines.append(impact)
            lines.append("")
    else:
        lines.append("*All expected document types were successfully ingested.*")
        lines.append("")

    # ── Data Quality Notes ───────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Data Quality Notes")
    lines.append("")

    failed = [d for d in documents if d.get("fetch_failed")]
    if failed:
        lines.append(f"**{len(failed)} document(s) could not be retrieved:**")
        for d in failed:
            lines.append(f"- {d['doc_type']} ({d.get('date', '?')}): {d.get('source', 'unknown URL')}")
        lines.append("")
        lines.append("These documents were identified as expected but could not be downloaded. "
                     "They may be behind paywalls, require authentication, or be unavailable in "
                     "the public domain. The analyst should obtain these directly.")
        lines.append("")

    if ticker == "AKSO":
        lines.append("**Language note:** Aker Solutions files in both Norwegian and English. "
                     "Norwegian-language documents were not processed in this run. "
                     "Material differences between Norwegian and English versions cannot be verified.")
        lines.append("")

    # ── Ingestion Log ────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Full Ingestion Log")
    lines.append("")
    lines.append("| Document Type | Date | Words | Source | Status |")
    lines.append("|---|---|---|---|---|")
    for doc in sorted(documents, key=lambda d: d.get("date", ""), reverse=True):
        status = "❌ FAILED" if doc.get("fetch_failed") else "✅ OK"
        source = doc.get("source", "")
        if len(source) > 60:
            source = source[:57] + "..."
        lines.append(
            f"| {doc.get('doc_type','?')} | {doc.get('date','?')} | "
            f"{doc.get('word_count',0):,} | {source} | {status} |"
        )
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*")

    return "\n".join(lines)
