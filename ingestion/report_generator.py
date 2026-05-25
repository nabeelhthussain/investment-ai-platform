"""
Phase 1 output: Ingestion Report.

Answers two questions precisely:
  1. What can the analyst expect to find in these documents if she read them herself?
  2. What important materials are missing, why do they matter, and how critical is each gap?
"""
from datetime import datetime
from models.router import call_llm_with_context, call_llm


# ── Document-level "what would you find" prompt ───────────────────────────────

DOC_WALKTHROUGH_SYSTEM = """You are a senior investment analyst briefing a colleague
on what she will find in a set of company documents.

Your job is to give her a precise, document-by-document walkthrough so she can decide
what to read in full versus rely on your summary.

Be specific. Name actual figures, dates, section titles, and page references where visible.
Do not be vague. "The filing covers financial performance" is useless.
"The FY2025 10-K covers the period ending December 31 2025, audited by Ernst & Young.
Section 7A quantifies interest rate risk on the $921.6M term loan at 15% PIK.
MD&A (pages 40-68) focuses almost entirely on the SYU restart timeline..." is useful.

Flag any quality issues: missing sections, redacted content, boilerplate-heavy filings,
or documents that appear incomplete."""

DOC_WALKTHROUGH_PROMPT = """You have been given chunks from {count} {doc_type} document(s)
for {company} ({ticker}), covering the period {date_range}.

Write a structured briefing covering:

**1. What these documents contain**
Walk through the major sections and what each covers. Be specific about:
- Key financial figures present (revenue, EBITDA, debt, cash — with actual numbers)
- Major topics and themes covered
- Time period and fiscal years represented
- Which sections are substantive vs boilerplate

**2. What an analyst would learn from reading them**
What are the 3-5 most important facts, trends, or disclosures in these documents?
What questions do these documents answer?

**3. Document quality assessment**
- Are these documents complete and well-structured?
- Is the financial data detailed or summary-level?
- Are there any notable omissions, redactions, or quality issues?
- How much can an analyst rely on these documents for investment analysis?

Be direct and specific. Use actual numbers from the documents."""


# ── Missing document explanation prompt ───────────────────────────────────────

MISSING_DOC_SYSTEM = """You are a senior research analyst explaining to a junior analyst
why a missing document matters for investment analysis.

Be specific to this company and this investment situation.
Explain: what the document would typically contain, why it matters for THIS company's
specific thesis, what analytical blind spots the analyst has without it, and what
decisions she cannot make reliably without it.

Do not be generic. "This document contains financial information" is not acceptable."""

MISSING_DOC_PROMPT = """The analyst is researching {company} ({ticker}), a {sector} company.
The following document is not available: {doc_type} ({description}).

Based on what you know about this company and this document type, explain:

**What this document would contain**
Describe specifically what a {doc_type} for {company} would typically include —
sections, data, disclosures, and certifications specific to this company type.

**Why it matters for this specific investment thesis**
{company} is a {sector} company. Why is this document particularly important
given this company's specific situation, risks, and the questions an analyst
would be trying to answer?

**What the analyst cannot determine without it**
List 3-4 specific analytical questions that CANNOT be answered reliably
without this document. Be precise.

**How critical is this gap**
Criticality: {criticality}
Explain what this criticality rating means in practice — can the analyst
proceed with caveats, or should research be paused until this is obtained?

**How to obtain it**
Where would an analyst typically find this document?"""


def _walkthrough_doc_type(
    ticker: str,
    company: dict,
    doc_type: str,
    doc_info: dict,
    chunks: list[dict],
) -> str:
    """
    Generate a detailed analyst walkthrough of a specific document type.
    This answers: "what would I find if I read these myself?"
    """
    if not chunks:
        return "*No content available for this document type.*"

    dates   = doc_info.get("dates", [])
    date_range = (
        f"{dates[-1]} to {dates[0]}" if len(dates) > 1
        else (dates[0] if dates else "unknown")
    )

    # Use more chunks for richer coverage — up to 12
    sample = chunks[:12]

    prompt = DOC_WALKTHROUGH_PROMPT.format(
        count      = doc_info.get("count", 1),
        doc_type   = doc_type,
        company    = company.get("name", ticker),
        ticker     = ticker,
        date_range = date_range,
    )

    return call_llm_with_context(
        query          = prompt,
        context_chunks = sample,
        system         = DOC_WALKTHROUGH_SYSTEM,
        max_tokens     = 800,
    )


def _explain_missing_doc(
    ticker: str,
    company: dict,
    missing: dict,
) -> str:
    """
    Generate a specific, company-aware explanation of why a missing document matters.
    This answers: "why does this gap matter for THIS company's thesis?"
    """
    prompt = MISSING_DOC_PROMPT.format(
        company     = company.get("name", ticker),
        ticker      = ticker,
        sector      = company.get("sector", ""),
        doc_type    = missing["type"],
        description = missing["description"],
        criticality = missing["criticality"],
    )

    return call_llm(
        prompt     = prompt,
        system     = MISSING_DOC_SYSTEM,
        max_tokens = 600,
        temperature = 0.1,
    )


def generate_ingestion_report(
    ticker: str,
    documents: list[dict],
    chunks: list[dict],
    audit_result: dict,
    company_config: dict = None,
) -> str:
    """
    Generate the full Phase 1 Ingestion Report as markdown.

    Answers:
      - What can the analyst expect to find in the documents?
      - What important materials are missing, why do they matter, how critical?
    """
    company = company_config or {}
    now     = datetime.now().strftime("%B %d, %Y")

    lines = []

    # ── Header ───────────────────────────────────────────────────────────
    lines.append(f"# Phase 1 Ingestion Report: {company.get('name', ticker)}")
    lines.append(
        f"**Ticker:** {ticker} | **Exchange:** {company.get('exchange', '')} | "
        f"**Sector:** {company.get('sector', '')} | **Generated:** {now}"
    )
    lines.append("")
    lines.append(
        "> This report answers two questions: what can the analyst expect to find "
        "in the available documents, and what important materials are missing and why "
        "does each gap matter to the investment thesis?"
    )
    lines.append("")

    # ── Coverage scorecard ────────────────────────────────────────────────
    lines.append("## Coverage Scorecard")
    lines.append("")
    total_docs   = audit_result.get("total_documents_ingested", 0)
    total_words  = audit_result.get("total_words", 0)
    total_chunks = len(chunks)
    coverage     = audit_result.get("coverage_pct", 0)
    crit_gaps    = audit_result.get("critical_missing_count", 0)
    high_gaps    = audit_result.get("high_missing_count", 0)

    # Confidence rating
    if crit_gaps > 0:
        confidence = "LOW — critical document gaps present"
        conf_icon  = "🔴"
    elif high_gaps > 1:
        confidence = "MEDIUM — multiple high-priority gaps"
        conf_icon  = "🟡"
    elif high_gaps == 1:
        confidence = "MEDIUM-HIGH — one high-priority gap"
        conf_icon  = "🟡"
    else:
        confidence = "HIGH — all critical documents present"
        conf_icon  = "🟢"

    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Documents ingested | {total_docs} |")
    lines.append(f"| Total words processed | {total_words:,} |")
    lines.append(f"| Retrieval chunks | {total_chunks:,} |")
    lines.append(f"| Document type coverage | {coverage}% of expected types |")
    lines.append(f"| Critical gaps | {crit_gaps} |")
    lines.append(f"| High-priority gaps | {high_gaps} |")
    lines.append(f"| Analysis confidence | {conf_icon} {confidence} |")
    lines.append("")
    lines.append(audit_result.get("analyst_summary", ""))
    lines.append("")

    # ── Section 1: What the analyst will find ────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Section 1: What the Analyst Will Find in These Documents")
    lines.append("")
    lines.append(
        "The following is a document-by-document briefing on what each ingested "
        "document type contains. This is designed to help the analyst decide what "
        "to read in full versus rely on the AI summary."
    )
    lines.append("")

    # Group chunks by doc_type
    by_type: dict[str, list[dict]] = {}
    for chunk in chunks:
        dt = chunk.get("doc_type", "unknown")
        by_type.setdefault(dt, []).append(chunk)

    present = audit_result.get("present", [])
    if present:
        for doc_info in present:
            dt         = doc_info["type"]
            dt_chunks  = by_type.get(dt, [])
            count      = doc_info.get("count", 0)
            words      = doc_info.get("total_words", 0)
            dates      = doc_info.get("dates", [])
            date_str   = (
                f"{dates[-1]} to {dates[0]}" if len(dates) > 1
                else (dates[0] if dates else "unknown")
            )

            lines.append(f"### {dt.upper().replace('_', ' ')}")
            lines.append(
                f"*{count} document(s) | {words:,} words | "
                f"Period: {date_str} | Criticality if missing: "
                f"{doc_info.get('criticality', 'N/A')}*"
            )
            lines.append("")

            print(f"  Generating analyst walkthrough for {dt}...")
            walkthrough = _walkthrough_doc_type(ticker, company, dt, doc_info, dt_chunks)
            lines.append(walkthrough)
            lines.append("")
    else:
        lines.append("*No documents were successfully ingested.*")
        lines.append("")

    # ── Section 2: Missing Context Report ────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Section 2: Missing Context Report")
    lines.append("")
    lines.append(
        "The following expected document types were not available in accessible "
        "public sources. For each gap, this section explains what the document "
        "would have contained, why it matters specifically for this company's "
        "investment thesis, and what analytical questions cannot be answered without it."
    )
    lines.append("")

    missing = audit_result.get("missing", [])
    if missing:
        for m in missing:
            crit   = m["criticality"]
            status = m["status"]

            # Icon by criticality
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪"}.get(crit, "⚪")

            lines.append(f"### {icon} {m['type'].upper().replace('_',' ')} — [{crit}]")
            lines.append(f"*{m['description']} | Status: {status}*")
            lines.append("")

            print(f"  Generating missing doc explanation for {m['type']}...")
            explanation = _explain_missing_doc(ticker, company, m)
            lines.append(explanation)
            lines.append("")
    else:
        lines.append("*All expected document types were successfully ingested.*")
        lines.append("")

    # ── Section 3: Data quality notes ────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Section 3: Data Quality Notes")
    lines.append("")

    failed = [d for d in documents if d.get("fetch_failed")]
    if failed:
        lines.append(f"**{len(failed)} document(s) attempted but inaccessible:**")
        lines.append("")
        for d in failed:
            lines.append(
                f"- **{d['doc_type']}** ({d.get('date', '?')}): "
                f"`{d.get('source', 'unknown URL')}`"
            )
        lines.append("")
        lines.append(
            "These documents were identified and attempted but could not be downloaded. "
            "Common causes: paywall, authentication required, or URL change. "
            "The analyst should obtain these directly from the company IR page or a "
            "data provider (Bloomberg, Refinitiv, FactSet)."
        )
        lines.append("")

    # Language note for non-English filers
    reporting_standard = company.get("reporting_standard", "")
    country            = company.get("country", "")
    if country and country not in ("USA", "United States") and "GAAP" not in reporting_standard:
        lines.append(
            f"**Language / translation note:** {company.get('name', ticker)} is based in "
            f"{country} and may file primary documents in a language other than English. "
            "Only English-language documents were processed in this run. "
            "Material differences between local-language and English versions cannot be verified."
        )
        lines.append("")

    lines.append(
        "**AI summary reliability:** Document summaries in Section 1 are generated "
        "by Claude from retrieved text chunks. They are accurate to the ingested text "
        "but may miss context from sections that were not retrieved. "
        "All material figures should be verified against source documents."
    )
    lines.append("")

    # ── Full ingestion log ────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Ingestion Log")
    lines.append("")
    lines.append("| Document Type | Date | Words | Status |")
    lines.append("|---|---|---|---|")
    for doc in sorted(documents, key=lambda d: d.get("date", ""), reverse=True):
        status = "❌ FAILED" if doc.get("fetch_failed") else "✅ OK"
        lines.append(
            f"| {doc.get('doc_type','?')} | {doc.get('date','?')} | "
            f"{doc.get('word_count',0):,} | {status} |"
        )
    lines.append("")

    lines.append("---")
    lines.append(
        f"*Report generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} "
        f"by Investment AI Platform v1.0*"
    )

    return "\n".join(lines)