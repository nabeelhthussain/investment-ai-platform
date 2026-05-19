"""
Ingestion audit: compares what was successfully ingested against the
expected document taxonomy and produces a structured Missing Context Report.
"""
from datetime import datetime
from config import EXPECTED_DOCS, COMPANIES


CRITICALITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

CRITICALITY_EXPLANATIONS = {
    "SOC": {
        "10-K": (
            "Primary annual disclosure. Contains audited financials, full risk factor list, "
            "MD&A, and legal proceedings. Without this, the analyst has no verified baseline "
            "for revenue, costs, or debt. CRITICAL gap."
        ),
        "10-Q": (
            "Intra-year trend data. For SOC, the quarterly cadence reveals progress on the "
            "Santa Ynez Unit restart — cost overruns and timeline slippage show up here first."
        ),
        "8-K": (
            "Material event disclosures filed within 4 days. Covers leadership changes, "
            "financing events, permit decisions, regulatory actions. Missing 8-Ks means "
            "the analyst may be unaware of material developments."
        ),
        "DEF14A": (
            "Proxy statement. Contains executive compensation, insider ownership, board "
            "independence, and shareholder vote outcomes. Relevant for governance risk."
        ),
        "earnings_call": (
            "Management tone and forward guidance. Transcripts allow contradiction detection "
            "across quarters — tone shifts on restart timelines or capex guidance are not "
            "visible in filings alone."
        ),
        "investor_pres": (
            "Management's self-presented bull case. Useful for identifying the claims "
            "that the bear case analysis should stress-test."
        ),
        "reserve_report": (
            "Independent certification of oil and gas reserves by a third-party engineer. "
            "CRITICAL for any E&P company — this is the primary valuation anchor. "
            "Without it, reserve claims are unverified."
        ),
        "credit_rating": (
            "Leverage and liquidity risk. Sable is pre-revenue and has drawn on its credit "
            "facility. A credit rating report reveals covenant structure and downgrade risk."
        ),
    },
    "AKSO": {
        "annual_report": (
            "IFRS annual report. Contains audited financials, order backlog, segment breakdown "
            "(subsea, renewables, electrification), and management commentary. "
            "Primary document for financial analysis."
        ),
        "quarterly": (
            "Order intake momentum and backlog conversion. For an oilfield services company, "
            "quarterly order intake is a leading indicator for 12–18 month revenue."
        ),
        "earnings_call": (
            "Management tone on order pipeline, contract wins/losses, and energy transition "
            "strategy. Tone shifts on renewables vs traditional oil services are high-value signals."
        ),
        "investor_pres": (
            "Capital Markets Day presentations lay out multi-year strategy and financial targets. "
            "Essential for assessing whether management is delivering on stated commitments."
        ),
        "esg_report": (
            "AKSO has significant ESG exposure as a supplier to oil majors. ESG report "
            "reveals Scope 1/2/3 commitments and transition risk management."
        ),
        "credit_rating": (
            "AKSO carries project finance exposure. Credit rating reveals covenant structure "
            "and how leverage is managed through the cycle."
        ),
        "reg_filing": (
            "Oslo Børs regulatory filings (MAR disclosures) are the primary source for "
            "insider transactions, major shareholder changes, and profit warnings. "
            "Norwegian-listed companies file these in lieu of 8-Ks."
        ),
    },
}


def run_audit(ticker: str, ingested_documents: list[dict]) -> dict:
    """
    Compare ingested documents against expected taxonomy.
    Returns audit result dict with present/missing/partial breakdowns.
    """
    expected = EXPECTED_DOCS.get(ticker, [])
    company_info = COMPANIES.get(ticker, {})

    # Build a set of what we have
    ingested_types = {}
    for doc in ingested_documents:
        dt = doc.get("doc_type", "unknown")
        if dt not in ingested_types:
            ingested_types[dt] = []
        ingested_types[dt].append(doc)

    present = []
    missing = []
    partial = []

    for expected_doc in expected:
        doc_type = expected_doc["type"]
        criticality = expected_doc["criticality"]
        description = expected_doc["description"]
        explanation = CRITICALITY_EXPLANATIONS.get(ticker, {}).get(doc_type, "")

        if doc_type in ingested_types:
            docs = ingested_types[doc_type]
            successful = [d for d in docs if not d.get("fetch_failed")]
            failed = [d for d in docs if d.get("fetch_failed")]

            if successful:
                total_words = sum(d.get("word_count", 0) for d in successful)
                present.append({
                    "type": doc_type,
                    "description": description,
                    "criticality": criticality,
                    "count": len(successful),
                    "total_words": total_words,
                    "dates": sorted([d.get("date", "") for d in successful], reverse=True),
                    "status": "PRESENT",
                })
            else:
                # Fetched but all failed
                missing.append({
                    "type": doc_type,
                    "description": description,
                    "criticality": criticality,
                    "reason": "Fetch attempted but document inaccessible (authentication or paywall required)",
                    "analyst_impact": explanation,
                    "status": "FETCH_FAILED",
                })
        else:
            missing.append({
                "type": doc_type,
                "description": description,
                "criticality": criticality,
                "reason": "Not found in accessible public sources",
                "analyst_impact": explanation,
                "status": "MISSING",
            })

    # Sort missing by criticality
    missing.sort(key=lambda x: CRITICALITY_ORDER.get(x["criticality"], 99))

    # Coverage score
    total = len(expected)
    found = len(present)
    coverage_pct = round((found / total * 100) if total else 0, 1)

    critical_missing = [m for m in missing if m["criticality"] == "CRITICAL"]
    high_missing = [m for m in missing if m["criticality"] == "HIGH"]

    return {
        "ticker": ticker,
        "company": company_info.get("name", ticker),
        "audit_date": datetime.now().strftime("%Y-%m-%d"),
        "total_documents_ingested": len(ingested_documents),
        "total_chunks": None,  # filled in by pipeline
        "total_words": sum(d.get("word_count", 0) for d in ingested_documents),
        "coverage_pct": coverage_pct,
        "present": present,
        "missing": missing,
        "critical_missing_count": len(critical_missing),
        "high_missing_count": len(high_missing),
        "ingested_by_type": {
            dt: len(docs) for dt, docs in ingested_types.items()
        },
        "analyst_summary": _build_analyst_summary(ticker, present, missing, coverage_pct),
    }


def _build_analyst_summary(ticker: str, present: list, missing: list, coverage_pct: float) -> str:
    critical_missing = [m for m in missing if m["criticality"] == "CRITICAL"]
    high_missing = [m for m in missing if m["criticality"] == "HIGH"]

    lines = []
    lines.append(f"Document coverage: {coverage_pct}% of expected document types found.")

    if present:
        lines.append(
            f"Available: {', '.join(p['type'] for p in present[:6])}."
        )

    if critical_missing:
        types = ", ".join(m["type"] for m in critical_missing)
        lines.append(
            f"CRITICAL gaps: {types}. These documents are required for a complete analysis "
            f"and their absence meaningfully limits the conclusions that can be drawn."
        )

    if high_missing:
        types = ", ".join(m["type"] for m in high_missing)
        lines.append(
            f"High-priority gaps: {types}. Obtaining these before finalizing any investment "
            f"thesis is strongly recommended."
        )

    if not critical_missing and not high_missing:
        lines.append(
            "No critical gaps identified. Analysis can proceed with standard caveats."
        )

    return " ".join(lines)
