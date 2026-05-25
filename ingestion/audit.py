"""
Ingestion audit: compares ingested documents against expected taxonomy
and produces a structured Missing Context Report.
Works for any company — taxonomy loaded from companies.yaml.
"""
from datetime import datetime
from config import get_company

CRITICALITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def run_audit(ticker: str, ingested_documents: list[dict]) -> dict:
    """
    Compare ingested documents against expected taxonomy.
    Returns audit result dict.
    """
    company_info = get_company(ticker)
    expected = company_info.get("expected_docs", [])

    # Build set of what we have
    ingested_types: dict[str, list] = {}
    for doc in ingested_documents:
        dt = doc.get("doc_type", "unknown")
        ingested_types.setdefault(dt, []).append(doc)

    present = []
    missing = []

    for expected_doc in expected:
        doc_type = expected_doc["type"]
        criticality = expected_doc["criticality"]
        description = expected_doc.get("description", "")

        if doc_type in ingested_types:
            docs = ingested_types[doc_type]
            successful = [d for d in docs if not d.get("fetch_failed")]

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
                missing.append({
                    "type": doc_type,
                    "description": description,
                    "criticality": criticality,
                    "reason": "Fetch attempted but document inaccessible",
                    "analyst_impact": f"{description} could not be retrieved. Obtain directly from company IR.",
                    "status": "FETCH_FAILED",
                })
        else:
            missing.append({
                "type": doc_type,
                "description": description,
                "criticality": criticality,
                "reason": "Not found in accessible public sources",
                "analyst_impact": f"{description} not available publicly. Analyst should obtain directly.",
                "status": "MISSING",
            })

    missing.sort(key=lambda x: CRITICALITY_ORDER.get(x["criticality"], 99))

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
        "total_chunks": None,
        "total_words": sum(d.get("word_count", 0) for d in ingested_documents),
        "coverage_pct": coverage_pct,
        "present": present,
        "missing": missing,
        "critical_missing_count": len(critical_missing),
        "high_missing_count": len(high_missing),
        "ingested_by_type": {dt: len(docs) for dt, docs in ingested_types.items()},
        "analyst_summary": _build_analyst_summary(ticker, present, missing, coverage_pct),
    }


def _build_analyst_summary(ticker, present, missing, coverage_pct):
    critical_missing = [m for m in missing if m["criticality"] == "CRITICAL"]
    high_missing = [m for m in missing if m["criticality"] == "HIGH"]

    lines = [f"Document coverage: {coverage_pct}% of expected document types found."]
    if present:
        lines.append(f"Available: {', '.join(p['type'] for p in present[:6])}.")
    if critical_missing:
        types = ", ".join(m["type"] for m in critical_missing)
        lines.append(f"CRITICAL gaps: {types}.")
    if high_missing:
        types = ", ".join(m["type"] for m in high_missing)
        lines.append(f"High-priority gaps: {types}.")
    if not critical_missing and not high_missing:
        lines.append("No critical gaps identified.")
    return " ".join(lines)
