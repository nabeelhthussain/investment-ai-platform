"""
Phase 4 Runner — Financial Model Extraction.

Works for any company:
  - US filers (SEC XBRL): automatic via EDGAR API
  - European ESEF filers: via iXBRL zip files defined in companies.yaml
  - Others: LLM extraction from PDF text
"""
import json
from pathlib import Path
from datetime import datetime

from config import OUTPUTS_DIR, get_company


def run_financial_model(ticker: str) -> dict:
    """Run Phase 4 financial model extraction for any company."""
    from financial_model.normalizer import format_financial_model_report, build_review_report

    output_dir = OUTPUTS_DIR / ticker
    fm_dir = output_dir / "financial_model"
    fm_dir.mkdir(parents=True, exist_ok=True)

    company = get_company(ticker)
    fetcher_type = company.get("fetcher", "sec_edgar")
    reporting_standard = company.get("reporting_standard", "US GAAP")

    print(f"\n{'='*60}")
    print(f"Phase 4: Financial Model Extraction — {ticker}")
    print(f"{'='*60}")
    print(f"  Method: {fetcher_type} / {reporting_standard}")

    # Route to correct extractor
    if fetcher_type == "sec_edgar" or reporting_standard == "US GAAP":
        result = _extract_sec(ticker, company, fm_dir)
    elif fetcher_type == "oslo_bors" or _has_esef_urls(company):
        result = _extract_esef(ticker, company, fm_dir)
    else:
        result = _extract_llm_fallback(ticker, company, fm_dir)

    # Generate report
    report = format_financial_model_report(result)
    report_path = output_dir / "phase4_financial_model.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  ✓ Financial model report → {report_path}")

    # Summary JSON
    summary = _extract_summary(result)
    summary_path = fm_dir / "financial_model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  ✓ Summary JSON → {summary_path}")

    flags = build_review_report(result)
    critical = sum(1 for f in flags if f["severity"] == "CRITICAL")
    high = sum(1 for f in flags if f["severity"] == "HIGH")
    print(f"  Review flags: {critical} CRITICAL, {high} HIGH, {len(flags)-critical-high} other")

    return {
        "ticker": ticker,
        "report_path": str(report_path),
        "flags": len(flags),
        "result": result,
    }


def _has_esef_urls(company: dict) -> bool:
    return bool(company.get("esef_urls"))


def _extract_sec(ticker: str, company: dict, fm_dir: Path) -> dict:
    """Extract via SEC EDGAR XBRL — works for any US filer with a CIK."""
    from financial_model.xbrl_extractor import (
        fetch_company_facts, build_financial_statement,
        calculate_derived_metrics, validate_balance_sheet,
        INCOME_STATEMENT_MAP, BALANCE_SHEET_MAP, CASH_FLOW_MAP,
    )

    cik = company.get("cik", "")
    if not cik:
        return {"error": f"No CIK found for {ticker}", "ticker": ticker}

    print(f"  Fetching XBRL facts (CIK: {cik})...")
    try:
        facts = fetch_company_facts(cik)
    except Exception as e:
        return {"error": f"XBRL fetch failed: {e}", "ticker": ticker}

    print("  Extracting statements...")
    income = build_financial_statement(facts, INCOME_STATEMENT_MAP, "annual")
    balance = build_financial_statement(facts, BALANCE_SHEET_MAP, "annual")
    cashflow = build_financial_statement(facts, CASH_FLOW_MAP, "annual")
    metrics = calculate_derived_metrics(income, balance, cashflow)
    validation = validate_balance_sheet(balance)

    income_q = build_financial_statement(facts, INCOME_STATEMENT_MAP, "quarterly")
    cashflow_q = build_financial_statement(facts, CASH_FLOW_MAP, "quarterly")

    result = {
        "ticker": ticker,
        "company": company.get("name", ticker),
        "extraction_method": "SEC EDGAR XBRL Company Facts API",
        "currency": company.get("currency", "USD"),
        "units": "as_reported",
        "annual": {
            "income_statement": income,
            "balance_sheet": balance,
            "cash_flow": cashflow,
        },
        "quarterly": {
            "income_statement": income_q,
            "cash_flow": cashflow_q,
        },
        "derived_metrics": metrics,
        "validation": validation,
        "xbrl_concepts_used": {
            "income_statement": income.get("source_concepts", {}),
            "balance_sheet": balance.get("source_concepts", {}),
            "cash_flow": cashflow.get("source_concepts", {}),
        },
    }

    fm_dir.mkdir(parents=True, exist_ok=True)
    (fm_dir / "financial_model_raw.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    return result


def _extract_esef(ticker: str, company: dict, fm_dir: Path) -> dict:
    """Extract via ESEF iXBRL zips — for European filers."""
    from financial_model.esef_extractor import (
        _download_zip, _extract_from_zip, _map_facts_to_schema,
        _validate_akso, extract_akso_financials_pdf,
        IFRS_INCOME_MAP, IFRS_BALANCE_MAP, IFRS_CASHFLOW_MAP,
    )

    esef_urls = company.get("esef_urls", [])
    if not esef_urls:
        print("  No ESEF URLs in config — falling back to PDF extraction")
        return extract_akso_financials_pdf(fm_dir)

    all_income, all_balance, all_cashflow = [], [], []

    for doc_meta in esef_urls:
        year = doc_meta["year"]
        print(f"  Downloading ESEF zip for {year}...")
        zip_bytes = _download_zip(doc_meta["url"])
        if not zip_bytes:
            continue

        facts = _extract_from_zip(zip_bytes, year)
        if not facts:
            continue

        print(f"  Extracted {len(facts)} iXBRL facts for {year}")
        period_end = doc_meta["period_end"]
        all_income.append({"year": year, "period_end": period_end,
                           "data": _map_facts_to_schema(facts, IFRS_INCOME_MAP, period_end)})
        all_balance.append({"year": year, "period_end": period_end,
                            "data": _map_facts_to_schema(facts, IFRS_BALANCE_MAP, period_end)})
        all_cashflow.append({"year": year, "period_end": period_end,
                             "data": _map_facts_to_schema(facts, IFRS_CASHFLOW_MAP, period_end)})

    if not all_income:
        print("  ESEF extraction failed — falling back to PDF extraction")
        return extract_akso_financials_pdf(fm_dir)

    result = {
        "ticker": ticker,
        "company": company.get("name", ticker),
        "extraction_method": "ESEF iXBRL",
        "currency": company.get("currency", "NOK"),
        "units": "as_reported_millions",
        "years_extracted": [d["year"] for d in all_income],
        "annual": {
            "income_statement": all_income,
            "balance_sheet": all_balance,
            "cash_flow": all_cashflow,
        },
        "validation": _validate_akso(all_income, all_balance),
    }

    fm_dir.mkdir(parents=True, exist_ok=True)
    (fm_dir / "financial_model_raw.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    return result


def _extract_llm_fallback(ticker: str, company: dict, fm_dir: Path) -> dict:
    """LLM-based extraction for companies without structured XBRL/ESEF."""
    from financial_model.esef_extractor import extract_akso_financials_pdf
    print("  No structured data source — using LLM extraction from available text")
    return extract_akso_financials_pdf(fm_dir)


def _extract_summary(result: dict) -> dict:
    ticker = result.get("ticker", "")
    metrics = result.get("derived_metrics", {})

    def get_latest(stmt_key, field):
        annual = result.get("annual", {})
        stmt = annual.get(stmt_key, {})
        if isinstance(stmt, list):
            stmt = stmt[0].get("data", {}) if stmt else {}
        series = stmt.get("data", {}).get(field, []) if isinstance(stmt, dict) else []
        return series[0]["value"] if series else None

    return {
        "ticker": ticker,
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "currency": result.get("currency", "USD"),
        "extraction_method": result.get("extraction_method"),
        "key_figures": {
            "revenue": get_latest("income_statement", "revenue"),
            "net_income": get_latest("income_statement", "net_income"),
            "cash": get_latest("balance_sheet", "cash"),
            "total_assets": get_latest("balance_sheet", "total_assets"),
            "operating_cf": get_latest("cash_flow", "operating_cf"),
        },
        "derived": {k: v.get("value") if isinstance(v, dict) else v
                   for k, v in metrics.items()},
    }
