"""
Financial Model Normalizer.

Takes raw extraction output and produces:
1. Clean formatted tables for human review
2. A review report flagging items that need analyst verification
3. Integration with the dossier bear case (quantitative grounding)
"""
import json
from datetime import datetime
from pathlib import Path


def _fmt(value, currency="USD", units_divisor=1_000_000) -> str:
    """Format a raw value as a human-readable financial figure."""
    if value is None:
        return "N/A"
    try:
        v = float(value) / units_divisor
        if abs(v) >= 1000:
            return f"{currency} {v:,.1f}B" if abs(v) >= 1000 else f"{currency} {v:,.1f}M"
        return f"{currency} {v:,.1f}M"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value)*100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _get_series(statement_data: dict, field: str) -> list:
    """Extract time series for a field from statement data dict."""
    return statement_data.get("data", {}).get(field, [])


def _get_latest(statement_data: dict, field: str):
    series = _get_series(statement_data, field)
    return series[0]["value"] if series else None


def build_review_report(extraction_result: dict) -> list[dict]:
    """
    Build a list of items requiring analyst review.
    Each item has: field, issue, severity, recommendation.
    """
    flags = []
    ticker = extraction_result.get("ticker", "")

    if "error" in extraction_result:
        flags.append({
            "field": "all",
            "issue": extraction_result["error"],
            "severity": "CRITICAL",
            "recommendation": "Re-run extraction after fixing the error",
        })
        return flags

    method = extraction_result.get("extraction_method", "")

    if "pdf_llm" in method:
        flags.append({
            "field": "all",
            "issue": "Extracted via LLM from PDF text — higher error probability",
            "severity": "HIGH",
            "recommendation": "Cross-check every figure against source PDF before use",
        })

    annual = extraction_result.get("annual", {})
    income = annual.get("income_statement", {})
    balance = annual.get("balance_sheet", {})
    cashflow = annual.get("cash_flow", {})
    metrics = extraction_result.get("derived_metrics", {})

    # Check for missing critical fields
    critical_fields = {
        "income_statement": ["revenue", "net_income"],
        "balance_sheet": ["total_assets", "total_liabilities", "cash"],
        "cash_flow": ["operating_cf"],
    }

    for stmt_name, fields in critical_fields.items():
        stmt = annual.get(stmt_name, {})
        for field in fields:
            if not _get_series(stmt, field):
                flags.append({
                    "field": f"{stmt_name}.{field}",
                    "issue": f"{field} not found in {stmt_name}",
                    "severity": "HIGH",
                    "recommendation": f"Manually extract {field} from source filing",
                })

    # Check balance sheet validation
    for msg in extraction_result.get("validation", []):
        if "imbalance" in msg.lower() or "⚠" in msg:
            flags.append({
                "field": "balance_sheet",
                "issue": msg,
                "severity": "HIGH",
                "recommendation": "Verify total assets = total liabilities + equity in source",
            })

    # Check for implausible values
    revenue = _get_latest(income, "revenue")
    net_income = _get_latest(income, "net_income")
    if revenue and net_income:
        margin = net_income / revenue
        if margin > 0.5 or margin < -1.0:
            flags.append({
                "field": "income_statement.net_margin",
                "issue": f"Net margin of {margin:.1%} appears implausible",
                "severity": "MEDIUM",
                "recommendation": "Verify net income and revenue figures in source",
            })

    # SOC specific: flag zero revenue
    if ticker == "SOC":
        if revenue is None or revenue == 0:
            flags.append({
                "field": "income_statement.revenue",
                "issue": "Zero revenue confirmed — pre-revenue company",
                "severity": "INFO",
                "recommendation": "Expected for SOC — no action needed",
            })

    return flags


def format_financial_model_report(extraction_result: dict) -> str:
    """
    Generate a human-readable markdown report of the extracted financial model.
    Includes review flags, formatted tables, and source tracing.
    """
    ticker = extraction_result.get("ticker", "")
    company = extraction_result.get("company", ticker)
    method = extraction_result.get("extraction_method", "unknown")
    currency = extraction_result.get("currency", "USD")
    now = datetime.now().strftime("%B %d, %Y")

    # Currency formatting
    if currency == "NOK":
        units_label = "NOK millions"
        units_div = 1
        curr_sym = "NOK"
    else:
        units_label = "USD millions"
        units_div = 1_000_000
        curr_sym = "USD"

    def fmt(v):
        if v is None:
            return "—"
        try:
            v = float(v) / units_div
            return f"{v:>12,.1f}"
        except (TypeError, ValueError):
            return str(v)

    lines = []
    lines.append(f"# Phase 4: Financial Model — {company} ({ticker})")
    lines.append(f"**Extraction method:** {method}  ")
    lines.append(f"**Currency:** {units_label}  ")
    lines.append(f"**Generated:** {now}")
    lines.append("")
    lines.append("> ⚠️ **All figures require analyst verification before use in a financial model.**")
    lines.append("> Source traces are provided for every figure. Review flagged items before proceeding.")
    lines.append("")

    # Review flags
    flags = build_review_report(extraction_result)
    if flags:
        lines.append("## ⚑ Review Flags")
        lines.append("")
        lines.append("| Field | Issue | Severity | Action |")
        lines.append("|---|---|---|---|")
        for f in flags:
            lines.append(f"| {f['field']} | {f['issue']} | **{f['severity']}** | {f['recommendation']} |")
        lines.append("")

    # Handle multi-year AKSO format
    if "years_extracted" in extraction_result:
        lines.extend(_format_akso_multiyear(extraction_result, fmt, curr_sym))
    elif "extracted_periods" in extraction_result:
        lines.extend(_format_akso_pdf_fallback(extraction_result))
    else:
        lines.extend(_format_soc_annual(extraction_result, fmt, curr_sym, units_div))

    # Derived metrics
    metrics = extraction_result.get("derived_metrics", {})
    if metrics:
        lines.append("## Derived Metrics")
        lines.append("")
        lines.append("| Metric | Value | Formula |")
        lines.append("|---|---|---|")
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                value = metric_data.get("value")
                formula = metric_data.get("formula", "")
                if metric_name.endswith("_margin") or metric_name.endswith("_ratio") or "growth" in metric_name:
                    formatted_val = f"{float(value)*100:.1f}%" if value is not None else "—"
                else:
                    formatted_val = fmt(value) if value is not None else "—"
                lines.append(f"| {metric_name.replace('_', ' ').title()} | {formatted_val} | {formula} |")
        lines.append("")

    # Source concepts used
    xbrl = extraction_result.get("xbrl_concepts_used", {})
    if xbrl:
        lines.append("## XBRL Source Concepts")
        lines.append("")
        lines.append("*Each field maps to a specific XBRL concept tag in the filing — full audit trail.*")
        lines.append("")
        for stmt, concepts in xbrl.items():
            if concepts:
                lines.append(f"**{stmt.replace('_', ' ').title()}:**")
                for field, concept in concepts.items():
                    lines.append(f"- `{field}` ← `{concept}`")
        lines.append("")

    # Validation
    validation = extraction_result.get("validation", [])
    if validation:
        lines.append("## Validation Results")
        lines.append("")
        for msg in validation:
            lines.append(f"- {msg}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Financial model extraction generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*  ")
    lines.append(f"*All figures must be verified against source filings before use in investment models.*")

    return "\n".join(lines)


def _format_soc_annual(result: dict, fmt, curr_sym: str, units_div: int) -> list[str]:
    """Format SOC multi-period annual statements."""
    lines = []
    annual = result.get("annual", {})
    income = annual.get("income_statement", {})
    balance = annual.get("balance_sheet", {})
    cashflow = annual.get("cash_flow", {})

    def get_series(stmt, field):
        return stmt.get("data", {}).get(field, [])

    def series_row(stmt, field, label, negate=False) -> str:
        values = get_series(stmt, field)
        if not values:
            return f"| {label:<35} | {'—':>12} | {'—':>12} | {'—':>12} | {'—':>12} |"
        cells = []
        for v in values[:4]:
            val = v.get("value")
            if val is not None and negate:
                val = -abs(float(val))
            cells.append(fmt(val))
        while len(cells) < 4:
            cells.append("—")
        return f"| {label:<35} | {cells[0]:>12} | {cells[1]:>12} | {cells[2]:>12} | {cells[3]:>12} |"

    # Get period headers
    rev_series = get_series(income, "revenue") or get_series(income, "net_income") or []
    periods = [v.get("period_end", "")[:4] for v in rev_series[:4]]
    while len(periods) < 4:
        periods.append("—")

    header = f"| {'':35} | {periods[0]:>12} | {periods[1]:>12} | {periods[2]:>12} | {periods[3]:>12} |"
    divider = "|" + "-"*37 + "|" + ("-"*14 + "|") * 4

    lines.append(f"## Income Statement ({curr_sym} millions)")
    lines.append("")
    lines.append(header)
    lines.append(divider)
    lines.append(series_row(income, "revenue", "Revenue"))
    lines.append(series_row(income, "gross_profit", "Gross Profit"))
    lines.append(series_row(income, "operating_income", "Operating Income (EBIT)"))
    lines.append(series_row(income, "ebitda", "EBITDA"))
    lines.append(series_row(income, "da", "D&A"))
    lines.append(series_row(income, "interest_expense", "Interest Expense", negate=True))
    lines.append(series_row(income, "net_income", "Net Income"))
    lines.append("")

    lines.append(f"## Balance Sheet ({curr_sym} millions)")
    lines.append("")
    lines.append(header)
    lines.append(divider)
    lines.append(series_row(balance, "cash", "Cash & Equivalents"))
    lines.append(series_row(balance, "current_assets", "Current Assets"))
    lines.append(series_row(balance, "total_assets", "Total Assets"))
    lines.append(series_row(balance, "current_liabilities", "Current Liabilities"))
    lines.append(series_row(balance, "long_term_debt", "Long-Term Debt"))
    lines.append(series_row(balance, "total_liabilities", "Total Liabilities"))
    lines.append(series_row(balance, "stockholders_equity", "Stockholders' Equity"))
    lines.append(series_row(balance, "accumulated_deficit", "Accumulated Deficit"))
    lines.append("")

    lines.append(f"## Cash Flow Statement ({curr_sym} millions)")
    lines.append("")
    lines.append(header)
    lines.append(divider)
    lines.append(series_row(cashflow, "operating_cf", "Operating Cash Flow"))
    lines.append(series_row(cashflow, "capex", "Capital Expenditures", negate=True))
    lines.append(series_row(cashflow, "investing_cf", "Investing Cash Flow"))
    lines.append(series_row(cashflow, "financing_cf", "Financing Cash Flow"))
    lines.append("")

    return lines


def _format_akso_multiyear(result: dict, fmt, curr_sym: str) -> list[str]:
    """Format AKSO multi-year ESEF extraction."""
    lines = []
    annual = result.get("annual", {})
    income_years = annual.get("income_statement", [])

    if not income_years:
        lines.append("*No structured financial data extracted from ESEF files.*")
        return lines

    lines.append(f"## Extracted Financial Data ({curr_sym} millions)")
    lines.append("")
    lines.append("| Metric | " + " | ".join(str(y["year"]) for y in income_years) + " |")
    lines.append("|---|" + "---|" * len(income_years))

    fields_to_show = [
        ("revenue", "Revenue"),
        ("gross_profit", "Gross Profit"),
        ("operating_income", "Operating Income"),
        ("ebitda", "EBITDA"),
        ("net_income", "Net Income"),
        ("personnel_costs", "Personnel Costs"),
    ]

    for field, label in fields_to_show:
        values = []
        for year_data in income_years:
            stmt_data = year_data.get("data", {}).get("data", {})
            series = stmt_data.get(field, [])
            val = series[0]["value"] if series else None
            values.append(fmt(val))
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.append("")

    # Balance sheet
    balance_years = annual.get("balance_sheet", [])
    if balance_years:
        lines.append(f"## Balance Sheet ({curr_sym} millions)")
        lines.append("")
        lines.append("| Metric | " + " | ".join(str(y["year"]) for y in balance_years) + " |")
        lines.append("|---|" + "---|" * len(balance_years))

        bs_fields = [
            ("cash", "Cash"),
            ("total_assets", "Total Assets"),
            ("long_term_debt", "Long-Term Debt"),
            ("equity", "Equity"),
        ]
        for field, label in bs_fields:
            values = []
            for year_data in balance_years:
                stmt_data = year_data.get("data", {}).get("data", {})
                series = stmt_data.get(field, [])
                val = series[0]["value"] if series else None
                values.append(fmt(val))
            lines.append(f"| {label} | " + " | ".join(values) + " |")
        lines.append("")

    return lines


def _format_akso_pdf_fallback(result: dict) -> list[str]:
    """Format AKSO PDF LLM extraction fallback."""
    lines = []
    periods = result.get("extracted_periods", [])

    if not periods:
        lines.append("*No financial data could be extracted.*")
        return lines

    lines.append("## Extracted Key Figures (NOK millions)")
    lines.append("")
    lines.append("*Extracted via LLM from PDF — requires verification*")
    lines.append("")

    fields = [
        ("period_end", "Period"),
        ("revenue", "Revenue"),
        ("ebitda", "EBITDA"),
        ("ebitda_margin_pct", "EBITDA Margin %"),
        ("net_income", "Net Income"),
        ("cash", "Cash"),
        ("net_debt", "Net Debt"),
        ("order_intake", "Order Intake"),
        ("order_backlog", "Order Backlog"),
        ("operating_cf", "Operating CF"),
        ("capex", "Capex"),
    ]

    for period in periods:
        lines.append(f"### {period.get('period_end', 'Unknown Period')}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for field, label in fields[1:]:
            val = period.get(field)
            if val is not None:
                if field == "ebitda_margin_pct":
                    formatted = f"{val:.1f}%"
                else:
                    formatted = f"{float(val):,.1f}"
                lines.append(f"| {label} | {formatted} |")
        lines.append("")

    return lines
