"""
XBRL Extractor for SEC Filers (SOC).

Uses the SEC EDGAR Company Facts API — no parsing, no PDF extraction.
Returns pre-labeled, period-aligned financial data directly from machine-readable filings.

API: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
Returns every reported financial figure with:
  - US-GAAP or IFRS concept label
  - Value, units, period start/end, form type, filing date
"""
import json
import time
import requests
from collections import defaultdict
from pathlib import Path

HEADERS = {"User-Agent": "InvestmentResearchPlatform research@example.com"}

# Canonical schema: maps US-GAAP XBRL concept → our internal field name
# Priority order matters — first match wins for each canonical field
INCOME_STATEMENT_MAP = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_expenses": [
        "OperatingExpenses",
        "GeneralAndAdministrativeExpense",
    ],
    "ebitda_proxy": [
        "OperatingIncomeLoss",  # EBITDA proxy — add back D&A separately
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "ebitda": ["EarningsBeforeInterestTaxesDepreciationAndAmortization"],
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
    ],
}

BALANCE_SHEET_MAP = {
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "total_liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermNotesPayable",
    ],
    "total_debt": [
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebt",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
    ],
    "accumulated_deficit": ["RetainedEarningsAccumulatedDeficit"],
}

CASH_FLOW_MAP = {
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "investing_cf": [
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",
    ],
    "financing_cf": [
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ],
    "free_cash_flow_proxy": [],  # Calculated: operating_cf - capex
}


def _get(url: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            time.sleep(0.15)
            return r.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def fetch_company_facts(cik: str) -> dict:
    """Fetch all XBRL facts for a company from SEC EDGAR."""
    cik_padded = str(int(cik.lstrip("0"))).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    print(f"  Fetching XBRL facts from EDGAR: {url}")
    return _get(url)


def _extract_concept_values(facts: dict, concept: str, taxonomy: str = "us-gaap") -> list[dict]:
    """
    Extract all reported values for a specific XBRL concept.
    Returns list sorted by end date descending.
    """
    try:
        units = facts["facts"][taxonomy][concept]["units"]
        # Financial figures are in USD
        values = units.get("USD", units.get("shares", []))
        # Filter to annual (10-K) and quarterly (10-Q) filings only
        filtered = [
            v for v in values
            if v.get("form") in ("10-K", "10-Q", "20-F", "40-F")
            and v.get("end")
        ]
        # Sort by end date descending
        return sorted(filtered, key=lambda x: x["end"], reverse=True)
    except (KeyError, TypeError):
        return []


def _pick_best_value(values: list[dict], period_type: str = "annual") -> list[dict]:
    """
    For each fiscal period, pick the most recent filing of that figure.
    period_type: 'annual' (full year) or 'quarterly'
    """
    if not values:
        return []

    # Group by end date
    by_end = defaultdict(list)
    for v in values:
        start = v.get("start", "")
        end = v.get("end", "")
        if not end:
            continue
        # Determine if annual or quarterly based on date span
        if start and end:
            try:
                from datetime import datetime
                s = datetime.strptime(start, "%Y-%m-%d")
                e = datetime.strptime(end, "%Y-%m-%d")
                days = (e - s).days
                is_annual = days > 300
            except Exception:
                is_annual = True
        else:
            is_annual = True

        if period_type == "annual" and is_annual:
            by_end[end].append(v)
        elif period_type == "quarterly" and not is_annual:
            by_end[end].append(v)

    # For each period, pick the value from the most recent filing
    result = []
    for end_date, period_values in sorted(by_end.items(), reverse=True):
        best = sorted(period_values, key=lambda x: x.get("filed", ""), reverse=True)[0]
        result.append({
            "period_end": end_date,
            "period_start": best.get("start", ""),
            "value": best.get("val"),
            "form": best.get("form"),
            "filed": best.get("filed"),
            "accn": best.get("accn"),
        })

    return result[:8]  # Last 8 periods


def build_financial_statement(
    facts: dict,
    field_map: dict,
    period_type: str = "annual",
    taxonomy: str = "us-gaap",
) -> dict:
    """
    Build a financial statement dict by mapping XBRL concepts to canonical fields.
    Returns dict of {field_name: [list of period values]}
    """
    statement = {}
    source_map = {}  # Track which XBRL concept was used for each field

    for canonical_field, xbrl_concepts in field_map.items():
        if not xbrl_concepts:
            continue
        for concept in xbrl_concepts:
            values = _extract_concept_values(facts, concept, taxonomy)
            if values:
                period_values = _pick_best_value(values, period_type)
                if period_values:
                    statement[canonical_field] = period_values
                    source_map[canonical_field] = concept
                    break  # First match wins

    return {"data": statement, "source_concepts": source_map}


def calculate_derived_metrics(income: dict, balance: dict, cashflow: dict) -> dict:
    """
    Calculate derived financial metrics from extracted statements.
    All calculations documented for auditability.
    """
    metrics = {}

    def get_latest(statement: dict, field: str):
        data = statement.get("data", {})
        values = data.get(field, [])
        return values[0]["value"] if values else None

    def get_series(statement: dict, field: str) -> list:
        data = statement.get("data", {})
        return statement.get("data", {}).get(field, [])

    # Net debt
    total_debt = get_latest(balance, "total_debt") or get_latest(balance, "long_term_debt")
    cash = get_latest(balance, "cash")
    if total_debt is not None and cash is not None:
        metrics["net_debt"] = {
            "value": total_debt - cash,
            "formula": "total_debt - cash",
            "total_debt": total_debt,
            "cash": cash,
        }

    # Free cash flow
    op_cf = get_latest(cashflow, "operating_cf")
    capex = get_latest(cashflow, "capex")
    if op_cf is not None and capex is not None:
        metrics["free_cash_flow"] = {
            "value": op_cf - abs(capex),
            "formula": "operating_cf - capex",
            "operating_cf": op_cf,
            "capex": capex,
        }

    # EBITDA (operating income + D&A if EBITDA not directly reported)
    ebitda = get_latest(income, "ebitda")
    if not ebitda:
        op_income = get_latest(income, "operating_income")
        da = get_latest(income, "da")
        if op_income is not None and da is not None:
            ebitda = op_income + abs(da)
            metrics["ebitda_calculated"] = {
                "value": ebitda,
                "formula": "operating_income + D&A",
                "operating_income": op_income,
                "da": da,
            }

    # EBITDA margin
    revenue = get_latest(income, "revenue")
    if ebitda and revenue and revenue != 0:
        metrics["ebitda_margin"] = {
            "value": round(ebitda / revenue, 4),
            "formula": "ebitda / revenue",
            "ebitda": ebitda,
            "revenue": revenue,
        }

    # Net debt / EBITDA
    net_debt_val = metrics.get("net_debt", {}).get("value")
    if net_debt_val and ebitda and ebitda != 0:
        metrics["net_debt_to_ebitda"] = {
            "value": round(net_debt_val / ebitda, 2),
            "formula": "net_debt / ebitda",
        }

    # Interest coverage
    op_income = get_latest(income, "operating_income")
    interest = get_latest(income, "interest_expense")
    if op_income and interest and interest != 0:
        metrics["interest_coverage"] = {
            "value": round(op_income / abs(interest), 2),
            "formula": "operating_income / interest_expense",
        }

    # Revenue growth (latest vs prior year)
    revenue_series = get_series(income, "revenue")
    if len(revenue_series) >= 2:
        latest = revenue_series[0]["value"]
        prior = revenue_series[1]["value"]
        if prior and prior != 0:
            metrics["revenue_growth_yoy"] = {
                "value": round((latest - prior) / abs(prior), 4),
                "formula": "(latest_revenue - prior_revenue) / prior_revenue",
                "latest_period": revenue_series[0]["period_end"],
                "prior_period": revenue_series[1]["period_end"],
            }

    return metrics


def validate_balance_sheet(balance: dict) -> list[str]:
    """
    Check that total assets ≈ total liabilities + equity.
    Returns list of validation issues.
    """
    issues = []
    data = balance.get("data", {})

    assets_series = data.get("total_assets", [])
    liabilities_series = data.get("total_liabilities", [])
    equity_series = data.get("stockholders_equity", [])

    if not (assets_series and liabilities_series and equity_series):
        issues.append("Cannot validate balance sheet — missing assets, liabilities, or equity data")
        return issues

    # Check most recent period
    assets = assets_series[0]["value"]
    liabilities = liabilities_series[0]["value"]
    equity = equity_series[0]["value"]

    if assets and liabilities and equity:
        diff = abs(assets - (liabilities + equity))
        tolerance = assets * 0.01  # 1% tolerance for rounding
        if diff > tolerance:
            issues.append(
                f"Balance sheet imbalance: Assets={assets:,.0f}, "
                f"Liabilities+Equity={liabilities+equity:,.0f}, "
                f"Difference={diff:,.0f}"
            )
        else:
            issues.append(f"✓ Balance sheet balances (within 1% tolerance)")

    return issues


def extract_soc_financials(output_dir: Path) -> dict:
    """
    Main entry point: extract SOC financials via XBRL.
    Returns complete financial model dict.
    """
    from config import COMPANIES
    cik = COMPANIES["SOC"]["cik"]

    print(f"\nPhase 4: Financial Model Extraction — SOC (XBRL)")
    print(f"CIK: {cik}")

    try:
        facts = fetch_company_facts(cik)
    except Exception as e:
        return {"error": f"Failed to fetch XBRL data: {e}", "ticker": "SOC"}

    print("  Extracting income statement...")
    income = build_financial_statement(facts, INCOME_STATEMENT_MAP, "annual")

    print("  Extracting balance sheet...")
    balance = build_financial_statement(facts, BALANCE_SHEET_MAP, "annual")

    print("  Extracting cash flow statement...")
    cashflow = build_financial_statement(facts, CASH_FLOW_MAP, "annual")

    print("  Calculating derived metrics...")
    metrics = calculate_derived_metrics(income, balance, cashflow)

    print("  Validating balance sheet...")
    validation = validate_balance_sheet(balance)

    # Also extract quarterly for recent trend
    print("  Extracting quarterly data...")
    income_q = build_financial_statement(facts, INCOME_STATEMENT_MAP, "quarterly")
    cashflow_q = build_financial_statement(facts, CASH_FLOW_MAP, "quarterly")

    result = {
        "ticker": "SOC",
        "company": "Sable Offshore Corp",
        "extraction_method": "SEC EDGAR XBRL Company Facts API",
        "currency": "USD",
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

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "financial_model_raw.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved raw financial model → {out_path}")

    return result
