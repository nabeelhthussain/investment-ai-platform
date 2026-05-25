"""
ESEF / iXBRL Extractor for European filers (AKSO).

Aker Solutions publishes annual reports in ESEF format (iXBRL embedded in HTML).
The .zip files on their annual reports page contain machine-readable financial data
tagged with IFRS XBRL concepts — same idea as SEC XBRL but for European filers.

Fallback: if ESEF parsing fails, extract from PDF using pdfplumber table detection.
"""
import io
import json
import re
import zipfile
import time
import requests
from pathlib import Path
from collections import defaultdict

HEADERS = {"User-Agent": "Mozilla/5.0 InvestmentResearch research@example.com"}

# ESEF zip URLs for AKSO annual reports
AKSO_ESEF_URLS = [
    {
        "url": "https://www.akersolutions.com/globalassets/investors/agm/2026/5967007lieexzxg42836-2025-12-31-en.zip",
        "year": 2025,
        "period_end": "2025-12-31",
    },
    {
        "url": "https://www.akersolutions.com/globalassets/investors/agm/2025/5967007lieexzxg42836-2024-12-31-en.zip",
        "year": 2024,
        "period_end": "2024-12-31",
    },
    {
        "url": "https://www.akersolutions.com/globalassets/investors/agm/2024/5967007lieexzxg42836-2023-12-31-en.zip",
        "year": 2023,
        "period_end": "2023-12-31",
    },
]

# IFRS concept → canonical field mapping
IFRS_INCOME_MAP = {
    "revenue": [
        "ifrs-full:Revenue",
        "ifrs-full:RevenueFromContractsWithCustomers",
        "ifrs-full:SalesAndOtherOperatingRevenue",
    ],
    "gross_profit": ["ifrs-full:GrossProfit"],
    "operating_income": [
        "ifrs-full:ProfitLossFromOperatingActivities",
        "ifrs-full:OperatingProfitLoss",
    ],
    "ebitda": ["ifrs-full:EarningsBeforeInterestTaxesDepreciationAndAmortisation"],
    "da": [
        "ifrs-full:DepreciationAndAmortisationExpense",
        "ifrs-full:DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss",
    ],
    "interest_expense": [
        "ifrs-full:FinanceCosts",
        "ifrs-full:InterestExpense",
    ],
    "net_income": [
        "ifrs-full:ProfitLoss",
        "ifrs-full:ProfitLossAttributableToOwnersOfParent",
    ],
    "personnel_costs": ["ifrs-full:EmployeeBenefitsExpense"],
}

IFRS_BALANCE_MAP = {
    "total_assets": ["ifrs-full:Assets"],
    "current_assets": ["ifrs-full:CurrentAssets"],
    "cash": [
        "ifrs-full:CashAndCashEquivalents",
        "ifrs-full:CashAndBankBalancesAtCentralBanks",
    ],
    "total_liabilities": ["ifrs-full:Liabilities"],
    "current_liabilities": ["ifrs-full:CurrentLiabilities"],
    "long_term_debt": [
        "ifrs-full:NoncurrentPortionOfLongtermBorrowings",
        "ifrs-full:Borrowings",
    ],
    "equity": [
        "ifrs-full:Equity",
        "ifrs-full:EquityAttributableToOwnersOfParent",
    ],
}

IFRS_CASHFLOW_MAP = {
    "operating_cf": ["ifrs-full:CashFlowsFromUsedInOperatingActivities"],
    "investing_cf": ["ifrs-full:CashFlowsFromUsedInInvestingActivities"],
    "financing_cf": ["ifrs-full:CashFlowsFromUsedInFinancingActivities"],
    "capex": [
        "ifrs-full:PurchaseOfPropertyPlantAndEquipment",
        "ifrs-full:AcquisitionOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    ],
}


def _download_zip(url: str) -> bytes | None:
    """Download a zip file, return raw bytes."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        r.raise_for_status()
        content = b""
        for chunk in r.iter_content(8192):
            content += chunk
            if len(content) > 100_000_000:  # 100MB cap
                break
        return content
    except Exception as e:
        print(f"  Could not download {url}: {e}")
        return None


def _parse_ixbrl(html_content: str) -> list[dict]:
    """
    Parse iXBRL HTML to extract tagged financial values.
    iXBRL uses inline tags like:
      <ix:nonFraction name="ifrs-full:Revenue" contextRef="..." decimals="-6">53200</ix:nonFraction>
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html_content, "lxml")
    facts = []

    # Find all iXBRL non-fraction elements (numeric financial data)
    for tag in soup.find_all(re.compile(r"ix:nonfraction", re.I)):
        name = tag.get("name", "")
        context_ref = tag.get("contextref", "")
        decimals = tag.get("decimals", "0")
        scale = tag.get("scale", "0")
        sign = tag.get("sign", "")

        text = tag.get_text(strip=True).replace(",", "").replace(" ", "")
        if not text or not name:
            continue

        try:
            value = float(text)
            # Apply scale (ESEF often uses scale=6 for millions)
            scale_int = int(scale) if scale else 0
            if scale_int:
                value = value * (10 ** scale_int)
            # Apply sign
            if sign == "-":
                value = -abs(value)
            # Apply decimals scaling
            decimals_int = int(decimals) if decimals and decimals != "INF" else 0
            if decimals_int < 0:
                value = value * (10 ** abs(decimals_int))

            facts.append({
                "concept": name,
                "value": value,
                "context_ref": context_ref,
                "decimals": decimals,
            })
        except (ValueError, TypeError):
            continue

    return facts


def _extract_from_zip(zip_bytes: bytes, year: int) -> list[dict]:
    """Extract iXBRL facts from an ESEF zip file."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Find the main iXBRL HTML file
            html_files = [
                name for name in zf.namelist()
                if name.endswith((".htm", ".html", ".xhtml"))
                and "report" in name.lower() or name.endswith("-en.htm")
            ]

            if not html_files:
                # Fall back to any HTML file
                html_files = [
                    name for name in zf.namelist()
                    if name.endswith((".htm", ".html", ".xhtml"))
                ]

            if not html_files:
                print(f"  No HTML files found in zip for {year}")
                return []

            # Use the largest HTML file (most likely the full report)
            html_files.sort(key=lambda x: zf.getinfo(x).file_size, reverse=True)
            main_file = html_files[0]

            print(f"  Parsing iXBRL from: {main_file} ({zf.getinfo(main_file).file_size:,} bytes)")
            html_content = zf.read(main_file).decode("utf-8", errors="replace")
            return _parse_ixbrl(html_content)

    except Exception as e:
        print(f"  Error parsing zip for {year}: {e}")
        return []


def _map_facts_to_schema(facts: list[dict], field_map: dict, period_end: str) -> dict:
    """Map extracted iXBRL facts to canonical schema fields."""
    # Group facts by concept
    concept_values = defaultdict(list)
    for fact in facts:
        concept_values[fact["concept"]].append(fact["value"])

    statement = {}
    source_map = {}

    for canonical_field, ifrs_concepts in field_map.items():
        for concept in ifrs_concepts:
            # Try exact match first, then suffix match
            values = concept_values.get(concept, [])
            if not values:
                # Try matching just the local name part (after colon)
                local_name = concept.split(":")[-1] if ":" in concept else concept
                for k, v in concept_values.items():
                    if k.endswith(local_name):
                        values = v
                        break

            if values:
                # Take the most common value (handles duplicates from multiple contexts)
                from collections import Counter
                most_common = Counter(values).most_common(1)[0][0]
                statement[canonical_field] = [{
                    "period_end": period_end,
                    "value": most_common,
                    "form": "annual_report",
                    "filed": period_end,
                }]
                source_map[canonical_field] = concept
                break

    return {"data": statement, "source_concepts": source_map}


def extract_akso_financials_esef(output_dir: Path) -> dict:
    """
    Extract AKSO financials from ESEF iXBRL zip files.
    Returns multi-year financial model dict.
    """
    print(f"\nPhase 4: Financial Model Extraction — AKSO (ESEF/iXBRL)")

    all_income = []
    all_balance = []
    all_cashflow = []

    for doc_meta in AKSO_ESEF_URLS:
        year = doc_meta["year"]
        print(f"  Downloading ESEF zip for {year}...")
        zip_bytes = _download_zip(doc_meta["url"])

        if not zip_bytes:
            print(f"  Skipping {year} — download failed")
            continue

        print(f"  Extracting iXBRL facts from {year}...")
        facts = _extract_from_zip(zip_bytes, year)

        if not facts:
            print(f"  No iXBRL facts extracted for {year}")
            continue

        print(f"  Extracted {len(facts)} iXBRL facts for {year}")

        period_end = doc_meta["period_end"]
        income = _map_facts_to_schema(facts, IFRS_INCOME_MAP, period_end)
        balance = _map_facts_to_schema(facts, IFRS_BALANCE_MAP, period_end)
        cashflow = _map_facts_to_schema(facts, IFRS_CASHFLOW_MAP, period_end)

        all_income.append({"year": year, "period_end": period_end, "data": income})
        all_balance.append({"year": year, "period_end": period_end, "data": balance})
        all_cashflow.append({"year": year, "period_end": period_end, "data": cashflow})

    if not all_income:
        print("  ESEF extraction failed for all years — falling back to PDF extraction")
        return extract_akso_financials_pdf(output_dir)

    # Build multi-year statements
    result = {
        "ticker": "AKSO",
        "company": "Aker Solutions ASA",
        "extraction_method": "ESEF iXBRL",
        "currency": "NOK",
        "units": "as_reported_millions",
        "years_extracted": [d["year"] for d in all_income],
        "annual": {
            "income_statement": all_income,
            "balance_sheet": all_balance,
            "cash_flow": all_cashflow,
        },
        "validation": _validate_akso(all_income, all_balance),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "financial_model_raw.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved raw financial model → {out_path}")

    return result


def extract_akso_financials_pdf(output_dir: Path) -> dict:
    """
    PDF fallback: extract key financial tables from AKSO annual report PDFs
    using pdfplumber. Less reliable than iXBRL but better than nothing.
    """
    print("  Attempting PDF table extraction as fallback...")

    # Look for already-downloaded PDFs in data dir
    data_dir = Path("data/AKSO")
    pdf_texts = list(data_dir.glob("*.txt")) if data_dir.exists() else []

    if not pdf_texts:
        return {
            "ticker": "AKSO",
            "error": "No PDF text files found. Run ingestion pipeline first.",
            "extraction_method": "pdf_fallback_unavailable",
        }

    # Use LLM to extract key figures from narrative text
    from models.router import call_llm
    all_results = []

    for txt_file in sorted(pdf_texts, reverse=True)[:2]:  # Most recent 2
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        # Focus on financial highlights section
        financial_section = _find_financial_section(text)

        if not financial_section:
            continue

        prompt = f"""Extract key financial figures from this Aker Solutions annual report text.
Return ONLY a JSON object with these fields (use null if not found):
{{
  "period_end": "YYYY-12-31",
  "currency": "NOK",
  "units": "millions",
  "revenue": number or null,
  "ebitda": number or null,
  "ebitda_margin_pct": number or null,
  "net_income": number or null,
  "total_assets": number or null,
  "cash": number or null,
  "net_debt": number or null,
  "order_intake": number or null,
  "order_backlog": number or null,
  "operating_cf": number or null,
  "capex": number or null,
  "headcount": number or null
}}

Text:
{financial_section[:4000]}

Return ONLY the JSON object, no other text."""

        try:
            response = call_llm(prompt, max_tokens=500, temperature=0.0)
            # Strip markdown code fences if present
            clean = response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"```[a-z]*\n?", "", clean).strip("` \n")
            figures = json.loads(clean)
            all_results.append(figures)
            print(f"  Extracted figures from {txt_file.name}")
        except Exception as e:
            print(f"  LLM extraction failed for {txt_file.name}: {e}")

    return {
        "ticker": "AKSO",
        "company": "Aker Solutions ASA",
        "extraction_method": "pdf_llm_extraction",
        "currency": "NOK",
        "units": "millions",
        "extracted_periods": all_results,
        "note": "Extracted via LLM from PDF text — verify against source before use in model",
    }


def _find_financial_section(text: str) -> str:
    """Find the financial highlights or key figures section in report text."""
    markers = [
        "financial highlights",
        "key figures",
        "financial summary",
        "consolidated income",
        "revenue",
        "ebitda",
    ]
    text_lower = text.lower()
    for marker in markers:
        idx = text_lower.find(marker)
        if idx > 0:
            return text[max(0, idx-200):idx+3000]
    return text[:3000]  # Fall back to start of document


def _validate_akso(all_income: list, all_balance: list) -> list[str]:
    """Basic validation checks for AKSO extracted data."""
    issues = []

    if not all_income:
        issues.append("No income statement data extracted")
        return issues

    # Check revenue is positive and reasonable for AKSO (NOK billions range)
    for year_data in all_income:
        income_data = year_data.get("data", {}).get("data", {})
        revenue_list = income_data.get("revenue", [])
        if revenue_list:
            rev = revenue_list[0].get("value", 0)
            if rev and rev > 0:
                issues.append(f"✓ Revenue extracted for {year_data['year']}: {rev:,.0f} NOK")
            else:
                issues.append(f"⚠ Revenue missing or zero for {year_data['year']}")

    return issues
