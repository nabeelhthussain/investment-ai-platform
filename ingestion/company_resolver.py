"""
Company Resolver — automatically resolves any ticker to full company metadata.

Resolution order:
  1. companies.yaml cache (instant, no network)
  2. SEC EDGAR tickers.json (covers all ~10k US-listed companies)
  3. LLM resolution (Claude identifies non-US companies)
  4. Raises ValueError with helpful message if all fail

Usage:
  from ingestion.company_resolver import resolve_ticker
  company = resolve_ticker("MSFT")
  company = resolve_ticker("SHEL", exchange="LSE")
"""
import json
import re
import time
import requests
import yaml
from pathlib import Path

HEADERS = {"User-Agent": "InvestmentResearchPlatform research@example.com"}
COMPANIES_YAML = Path(__file__).parent.parent / "companies.yaml"


# ── Cache layer ────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not COMPANIES_YAML.exists():
        return {}
    try:
        with open(COMPANIES_YAML) as f:
            data = yaml.safe_load(f) or {}
        return data.get("companies", {})
    except Exception:
        return {}


def _save_to_cache(ticker: str, company: dict):
    """Persist a resolved company to companies.yaml."""
    data = {}
    if COMPANIES_YAML.exists():
        try:
            with open(COMPANIES_YAML) as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            pass

    data.setdefault("companies", {})[ticker.upper()] = company

    with open(COMPANIES_YAML, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  Saved {ticker} to companies.yaml cache")


def _load_from_cache(ticker: str) -> dict | None:
    cache = _load_cache()
    return cache.get(ticker.upper())


# ── SEC EDGAR resolution ───────────────────────────────────────────────────

_edgar_tickers_cache = None


def _get_edgar_tickers() -> dict:
    """Fetch SEC EDGAR company_tickers.json — maps ticker → CIK + name."""
    global _edgar_tickers_cache
    if _edgar_tickers_cache is not None:
        return _edgar_tickers_cache

    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        _edgar_tickers_cache = r.json()
        return _edgar_tickers_cache
    except Exception as e:
        print(f"  Could not fetch EDGAR tickers: {e}")
        return {}


def _resolve_via_edgar(ticker: str) -> dict | None:
    """
    Look up a ticker in SEC EDGAR.
    Returns company config dict if found, None otherwise.
    """
    tickers_data = _get_edgar_tickers()
    if not tickers_data:
        return None

    ticker_upper = ticker.upper()
    match = None

    for _, entry in tickers_data.items():
        if entry.get("ticker", "").upper() == ticker_upper:
            match = entry
            break

    if not match:
        return None

    cik_raw = str(match["cik_str"])
    cik_padded = cik_raw.zfill(10)
    name = match["title"]

    # Get submission details for exchange and SIC
    try:
        sub_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        r = requests.get(sub_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        sub = r.json()
        time.sleep(0.15)

        sic_desc = sub.get("sicDescription", "")
        exchanges = sub.get("exchanges", [])
        exchange = exchanges[0] if exchanges else "NYSE"

    except Exception:
        sic_desc = ""
        exchange = "NYSE"

    return {
        "name": name,
        "exchange": exchange,
        "sector": sic_desc,
        "description": f"{name} — {sic_desc}",
        "currency": "USD",
        "reporting_standard": "US GAAP",
        "country": "USA",
        "cik": f"0{cik_raw.lstrip('0') or '0'}",
        "fetcher": "sec_edgar",
        "expected_docs": _default_sec_docs(),
        "auto_resolved": True,
        "resolution_method": "sec_edgar",
    }


# ── LLM resolution ─────────────────────────────────────────────────────────

def _resolve_via_llm(ticker: str, exchange: str = None) -> dict | None:
    """
    Use Claude to identify a non-US or unrecognised company from its ticker.
    Populates enough metadata to run the generic_web fetcher.
    """
    try:
        from models.router import call_llm_fast
    except ImportError:
        return None

    exchange_hint = f" listed on {exchange}" if exchange else ""

    prompt = f"""I need to identify the company with stock ticker: {ticker}{exchange_hint}

Return ONLY a valid JSON object. No markdown, no explanation, just JSON.

{{
  "name": "Full legal company name",
  "exchange": "Primary exchange code e.g. NYSE, NASDAQ, OSE, LSE, TSX, ASX",
  "sector": "Industry sector description",
  "description": "One sentence description of what the company does",
  "currency": "Primary reporting currency e.g. USD, EUR, GBP, NOK, AUD",
  "reporting_standard": "US GAAP or IFRS",
  "country": "Country of incorporation",
  "fetcher": "sec_edgar if US/Canada SEC filer, oslo_bors if Oslo Stock Exchange, generic_web otherwise",
  "ir_url": "Investor relations page URL or null",
  "cik": "SEC CIK number if US-listed, else null"
}}

If you don't recognise this ticker at all, return {{"error": "unknown"}}"""

    try:
        response = call_llm_fast(prompt, max_tokens=300)
        clean = response.strip()
        # Strip markdown fences if present
        clean = re.sub(r"```[a-z]*\n?", "", clean).strip("`").strip()
        data = json.loads(clean)

        if "error" in data:
            return None

        # Fill in defaults for missing fields
        data.setdefault("fetcher", "generic_web")
        data.setdefault("expected_docs", _default_intl_docs())
        data["auto_resolved"] = True
        data["resolution_method"] = "llm"
        return data

    except Exception as e:
        print(f"  LLM resolution failed: {e}")
        return None


# ── Generic web fetcher config ─────────────────────────────────────────────

def _build_generic_web_config(company: dict, ticker: str) -> dict:
    """
    For companies resolved via LLM, add generic web fetcher configuration.
    The generic fetcher will use the IR URL to scrape documents.
    """
    ir_url = company.get("ir_url")
    if not ir_url:
        # Try to construct a plausible IR URL search
        name_slug = company.get("name", ticker).lower().replace(" ", "")
        company["ir_search_query"] = f"{company.get('name', ticker)} investor relations annual report"

    return company


# ── Default expected doc taxonomies ───────────────────────────────────────

def _default_sec_docs() -> list:
    return [
        {"type": "10-K",         "description": "Annual report",               "criticality": "CRITICAL"},
        {"type": "10-Q",         "description": "Quarterly reports",            "criticality": "HIGH"},
        {"type": "8-K",          "description": "Material event disclosures",   "criticality": "HIGH"},
        {"type": "DEF14A",       "description": "Proxy statement",              "criticality": "MEDIUM"},
        {"type": "earnings_call","description": "Earnings call transcripts",    "criticality": "HIGH"},
        {"type": "credit_rating","description": "Credit rating report",         "criticality": "HIGH"},
    ]


def _default_intl_docs() -> list:
    return [
        {"type": "annual_report", "description": "Annual report",              "criticality": "CRITICAL"},
        {"type": "quarterly",     "description": "Quarterly/interim reports",  "criticality": "HIGH"},
        {"type": "earnings_call", "description": "Earnings call transcripts",  "criticality": "HIGH"},
        {"type": "esg_report",    "description": "Sustainability/ESG report",  "criticality": "MEDIUM"},
        {"type": "credit_rating", "description": "Credit rating report",       "criticality": "HIGH"},
    ]


# ── Main entry point ───────────────────────────────────────────────────────

def resolve_ticker(ticker: str, exchange: str = None) -> dict:
    """
    Resolve any ticker to a full company config dict.

    Args:
        ticker:   Stock ticker symbol (e.g. "MSFT", "BP", "AKSO")
        exchange: Optional exchange hint for ambiguous non-US tickers

    Returns:
        Company config dict ready for use by fetchers and agents.

    Raises:
        ValueError: If the ticker cannot be resolved by any method.
    """
    ticker = ticker.upper().strip()

    # 1. Cache
    cached = _load_from_cache(ticker)
    if cached:
        print(f"  {ticker}: loaded from companies.yaml cache")
        return cached

    # 2. SEC EDGAR (all US-listed companies)
    print(f"  {ticker}: querying SEC EDGAR...")
    company = _resolve_via_edgar(ticker)

    if company:
        _save_to_cache(ticker, company)
        return company

    # 3. LLM resolution (non-US or unrecognised)
    print(f"  {ticker}: not found in EDGAR, trying LLM resolution...")
    company = _resolve_via_llm(ticker, exchange)

    if company:
        company = _build_generic_web_config(company, ticker)
        _save_to_cache(ticker, company)
        return company

    # 4. Give up with a helpful message
    raise ValueError(
        f"\nCould not resolve ticker '{ticker}'.\n"
        f"Options:\n"
        f"  1. Add --exchange flag: --ticker {ticker} --exchange LSE\n"
        f"  2. Add manually to companies.yaml\n"
        f"  3. Check the ticker is correct"
    )


def list_cached_companies() -> list[str]:
    """Return list of tickers currently in the cache."""
    return list(_load_cache().keys())
