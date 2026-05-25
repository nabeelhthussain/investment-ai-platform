"""
SEC EDGAR fetcher — works for any US-listed company.

Fetches: 10-K, 10-Q, 8-K, DEF14A and other filings via EDGAR API.
Company CIK is resolved from companies.yaml (auto-populated by company_resolver).
"""
import json
import time
import re
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from config import DATA_DIR, get_company

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {
    "User-Agent": "InvestmentResearchPlatform research@example.com",
    "Accept-Encoding": "gzip, deflate",
}


def _get(url: str, params: dict = None, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            time.sleep(0.15)
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_submissions(cik: str) -> dict:
    cik_padded = str(int(cik.lstrip("0") or "0")).zfill(10)
    url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
    r = _get(url)
    return r.json()


def get_recent_filings(cik: str, form_types: list[str], max_per_type: int = 5) -> list[dict]:
    subs = fetch_submissions(cik)
    filings = subs.get("filings", {}).get("recent", {})

    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    results = []
    counts = {}
    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form in form_types:
            counts[form] = counts.get(form, 0)
            if counts[form] < max_per_type:
                results.append({
                    "form_type": form,
                    "filing_date": date,
                    "accession_number": acc,
                    "primary_document": doc,
                    "accession_clean": acc.replace("-", ""),
                })
                counts[form] += 1

    return sorted(results, key=lambda x: x["filing_date"], reverse=True)


def build_filing_url(cik: str, accession_clean: str, filename: str) -> str:
    cik_int = int(cik.lstrip("0") or "0")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{filename}"


def download_filing_text(filing: dict, cik: str) -> str:
    acc_clean = filing["accession_clean"]
    doc = filing["primary_document"]
    url = build_filing_url(cik, acc_clean, doc)

    try:
        r = _get(url)
        content_type = r.headers.get("content-type", "")

        if "html" in content_type or doc.endswith(".htm"):
            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = r.text

        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r' {3,}', ' ', text)
        return text.strip()

    except Exception as e:
        print(f"  Warning: could not download {url}: {e}")
        return ""


def fetch_sec_filings(ticker: str, output_dir: Path = None) -> list[dict]:
    """
    Fetch SEC filings for any ticker.
    CIK is resolved from company config (companies.yaml).
    """
    company = get_company(ticker)
    cik = company.get("cik", "")

    if not cik:
        print(f"  No CIK found for {ticker} — cannot fetch SEC filings")
        return []

    if output_dir is None:
        output_dir = DATA_DIR / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {ticker} filings from SEC EDGAR (CIK: {cik})...")

    form_types = ["10-K", "10-Q", "8-K", "DEF 14A", "S-1/A", "S-1", "20-F"]
    filings = get_recent_filings(cik, form_types, max_per_type=4)

    documents = []
    for filing in filings:
        form = filing["form_type"]
        date = filing["filing_date"]
        print(f"  Downloading {form} ({date})...")

        text = download_filing_text(filing, cik)
        if not text or len(text) < 500:
            print(f"  Warning: {form} {date} returned minimal text, skipping.")
            continue

        safe_date = date.replace("-", "")
        safe_form = form.replace("/", "_").replace(" ", "_")
        filename = output_dir / f"{ticker}_{safe_form}_{safe_date}.txt"
        filename.write_text(text, encoding="utf-8")

        documents.append({
            "ticker": ticker,
            "company": company.get("name", ticker),
            "doc_type": form,
            "date": date,
            "source": build_filing_url(cik, filing["accession_clean"], filing["primary_document"]),
            "text": text,
            "filename": str(filename),
            "section": "full_filing",
            "word_count": len(text.split()),
        })

    print(f"  Fetched {len(documents)} {ticker} documents.")
    return documents


# Backwards-compatible alias
def fetch_soc_filings(output_dir: Path = None) -> list[dict]:
    return fetch_sec_filings("SOC", output_dir)
