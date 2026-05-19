"""
SEC EDGAR fetcher for Sable Offshore Corp (SOC).

Uses the EDGAR full-text search and submissions API — no API key required.
Fetches: 10-K, 10-Q, 8-K, DEF14A filings + downloads PDFs/HTMs.
"""
import json
import time
import re
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from config import DATA_DIR

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
HEADERS = {
    "User-Agent": "InvestmentResearchPlatform research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

SOC_CIK = "0001831481"


def _get(url: str, params: dict = None, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            time.sleep(0.15)  # EDGAR rate limit: 10 req/sec max
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_submissions(cik: str) -> dict:
    """Get all filing metadata for a company from EDGAR submissions API."""
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
    r = _get(url)
    return r.json()


def get_recent_filings(cik: str, form_types: list[str], max_per_type: int = 5) -> list[dict]:
    """
    Return metadata for the most recent filings of each form type.
    Returns list of dicts with: accession_number, form_type, filing_date, primary_document.
    """
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
    cik_padded = cik.lstrip("0").zfill(10)
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{accession_clean}/{filename}"


def download_filing_text(filing: dict, cik: str) -> str:
    """Download and return the text content of a filing."""
    acc_clean = filing["accession_clean"]
    doc = filing["primary_document"]
    url = build_filing_url(cik, acc_clean, doc)

    try:
        r = _get(url)
        content_type = r.headers.get("content-type", "")

        if "html" in content_type or doc.endswith(".htm"):
            soup = BeautifulSoup(r.text, "lxml")
            # Remove scripts, styles, navigation
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = r.text

        # Clean up excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r' {3,}', ' ', text)
        return text.strip()

    except Exception as e:
        print(f"  Warning: could not download {url}: {e}")
        return ""


def fetch_soc_filings(output_dir: Path = None) -> list[dict]:
    """
    Main entry point: fetch all relevant SOC filings.
    Returns list of document dicts ready for the ingestion pipeline.
    """
    if output_dir is None:
        output_dir = DATA_DIR / "SOC"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching SOC filings from SEC EDGAR...")

    form_types = ["10-K", "10-Q", "8-K", "DEF 14A", "S-1/A", "S-1"]
    filings = get_recent_filings(SOC_CIK, form_types, max_per_type=4)

    documents = []
    for filing in filings:
        form = filing["form_type"]
        date = filing["filing_date"]
        print(f"  Downloading {form} ({date})...")

        text = download_filing_text(filing, SOC_CIK)
        if not text or len(text) < 500:
            print(f"  Warning: {form} {date} returned minimal text, skipping.")
            continue

        # Save raw text
        safe_date = date.replace("-", "")
        safe_form = form.replace("/", "_").replace(" ", "_")
        filename = output_dir / f"SOC_{safe_form}_{safe_date}.txt"
        filename.write_text(text, encoding="utf-8")

        documents.append({
            "ticker": "SOC",
            "company": "Sable Offshore Corp",
            "doc_type": form,
            "date": date,
            "source": build_filing_url(
                SOC_CIK,
                filing["accession_clean"],
                filing["primary_document"]
            ),
            "text": text,
            "filename": str(filename),
            "section": "full_filing",
            "word_count": len(text.split()),
        })

    print(f"  Fetched {len(documents)} SOC documents.")
    return documents


def fetch_soc_news(max_items: int = 10) -> list[dict]:
    """
    Search SEC EDGAR full-text search for recent SOC mentions.
    Supplements filings with 8-K press releases.
    """
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": '"Sable Offshore"',
        "dateRange": "custom",
        "startdt": (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"),
        "enddt": datetime.now().strftime("%Y-%m-%d"),
        "forms": "8-K",
    }
    try:
        r = _get("https://efts.sec.gov/LATEST/search-index", params=params)
        # EDGAR returns JSON for this endpoint
        return []  # Placeholder — 8-Ks already captured in fetch_soc_filings
    except Exception:
        return []
