"""
Oslo Børs / Aker Solutions fetcher.

Sources:
  1. Newsweb API (Oslo Børs regulatory filings) — structured JSON
  2. Aker Solutions IR page (annual reports, presentations)
  3. Fallback: direct PDF links for known annual reports
"""
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import html2text

from config import DATA_DIR

NEWSWEB_API = "https://newsweb.oslobors.no/api"
AKSO_IR_URL = "https://www.akersolutions.com/investors"
HEADERS = {
    "User-Agent": "Mozilla/5.0 InvestmentResearch research@example.com",
    "Accept": "application/json, text/html, */*",
}

AKSO_SUPPLEMENTAL_URLS = [
    ("https://www.akersolutions.com/news/news-archive/2025/aker-solutions-asa-annual-remuneration-and-corporate-governance-reports-for-2024/", "reg_filing"),
    ("https://www.akersolutions.com/news/news-archive/2025/aker-solutions-asafourth-quarter-and-full-year-2024-results/", "quarterly"),
    ("https://www.akersolutions.com/news/news-archive/", "news"),
]

# Known direct PDF URLs for AKSO annual reports (fallback)
AKSO_KNOWN_DOCS = [
    {
        "url": "https://www.akersolutions.com/globalassets/investors/agm/2026/aker-solutions-annual-report-2025.pdf",
        "doc_type": "annual_report",
        "date": "2025-03-14",
        "section": "Annual Report 2025",
    },
    {
        "url": "https://www.akersolutions.com/globalassets/investors/agm/2025/akso---annual-report-2024.pdf",
        "doc_type": "annual_report",
        "date": "2024-03-14",
        "section": "Annual Report 2024",
    },
    {
        "url": "https://www.akersolutions.com/globalassets/investors/agm/2024/akso---annual-report-2023---doc.pdf",
        "doc_type": "annual_report",
        "date": "2023-03-09",
        "section": "Annual Report 2023",
    },
    {
        "url": "https://www.akersolutions.com/globalassets/huginreport/2022/annual-report-2022.pdf",
        "doc_type": "annual_report",
        "date": "2022-03-10",
        "section": "Annual Report 2022",
    },
    {
        "url": "https://www.akersolutions.com/globalassets/global/downloads/half-year-results-2024.pdf",
        "doc_type": "quarterly",
        "date": "2024-08-15",
        "section": "Half Year Results 2024",
    },
    {
        "url": "https://www.akersolutions.com/globalassets/investors/agm/2025/remuneration-report-2024.pdf",
        "doc_type": "esg_report",
        "date": "2025-03-31",
        "section": "Remuneration Report 2024",
    },
]


def _get(url: str, params: dict = None, timeout: int = 30) -> requests.Response:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
            r.raise_for_status()
            time.sleep(0.3)
            return r
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def fetch_newsweb_filings(issuer: str = "AKSO", max_items: int = 30) -> list[dict]:
    """
    Fetch regulatory announcements from Oslo Børs Newsweb API.
    Returns list of filing metadata dicts.
    """
    try:
        params = {
            "issuer": issuer,
            "fromDate": (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d"),
            "toDate": datetime.now().strftime("%Y-%m-%d"),
            "limit": max_items,
        }
        # Try the newsweb search endpoint
        url = f"{NEWSWEB_API}/messages"
        r = _get(url, params=params)
        data = r.json()
        return data.get("messages", data if isinstance(data, list) else [])
    except Exception as e:
        print(f"  Newsweb API unavailable ({e}), using fallback sources.")
        return []


def scrape_akso_ir_page() -> list[dict]:
    """
    Scrape the Aker Solutions investor relations page for document links.
    Returns list of discovered document URLs with metadata.
    """
    try:
        r = _get(AKSO_IR_URL, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")

        docs = []
        # Look for PDF links and report links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()

            if any(kw in text for kw in ["annual report", "quarterly", "interim", "presentation", "q1", "q2", "q3", "q4"]):
                if href.startswith("/"):
                    href = "https://www.akersolutions.com" + href
                docs.append({
                    "url": href,
                    "title": a.get_text(strip=True),
                    "doc_type": _classify_doc(text),
                })

        return docs[:20]
    except Exception as e:
        print(f"  IR page scraping failed ({e}), using known docs only.")
        return []


def _classify_doc(text: str) -> str:
    text = text.lower()
    if "annual" in text:
        return "annual_report"
    if any(q in text for q in ["q1", "q2", "q3", "q4", "quarterly", "interim"]):
        return "quarterly"
    if "presentation" in text or "capital market" in text:
        return "investor_pres"
    if "esg" in text or "sustainability" in text:
        return "esg_report"
    return "regulatory"


def extract_text_from_html(html_content: str) -> str:
    """Convert HTML to clean markdown-ish text."""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0  # Don't wrap
    text = h.handle(html_content)
    # Clean up
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


def fetch_pdf_text(url: str) -> str:
    """
    Download a PDF and extract text using pypdf.
    Falls back to empty string on failure.
    """
    try:
        import io
        from pypdf import PdfReader

        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        r.raise_for_status()

        content = b""
        for chunk in r.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 50_000_000:  # 50MB cap
                break

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pass

        text = "\n\n".join(pages)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text.strip()

    except Exception as e:
        print(f"  PDF extraction failed for {url}: {e}")
        return ""


def fetch_akso_web_content() -> list[dict]:
    """
    Fetch publicly available web content about Aker Solutions:
    press releases, news, and investor content from their website.
    """
    documents = []

    # Try fetching quarterly report pages
    quarterly_urls = [
        ("https://www.akersolutions.com/investors/reports-and-presentations/quarterly-reports/", "quarterly_index"),
    ]

    for url, doc_type in quarterly_urls:
        try:
            r = _get(url, timeout=20)
            text = extract_text_from_html(r.text)
            if len(text) > 500:
                documents.append({
                    "ticker": "AKSO",
                    "company": "Aker Solutions ASA",
                    "doc_type": doc_type,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": url,
                    "text": text[:50000],
                    "section": "IR website",
                    "word_count": len(text.split()),
                })
        except Exception as e:
            print(f"  Could not fetch {url}: {e}")

    return documents


def fetch_akso_filings(output_dir: Path = None) -> list[dict]:
    """
    Main entry point: fetch all available AKSO documents.
    Returns list of document dicts ready for the ingestion pipeline.
    """
    if output_dir is None:
        output_dir = DATA_DIR / "AKSO"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching AKSO filings from Oslo Børs and Aker Solutions IR...")

    documents = []

    # 1. Try Newsweb API
    newsweb_items = fetch_newsweb_filings()
    for item in newsweb_items[:15]:
        try:
            # Newsweb items have a messageId and can be fetched
            msg_id = item.get("messageId") or item.get("id")
            if not msg_id:
                continue

            url = f"https://newsweb.oslobors.no/message/{msg_id}"
            r = _get(url, timeout=20)
            text = extract_text_from_html(r.text)

            if len(text) < 200:
                continue

            pub_date = item.get("publishedTime", item.get("date", ""))[:10]
            title = item.get("header", item.get("title", "Regulatory filing"))

            safe_date = pub_date.replace("-", "")
            filename = output_dir / f"AKSO_newsweb_{msg_id}_{safe_date}.txt"
            filename.write_text(text, encoding="utf-8")

            documents.append({
                "ticker": "AKSO",
                "company": "Aker Solutions ASA",
                "doc_type": "reg_filing",
                "date": pub_date,
                "source": url,
                "text": text,
                "section": title,
                "filename": str(filename),
                "word_count": len(text.split()),
            })
        except Exception as e:
            print(f"  Newsweb item error: {e}")

    # 2. Try known PDF annual reports
    for doc_meta in AKSO_KNOWN_DOCS:
        print(f"  Downloading AKSO annual report ({doc_meta['date'][:4]})...")
        text = fetch_pdf_text(doc_meta["url"])

        if not text or len(text) < 1000:
            print(f"  Warning: annual report {doc_meta['date'][:4]} empty or inaccessible.")
            # Create a placeholder so the audit knows we tried
            documents.append({
                "ticker": "AKSO",
                "company": "Aker Solutions ASA",
                "doc_type": doc_meta["doc_type"],
                "date": doc_meta["date"],
                "source": doc_meta["url"],
                "text": f"[Document inaccessible: {doc_meta['url']}]",
                "section": doc_meta["section"],
                "word_count": 0,
                "fetch_failed": True,
            })
            continue

        safe_date = doc_meta["date"].replace("-", "")
        filename = output_dir / f"AKSO_annual_{safe_date}.txt"
        filename.write_text(text, encoding="utf-8")

        documents.append({
            "ticker": "AKSO",
            "company": "Aker Solutions ASA",
            "doc_type": doc_meta["doc_type"],
            "date": doc_meta["date"],
            "source": doc_meta["url"],
            "text": text,
            "section": doc_meta["section"],
            "filename": str(filename),
            "word_count": len(text.split()),
        })

    # 3. Web content as supplement
    web_docs = fetch_akso_web_content()
    documents.extend(web_docs)

    print(f"  Fetched {len(documents)} AKSO documents ({sum(1 for d in documents if not d.get('fetch_failed'))} successful).")
    return documents
