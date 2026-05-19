"""
Web scraper for supplemental content: news articles, press releases,
analyst summaries from public sources.
"""
import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import html2text

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; InvestmentResearch/1.0)",
    "Accept": "text/html,application/xhtml+xml,*/*",
}

# Public sources to check per company
SOC_SUPPLEMENTAL_URLS = [
    ("https://www.sableoffshore.com/investors", "ir_website"),
]

AKSO_SUPPLEMENTAL_URLS = [
    ("https://www.akersolutions.com/news/", "news"),
]


def _safe_get(url: str, timeout: int = 20) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        time.sleep(0.5)
        return r.text
    except Exception as e:
        print(f"  Could not fetch {url}: {e}")
        return None


def html_to_text(html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0
    text = h.handle(html)
    return re.sub(r'\n{4,}', '\n\n\n', text).strip()


def fetch_supplemental(ticker: str) -> list[dict]:
    """Fetch supplemental web content for a given ticker."""
    urls = SOC_SUPPLEMENTAL_URLS if ticker == "SOC" else AKSO_SUPPLEMENTAL_URLS
    documents = []

    for url, doc_type in urls:
        print(f"  Fetching supplemental: {url}")
        html = _safe_get(url)
        if not html:
            continue

        text = html_to_text(html)
        if len(text) < 300:
            continue

        documents.append({
            "ticker": ticker,
            "company": "Sable Offshore Corp" if ticker == "SOC" else "Aker Solutions ASA",
            "doc_type": doc_type,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": url,
            "text": text[:30000],
            "section": "Web content",
            "word_count": len(text.split()),
        })

    return documents
