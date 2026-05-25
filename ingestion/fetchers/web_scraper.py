"""
Generic web scraper for supplemental IR content.
Works for any company — uses IR URL from company config if available.
"""
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import html2text

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; InvestmentResearch/1.0)",
    "Accept": "text/html,application/xhtml+xml,*/*",
}


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
    """
    Fetch supplemental web content for any ticker.
    Uses IR URL from company config if available.
    """
    from config import get_company
    try:
        company = get_company(ticker)
    except Exception:
        return []

    company_name = company.get("name", ticker)
    ir_url = company.get("ir_url")

    if not ir_url:
        # Try common IR URL patterns
        domain_guesses = [
            f"https://www.{company_name.lower().replace(' ', '').replace(',', '')[:20]}.com/investors",
        ]
        urls_to_try = domain_guesses
    else:
        urls_to_try = [ir_url]

    documents = []
    for url in urls_to_try[:2]:
        print(f"  Fetching supplemental: {url}")
        html = _safe_get(url)
        if not html:
            continue
        text = html_to_text(html)
        if len(text) < 300:
            continue
        documents.append({
            "ticker": ticker,
            "company": company_name,
            "doc_type": "ir_website",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": url,
            "text": text[:30000],
            "section": "IR website",
            "word_count": len(text.split()),
        })

    return documents
