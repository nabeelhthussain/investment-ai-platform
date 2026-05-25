"""
Oslo Bors fetcher — for AKSO and other Oslo-listed companies.
Document URLs and ESEF zip locations are loaded from companies.yaml.
"""
import re
import time
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import html2text

from config import DATA_DIR, get_company

NEWSWEB_API = "https://newsweb.oslobors.no/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 InvestmentResearch research@example.com",
    "Accept": "application/json, text/html, */*",
}


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


def extract_text_from_html(html_content: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    text = h.handle(html_content)
    return re.sub(r'\n{4,}', '\n\n\n', text).strip()


def fetch_pdf_text(url: str) -> str:
    try:
        import io
        from pypdf import PdfReader
        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        r.raise_for_status()
        content = b""
        for chunk in r.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 50_000_000:
                break
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pass
        text = "\n\n".join(pages)
        return re.sub(r'\n{4,}', '\n\n\n', text).strip()
    except Exception as e:
        print(f"  PDF extraction failed for {url}: {e}")
        return ""


def fetch_newsweb_filings(ticker: str, max_items: int = 30) -> list[dict]:
    """Fetch regulatory announcements from Oslo Bors Newsweb API."""
    try:
        params = {
            "issuer": ticker,
            "fromDate": "2022-01-01",
            "toDate": datetime.now().strftime("%Y-%m-%d"),
            "limit": max_items,
        }
        r = _get(f"{NEWSWEB_API}/messages", params=params)
        data = r.json()
        return data.get("messages", data if isinstance(data, list) else [])
    except Exception as e:
        print(f"  Newsweb API unavailable ({e}), using fallback sources.")
        return []


def fetch_supplemental_urls(ticker: str, company: dict) -> list[dict]:
    """Fetch supplemental URLs defined in companies.yaml."""
    documents = []
    supplemental = company.get("supplemental_urls", [])

    for item in supplemental:
        url = item.get("url", "")
        doc_type = item.get("doc_type", "web")
        if not url:
            continue
        try:
            r = _get(url, timeout=20)
            text = extract_text_from_html(r.text)
            if len(text) < 300:
                continue
            documents.append({
                "ticker": ticker,
                "company": company.get("name", ticker),
                "doc_type": doc_type,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": url,
                "text": text[:50000],
                "section": "Web content",
                "word_count": len(text.split()),
            })
        except Exception as e:
            print(f"  Could not fetch {url}: {e}")

    return documents


def fetch_oslo_filings(ticker: str, output_dir: Path = None) -> list[dict]:
    """
    Fetch filings for any Oslo-listed company.
    Document URLs come from companies.yaml under annual_report_urls.
    """
    company = get_company(ticker)

    if output_dir is None:
        output_dir = DATA_DIR / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {ticker} filings from Oslo Bors and IR website...")

    documents = []

    # 1. Newsweb regulatory filings
    newsweb_items = fetch_newsweb_filings(ticker)
    for item in newsweb_items[:15]:
        try:
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
            filename = output_dir / f"{ticker}_newsweb_{msg_id}_{safe_date}.txt"
            filename.write_text(text, encoding="utf-8")
            documents.append({
                "ticker": ticker,
                "company": company.get("name", ticker),
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

    # 2. Annual report PDFs from companies.yaml
    annual_report_urls = company.get("annual_report_urls", [])
    for doc_meta in annual_report_urls:
        url = doc_meta.get("url", "")
        doc_type = doc_meta.get("doc_type", "annual_report")
        date = doc_meta.get("date", "")
        section = doc_meta.get("section", "")

        print(f"  Downloading {ticker} {section}...")
        text = fetch_pdf_text(url)

        if not text or len(text) < 1000:
            print(f"  Warning: {section} empty or inaccessible.")
            documents.append({
                "ticker": ticker,
                "company": company.get("name", ticker),
                "doc_type": doc_type,
                "date": date,
                "source": url,
                "text": f"[Document inaccessible: {url}]",
                "section": section,
                "word_count": 0,
                "fetch_failed": True,
            })
            continue

        safe_date = date.replace("-", "")
        filename = output_dir / f"{ticker}_{doc_type}_{safe_date}.txt"
        filename.write_text(text, encoding="utf-8")

        documents.append({
            "ticker": ticker,
            "company": company.get("name", ticker),
            "doc_type": doc_type,
            "date": date,
            "source": url,
            "text": text,
            "section": section,
            "filename": str(filename),
            "word_count": len(text.split()),
        })

    # 3. Supplemental web content from companies.yaml
    supp_docs = fetch_supplemental_urls(ticker, company)
    documents.extend(supp_docs)

    successful = sum(1 for d in documents if not d.get("fetch_failed"))
    print(f"  Fetched {len(documents)} {ticker} documents ({successful} successful).")
    return documents


# Backwards-compatible alias
def fetch_akso_filings(output_dir: Path = None) -> list[dict]:
    return fetch_oslo_filings("AKSO", output_dir)
