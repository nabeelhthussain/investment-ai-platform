"""
Central configuration for the Investment AI Platform.
All API keys loaded from .env — never hardcode them here.
Company registry loaded from companies.yaml — auto-populated on first run.
"""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
COMPANIES_YAML = BASE_DIR / "companies.yaml"

# ── API ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────
PRIMARY_MODEL = "claude-sonnet-4-5"
FAST_MODEL = "claude-haiku-4-5-20251001"

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS_PER_QUERY = 12

# ── Retrieval ──────────────────────────────────────────────────────────────
BM25_TOP_K = 20
DENSE_TOP_K = 20
FINAL_TOP_K = 12


# ── Company registry ───────────────────────────────────────────────────────

def _load_companies() -> dict:
    """Load company registry from companies.yaml. Returns empty dict if not found."""
    if not COMPANIES_YAML.exists():
        return {}
    try:
        with open(COMPANIES_YAML) as f:
            data = yaml.safe_load(f) or {}
        return data.get("companies", {})
    except Exception as e:
        print(f"Warning: could not load companies.yaml: {e}")
        return {}


def get_company(ticker: str, exchange: str = None) -> dict:
    """
    Get company config for any ticker.
    Auto-resolves unknown tickers via SEC EDGAR or LLM.
    Results cached in companies.yaml for future runs.
    """
    from ingestion.company_resolver import resolve_ticker
    return resolve_ticker(ticker.upper(), exchange)


def get_expected_docs(ticker: str) -> list:
    """Get expected document taxonomy for a ticker."""
    company = _load_companies().get(ticker.upper(), {})
    return company.get("expected_docs", [])


# Preload known companies for backwards compatibility
COMPANIES = _load_companies()
EXPECTED_DOCS = {t: c.get("expected_docs", []) for t, c in COMPANIES.items()}
