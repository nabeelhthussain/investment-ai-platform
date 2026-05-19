"""
Central configuration for the Investment AI Platform.
All API keys loaded from .env — never hardcode them here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── API ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────
# Primary reasoning model (bear case, contradiction, verdict, synthesis)
PRIMARY_MODEL = "claude-sonnet-4-5"
# Fast model for cheaper tasks (metadata extraction, brief generation)
FAST_MODEL = "claude-haiku-4-5-20251001"

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 800          # tokens (approximate, using word count proxy)
CHUNK_OVERLAP = 100       # tokens overlap between chunks
MAX_CHUNKS_PER_QUERY = 12 # top-k chunks passed to LLM

# ── Retrieval ──────────────────────────────────────────────────────────────
BM25_TOP_K = 20           # BM25 candidates before RRF merge
DENSE_TOP_K = 20          # keyword dense candidates
FINAL_TOP_K = 12          # after RRF reranking

# ── Company registry ───────────────────────────────────────────────────────
COMPANIES = {
    "AKSO": {
        "name": "Aker Solutions ASA",
        "exchange": "OSE",
        "sector": "Oilfield Services",
        "description": "Norwegian oilfield services company listed on Oslo Stock Exchange",
        "currency": "NOK",
        "reporting_standard": "IFRS",
        "country": "Norway",
    },
    "SOC": {
        "name": "Sable Offshore Corp",
        "exchange": "NYSE",
        "sector": "Oil & Gas Exploration & Production",
        "description": "Oil and gas company focused on restarting offshore California assets",
        "currency": "USD",
        "reporting_standard": "US GAAP",
        "country": "USA",
        "cik": "0001831481",  # SEC CIK for EDGAR
    },
}

# ── Document taxonomy (for missing-doc audit) ──────────────────────────────
EXPECTED_DOCS = {
    "SOC": [
        {"type": "10-K",        "description": "Annual report",                    "criticality": "CRITICAL"},
        {"type": "10-Q",        "description": "Quarterly reports (last 3)",       "criticality": "HIGH"},
        {"type": "8-K",         "description": "Material event disclosures",       "criticality": "HIGH"},
        {"type": "DEF14A",      "description": "Proxy statement / governance",     "criticality": "MEDIUM"},
        {"type": "earnings_call","description": "Earnings call transcripts",       "criticality": "HIGH"},
        {"type": "investor_pres","description": "Investor day presentations",      "criticality": "MEDIUM"},
        {"type": "reserve_report","description": "Independent reserve certification","criticality": "CRITICAL"},
        {"type": "credit_rating","description": "Credit rating report",            "criticality": "HIGH"},
    ],
    "AKSO": [
        {"type": "annual_report","description": "Annual report (English)",         "criticality": "CRITICAL"},
        {"type": "quarterly",   "description": "Quarterly/interim reports",        "criticality": "HIGH"},
        {"type": "earnings_call","description": "Earnings call transcripts",       "criticality": "HIGH"},
        {"type": "investor_pres","description": "Capital markets day presentation","criticality": "MEDIUM"},
        {"type": "esg_report",  "description": "Sustainability / ESG report",      "criticality": "MEDIUM"},
        {"type": "credit_rating","description": "Credit rating report",            "criticality": "HIGH"},
        {"type": "reg_filing",  "description": "Oslo Børs regulatory filings",     "criticality": "HIGH"},
    ],
}

# ── Output templates ───────────────────────────────────────────────────────
INGESTION_REPORT_TEMPLATE = "ingestion_report.md.j2"
DOSSIER_TEMPLATE = "dossier.md.j2"
BRIEF_TEMPLATE = "analyst_brief.md.j2"
