# Investment AI Platform

Adversarial investment research pipeline for Aker Solutions (AKSO) and Sable Offshore (SOC).

## What it does

Three-phase pipeline that turns raw public filings into a structured investment research package:

| Phase | Output | Description |
|---|---|---|
| 1 | `phase1_ingestion_report.md` | Document inventory, quality assessment, missing context report |
| 2 | `phase2_dossier.md` | Adversarial deep research dossier: verdict, bear risks, contradictions, gaps |
| 3 | `phase3_analyst_brief.md` | One-page analyst brief synthesizing phases 1 + 2 |

## Quick start

```bash
# 1. Clone and install
git clone <repo>
cd investment-ai-platform
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Run for both companies
python run_pipeline.py --all

# Or run individually
python run_pipeline.py --ticker SOC
python run_pipeline.py --ticker AKSO
```

Outputs appear in `outputs/SOC/` and `outputs/AKSO/`.

## Architecture

```
run_pipeline.py
├── ingestion/
│   ├── fetchers/sec_edgar.py      # SEC EDGAR API (no key required)
│   ├── fetchers/oslo_bors.py      # Oslo Børs Newsweb + AKSO IR page
│   ├── fetchers/web_scraper.py    # Supplemental web content
│   ├── chunker.py                 # Section-aware chunking
│   ├── audit.py                   # Missing document report
│   └── report_generator.py        # Phase 1 output
├── retrieval/hybrid.py            # BM25 + TF-IDF + RRF merge
├── dossier/
│   ├── agents/bear_case.py        # Adversarial risk analysis
│   ├── agents/contradiction.py    # Cross-temporal consistency check
│   ├── agents/verdict.py          # PROCEED / STOP / NEEDS_MORE_INFO
│   ├── agents/gap_analysis.py     # Information gap identification
│   ├── orchestrator.py            # Runs all agents
│   └── synthesis.py               # Assembles final dossier
├── brief/generator.py             # Phase 3 one-pager
└── models/router.py               # Single LLM interface (swappable)
```

## Design principles

**All claims cited.** Every factual claim in AI-generated outputs includes `[Source N]` pointing to a specific retrieved chunk. Ungrounded inferences are flagged `[UNCERTAIN — not grounded in documents]`.

**Adversarial by default.** Agents are explicitly instructed to stress-test the thesis, not summarize management's view. The burden of proof is on the bull case.

**LLM-agnostic.** All LLM calls route through `models/router.py`. Swapping from Claude to GPT-4 or a self-hosted model requires changing two lines in `config.py`.

**No external databases required.** Retrieval uses in-memory BM25 + TF-IDF. No Qdrant, no Pinecone, no Docker. Production scaling to Qdrant is documented in `scaling/architecture.md`.

## Data sources

| Company | Sources |
|---|---|
| Sable Offshore (SOC) | SEC EDGAR (10-K, 10-Q, 8-K, DEF 14A) |
| Aker Solutions (AKSO) | Oslo Børs Newsweb API, AKSO IR website, annual report PDFs |

## Scaling

See `scaling/architecture.md` for:
- Ingestion pipeline for 1,000+ companies
- Passive monitoring design for parked ideas
- LLM fallback chain design
- Cost structure (~$16/company/year at scale)
- Security and compliance controls
- Where NOT to deploy AI

## Requirements

- Python 3.11+
- Anthropic API key (~$2–5 in credits for both companies)
- Internet access (fetches live public data)

## Known limitations

1. AKSO annual report PDFs may be inaccessible without authentication — analysis will rely on available web content and regulatory filings
2. Earnings call transcripts require a paid provider (Seeking Alpha, Refinitiv) — not included
3. BM25 retrieval is adequate for this demo but dense embedding retrieval would improve recall on semantic queries
4. Norwegian-language content is not translated — AKSO analysis is based on English-language materials only
