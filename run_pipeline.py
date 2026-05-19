"""
Investment AI Platform — Main Pipeline

Usage:
    python run_pipeline.py --ticker SOC
    python run_pipeline.py --ticker AKSO
    python run_pipeline.py --ticker SOC --ticker AKSO
    python run_pipeline.py --all

Outputs (per company):
    outputs/{TICKER}/phase1_ingestion_report.md
    outputs/{TICKER}/phase2_dossier.md
    outputs/{TICKER}/phase3_analyst_brief.md
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import track

from config import OUTPUTS_DIR, COMPANIES, ANTHROPIC_API_KEY

console = Console()


def check_env():
    if not ANTHROPIC_API_KEY:
        console.print("[red]ERROR: ANTHROPIC_API_KEY not set.[/red]")
        console.print("Copy .env.example to .env and add your API key.")
        sys.exit(1)


def run_company_pipeline(ticker: str) -> dict:
    """Full pipeline for one company. Returns result summary."""
    from ingestion.fetchers.sec_edgar import fetch_soc_filings
    from ingestion.fetchers.oslo_bors import fetch_akso_filings
    from ingestion.fetchers.web_scraper import fetch_supplemental
    from ingestion.chunker import chunk_all_documents
    from ingestion.audit import run_audit
    from ingestion.report_generator import generate_ingestion_report
    from dossier.orchestrator import run_dossier_pipeline
    from brief.generator import generate_analyst_brief

    output_dir = OUTPUTS_DIR / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    company = COMPANIES.get(ticker, {})
    console.print(Panel(
        f"[bold]{company.get('name', ticker)}[/bold] ({ticker})\n"
        f"{company.get('exchange', '')} | {company.get('sector', '')}",
        title=f"Processing {ticker}",
        border_style="blue",
    ))

    # ── Phase 1: Ingestion ────────────────────────────────────────────────
    console.print("\n[bold blue]Phase 1: Ingestion[/bold blue]")

    # Fetch documents
    if ticker == "SOC":
        documents = fetch_soc_filings()
    else:
        documents = fetch_akso_filings()

    # Supplemental web content
    console.print("Fetching supplemental web content...")
    supp = fetch_supplemental(ticker)
    documents.extend(supp)

    console.print(f"Total documents fetched: [green]{len(documents)}[/green]")

    # Chunk
    console.print("Chunking documents...")
    chunks = chunk_all_documents(documents)
    console.print(f"Total chunks: [green]{len(chunks)}[/green]")

    # Audit
    console.print("Running document audit...")
    audit_result = run_audit(ticker, documents)
    audit_result["total_chunks"] = len(chunks)

    # Generate Phase 1 report
    console.print("Generating Phase 1 ingestion report...")
    ingestion_report = generate_ingestion_report(ticker, documents, chunks, audit_result)

    p1_path = output_dir / "phase1_ingestion_report.md"
    p1_path.write_text(ingestion_report, encoding="utf-8")
    console.print(f"[green]✓[/green] Phase 1 report → {p1_path}")

    # ── Phase 2: Deep Research Dossier ───────────────────────────────────
    console.print("\n[bold blue]Phase 2: Deep Research Dossier[/bold blue]")

    dossier_result = run_dossier_pipeline(ticker, chunks, audit_result)

    p2_path = output_dir / "phase2_dossier.md"
    p2_path.write_text(dossier_result["dossier_text"], encoding="utf-8")
    console.print(f"[green]✓[/green] Dossier → {p2_path}")

    # ── Phase 3: Analyst Brief ────────────────────────────────────────────
    console.print("\n[bold blue]Phase 3: Analyst Brief[/bold blue]")

    brief = generate_analyst_brief(ticker, audit_result, dossier_result)

    p3_path = output_dir / "phase3_analyst_brief.md"
    p3_path.write_text(brief, encoding="utf-8")
    console.print(f"[green]✓[/green] Brief → {p3_path}")

    return {
        "ticker": ticker,
        "verdict": dossier_result.get("verdict"),
        "confidence": dossier_result.get("confidence"),
        "documents": len(documents),
        "chunks": len(chunks),
        "coverage_pct": audit_result.get("coverage_pct"),
        "outputs": {
            "ingestion_report": str(p1_path),
            "dossier": str(p2_path),
            "brief": str(p3_path),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Investment AI Platform — Research Pipeline"
    )
    parser.add_argument(
        "--ticker",
        action="append",
        choices=list(COMPANIES.keys()),
        help="Ticker(s) to process. Can be specified multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all companies in config.",
    )
    args = parser.parse_args()

    check_env()

    tickers = list(COMPANIES.keys()) if args.all else (args.ticker or [])

    if not tickers:
        console.print("[yellow]No tickers specified. Use --ticker SOC or --all[/yellow]")
        parser.print_help()
        sys.exit(0)

    console.print(Panel(
        f"[bold]Investment AI Research Platform[/bold]\n"
        f"Companies: {', '.join(tickers)}\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        border_style="green",
    ))

    results = []
    for ticker in tickers:
        try:
            result = run_company_pipeline(ticker)
            results.append(result)
        except Exception as e:
            console.print(f"[red]ERROR processing {ticker}: {e}[/red]")
            import traceback
            traceback.print_exc()
            results.append({"ticker": ticker, "error": str(e)})

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Pipeline Complete — Summary[/bold]")
    console.print("=" * 60)
    for r in results:
        if "error" in r:
            console.print(f"[red]✗ {r['ticker']}: {r['error']}[/red]")
        else:
            console.print(
                f"[green]✓ {r['ticker']}[/green] | "
                f"Verdict: {r.get('verdict', '?')} ({r.get('confidence', '?')}) | "
                f"{r.get('documents', 0)} docs | "
                f"{r.get('chunks', 0)} chunks | "
                f"Coverage: {r.get('coverage_pct', 0)}%"
            )
            for label, path in r.get("outputs", {}).items():
                console.print(f"   → {path}")


if __name__ == "__main__":
    main()
