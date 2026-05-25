"""
Investment AI Platform — Main Pipeline

Run for any publicly traded company:

  python3 run_pipeline.py --ticker SOC
  python3 run_pipeline.py --ticker AKSO
  python3 run_pipeline.py --ticker MSFT
  python3 run_pipeline.py --ticker BP --exchange LSE
  python3 run_pipeline.py --ticker SOC --ticker AKSO
  python3 run_pipeline.py --ticker SOC --phase4-only

Outputs (per company):
  outputs/{TICKER}/phase1_ingestion_report.md
  outputs/{TICKER}/phase2_dossier.md
  outputs/{TICKER}/phase3_analyst_brief.md
  outputs/{TICKER}/phase4_financial_model.md
"""
import argparse
import sys
import traceback
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from config import ANTHROPIC_API_KEY, OUTPUTS_DIR

console = Console()


def check_env():
    if not ANTHROPIC_API_KEY:
        console.print("[red]ERROR: ANTHROPIC_API_KEY not set.[/red]")
        console.print("Copy .env.example to .env and add your API key.")
        sys.exit(1)


def get_fetcher(ticker: str, company: dict):
    """Return the right fetcher function based on company config."""
    fetcher_type = company.get("fetcher", "sec_edgar")

    if fetcher_type == "sec_edgar":
        from ingestion.fetchers.sec_edgar import fetch_sec_filings
        return lambda: fetch_sec_filings(ticker)

    elif fetcher_type == "oslo_bors":
        from ingestion.fetchers.oslo_bors import fetch_oslo_filings
        return lambda: fetch_oslo_filings(ticker)

    else:
        # generic_web fallback — tries IR URL scraping
        console.print(f"  [yellow]No dedicated fetcher for {fetcher_type} — using generic web fetcher[/yellow]")
        from ingestion.fetchers.web_scraper import fetch_supplemental
        return lambda: fetch_supplemental(ticker)


def run_company_pipeline(ticker: str, exchange: str = None) -> dict:
    """Full pipeline for one company."""
    from ingestion.company_resolver import resolve_ticker
    from ingestion.chunker import chunk_all_documents
    from ingestion.audit import run_audit
    from ingestion.report_generator import generate_ingestion_report
    from ingestion.fetchers.web_scraper import fetch_supplemental
    from dossier.orchestrator import run_dossier_pipeline
    from brief.generator import generate_analyst_brief

    # Resolve company (cache hit or auto-discover)
    console.print(f"\n[dim]Resolving {ticker}...[/dim]")
    try:
        company = resolve_ticker(ticker, exchange)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return {"ticker": ticker, "error": str(e)}

    output_dir = OUTPUTS_DIR / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold]{company.get('name', ticker)}[/bold] ({ticker})\n"
        f"{company.get('exchange', '')} | {company.get('sector', '')} | "
        f"{company.get('currency', '')} | {company.get('reporting_standard', '')}",
        title=f"Processing {ticker}",
        border_style="blue",
    ))

    # ── Phase 1: Ingestion ────────────────────────────────────────────────
    console.print("\n[bold blue]Phase 1: Ingestion[/bold blue]")

    fetch_fn = get_fetcher(ticker, company)
    documents = fetch_fn()

    # Supplemental web content (always try)
    console.print("Fetching supplemental web content...")
    supp = fetch_supplemental(ticker)
    documents.extend(supp)

    console.print(f"Total documents fetched: [green]{len(documents)}[/green]")

    console.print("Chunking documents...")
    from ingestion.chunker import chunk_all_documents
    chunks = chunk_all_documents(documents)
    console.print(f"Total chunks: [green]{len(chunks)}[/green]")

    console.print("Running document audit...")
    audit_result = run_audit(ticker, documents)
    audit_result["total_chunks"] = len(chunks)

    console.print("Generating Phase 1 ingestion report...")
    ingestion_report = generate_ingestion_report(ticker, documents, chunks, audit_result)
    p1_path = output_dir / "phase1_ingestion_report.md"
    p1_path.write_text(ingestion_report, encoding="utf-8")
    console.print(f"[green]✓[/green] Phase 1 report → {p1_path}")

    # ── Phase 2: Deep Research Dossier ───────────────────────────────────
    console.print("\n[bold blue]Phase 2: Deep Research Dossier[/bold blue]")
    dossier_result = run_dossier_pipeline(ticker, chunks, audit_result, company)
    p2_path = output_dir / "phase2_dossier.md"
    p2_path.write_text(dossier_result["dossier_text"], encoding="utf-8")
    console.print(f"[green]✓[/green] Dossier → {p2_path}")

    # ── Phase 3: Analyst Brief ────────────────────────────────────────────
    console.print("\n[bold blue]Phase 3: Analyst Brief[/bold blue]")
    brief = generate_analyst_brief(ticker, audit_result, dossier_result, company)
    p3_path = output_dir / "phase3_analyst_brief.md"
    p3_path.write_text(brief, encoding="utf-8")
    console.print(f"[green]✓[/green] Brief → {p3_path}")

    # ── Phase 4: Financial Model ──────────────────────────────────────────
    console.print("\n[bold blue]Phase 4: Financial Model Extraction[/bold blue]")
    p4_path = ""
    try:
        from financial_model.runner import run_financial_model
        fm_result = run_financial_model(ticker)
        p4_path = fm_result.get("report_path", "")
        console.print(f"[green]✓[/green] Financial model → {p4_path}")
        console.print(f"  Review flags: {fm_result.get('flags', 0)}")
    except Exception as e:
        console.print(f"[yellow]Phase 4 skipped: {e}[/yellow]")

    return {
        "ticker": ticker,
        "company": company.get("name", ticker),
        "verdict": dossier_result.get("verdict"),
        "confidence": dossier_result.get("confidence"),
        "documents": len(documents),
        "chunks": len(chunks),
        "coverage_pct": audit_result.get("coverage_pct"),
        "outputs": {
            "ingestion_report": str(p1_path),
            "dossier": str(p2_path),
            "brief": str(p3_path),
            "financial_model": p4_path,
        },
    }


def run_phase4_only(ticker: str, exchange: str = None):
    """Run only Phase 4 for a ticker."""
    from ingestion.company_resolver import resolve_ticker
    try:
        resolve_ticker(ticker, exchange)  # Ensure company is cached
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return
    from financial_model.runner import run_financial_model
    run_financial_model(ticker)


def main():
    parser = argparse.ArgumentParser(
        description="Investment AI Research Platform — works with any publicly traded company",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_pipeline.py --ticker SOC
  python3 run_pipeline.py --ticker AKSO
  python3 run_pipeline.py --ticker MSFT
  python3 run_pipeline.py --ticker BP --exchange LSE
  python3 run_pipeline.py --ticker SOC --ticker AKSO
  python3 run_pipeline.py --ticker SOC --phase4-only
        """
    )
    parser.add_argument(
        "--ticker",
        action="append",
        metavar="TICKER",
        help="Stock ticker (any publicly traded company). Can be specified multiple times.",
    )
    parser.add_argument(
        "--exchange",
        default=None,
        metavar="EXCHANGE",
        help="Exchange hint for non-US tickers (e.g. LSE, OSE, ASX). Helps with resolution.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all companies currently in companies.yaml.",
    )
    parser.add_argument(
        "--phase4-only",
        action="store_true",
        help="Run only Phase 4 financial model extraction (skip ingestion and dossier).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all cached companies in companies.yaml and exit.",
    )

    args = parser.parse_args()

    # List cached companies
    if args.list:
        from ingestion.company_resolver import list_cached_companies
        cached = list_cached_companies()
        if cached:
            console.print(f"[bold]Cached companies ({len(cached)}):[/bold] {', '.join(cached)}")
        else:
            console.print("No companies cached yet. Run with --ticker to add one.")
        sys.exit(0)

    check_env()

    # Determine tickers to process
    if args.all:
        from ingestion.company_resolver import list_cached_companies
        tickers = list_cached_companies()
        if not tickers:
            console.print("[yellow]No companies in cache. Use --ticker to add one.[/yellow]")
            sys.exit(0)
    else:
        tickers = args.ticker or []

    if not tickers:
        console.print("[yellow]No tickers specified.[/yellow]")
        parser.print_help()
        sys.exit(0)

    console.print(Panel(
        f"[bold]Investment AI Research Platform[/bold]\n"
        f"Tickers: {', '.join(t.upper() for t in tickers)}\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        border_style="green",
    ))

    results = []
    for ticker in tickers:
        ticker = ticker.upper()
        try:
            if args.phase4_only:
                run_phase4_only(ticker, args.exchange)
                results.append({"ticker": ticker, "phase4_only": True})
            else:
                result = run_company_pipeline(ticker, args.exchange)
                results.append(result)
        except Exception as e:
            console.print(f"[red]ERROR processing {ticker}: {e}[/red]")
            traceback.print_exc()
            results.append({"ticker": ticker, "error": str(e)})

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Pipeline Complete — Summary[/bold]")
    console.print("=" * 60)
    for r in results:
        if "error" in r:
            console.print(f"[red]✗ {r['ticker']}: {r['error']}[/red]")
        elif r.get("phase4_only"):
            console.print(f"[green]✓ {r['ticker']}: Phase 4 complete[/green]")
        else:
            console.print(
                f"[green]✓ {r['ticker']}[/green] — "
                f"{r.get('company', '')} | "
                f"Verdict: {r.get('verdict', '?')} ({r.get('confidence', '?')}) | "
                f"{r.get('documents', 0)} docs | "
                f"Coverage: {r.get('coverage_pct', 0)}%"
            )
            for label, path in r.get("outputs", {}).items():
                if path:
                    console.print(f"   → {path}")


if __name__ == "__main__":
    main()
