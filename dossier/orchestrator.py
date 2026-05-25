"""
Dossier Orchestrator.

Execution order:
  1. Bear case, contradiction, gap analysis — run in parallel against corpus
  2. Verdict — reads all three memos and makes a final call
  3. Synthesis — assembles everything into the dossier

The verdict agent does NOT retrieve independently. It reasons over
what the first three agents found, like a committee chair reading memos.
"""
from retrieval.hybrid import HybridRetriever
from dossier.agents.bear_case import run_bear_case
from dossier.agents.contradiction import run_contradiction_detection
from dossier.agents.verdict import run_verdict
from dossier.agents.gap_analysis import run_gap_analysis
from dossier.synthesis import synthesize_dossier


def run_dossier_pipeline(
    ticker: str,
    chunks: list[dict],
    audit_result: dict,
    company_config: dict = None,
) -> dict:
    """
    Full dossier pipeline for one company.

    Args:
        ticker:         Company ticker
        chunks:         All document chunks from ingestion
        audit_result:   Output from the ingestion audit
        company_config: Resolved company metadata dict

    Returns:
        Dossier dict with all agent outputs and final markdown text
    """
    company_config = company_config or {}

    print(f"\n{'='*60}")
    print(f"Running Phase 2: Deep Research Dossier for {ticker}")
    print(f"{'='*60}")

    if not chunks:
        print(f"  WARNING: No chunks available for {ticker}. Dossier will be limited.")

    # Build retriever once — shared by bear, contradiction, gap
    retriever = HybridRetriever(chunks)

    # ── Step 1: Three independent analyses against the corpus ─────────────
    print("\n[1/4] Bear case analysis...")
    bear_result = run_bear_case(ticker, retriever, company_config)

    print("\n[2/4] Contradiction & tone detection...")
    contradiction_result = run_contradiction_detection(ticker, retriever, company_config)

    print("\n[3/4] Gap analysis...")
    gap_result = run_gap_analysis(ticker, retriever, audit_result, company_config)

    # ── Step 2: Verdict reads all three memos ────────────────────────────
    print("\n[4/4] Verdict (synthesising agent memos)...")
    verdict_result = run_verdict(
        ticker             = ticker,
        company_config     = company_config,
        bear_result        = bear_result,
        contradiction_result = contradiction_result,
        gap_result         = gap_result,
    )

    # ── Step 3: Assemble final dossier ───────────────────────────────────
    print("\n[Assembling dossier...]")
    dossier_text = synthesize_dossier(
        ticker               = ticker,
        audit_result         = audit_result,
        bear_result          = bear_result,
        contradiction_result = contradiction_result,
        verdict_result       = verdict_result,
        gap_result           = gap_result,
        company_config       = company_config,
    )

    return {
        "ticker":       ticker,
        "verdict":      verdict_result.get("verdict", "NEEDS_MORE_INFO"),
        "confidence":   verdict_result.get("confidence", "LOW"),
        "dossier_text": dossier_text,
        "agent_outputs": {
            "bear_case":     bear_result,
            "contradiction": contradiction_result,
            "verdict":       verdict_result,
            "gap_analysis":  gap_result,
        },
    }
