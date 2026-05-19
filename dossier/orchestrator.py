"""
Dossier Orchestrator.

Runs all four agents in a logical sequence, then passes results
to the synthesis layer to produce the final Deep Research Dossier.

Sequence:
  1. Bear case (retrieval-heavy, sets adversarial tone)
  2. Contradiction detection (cross-temporal analysis)
  3. Verdict (synthesizes bear + context)
  4. Gap analysis (what we still don't know)
  5. Synthesis (assembles into final dossier)
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
) -> dict:
    """
    Full dossier pipeline for one company.

    Args:
        ticker: Company ticker (AKSO or SOC)
        chunks: All document chunks from ingestion
        audit_result: Output from the ingestion audit

    Returns:
        Dossier dict with all agent outputs and final markdown text
    """
    print(f"\n{'='*60}")
    print(f"Running Phase 2: Deep Research Dossier for {ticker}")
    print(f"{'='*60}")

    if not chunks:
        print(f"  WARNING: No chunks available for {ticker}. Dossier will be limited.")

    # Build retriever once, share across all agents
    retriever = HybridRetriever(chunks)

    # Run all agents
    print("\n[1/4] Bear case analysis...")
    bear_result = run_bear_case(ticker, retriever)

    print("\n[2/4] Contradiction detection...")
    contradiction_result = run_contradiction_detection(ticker, retriever)

    print("\n[3/4] Verdict analysis...")
    verdict_result = run_verdict(ticker, retriever)

    print("\n[4/4] Gap analysis...")
    gap_result = run_gap_analysis(ticker, retriever, audit_result)

    print("\n[Synthesizing dossier...]")
    dossier_text = synthesize_dossier(
        ticker=ticker,
        audit_result=audit_result,
        bear_result=bear_result,
        contradiction_result=contradiction_result,
        verdict_result=verdict_result,
        gap_result=gap_result,
    )

    return {
        "ticker": ticker,
        "verdict": verdict_result.get("verdict", "NEEDS_MORE_INFO"),
        "confidence": verdict_result.get("confidence", "LOW"),
        "dossier_text": dossier_text,
        "agent_outputs": {
            "bear_case": bear_result,
            "contradiction": contradiction_result,
            "verdict": verdict_result,
            "gap_analysis": gap_result,
        },
    }
