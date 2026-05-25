"""
Contradiction Detector Agent.

Two-stage retrieval:
  1. Orientation pass — agent generates queries targeting cross-temporal topics
  2. Targeted retrieval sorted by date — surfaces the same topics across periods
  3. Analysis — Claude finds contradictions, tone shifts, guidance misses, omissions

Enhanced with FinBERT quantitative tone scoring as a pre-screening layer.
"""
from models.router import call_llm_with_context, call_llm_fast
from retrieval.hybrid import HybridRetriever

try:
    from ingestion.tone_analyzer import analyze_tone_trends, format_tone_section
    TONE_AVAILABLE = True
except ImportError:
    TONE_AVAILABLE = False

# ── Universal cross-temporal queries — always included ───────────────────────
UNIVERSAL_QUERIES = [
    "management guidance forecast outlook target timeline",
    "risk factors material risks disclosure language",
    "strategy priorities focus areas direction",
    "financial position liquidity cash debt covenant",
]

ORIENTATION_SYSTEM = """You are a forensic analyst looking for management credibility 
issues in corporate disclosures. You specialise in detecting contradictions, 
tone shifts, and strategic pivots across filings."""

ORIENTATION_PROMPT = """You are analyzing {company} ({ticker}) for management 
contradictions and tone shifts across multiple time periods.

Here is a sample of documents from their filings, spanning different dates:

{sample_text}

Generate exactly 8 retrieval queries that would surface the same specific topics 
from DIFFERENT time periods — so you can compare how management discussed them 
earlier versus later.

Good queries target: guidance given at one point in time, strategic commitments, 
specific milestones, financial targets, risk factor language, and named projects 
or initiatives.

Return exactly 8 queries, one per line, no numbering, no explanation."""

CONTRADICTION_SYSTEM = """You are a forensic analyst specializing in detecting 
management credibility issues in corporate disclosures.

Find:
1. Direct contradictions — statements that logically conflict across time periods
2. Guidance misses — forecasts that were subsequently missed
3. Tone shifts — changes from optimistic to cautious (or vice versa) on key topics
4. Strategic omissions — topics that quietly disappear from later communications

Rules:
- Be specific. Quote or closely paraphrase relevant language.
- Always cite sources with [Source N].
- Distinguish: CONTRADICTION / GUIDANCE_MISS / TONE_SHIFT / OMISSION
- Do NOT flag normal updates or expected revisions as contradictions.
- If you find no credible contradictions, say so clearly."""

CONTRADICTION_PROMPT = """Analyze the following documents from {company} ({ticker})
spanning multiple time periods.

Look specifically for:

**1. Direct Contradictions**
Statements where management says one thing in one document and something 
materially different in another. Quote or closely paraphrase both sides.

**2. Guidance Misses**
Cases where management gave a forecast or target that was subsequently missed.
What was the guidance? What was the outcome?

**3. Tone Shifts**
Topics where tone changed significantly without a clear external event explaining it.

**4. Strategic Omissions**
Major themes from earlier documents that quietly disappeared from later ones.

For each finding:
- Type: CONTRADICTION / GUIDANCE_MISS / TONE_SHIFT / OMISSION
- Earlier statement (with [Source N])
- Later statement or absence (with [Source N])
- Significance: what should an analyst read into this?

If you find NO credible contradictions or shifts, state that clearly."""


def _generate_queries(
    ticker: str,
    company_config: dict,
    retriever: HybridRetriever,
) -> list[str]:
    """Generate cross-temporal retrieval queries from corpus orientation."""
    name   = company_config.get("name", ticker)

    # For contradiction detection, we want diversity across time periods
    # Use a broad query to get chunks from multiple dates
    orientation_chunks = retriever.retrieve(
        "guidance outlook strategy target milestone timeline forecast",
        top_k=20,
    )

    if not orientation_chunks:
        return UNIVERSAL_QUERIES

    # Sort by date for the sample — show temporal spread
    sorted_chunks = sorted(
        orientation_chunks,
        key=lambda x: x.get("date", ""),
    )

    # Pick from early, middle, and recent periods
    n = len(sorted_chunks)
    sample_indices = list({0, n//4, n//2, 3*n//4, n-1})
    sample_chunks  = [sorted_chunks[i] for i in sorted(sample_indices) if i < n]

    sample_parts = []
    for chunk in sample_chunks[:8]:
        label = f"[{chunk.get('doc_type','')} | {chunk.get('date','')[:7]} | {chunk.get('section','')[:40]}]"
        sample_parts.append(f"{label}\n{chunk['text'][:300]}")

    sample_text = "\n\n---\n\n".join(sample_parts)

    prompt = ORIENTATION_PROMPT.format(
        company     = name,
        ticker      = ticker,
        sample_text = sample_text,
    )

    response = call_llm_fast(
        prompt     = prompt,
        system     = ORIENTATION_SYSTEM,
        max_tokens = 300,
    )

    generated = [
        q.strip().lstrip("-•·123456789. ")
        for q in response.strip().split("\n")
        if q.strip() and len(q.strip()) > 10
    ][:8]

    print(f"  Agent generated {len(generated)} contradiction queries")

    all_queries = generated + [
        q for q in UNIVERSAL_QUERIES
        if not any(word in " ".join(generated).lower()
                   for word in q.split()[:2])
    ]

    return all_queries


def run_contradiction_detection(
    ticker: str,
    retriever: HybridRetriever,
    company_config: dict = None,
) -> dict:
    """
    Run contradiction detection for a company.

    Stage 1: Agent generates cross-temporal retrieval queries.
    Stage 2: Retrieval sorted by date to surface temporal patterns.
    Stage 3: Claude identifies contradictions, tone shifts, omissions.
    Stage 4: FinBERT quantitative tone scoring (if available).
    """
    company_config = company_config or {}
    name = company_config.get("name", ticker)

    print(f"  Running contradiction detection for {ticker}...")

    if not retriever.chunks:
        return {
            "ticker":       ticker,
            "analysis":     "[UNCERTAIN — No source documents available]",
            "chunks_used":  0,
            "tone_section": "",
            "tone_result":  None,
        }

    # ── Stage 1: Generate queries ─────────────────────────────────────────────
    print(f"  Generating contradiction queries from corpus orientation...")
    queries = _generate_queries(ticker, company_config, retriever)

    # ── Stage 2: Targeted retrieval ───────────────────────────────────────────
    chunks = retriever.retrieve_multi(queries, top_k_per_query=4)

    if not chunks:
        return {
            "ticker":       ticker,
            "analysis":     "[UNCERTAIN — No relevant chunks retrieved]",
            "chunks_used":  0,
            "tone_section": "",
            "tone_result":  None,
            "queries_used": queries,
        }

    # Sort by date — critical for temporal analysis
    chunks_sorted = sorted(chunks, key=lambda x: x.get("date", ""))

    # ── Stage 3: FinBERT tone pre-screening ───────────────────────────────────
    tone_result  = None
    tone_section = ""
    shift_summary = ""

    if TONE_AVAILABLE:
        print(f"  Running FinBERT tone analysis...")
        all_chunks  = retriever.retrieve_multi(queries, top_k_per_query=8)
        tone_result = analyze_tone_trends(all_chunks)
        tone_section = format_tone_section(tone_result)

        shifts = tone_result.get("significant_shifts", [])
        if shifts:
            shift_summary = "\n\nFinBERT pre-screening detected these quantitative tone shifts to investigate:\n"
            for s in shifts[:5]:
                shift_summary += (
                    f"- {s['topic'].replace('_',' ').title()}: became "
                    f"{'MORE NEGATIVE' if s['neg_delta'] > 0 else 'MORE POSITIVE'} "
                    f"from {s['from_period']} to {s['to_period']} "
                    f"(Δneg={s['neg_delta']:+.3f}, {s['magnitude']} shift). "
                    f"Investigate this topic specifically.\n"
                )
    else:
        print("  Skipping FinBERT (install: pip install transformers torch)")

    # ── Stage 4: Claude qualitative analysis ──────────────────────────────────
    prompt = CONTRADICTION_PROMPT.format(
        company = name,
        ticker  = ticker,
    ) + shift_summary

    analysis = call_llm_with_context(
        query          = prompt,
        context_chunks = chunks_sorted,
        system         = CONTRADICTION_SYSTEM,
        max_tokens     = 3000,
    )

    return {
        "ticker":        ticker,
        "analysis":      analysis,
        "tone_section":  tone_section,
        "tone_result":   tone_result,
        "chunks_used":   len(chunks),
        "queries_used":  queries,
        "date_range":    _date_range(chunks),
        "source_docs":   list({c.get("doc_type") for c in chunks}),
    }


def _date_range(chunks: list[dict]) -> str:
    dates = sorted([c.get("date", "") for c in chunks if c.get("date")])
    return f"{dates[0]} to {dates[-1]}" if dates else "unknown"
