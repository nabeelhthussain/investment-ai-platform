"""
Bear Case Agent.

Two-stage retrieval:
  1. Orientation pass — agent sees a broad sample and generates its own queries
  2. Targeted retrieval — queries used to surface the most relevant chunks
  3. Analysis — adversarial bear case written against retrieved evidence

Universal fallback queries are always appended to agent-generated ones,
ensuring coverage of risks that generic language might obscure.
"""
from models.router import call_llm_with_context, call_llm, call_llm_fast
from retrieval.hybrid import HybridRetriever

# ── Universal queries — always run regardless of company ─────────────────────
# These catch risks that domain-specific language might obscure
UNIVERSAL_QUERIES = [
    "debt maturity liquidity risk going concern cash runway",
    "litigation legal proceedings regulatory enforcement action",
    "management guidance miss timeline delay operational failure",
    "customer concentration revenue dependency key contract loss",
    "audit qualification going concern substantial doubt",
]

# ── Prompts ───────────────────────────────────────────────────────────────────
ORIENTATION_SYSTEM = """You are a skeptical short-seller preparing to stress-test 
an investment thesis. You have been given a sample of documents from a company's 
public filings. Your job is to identify what specific risks to investigate."""

ORIENTATION_PROMPT = """You are conducting bear case analysis on {company} ({ticker}),
a {sector} company.

Here is a sample from their public filings:

{sample_text}

Based on what you see in these documents, generate exactly 8 targeted retrieval 
queries that would surface the most important downside risks for this specific company.

Your queries should be:
- Specific to what you actually see in the documents above, not generic
- Designed to retrieve chunks about financial risk, operational risk, 
  regulatory risk, management credibility, and competitive threats
- Phrased as keyword-rich search strings, not questions

Return exactly 8 queries, one per line, no numbering, no explanation."""

BEAR_SYSTEM = """You are a skeptical, adversarial investment analyst. Your job is to 
stress-test the investment thesis and identify every reason why this investment could fail.

You are NOT summarizing management's view. You are explicitly looking for:
- Risks that management mentions briefly or downplays
- Assumptions in the bull case that could prove wrong  
- Operational, financial, regulatory, or competitive risks
- Red flags in the financials or disclosures
- What bears would say about this company

Rules:
1. Every claim must be followed by [Source N] citing the retrieved document.
2. If you cannot find supporting evidence, write [UNCERTAIN — not grounded in documents].
3. Do NOT fabricate risks not evidenced in the documents.
4. Be specific — vague risks like "macroeconomic headwinds" are useless unless 
   they appear in the documents.
5. Prioritize non-consensus risks — things the market may not be pricing in."""

BEAR_PROMPT = """Analyze the following retrieved documents and identify the TOP bear 
case risks for {company} ({ticker}), a {sector} company.

Focus specifically on:
1. **Existential or near-term risks** — what could cause permanent capital loss?
2. **Timeline/execution risks** — what specific milestones could slip?
3. **Financial risks** — leverage, liquidity, cash burn, covenant risk
4. **Regulatory/legal risks** — proceedings, liabilities, permit dependencies
5. **Management credibility** — guidance misses, overly optimistic framing
6. **Non-consensus risks** — risks the market may be underpricing

For each risk:
- State the risk clearly in one sentence
- Provide specific evidence from the source documents
- Assess severity: HIGH / MEDIUM / LOW
- Cite the source with [Source N]

If a risk cannot be grounded in the provided documents, flag it as [UNCERTAIN]."""


def _generate_queries(
    ticker: str,
    company_config: dict,
    retriever: HybridRetriever,
) -> list[str]:
    """
    Stage 1: Give the agent a broad orientation sample and let it
    generate its own targeted retrieval queries.
    """
    name   = company_config.get("name", ticker)
    sector = company_config.get("sector", "")

    # Broad orientation retrieval — diverse sample across doc types
    orientation_chunks = retriever.retrieve(
        "company overview business operations financial position risks outlook",
        top_k=20,
    )

    if not orientation_chunks:
        return UNIVERSAL_QUERIES

    # Build a compact sample — first 300 chars per chunk, diverse doc types
    seen_types = set()
    sample_parts = []
    for chunk in orientation_chunks:
        dt = chunk.get("doc_type", "")
        label = f"[{dt} | {chunk.get('date','')[:7]} | {chunk.get('section','')[:40]}]"
        sample_parts.append(f"{label}\n{chunk['text'][:300]}")
        seen_types.add(dt)
        if len(sample_parts) >= 10:
            break

    sample_text = "\n\n---\n\n".join(sample_parts)

    prompt = ORIENTATION_PROMPT.format(
        company     = name,
        ticker      = ticker,
        sector      = sector,
        sample_text = sample_text,
    )

    response = call_llm_fast(
        prompt     = prompt,
        system     = ORIENTATION_SYSTEM,
        max_tokens = 300,
    )

    # Parse one query per line
    generated = [
        q.strip().lstrip("-•·123456789. ")
        for q in response.strip().split("\n")
        if q.strip() and len(q.strip()) > 10
    ][:8]

    print(f"  Agent generated {len(generated)} bear case queries")

    # Combine agent-generated + universal (deduped)
    all_queries = generated + [
        q for q in UNIVERSAL_QUERIES
        if not any(word in " ".join(generated).lower()
                   for word in q.split()[:2])
    ]

    return all_queries


def run_bear_case(
    ticker: str,
    retriever: HybridRetriever,
    company_config: dict = None,
) -> dict:
    """
    Run the bear case agent for a given company.

    Stage 1: Agent generates its own retrieval queries from an orientation sample.
    Stage 2: Targeted retrieval using generated + universal queries.
    Stage 3: Adversarial analysis written against retrieved evidence.
    """
    company_config = company_config or {}
    name   = company_config.get("name", ticker)
    sector = company_config.get("sector", "")

    print(f"  Running bear case analysis for {ticker}...")

    if not retriever.chunks:
        return {
            "ticker":   ticker,
            "analysis": "[UNCERTAIN — No source documents available for bear case analysis]",
            "chunks_used": 0,
            "queries_used": [],
            "error": "No chunks in corpus",
        }

    # ── Stage 1: Generate queries ─────────────────────────────────────────────
    print(f"  Generating bear case queries from corpus orientation...")
    queries = _generate_queries(ticker, company_config, retriever)
    print(queries)
    
    # ── Stage 2: Targeted retrieval ───────────────────────────────────────────
    chunks = retriever.retrieve_multi(queries, top_k_per_query=5)

    if not chunks:
        return {
            "ticker":      ticker,
            "analysis":    "[UNCERTAIN — No relevant chunks retrieved for bear case]",
            "chunks_used": 0,
            "queries_used": queries,
        }

    # ── Stage 3: Analysis ─────────────────────────────────────────────────────
    prompt = BEAR_PROMPT.format(
        company = name,
        ticker  = ticker,
        sector  = sector,
    )

    analysis = call_llm_with_context(
        query          = prompt,
        context_chunks = chunks,
        system         = BEAR_SYSTEM,
        max_tokens     = 4000,
    )

    return {
        "ticker":       ticker,
        "analysis":     analysis,
        "chunks_used":  len(chunks),
        "queries_used": queries,
        "source_docs":  list({c.get("doc_type") for c in chunks}),
    }
