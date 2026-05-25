"""
Tone Analyzer — FinBERT-based sentiment scoring for financial text.

Uses ProsusAI/finbert, fine-tuned on financial communications.
Scores each chunk as positive/negative/neutral and tracks shifts over time.

Slots into the contradiction detection pipeline as a pre-processing step:
  1. Score all chunks with FinBERT
  2. Group by topic and time period
  3. Flag statistically significant tone shifts
  4. Pass flagged passages to Claude for qualitative interpretation
"""

import re
from collections import defaultdict
from typing import Optional

# Lazy imports — only load torch/transformers when actually called
_pipeline = None
_model_loaded = False


def _get_pipeline():
    """Load FinBERT pipeline once, reuse across calls."""
    global _pipeline, _model_loaded
    if not _model_loaded:
        try:
            from transformers import pipeline
            print("  Loading FinBERT model (first run downloads ~440MB)...")
            _pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
            _model_loaded = True
            print("  FinBERT loaded.")
        except ImportError:
            raise ImportError(
                "transformers and torch are required for tone analysis.\n"
                "Install with: pip install transformers torch"
            )
    return _pipeline


def score_text(text: str) -> dict:
    """
    Score a single text passage with FinBERT.
    Returns dict with positive, negative, neutral scores (sum to 1.0).
    """
    pipe = _get_pipeline()

    # FinBERT max 512 tokens — truncate long text to first ~1800 chars
    text = text[:1800].strip()
    if not text:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "dominant": "neutral"}

    try:
        results = pipe(text)[0]
        scores = {r["label"].lower(): round(r["score"], 4) for r in results}
        scores["dominant"] = max(scores, key=lambda k: scores[k] if k != "dominant" else -1)
        return scores
    except Exception as e:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "dominant": "neutral", "error": str(e)}


# Topic keywords for grouping chunks by subject matter
TOPIC_KEYWORDS = {
    "guidance_outlook": [
        "expect", "outlook", "guidance", "forecast", "anticipate",
        "project", "target", "plan", "going forward", "future"
    ],
    "liquidity_financing": [
        "cash", "liquidity", "debt", "credit", "financing", "capital",
        "covenant", "maturity", "refinanc", "borrow", "facility"
    ],
    "operations_execution": [
        "production", "restart", "pipeline", "permit", "regulatory",
        "operations", "project", "execution", "deliver", "milestone"
    ],
    "risk_factors": [
        "risk", "uncertain", "contingent", "may not", "no assurance",
        "subject to", "could adversely", "material adverse", "going concern"
    ],
    "revenue_growth": [
        "revenue", "order", "backlog", "intake", "growth", "demand",
        "contract", "award", "win", "book-to-bill"
    ],
    "margins_profitability": [
        "margin", "ebitda", "profit", "earnings", "cost", "inflation",
        "efficiency", "improvement", "decline", "compress"
    ],
}


def classify_topic(text: str) -> str:
    """Assign a chunk to its most relevant topic based on keyword density."""
    text_lower = text.lower()
    topic_scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        topic_scores[topic] = score

    best = max(topic_scores, key=lambda k: topic_scores[k])
    return best if topic_scores[best] > 0 else "general"


def analyze_tone_trends(chunks: list[dict]) -> dict:
    """
    Run FinBERT across all chunks and build a tone trend report.

    Args:
        chunks: List of chunk dicts with text, date, doc_type, section fields.

    Returns:
        Dict with:
          - scored_chunks: all chunks with sentiment scores added
          - topic_trends: sentiment by topic over time
          - significant_shifts: list of detected tone shifts
          - summary: human-readable summary string
    """
    if not chunks:
        return {
            "scored_chunks": [],
            "topic_trends": {},
            "significant_shifts": [],
            "summary": "No chunks available for tone analysis.",
        }

    print(f"  Running FinBERT tone analysis on {len(chunks)} chunks...")

    # Score all chunks
    scored = []
    for i, chunk in enumerate(chunks):
        if i % 50 == 0 and i > 0:
            print(f"    Scored {i}/{len(chunks)} chunks...")

        scores = score_text(chunk.get("text", ""))
        topic = classify_topic(chunk.get("text", ""))

        scored.append({
            **chunk,
            "sentiment": scores,
            "topic": topic,
            "negativity": scores.get("negative", 0.0),
            "positivity": scores.get("positive", 0.0),
        })

    # Group by topic and sort by date within each topic
    topic_groups = defaultdict(list)
    for chunk in scored:
        topic_groups[chunk["topic"]].append(chunk)

    for topic in topic_groups:
        topic_groups[topic].sort(key=lambda x: x.get("date", ""))

    # Build topic trends — average sentiment per (topic, doc_type, date)
    topic_trends = {}
    for topic, topic_chunks in topic_groups.items():
        # Group by document (date + doc_type)
        doc_groups = defaultdict(list)
        for chunk in topic_chunks:
            doc_key = f"{chunk.get('date', 'unknown')}|{chunk.get('doc_type', 'unknown')}"
            doc_groups[doc_key].append(chunk)

        periods = []
        for doc_key, doc_chunks in sorted(doc_groups.items()):
            date, doc_type = doc_key.split("|")
            avg_neg = sum(c["negativity"] for c in doc_chunks) / len(doc_chunks)
            avg_pos = sum(c["positivity"] for c in doc_chunks) / len(doc_chunks)
            avg_neu = 1.0 - avg_neg - avg_pos

            periods.append({
                "date": date,
                "doc_type": doc_type,
                "avg_negative": round(avg_neg, 4),
                "avg_positive": round(avg_pos, 4),
                "avg_neutral": round(avg_neu, 4),
                "chunk_count": len(doc_chunks),
                "dominant": "negative" if avg_neg > avg_pos and avg_neg > avg_neu
                           else "positive" if avg_pos > avg_neg and avg_pos > avg_neu
                           else "neutral",
            })

        topic_trends[topic] = periods

    # Detect significant shifts
    significant_shifts = _detect_shifts(topic_trends, scored)

    # Build summary
    summary = _build_summary(significant_shifts, topic_trends, scored)

    return {
        "scored_chunks": scored,
        "topic_trends": topic_trends,
        "significant_shifts": significant_shifts,
        "summary": summary,
    }


def _detect_shifts(topic_trends: dict, scored_chunks: list) -> list:
    """
    Find topic+period pairs where sentiment shifted significantly.
    Threshold: >0.15 change in negativity score between consecutive periods.
    """
    SHIFT_THRESHOLD = 0.15
    shifts = []

    for topic, periods in topic_trends.items():
        if len(periods) < 2:
            continue

        for i in range(1, len(periods)):
            prev = periods[i - 1]
            curr = periods[i]

            neg_delta = curr["avg_negative"] - prev["avg_negative"]
            pos_delta = curr["avg_positive"] - prev["avg_positive"]

            if abs(neg_delta) >= SHIFT_THRESHOLD or abs(pos_delta) >= SHIFT_THRESHOLD:
                # Find the most negative chunk in the current period for evidence
                period_chunks = [
                    c for c in scored_chunks
                    if c.get("topic") == topic
                    and c.get("date") == curr["date"]
                ]
                period_chunks.sort(key=lambda x: x["negativity"], reverse=True)
                example_chunk = period_chunks[0] if period_chunks else None

                shifts.append({
                    "topic": topic,
                    "from_period": prev["date"],
                    "from_doc": prev["doc_type"],
                    "to_period": curr["date"],
                    "to_doc": curr["doc_type"],
                    "neg_delta": round(neg_delta, 4),
                    "pos_delta": round(pos_delta, 4),
                    "direction": "MORE_NEGATIVE" if neg_delta > 0 else "MORE_POSITIVE",
                    "magnitude": "LARGE" if abs(neg_delta) > 0.25 else "MODERATE",
                    "example_text": example_chunk["text"][:400] if example_chunk else None,
                    "example_section": example_chunk.get("section", "") if example_chunk else None,
                })

    # Sort by magnitude of shift
    shifts.sort(key=lambda x: abs(x["neg_delta"]), reverse=True)
    return shifts[:10]  # Return top 10 most significant


def _build_summary(shifts: list, topic_trends: dict, scored_chunks: list) -> str:
    """Build a human-readable tone trend summary."""
    lines = []

    if not scored_chunks:
        return "Insufficient data for tone analysis."

    # Overall corpus sentiment
    avg_neg = sum(c["negativity"] for c in scored_chunks) / len(scored_chunks)
    avg_pos = sum(c["positivity"] for c in scored_chunks) / len(scored_chunks)
    lines.append(
        f"Overall corpus tone: {avg_pos:.1%} positive, "
        f"{avg_neg:.1%} negative, "
        f"{1-avg_pos-avg_neg:.1%} neutral across {len(scored_chunks)} passages."
    )

    if not shifts:
        lines.append("No significant tone shifts detected across document periods.")
        return " ".join(lines)

    lines.append(f"\n{len(shifts)} significant tone shift(s) detected:\n")

    for s in shifts[:5]:
        direction_label = "deteriorated" if s["direction"] == "MORE_NEGATIVE" else "improved"
        lines.append(
            f"• [{s['magnitude']}] {s['topic'].replace('_', ' ').title()}: "
            f"tone {direction_label} from {s['from_period']} ({s['from_doc']}) "
            f"to {s['to_period']} ({s['to_doc']}) — "
            f"negativity {'+' if s['neg_delta'] > 0 else ''}{s['neg_delta']:.3f}"
        )

    return "\n".join(lines)


def format_tone_section(tone_result: dict) -> str:
    """
    Format the tone analysis result as a markdown section
    for inclusion in the dossier.
    """
    lines = []
    lines.append("## Quantitative Tone Analysis (FinBERT)")
    lines.append("")
    lines.append(
        "*FinBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial text. "
        "Scores reflect sentiment of language used — not fundamental quality of the business. "
        "Significant shifts in tone across periods warrant qualitative follow-up.*"
    )
    lines.append("")
    lines.append(tone_result.get("summary", "No tone data available."))
    lines.append("")

    shifts = tone_result.get("significant_shifts", [])
    if shifts:
        lines.append("### Detected Tone Shifts")
        lines.append("")
        lines.append("| Topic | From | To | Direction | Δ Negativity | Magnitude |")
        lines.append("|---|---|---|---|---|---|")
        for s in shifts:
            lines.append(
                f"| {s['topic'].replace('_',' ').title()} "
                f"| {s['from_period']} ({s['from_doc']}) "
                f"| {s['to_period']} ({s['to_doc']}) "
                f"| {s['direction']} "
                f"| {'+' if s['neg_delta'] > 0 else ''}{s['neg_delta']:.3f} "
                f"| {s['magnitude']} |"
            )
        lines.append("")

        # Show top 3 example passages
        lines.append("### Most Negative Passages by Topic")
        lines.append("")
        shown_topics = set()
        for s in shifts[:3]:
            if s["topic"] in shown_topics:
                continue
            shown_topics.add(s["topic"])
            if s.get("example_text"):
                lines.append(
                    f"**{s['topic'].replace('_',' ').title()}** "
                    f"({s['to_period']}, {s['to_doc']}, {s.get('example_section','')}):"
                )
                lines.append(f"> {s['example_text'][:300]}...")
                lines.append("")

    topic_trends = tone_result.get("topic_trends", {})
    if topic_trends:
        lines.append("### Sentiment by Topic Over Time")
        lines.append("")
        lines.append("| Topic | Period | Doc Type | Avg Positive | Avg Negative | Dominant |")
        lines.append("|---|---|---|---|---|---|")
        for topic, periods in sorted(topic_trends.items()):
            for p in periods:
                lines.append(
                    f"| {topic.replace('_',' ').title()} "
                    f"| {p['date']} "
                    f"| {p['doc_type']} "
                    f"| {p['avg_positive']:.3f} "
                    f"| {p['avg_negative']:.3f} "
                    f"| {p['dominant']} |"
                )
        lines.append("")

    return "\n".join(lines)
