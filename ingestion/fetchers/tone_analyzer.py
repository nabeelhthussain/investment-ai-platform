"""
Tone Analyzer — FinBERT-based sentiment scoring for financial text.
"""
import re
from collections import defaultdict

_pipeline = None
_model_loaded = False


def _get_pipeline():
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
    pipe = _get_pipeline()
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


TOPIC_KEYWORDS = {
    "guidance_outlook": ["expect", "outlook", "guidance", "forecast", "anticipate", "project", "target", "plan", "going forward", "future"],
    "liquidity_financing": ["cash", "liquidity", "debt", "credit", "financing", "capital", "covenant", "maturity", "refinanc", "borrow", "facility"],
    "operations_execution": ["production", "restart", "pipeline", "permit", "regulatory", "operations", "project", "execution", "deliver", "milestone"],
    "risk_factors": ["risk", "uncertain", "contingent", "may not", "no assurance", "subject to", "could adversely", "material adverse", "going concern"],
    "revenue_growth": ["revenue", "order", "backlog", "intake", "growth", "demand", "contract", "award", "win", "book-to-bill"],
    "margins_profitability": ["margin", "ebitda", "profit", "earnings", "cost", "inflation", "efficiency", "improvement", "decline", "compress"],
}


def classify_topic(text: str) -> str:
    text_lower = text.lower()
    topic_scores = {topic: sum(1 for kw in keywords if kw in text_lower) for topic, keywords in TOPIC_KEYWORDS.items()}
    best = max(topic_scores, key=lambda k: topic_scores[k])
    return best if topic_scores[best] > 0 else "general"


def analyze_tone_trends(chunks: list[dict]) -> dict:
    if not chunks:
        return {"scored_chunks": [], "topic_trends": {}, "significant_shifts": [], "summary": "No chunks available."}

    print(f"  Running FinBERT tone analysis on {len(chunks)} chunks...")
    scored = []
    for i, chunk in enumerate(chunks):
        if i % 50 == 0 and i > 0:
            print(f"    Scored {i}/{len(chunks)} chunks...")
        scores = score_text(chunk.get("text", ""))
        topic = classify_topic(chunk.get("text", ""))
        scored.append({**chunk, "sentiment": scores, "topic": topic,
                       "negativity": scores.get("negative", 0.0),
                       "positivity": scores.get("positive", 0.0)})

    topic_groups = defaultdict(list)
    for chunk in scored:
        topic_groups[chunk["topic"]].append(chunk)
    for topic in topic_groups:
        topic_groups[topic].sort(key=lambda x: x.get("date", ""))

    topic_trends = {}
    for topic, topic_chunks in topic_groups.items():
        doc_groups = defaultdict(list)
        for chunk in topic_chunks:
            doc_key = f"{chunk.get('date','unknown')}|{chunk.get('doc_type','unknown')}"
            doc_groups[doc_key].append(chunk)
        periods = []
        for doc_key, doc_chunks in sorted(doc_groups.items()):
            date, doc_type = doc_key.split("|")
            avg_neg = sum(c["negativity"] for c in doc_chunks) / len(doc_chunks)
            avg_pos = sum(c["positivity"] for c in doc_chunks) / len(doc_chunks)
            periods.append({
                "date": date, "doc_type": doc_type,
                "avg_negative": round(avg_neg, 4), "avg_positive": round(avg_pos, 4),
                "avg_neutral": round(1.0 - avg_neg - avg_pos, 4),
                "chunk_count": len(doc_chunks),
                "dominant": "negative" if avg_neg > avg_pos and avg_neg > (1-avg_neg-avg_pos)
                            else "positive" if avg_pos > avg_neg and avg_pos > (1-avg_neg-avg_pos)
                            else "neutral",
            })
        topic_trends[topic] = periods

    significant_shifts = _detect_shifts(topic_trends, scored)
    summary = _build_summary(significant_shifts, topic_trends, scored)
    return {"scored_chunks": scored, "topic_trends": topic_trends,
            "significant_shifts": significant_shifts, "summary": summary}


def _detect_shifts(topic_trends: dict, scored_chunks: list) -> list:
    SHIFT_THRESHOLD = 0.15
    shifts = []
    for topic, periods in topic_trends.items():
        if len(periods) < 2:
            continue
        for i in range(1, len(periods)):
            prev, curr = periods[i-1], periods[i]
            neg_delta = curr["avg_negative"] - prev["avg_negative"]
            pos_delta = curr["avg_positive"] - prev["avg_positive"]
            if abs(neg_delta) >= SHIFT_THRESHOLD or abs(pos_delta) >= SHIFT_THRESHOLD:
                period_chunks = sorted(
                    [c for c in scored_chunks if c.get("topic") == topic and c.get("date") == curr["date"]],
                    key=lambda x: x["negativity"], reverse=True
                )
                example_chunk = period_chunks[0] if period_chunks else None
                shifts.append({
                    "topic": topic,
                    "from_period": prev["date"], "from_doc": prev["doc_type"],
                    "to_period": curr["date"], "to_doc": curr["doc_type"],
                    "neg_delta": round(neg_delta, 4), "pos_delta": round(pos_delta, 4),
                    "direction": "MORE_NEGATIVE" if neg_delta > 0 else "MORE_POSITIVE",
                    "magnitude": "LARGE" if abs(neg_delta) > 0.25 else "MODERATE",
                    "example_text": example_chunk["text"][:400] if example_chunk else None,
                    "example_section": example_chunk.get("section", "") if example_chunk else None,
                })
    shifts.sort(key=lambda x: abs(x["neg_delta"]), reverse=True)
    return shifts[:10]


def _build_summary(shifts, topic_trends, scored_chunks):
    if not scored_chunks:
        return "Insufficient data for tone analysis."
    avg_neg = sum(c["negativity"] for c in scored_chunks) / len(scored_chunks)
    avg_pos = sum(c["positivity"] for c in scored_chunks) / len(scored_chunks)
    lines = [f"Overall corpus tone: {avg_pos:.1%} positive, {avg_neg:.1%} negative, "
             f"{1-avg_pos-avg_neg:.1%} neutral across {len(scored_chunks)} passages."]
    if not shifts:
        lines.append("No significant tone shifts detected.")
        return " ".join(lines)
    lines.append(f"\n{len(shifts)} significant tone shift(s) detected:\n")
    for s in shifts[:5]:
        direction_label = "deteriorated" if s["direction"] == "MORE_NEGATIVE" else "improved"
        lines.append(f"• [{s['magnitude']}] {s['topic'].replace('_',' ').title()}: tone {direction_label} "
                     f"from {s['from_period']} to {s['to_period']} — negativity {s['neg_delta']:+.3f}")
    return "\n".join(lines)


def format_tone_section(tone_result: dict) -> str:
    lines = ["## Quantitative Tone Analysis (FinBERT)", "",
             "*FinBERT (ProsusAI/finbert) scores financial text sentiment. "
             "Significant shifts warrant qualitative follow-up.*", "",
             tone_result.get("summary", "No tone data available."), ""]

    shifts = tone_result.get("significant_shifts", [])
    if shifts:
        lines += ["### Detected Tone Shifts", "",
                  "| Topic | From | To | Direction | Δ Negativity | Magnitude |",
                  "|---|---|---|---|---|---|"]
        for s in shifts:
            lines.append(f"| {s['topic'].replace('_',' ').title()} "
                         f"| {s['from_period']} ({s['from_doc']}) "
                         f"| {s['to_period']} ({s['to_doc']}) "
                         f"| {s['direction']} | {s['neg_delta']:+.3f} | {s['magnitude']} |")
        lines.append("")

        lines += ["### Most Negative Passages by Topic", ""]
        shown = set()
        for s in shifts[:3]:
            if s["topic"] in shown or not s.get("example_text"):
                continue
            shown.add(s["topic"])
            lines.append(f"**{s['topic'].replace('_',' ').title()}** ({s['to_period']}, {s['to_doc']}):")
            lines.append(f"> {s['example_text'][:300]}...")
            lines.append("")

    topic_trends = tone_result.get("topic_trends", {})
    if topic_trends:
        lines += ["### Sentiment by Topic Over Time", "",
                  "| Topic | Period | Doc Type | Avg Positive | Avg Negative | Dominant |",
                  "|---|---|---|---|---|---|"]
        for topic, periods in sorted(topic_trends.items()):
            for p in periods:
                lines.append(f"| {topic.replace('_',' ').title()} | {p['date']} | {p['doc_type']} "
                             f"| {p['avg_positive']:.3f} | {p['avg_negative']:.3f} | {p['dominant']} |")
        lines.append("")

    return "\n".join(lines)