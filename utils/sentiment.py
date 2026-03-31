from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """Returns compound sentiment score: -1.0 (negative) to +1.0 (positive)."""
    if not text or not text.strip():
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def extract_highlight(reviews: list[str], max_len: int = 150) -> str:
    """Pick the most positive short review as the highlight snippet."""
    if not reviews:
        return ""

    scored = [(r, score_text(r)) for r in reviews if r and len(r.strip()) > 15]
    if not scored:
        return reviews[0][:max_len] if reviews else ""

    best = max(scored, key=lambda x: x[1])
    snippet = best[0].strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rsplit(" ", 1)[0] + "…"
    return snippet
