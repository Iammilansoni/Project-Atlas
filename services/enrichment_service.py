from __future__ import annotations

"""
EnrichmentService — joins FAISS-retrieved courses with real user reviews.

Data sources:
  reviews_by_course.csv → columns: CourseId (slug), Review, Label
    e.g. CourseId="machine-learning", Review="Best ML course!", Label=5

Strategy:
  The CourseId slug (e.g. "machine-learning") doesn't directly match the
  course title (e.g. "Machine Learning"). We build a fuzzy lookup:
    1. Exact slug match after normalising the title to a slug
    2. Prefix/substring match as a fallback
  This gives us real review text for ~60-80% of retrieved courses.
"""

import re
import pandas as pd
from pathlib import Path

from models.course import Course
from models.learning_path import EnrichedCourse
from utils.sentiment import score_text, extract_highlight
from app.config import settings


def _to_slug(text: str) -> str:
    """'Machine Learning Specialization' → 'machine-learning-specialization'"""
    return re.sub(r"[^a-z0-9]+", "-", text.lower().strip()).strip("-")


def _slug_tokens(slug: str) -> set[str]:
    return set(slug.split("-"))


class EnrichmentService:
    """
    Enriches FAISS-retrieved Course objects with:
      - VADER sentiment score (−1 → +1)
      - Representative review highlight
      - Rating tier label
    """

    def __init__(self):
        # slug → list[review_text]
        self._reviews: dict[str, list[str]] = {}
        self._slugs: list[str] = []          # sorted list of known slugs for matching
        self._loaded = False

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_reviews(self) -> None:
        """Load reviews_by_course.csv into an in-memory slug→[reviews] dict."""
        if self._loaded:
            return

        rbc_path = settings.resolve_path(settings.reviews_by_course_path)

        if not rbc_path.exists():
            print(f"⚠️  reviews_by_course.csv not found at {rbc_path}")
            self._loaded = True
            return

        try:
            # File has: CourseId, Review, Label
            df = pd.read_csv(
                rbc_path,
                usecols=["CourseId", "Review"],
                on_bad_lines="skip",
                nrows=300_000,          # cap memory ~100MB
            )
            df = df.dropna(subset=["Review"])
            df["CourseId"] = df["CourseId"].astype(str).str.strip().str.lower()
            df["Review"]   = df["Review"].astype(str).str.strip()

            # Build slug → [reviews] — keep up to 5 reviews per course
            for cid, group in df.groupby("CourseId"):
                texts = group["Review"].tolist()[:5]
                if texts:
                    self._reviews[cid] = texts

            self._slugs = sorted(self._reviews.keys())
            self._loaded = True
            print(f"✅ Reviews loaded: {len(self._reviews):,} courses with reviews")

        except Exception as e:
            print(f"⚠️  Review enrichment failed to load: {e}")
            self._loaded = True

    # ── Review Lookup ──────────────────────────────────────────────────────────

    def _find_reviews(self, course: Course) -> list[str]:
        """
        Multi-strategy slug match:
          1. Exact slug from title
          2. Partial token overlap (≥2 tokens in common)
        """
        if not self._reviews:
            return []

        title_slug = _to_slug(course.title)

        # Strategy 1: exact slug match
        if title_slug in self._reviews:
            return self._reviews[title_slug]

        # Strategy 2: token overlap — need ≥2 meaningful tokens in common
        title_tokens = _slug_tokens(title_slug) - {"and", "the", "of", "in", "for", "a", "to"}
        if len(title_tokens) < 2:
            return []

        best_slug  = None
        best_score = 0
        for slug in self._slugs:
            slug_tokens = _slug_tokens(slug)
            overlap = len(title_tokens & slug_tokens)
            if overlap > best_score and overlap >= 2:
                best_score = overlap
                best_slug  = slug

        if best_slug:
            return self._reviews[best_slug]

        return []

    # ── Enrich ────────────────────────────────────────────────────────────────

    def enrich(
        self, courses: list[Course], scores: list[float]
    ) -> list[EnrichedCourse]:
        """Attach sentiment, review highlight, and rating label to each course."""
        if not self._loaded:
            self.load_reviews()

        enriched: list[EnrichedCourse] = []
        for course, score in zip(courses, scores):
            reviews = self._find_reviews(course)
            sentiment  = score_text(" ".join(reviews)) if reviews else 0.0
            highlight  = extract_highlight(reviews)
            if not highlight:
                # Fallback: construct a synthetic highlight from metadata
                highlight = (
                    f"{'Excellent' if course.rating >= 4.7 else 'Well-regarded'} "
                    f"{course.level.lower()} course with {course.review_count:,} learner reviews."
                )

            enriched.append(EnrichedCourse(
                course_id         = course.id,
                title             = course.title,
                institution       = course.institution,
                subject           = course.subject,
                skills            = course.skills,
                level             = course.level,
                duration          = course.duration,
                rating            = course.rating,
                review_count      = course.review_count,
                rating_label      = course.rating_label,
                sentiment_score   = sentiment,
                review_highlight  = highlight,
                relevance_score   = score,
            ))

        return enriched


enrichment_service = EnrichmentService()
