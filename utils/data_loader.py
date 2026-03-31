from __future__ import annotations

import pandas as pd
import hashlib
import re
from pathlib import Path
from typing import Optional

from models.course import Course


def _parse_rating(val) -> float:
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return 0.0


def _parse_reviews(val) -> int:
    try:
        cleaned = re.sub(r"[^\d]", "", str(val))
        return int(cleaned) if cleaned else 0
    except (ValueError, TypeError):
        return 0


def _parse_skills(val) -> list[str]:
    if pd.isna(val) or not str(val).strip():
        return []
    return [s.strip() for s in str(val).split(",") if s.strip()]


def _make_id(title: str, institution: str) -> str:
    raw = f"{title}|{institution}".lower().strip()
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_courses(data_path: str | Path) -> list[Course]:
    """Load Coursera.csv and return a list of Course objects."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Expected columns from Coursera.csv:
    # subject, title, institution, learning_product, level, duration, gained_skills, rate, reviews
    col_map = {
        "subject": "subject",
        "title": "title",
        "institution": "institution",
        "learning_product": "learning_product",
        "level": "level",
        "duration": "duration",
        "gained_skills": "skills_raw",
        "rate": "rating",
        "reviews": "reviews",
    }

    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Fill missing columns with defaults
    for col in ["title", "institution", "subject", "level", "duration", "learning_product"]:
        if col not in df.columns:
            df[col] = ""

    df["rating"] = df.get("rating", pd.Series([0.0] * len(df))).apply(_parse_rating)
    df["review_count"] = df.get("reviews", pd.Series([0] * len(df))).apply(_parse_reviews)
    df["skills_raw"] = df.get("skills_raw", pd.Series([""] * len(df)))

    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]

    courses: list[Course] = []
    for _, row in df.iterrows():
        skills = _parse_skills(row.get("skills_raw", ""))
        title = str(row["title"]).strip()
        institution = str(row.get("institution", "")).strip()
        subject = str(row.get("subject", "")).strip()
        level = str(row.get("level", "")).strip()
        duration = str(row.get("duration", "")).strip()
        rating = float(row["rating"])
        review_count = int(row["review_count"])
        lp = str(row.get("learning_product", "")).strip()

        embed_text = f"{title} {subject} {' '.join(skills)} {level}".strip()

        course = Course(
            id=_make_id(title, institution),
            title=title,
            institution=institution,
            subject=subject,
            skills=skills,
            level=level,
            duration=duration,
            rating=rating,
            review_count=review_count,
            learning_product=lp,
            embed_text=embed_text,
        )
        courses.append(course)

    return courses
