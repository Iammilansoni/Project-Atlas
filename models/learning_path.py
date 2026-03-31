from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnrichedCourse:
    course_id: str
    title: str
    institution: str
    subject: str
    skills: list[str]
    level: str
    duration: str
    rating: float
    review_count: int
    rating_label: str
    sentiment_score: float        # -1.0 to 1.0
    review_highlight: str         # best review snippet
    relevance_score: float        # FAISS cosine score


@dataclass
class CourseStep:
    step: int
    course_title: str
    institution: str
    why_this_course: str
    skills_gained: list[str]
    duration: str
    rating: float
    rating_label: str
    review_highlight: str
    level: str


@dataclass
class SkillGapAnalysis:
    current_level: str
    target_skills: list[str]
    missing_skills: list[str]
    recommended_focus: str


@dataclass
class LearningPath:
    goal: str
    skill_gap_analysis: SkillGapAnalysis
    learning_path: list[CourseStep]
    total_hours_estimate: str
    estimated_timeline: str
    pro_tip: str
    llm_mode: str                 # "claude" | "gemini" | "deterministic"
