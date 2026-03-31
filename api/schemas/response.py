from pydantic import BaseModel
from typing import Literal


class CourseStepResponse(BaseModel):
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


class SkillGapResponse(BaseModel):
    current_level: str
    target_skills: list[str]
    missing_skills: list[str]
    recommended_focus: str


class LearningPathResponse(BaseModel):
    goal: str
    skill_gap_analysis: SkillGapResponse
    learning_path: list[CourseStepResponse]
    total_hours_estimate: str
    estimated_timeline: str
    pro_tip: str
    llm_mode: Literal["claude", "gemini", "deterministic"]


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    model_loaded: bool
    llm_mode: str
    course_count: int


class IndexStatusResponse(BaseModel):
    index_loaded: bool
    course_count: int
    index_path: str
    last_built: str | None
