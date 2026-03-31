from __future__ import annotations

import dataclasses
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.schemas.request import RecommendRequest
from api.schemas.response import LearningPathResponse, CourseStepResponse, SkillGapResponse
from services.rag_service import rag_service

router = APIRouter()


def _to_response(path) -> LearningPathResponse:
    return LearningPathResponse(
        goal=path.goal,
        skill_gap_analysis=SkillGapResponse(
            current_level=path.skill_gap_analysis.current_level,
            target_skills=path.skill_gap_analysis.target_skills,
            missing_skills=path.skill_gap_analysis.missing_skills,
            recommended_focus=path.skill_gap_analysis.recommended_focus,
        ),
        learning_path=[
            CourseStepResponse(
                step=s.step,
                course_title=s.course_title,
                institution=s.institution,
                why_this_course=s.why_this_course,
                skills_gained=s.skills_gained,
                duration=s.duration,
                rating=s.rating,
                rating_label=s.rating_label,
                review_highlight=s.review_highlight,
                level=s.level,
            )
            for s in path.learning_path
        ],
        total_hours_estimate=path.total_hours_estimate,
        estimated_timeline=path.estimated_timeline,
        pro_tip=path.pro_tip,
        llm_mode=path.llm_mode,
    )


@router.post("/recommend-path", response_model=LearningPathResponse)
async def recommend_path(request: RecommendRequest):
    """Generate a personalized 5-step learning path for the given goal."""
    if not rag_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Index not built yet. Run: python scripts/build_index.py",
        )
    try:
        learning_path = await rag_service.recommend(request.goal)
        return _to_response(learning_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
