from __future__ import annotations

"""
LLM Service — LangChain-powered hybrid intelligence with three-tier fallback:

  Tier 1: Claude Sonnet   via langchain-anthropic   (ChatAnthropic)
  Tier 2: Gemini 2.5 Flash via langchain-openai     (ChatOpenAI → OpenAI-compat)
  Tier 3: Deterministic Ranker                      (zero-dependency guarantee)

The full pipeline uses LangChain LCEL (LangChain Expression Language):
  prompt | llm | StrOutputParser()
"""

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.learning_path import (
    EnrichedCourse, CourseStep, SkillGapAnalysis, LearningPath
)
from app.config import settings


# ─── Prompt Template (LangChain ChatPromptTemplate) ───────────────────────────

_SYSTEM_MSG = (
    "You are an expert learning path designer with deep knowledge of online education. "
    "You always respond with valid JSON only — no markdown, no prose, just the JSON object."
)

_HUMAN_TEMPLATE = """\
A learner has the following goal:

GOAL: "{goal}"

Here are the most relevant Coursera courses retrieved via semantic search:
{course_context}

Design a structured 5-step learning path from the courses above.
Return ONLY valid JSON in this exact schema:

{{
  "skill_gap_analysis": {{
    "current_level": "what the learner likely knows now",
    "target_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
    "missing_skills": ["skill1", "skill2", "skill3"],
    "recommended_focus": "one sentence of targeted advice"
  }},
  "learning_path": [
    {{
      "step": 1,
      "course_title": "exact course title from the list above",
      "institution": "institution name",
      "why_this_course": "2-3 sentences explaining WHY this step matters NOW for this specific goal",
      "skills_gained": ["skill1", "skill2", "skill3"],
      "duration": "duration string from the list",
      "rating": 4.8,
      "rating_label": "⭐ Exceptional",
      "review_highlight": "review snippet from the list",
      "level": "Beginner / Intermediate / Advanced"
    }}
  ],
  "total_hours_estimate": "e.g. 120-150 hours",
  "estimated_timeline": "e.g. 3-4 months at 10 hrs/week",
  "pro_tip": "one actionable tip specific to this learning path"
}}

Rules:
- Select exactly 5 courses from the list
- Order from foundational → advanced (respect prerequisites)
- Each why_this_course must be specific to the learner's goal, not generic
- Return ONLY the JSON object"""

# Build the LangChain ChatPromptTemplate once
_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_MSG),
    ("human", _HUMAN_TEMPLATE),
])

_OUTPUT_PARSER = StrOutputParser()


# ─── Course context builder ────────────────────────────────────────────────────

def _build_course_context(courses: list[EnrichedCourse]) -> str:
    lines = []
    for i, c in enumerate(courses[:10], 1):
        skills_str = ", ".join(c.skills[:6]) if c.skills else "General skills"
        lines.append(
            f"[{i}] {c.title}\n"
            f"    Institution: {c.institution} | Level: {c.level} | Duration: {c.duration}\n"
            f"    Rating: {c.rating} ({c.rating_label}) | Skills: {skills_str}\n"
            f"    Review sentiment: {c.sentiment_score:+.2f} | Highlight: \"{c.review_highlight}\""
        )
    return "\n".join(lines)


# ─── JSON Parser ───────────────────────────────────────────────────────────────

def _parse_response(raw: str, goal: str, mode: str) -> LearningPath:
    """Parse LLM JSON string → LearningPath domain model."""
    # Strip code fences if the LLM ignores instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    data = json.loads(raw)

    sg = data["skill_gap_analysis"]
    skill_gap = SkillGapAnalysis(
        current_level=sg.get("current_level", "Beginner"),
        target_skills=sg.get("target_skills", []),
        missing_skills=sg.get("missing_skills", []),
        recommended_focus=sg.get("recommended_focus", ""),
    )

    steps: list[CourseStep] = []
    for item in data.get("learning_path", [])[:5]:
        steps.append(CourseStep(
            step=item.get("step", len(steps) + 1),
            course_title=item.get("course_title", ""),
            institution=item.get("institution", ""),
            why_this_course=item.get("why_this_course", ""),
            skills_gained=item.get("skills_gained", []),
            duration=item.get("duration", ""),
            rating=float(item.get("rating", 0.0)),
            rating_label=item.get("rating_label", ""),
            review_highlight=item.get("review_highlight", ""),
            level=item.get("level", ""),
        ))

    return LearningPath(
        goal=goal,
        skill_gap_analysis=skill_gap,
        learning_path=steps,
        total_hours_estimate=data.get("total_hours_estimate", ""),
        estimated_timeline=data.get("estimated_timeline", ""),
        pro_tip=data.get("pro_tip", ""),
        llm_mode=mode,
    )


# ─── Tier 1: Claude Sonnet via LangChain ──────────────────────────────────────

def _call_claude(goal: str, courses: list[EnrichedCourse]) -> LearningPath:
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )
    # LangChain LCEL chain: prompt | llm | parser
    chain = _PROMPT | llm | _OUTPUT_PARSER
    raw = chain.invoke({
        "goal": goal,
        "course_context": _build_course_context(courses),
    })
    return _parse_response(raw, goal, "claude")


# ─── Tier 2: Gemini 2.5 Flash via LangChain (OpenAI-compat) ──────────────────

def _call_gemini(goal: str, courses: list[EnrichedCourse]) -> LearningPath:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        temperature=0.3,
        max_tokens=4096,
    )
    chain = _PROMPT | llm | _OUTPUT_PARSER
    raw = chain.invoke({
        "goal": goal,
        "course_context": _build_course_context(courses),
    })
    return _parse_response(raw, goal, "gemini")


# ─── Tier 3: Deterministic Ranker (no LLM) ───────────────────────────────────

def _deterministic_path(goal: str, courses: list[EnrichedCourse]) -> LearningPath:
    """
    Scores courses by: FAISS relevance × (rating / 5) × (1 + sentiment).
    Selects best 7, then sorts by level: Beginner → Intermediate → Advanced.
    """
    level_order = {"beginner": 0, "intermediate": 1, "mixed": 2, "advanced": 3}

    def composite_score(c: EnrichedCourse) -> float:
        return c.relevance_score * (c.rating / 5.0) * (1.0 + max(0, c.sentiment_score))

    top7 = sorted(courses, key=composite_score, reverse=True)[:7]
    top7.sort(key=lambda c: level_order.get(c.level.lower().split()[0], 2))
    top5 = top7[:5]

    all_skills = [s for c in top5 for s in c.skills]
    unique_skills = list(dict.fromkeys(all_skills))[:8]

    skill_gap = SkillGapAnalysis(
        current_level="Beginner",
        target_skills=unique_skills[:5],
        missing_skills=unique_skills[:3],
        recommended_focus=(
            f"Build strong foundations before tackling advanced "
            f"{goal.split()[-1] if goal.split() else 'topics'} topics."
        ),
    )

    steps = [
        CourseStep(
            step=i,
            course_title=c.title,
            institution=c.institution,
            why_this_course=(
                f"Step {i} of your path: this {c.level.lower()} course builds "
                f"essential skills in {', '.join(c.skills[:2]) or 'this area'}. "
                f"Rated {c.rating}/5 by {c.review_count:,} learners."
            ),
            skills_gained=c.skills[:4],
            duration=c.duration,
            rating=c.rating,
            rating_label=c.rating_label,
            review_highlight=c.review_highlight,
            level=c.level,
        )
        for i, c in enumerate(top5, 1)
    ]

    return LearningPath(
        goal=goal,
        skill_gap_analysis=skill_gap,
        learning_path=steps,
        total_hours_estimate="80–150 hours (varies by pace)",
        estimated_timeline="2–4 months at 8–10 hrs/week",
        pro_tip="Complete each course fully before advancing, and build a mini-project after each step.",
        llm_mode="deterministic",
    )


# ─── Public Interface ──────────────────────────────────────────────────────────

class LLMService:
    """
    Hybrid LangChain LLM service.
    Chain: ChatPromptTemplate | ChatAnthropic/ChatOpenAI | StrOutputParser
    Fallback: Claude → Gemini → Deterministic
    """

    def generate_path(self, goal: str, courses: list[EnrichedCourse]) -> LearningPath:
        """Try Claude (LangChain) → Gemini (LangChain) → Deterministic. Never raises."""
        print(f"🤖 LLM mode: {settings.active_llm_mode.upper()}")

        if settings.anthropic_api_key:
            try:
                return _call_claude(goal, courses)
            except Exception as e:
                print(f"⚠️  Claude (LangChain) failed ({type(e).__name__}: {e}) → Gemini…")

        if settings.gemini_api_key:
            try:
                return _call_gemini(goal, courses)
            except Exception as e:
                print(f"⚠️  Gemini (LangChain) failed ({type(e).__name__}: {e}) → Deterministic…")

        print("📊 Using deterministic ranker")
        return _deterministic_path(goal, courses)


llm_service = LLMService()
