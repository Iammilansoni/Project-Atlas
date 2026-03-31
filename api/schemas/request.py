from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    goal: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The learner's goal, e.g. 'Learn Machine Learning from scratch'",
        examples=["Learn Machine Learning from scratch", "Become a Full Stack Web Developer"],
    )
