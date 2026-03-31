from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Course:
    id: str
    title: str
    institution: str
    subject: str
    skills: list[str]
    level: str
    duration: str
    rating: float
    review_count: int
    learning_product: str
    embed_text: str = ""          # text used for FAISS embedding

    @property
    def rating_label(self) -> str:
        if self.rating >= 4.7:
            return "⭐ Exceptional"
        elif self.rating >= 4.4:
            return "⭐ Highly Rated"
        elif self.rating >= 4.0:
            return "✅ Well Rated"
        return "📘 Standard"
