from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Keys
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude Sonnet")
    gemini_api_key: str = Field(default="", description="Google AI Studio key for Gemini fallback")

    # Model names
    claude_model: str = "claude-3-5-sonnet-20241022"
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # fastembed ONNX model (384-dim, ~80MB)

    # Data paths (resolved relative to repo root)
    data_path: str = "data/Coursera.csv"
    reviews_path: str = "data/reviews.csv"
    reviews_by_course_path: str = "data/reviews_by_course.csv"

    # Index directory (LangChain FAISS uses a folder: index.faiss + index.pkl)
    faiss_index_dir: str = "indexes/faiss_store"

    # Retrieval
    top_k_retrieval: int = 15
    max_reviews_per_course: int = 3

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @property
    def active_llm_mode(self) -> str:
        """Returns which LLM mode will be used."""
        if self.anthropic_api_key:
            return "claude"
        elif self.gemini_api_key:
            return "gemini"
        return "deterministic"

    def resolve_path(self, p: str) -> Path:
        """Resolve a path relative to the repo root (parent of app/)."""
        base = Path(__file__).resolve().parent.parent
        return (base / p).resolve()


settings = Settings()
