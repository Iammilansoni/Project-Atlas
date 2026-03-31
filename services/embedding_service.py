from __future__ import annotations

"""
EmbeddingService — LangChain HuggingFaceEmbeddings wrapper.

Backed by all-MiniLM-L6-v2 (384-dim). Used by both the index builder
and the FAISS vector store for query encoding at inference time.
"""

import threading
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


class EmbeddingService:
    """Singleton LangChain HuggingFaceEmbeddings instance."""

    _instance: "EmbeddingService | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "EmbeddingService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._embeddings = None
                cls._instance._loaded = False
        return cls._instance

    def load(self) -> None:
        if not self._loaded:
            print(f"⏳ Loading embedding model via LangChain: {settings.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._loaded = True
            print("✅ LangChain HuggingFaceEmbeddings loaded.")

    @property
    def lc_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the raw LangChain embeddings object (for FAISS build/load)."""
        if not self._loaded:
            self.load()
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text string → list[float] (LangChain interface)."""
        if not self._loaded:
            self.load()
        return self._embeddings.embed_query(text)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


embedding_service = EmbeddingService()
