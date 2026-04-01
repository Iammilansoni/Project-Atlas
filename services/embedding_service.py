from __future__ import annotations

"""
EmbeddingService — native fastembed (ONNX Runtime only, no PyTorch).

Uses fastembed.TextEmbedding directly and wraps it in a LangChain-compatible
Embeddings class so FAISSIndex.build / FAISSIndex.load continue to work
unchanged.  This completely avoids importing sentence-transformers or torch,
keeping resident memory well under Render's 512 MB free-tier limit.
"""

import threading
from typing import List

from langchain_core.embeddings import Embeddings

from app.config import settings


# ── LangChain-compatible wrapper ───────────────────────────────────────────────

class FastEmbedLangChain(Embeddings):
    """
    Thin LangChain Embeddings adapter around fastembed.TextEmbedding.

    Satisfies the langchain_core.embeddings.Embeddings interface
    (embed_documents / embed_query) so it can be passed directly to
    langchain_community.vectorstores.FAISS.
    """

    def __init__(self, model_name: str):
        # Import here so the heavy ONNX model is only loaded when needed
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)

    # LangChain Embeddings interface ──────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self._model.embed([text]))[0].tolist()


# ── Singleton service ──────────────────────────────────────────────────────────

class EmbeddingService:
    """Singleton that lazily initialises FastEmbedLangChain on first use."""

    _instance: "EmbeddingService | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "EmbeddingService":
        with cls._lock:
            if cls._instance is None:
                ins = super().__new__(cls)
                ins._embeddings: FastEmbedLangChain | None = None
                ins._loaded = False
                cls._instance = ins
        return cls._instance

    def load(self) -> None:
        if not self._loaded:
            print(
                f"⏳ Loading embedding model via fastembed (ONNX): "
                f"{settings.embedding_model}"
            )
            self._embeddings = FastEmbedLangChain(model_name=settings.embedding_model)
            self._loaded = True
            print("✅ FastEmbedLangChain loaded.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def lc_embeddings(self) -> FastEmbedLangChain:
        """Return the LangChain-compatible embeddings object."""
        if not self._loaded:
            self.load()
        return self._embeddings  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        if not self._loaded:
            self.load()
        return self._embeddings.embed_query(text)  # type: ignore[union-attr]

    @property
    def is_loaded(self) -> bool:
        return self._loaded


embedding_service = EmbeddingService()
