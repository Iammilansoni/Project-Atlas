from __future__ import annotations

"""
RAGService — orchestrates the full pipeline with proper async isolation.

All blocking operations (embedding, FAISS search, LLM call) are run
in a thread pool via asyncio.to_thread() to avoid blocking the event loop.

Pipeline:
  goal → [thread] embed → [thread] FAISS search → enrich → [thread] LLM
"""

import asyncio

from utils.faiss_index import FAISSIndex
from services.embedding_service import embedding_service
from services.enrichment_service import enrichment_service
from services.llm_service import llm_service
from models.learning_path import LearningPath
from app.config import settings


class RAGService:
    """LangChain-powered RAG orchestrator with async-safe thread offloading."""

    def __init__(self):
        self.faiss_index = FAISSIndex(
            index_dir=str(settings.resolve_path(settings.faiss_index_dir)),
        )
        self._index_loaded = False

    def initialize(self) -> None:
        """Blocking startup: load LangChain FAISS index + reviews + embeddings."""
        # Load embedding model first (needed by FAISS load)
        embedding_service.load()
        # Load LangChain FAISS vectorstore
        self._index_loaded = self.faiss_index.load(embedding_service.lc_embeddings)
        if not self._index_loaded:
            print("⚠️  FAISS index not found. Run: python scripts/build_index.py")
        # Pre-load reviews for enrichment
        enrichment_service.load_reviews()

    async def recommend(self, goal: str) -> LearningPath:
        """
        Full async pipeline. Blocking operations run in thread pool.

        1. FAISS semantic search    → asyncio.to_thread (blocks on CPU/IO)
        2. EnrichmentService.enrich → sync (fast, in-memory)
        3. LLMService.generate_path → asyncio.to_thread (blocks on HTTP)
        """
        if not self._index_loaded:
            raise RuntimeError(
                "FAISS index not loaded. Run: python scripts/build_index.py"
            )

        # Step 1 — LangChain FAISS search (CPU-bound, off main thread)
        raw_results = await asyncio.to_thread(
            self.faiss_index.search, goal, settings.top_k_retrieval
        )
        courses = [r[0] for r in raw_results]
        scores  = [r[1] for r in raw_results]

        # Step 2 — Enrich with VADER sentiment + review highlights (fast, sync OK)
        enriched = enrichment_service.enrich(courses, scores)

        # Step 3 — LangChain LLM chain (IO-bound HTTP, off main thread)
        path = await asyncio.to_thread(
            llm_service.generate_path, goal, enriched
        )
        return path

    @property
    def is_ready(self) -> bool:
        return self._index_loaded

    @property
    def course_count(self) -> int:
        return self.faiss_index.course_count

    @property
    def last_built(self) -> str | None:
        return self.faiss_index.last_built


rag_service = RAGService()
