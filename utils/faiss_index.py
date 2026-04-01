from __future__ import annotations

"""
FAISSIndex — LangChain vectorstore wrapper.

Uses langchain_community.vectorstores.FAISS with HuggingFaceEmbeddings.
Stores Course metadata inside each Document so reconstructing Course
objects after search requires no separate pickle lookup.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from models.course import Course
from app.config import settings


def _course_to_doc(course: Course) -> Document:
    """Serialise a Course into a LangChain Document."""
    return Document(
        page_content=course.embed_text,
        metadata={
            "id":               course.id,
            "title":            course.title,
            "institution":      course.institution,
            "subject":          course.subject,
            "skills":           course.skills,          # list[str] — JSON-safe
            "level":            course.level,
            "duration":         course.duration,
            "rating":           course.rating,
            "review_count":     course.review_count,
            "learning_product": course.learning_product,
        },
    )


def _doc_to_course(doc: Document) -> Course:
    """Reconstruct a Course from a LangChain Document's metadata."""
    m = doc.metadata
    return Course(
        id=m.get("id", ""),
        title=m.get("title", ""),
        institution=m.get("institution", ""),
        subject=m.get("subject", ""),
        skills=m.get("skills", []),
        level=m.get("level", ""),
        duration=m.get("duration", ""),
        rating=float(m.get("rating", 0.0)),
        review_count=int(m.get("review_count", 0)),
        learning_product=m.get("learning_product", ""),
    )


class FAISSIndex:
    """LangChain-backed FAISS vector store."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self._vs: Optional[FAISS] = None
        self._doc_count: int = 0
        self._built_at: Optional[str] = None

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, courses: list[Course], lc_embeddings) -> None:
        """
        Embed all courses via LangChain and persist the FAISS vectorstore.
        `lc_embeddings` is a LangChain Embeddings instance (HuggingFaceEmbeddings).
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)

        docs = [_course_to_doc(c) for c in courses]
        print(f"⚙️  Building LangChain FAISS index from {len(docs):,} documents…")

        self._vs = FAISS.from_documents(docs, lc_embeddings)
        self._vs.save_local(str(self.index_dir))
        self._doc_count = len(docs)

        from datetime import datetime, timezone
        self._built_at = datetime.now(timezone.utc).isoformat()
        print(f"✅ LangChain FAISS built & saved → {self.index_dir}")

    # ── Load ───────────────────────────────────────────────────────────────────

    def load(self, lc_embeddings) -> bool:
        """Load persisted FAISS vectorstore. Returns True on success."""
        faiss_file = self.index_dir / "index.faiss"
        if not faiss_file.exists():
            return False
        try:
            self._vs = FAISS.load_local(
                str(self.index_dir),
                lc_embeddings,
                allow_dangerous_deserialization=True,   # required by LangChain FAISS
            )
            self._doc_count = self._vs.index.ntotal
            mtime = os.path.getmtime(faiss_file)
            from datetime import datetime, timezone
            self._built_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            print(f"✅ LangChain FAISS loaded: {self._doc_count:,} vectors")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load LangChain FAISS index: {e}")
            return False

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 15) -> list[tuple[Course, float]]:
        """
        Semantic search on the vectorstore.
        Returns (Course, similarity_score) pairs in [0, 1].

        Uses similarity_search_with_score which returns L2 distances, then
        converts via similarity = 1 / (1 + distance) so:
          - distance 0  → similarity 1.0  (perfect match)
          - distance ∞  → similarity 0.0  (unrelated)
        This avoids the LangChain relevance-score normalizer that breaks on
        fastembed vectors and emits spurious UserWarning with negative scores.
        """
        if self._vs is None:
            raise RuntimeError("Index not loaded. Run: python scripts/build_index.py")

        # Returns (Document, L2_distance) — lower distance = better match
        results: list[tuple[Document, float]] = \
            self._vs.similarity_search_with_score(query, k=k)

        return [(_doc_to_course(doc), 1.0 / (1.0 + dist)) for doc, dist in results]

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._vs is not None

    @property
    def course_count(self) -> int:
        return self._doc_count

    @property
    def last_built(self) -> Optional[str]:
        return self._built_at
