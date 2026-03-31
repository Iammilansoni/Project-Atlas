#!/usr/bin/env python
"""
build_index.py — One-time LangChain FAISS index builder.

Usage (from project-atlas/ directory):
    python scripts/build_index.py

What it does:
  1. Loads Coursera.csv → list[Course]
  2. Initialises LangChain HuggingFaceEmbeddings (all-MiniLM-L6-v2)
  3. Builds a LangChain FAISS vectorstore (IndexFlatL2) from course documents
  4. Saves to indexes/faiss_store/ (index.faiss + index.pkl)
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project-atlas root to sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings
from utils.data_loader import load_courses
from utils.faiss_index import FAISSIndex
from services.embedding_service import embedding_service


def main() -> None:
    print("=" * 60)
    print("  PROJECT ATLAS — LangChain FAISS Index Builder")
    print("=" * 60)

    # 1. Load courses from Coursera.csv
    data_path = settings.resolve_path(settings.data_path)
    print(f"\n📂 Loading dataset: {data_path}")
    courses = load_courses(data_path)
    print(f"   ✅ {len(courses):,} courses loaded")

    if not courses:
        print("❌ No courses found. Check DATA_PATH in .env")
        sys.exit(1)

    # 2. Load LangChain HuggingFaceEmbeddings
    print(f"\n🔤 Initialising LangChain HuggingFaceEmbeddings ({settings.embedding_model})")
    embedding_service.load()
    lc_embeddings = embedding_service.lc_embeddings

    # 3. Build + save LangChain FAISS vectorstore
    index_dir = settings.resolve_path(settings.faiss_index_dir)
    print(f"\n💾 Building LangChain FAISS vectorstore → {index_dir}")
    index = FAISSIndex(index_dir=str(index_dir))
    index.build(courses, lc_embeddings)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Done!")
    print(f"  Courses indexed : {len(courses):,}")
    print(f"  Index location  : {index_dir}")
    print(f"\n  Start the server:")
    print(f"  uvicorn app.main:app --reload")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
