from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes.recommend import router as recommend_router
from api.routes.health import router as health_router
from services.rag_service import rag_service
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    print("🚀 PROJECT ATLAS — Starting up…")
    print(f"   LLM mode: {settings.active_llm_mode.upper()}")
    rag_service.initialize()
    yield
    print("👋 PROJECT ATLAS — Shutting down.")


app = FastAPI(
    title="PROJECT ATLAS — AI Learning Path Recommender",
    description=(
        "RAG-powered learning path generation using FAISS + MiniLM embeddings "
        "with Claude Sonnet / Gemini 2.5 Flash reasoning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# API routes
app.include_router(recommend_router, tags=["Recommendations"])
app.include_router(health_router, tags=["System"])

# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
