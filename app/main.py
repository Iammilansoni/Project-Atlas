from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from api.routes.recommend import router as recommend_router
from api.routes.health import router as health_router
from app.config import settings


# ── Background initialisation ──────────────────────────────────────────────────
# We deliberately do NOT import rag_service at module level so that the ONNX
# model download + FAISS load happen AFTER Render detects the open port.
# Without this, the process gets OOM-killed before the health-check passes.

_init_done = threading.Event()
_init_error: Exception | None = None


def _background_init() -> None:
    """Run the heavy startup in a daemon thread so the port opens first."""
    global _init_error
    try:
        from services.rag_service import rag_service  # deferred import
        rag_service.initialize()
    except Exception as exc:
        _init_error = exc
        print(f"❌ Background init failed: {exc}")
    finally:
        _init_done.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bind port immediately, then init services in the background."""
    print("🚀 PROJECT ATLAS — Starting up…")
    print(f"   LLM mode : {settings.active_llm_mode.upper()}")
    print("   Port bound — launching background initialisation…")

    # Kick off heavy init in a daemon thread; don't await it here
    t = threading.Thread(target=_background_init, daemon=True, name="atlas-init")
    t.start()

    yield  # ← server is live and accepting requests from here

    print("👋 PROJECT ATLAS — Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PROJECT ATLAS — AI Learning Path Recommender",
    description=(
        "RAG-powered learning path generation using FAISS + FastEmbed (ONNX) "
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


# ── Readiness probe used by Render health-check ───────────────────────────────
# /health is defined in health_router; this extra endpoint lets callers poll
# whether the RAG service has finished loading.

@app.get("/ready", tags=["System"])
async def readiness() -> JSONResponse:
    if _init_done.is_set():
        if _init_error:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "detail": str(_init_error)},
            )
        return JSONResponse({"status": "ready"})
    return JSONResponse(status_code=503, content={"status": "initialising"})


# ── Static frontend ────────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        return FileResponse(str(frontend_dir / "index.html"))
