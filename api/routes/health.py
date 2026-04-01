from __future__ import annotations

from fastapi import APIRouter
from api.schemas.response import HealthResponse, IndexStatusResponse
from app.config import settings

router = APIRouter()


def _get_rag_service():
    """Deferred import — prevents torch/ONNX loading at module-import time."""
    from services.rag_service import rag_service
    return rag_service


def _get_embedding_service():
    from services.embedding_service import embedding_service
    return embedding_service


@router.get("/health", response_model=HealthResponse)
async def health():
    svc = _get_rag_service()
    emb = _get_embedding_service()
    return HealthResponse(
        status="ok",
        index_loaded=svc.is_ready,
        model_loaded=emb.is_loaded,
        llm_mode=settings.active_llm_mode,
        course_count=svc.course_count,
    )


@router.get("/index/status", response_model=IndexStatusResponse)
async def index_status():
    svc = _get_rag_service()
    return IndexStatusResponse(
        index_loaded=svc.is_ready,
        course_count=svc.course_count,
        index_path=str(settings.resolve_path(settings.faiss_index_dir)),
        last_built=svc.last_built,
    )
