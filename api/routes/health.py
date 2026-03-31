from fastapi import APIRouter
from api.schemas.response import HealthResponse, IndexStatusResponse
from services.rag_service import rag_service
from services.embedding_service import embedding_service
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        index_loaded=rag_service.is_ready,
        model_loaded=embedding_service.is_loaded,
        llm_mode=settings.active_llm_mode,
        course_count=rag_service.course_count,
    )


@router.get("/index/status", response_model=IndexStatusResponse)
async def index_status():
    return IndexStatusResponse(
        index_loaded=rag_service.is_ready,
        course_count=rag_service.course_count,
        index_path=str(settings.resolve_path(settings.faiss_index_path)),
        last_built=rag_service.last_built,
    )
