from fastapi import APIRouter
from api.schemas import HealthResponse
from api.services.model_service import ModelService
from api.config import DB_PATH

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    db_ok = DB_PATH.exists()
    return HealthResponse(
        status="ok",
        model_loaded=ModelService.is_loaded(),
        db_connected=db_ok,
    )
