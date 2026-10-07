from fastapi import APIRouter, Depends

from llm_lab.api.dependencies import model_service
from llm_lab.models.manager import ModelService
from llm_lab.schemas.model import HealthResponse, ServiceStatus

router = APIRouter()


@router.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
@router.get("/api/health", response_model=HealthResponse, include_in_schema=False)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/api/v1/status", response_model=ServiceStatus, tags=["health"])
def status(service: ModelService = Depends(model_service)) -> ServiceStatus:
    model = service.status()
    return ServiceStatus(inference_ready=model.state == "ready", model=model)
