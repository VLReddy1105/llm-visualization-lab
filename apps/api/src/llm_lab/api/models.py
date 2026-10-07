from fastapi import APIRouter, Depends, Response

from llm_lab.api.dependencies import model_service
from llm_lab.config import MODEL_ID
from llm_lab.errors import ServiceError
from llm_lab.models.manager import ModelService
from llm_lab.schemas.model import ErrorResponse, ModelLoadRequest, ModelStatus

router = APIRouter()


def check_model_id(model_id: str) -> None:
    if model_id != MODEL_ID:
        raise ServiceError("MODEL_NOT_FOUND", "Model is not in the Stage 1 registry.", 404)


@router.get("/api/v1/models", response_model=list[ModelStatus], tags=["models"])
def list_models(service: ModelService = Depends(model_service)) -> list[ModelStatus]:
    return [service.status()]


@router.get("/api/model/status", response_model=ModelStatus, include_in_schema=False)
def model_status(service: ModelService = Depends(model_service)) -> ModelStatus:
    return service.status()


@router.get(
    "/api/v1/models/{model_id}",
    response_model=ModelStatus,
    tags=["models"],
    responses={404: {"model": ErrorResponse}},
)
def get_model(model_id: str, service: ModelService = Depends(model_service)) -> ModelStatus:
    check_model_id(model_id)
    return service.status()


@router.post(
    "/api/v1/models/{model_id}/load",
    response_model=ModelStatus,
    status_code=202,
    tags=["models"],
    responses={
        200: {"model": ModelStatus},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def load_model(
    model_id: str,
    body: ModelLoadRequest,
    response: Response,
    service: ModelService = Depends(model_service),
) -> ModelStatus:
    check_model_id(model_id)
    status = service.start_loading(body.device, body.revision)
    response.status_code = 200 if status.state == "ready" else 202
    return status
