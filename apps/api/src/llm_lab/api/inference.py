import asyncio

from fastapi import APIRouter, Depends, Request

from llm_lab.api.dependencies import model_service
from llm_lab.models.manager import ModelService
from llm_lab.schemas.inference import NextTokenRequest, NextTokenResponse
from llm_lab.schemas.model import ErrorResponse

router = APIRouter()
error_responses = {code: {"model": ErrorResponse} for code in (409, 413, 422, 429, 500, 503)}


@router.post(
    "/api/v1/inference",
    response_model=NextTokenResponse,
    tags=["inference"],
    responses=error_responses,
)
@router.post(
    "/api/inference/next-token",
    response_model=NextTokenResponse,
    include_in_schema=False,
    responses=error_responses,
)
async def next_token(
    body: NextTokenRequest,
    request: Request,
    service: ModelService = Depends(model_service),
) -> NextTokenResponse:
    future = service.infer(body.input.text, body.top_k, request.state.request_id)
    return await asyncio.wrap_future(future)
