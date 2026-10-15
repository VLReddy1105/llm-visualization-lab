"""Bounded autoregressive generation routes; GET and DELETE never execute a forward."""

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response

from llm_lab.api.traces import ERRORS
from llm_lab.schemas.generation import (
    GenerationRequest,
    GenerationSession,
    GenerationStep,
    GenerationStepPage,
    GenerationStoreStatus,
)
from llm_lab.schemas.model import ErrorResponse

router = APIRouter(tags=["generation"])
ERRORS = {**ERRORS, 410: {"model": ErrorResponse}}


def route(path, method, model, *, status_code=200, responses=None):
    def register(function):
        router.add_api_route(
            "/api/v1/generations" + path,
            function,
            methods=[method],
            response_model=model,
            status_code=status_code,
            responses={**ERRORS, **(responses or {})},
        )
        router.add_api_route(
            "/api/generations" + path,
            function,
            methods=[method],
            response_model=model,
            status_code=status_code,
            include_in_schema=False,
        )
        return function

    return register


@route("", "POST", GenerationSession, status_code=202)
def create(body: GenerationRequest, request: Request):
    result = request.app.state.generation_service.create(body)
    return request.app.state.trace_serializer.response(
        result, status=202, headers={"Location": f"/api/v1/generations/{result.generation_id}"}
    )


@route("/status", "GET", GenerationStoreStatus)
def status(request: Request):
    result = request.app.state.generation_service.status()
    return request.app.state.trace_serializer.response(result)


@route("/{generation_id}", "GET", GenerationSession)
def get(generation_id: str, request: Request):
    result = request.app.state.generation_service.get(generation_id)
    return request.app.state.trace_serializer.response(result)


@route("/{generation_id}/steps", "GET", GenerationStepPage)
def steps(
    generation_id: str,
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=32)] = 16,
):
    session = request.app.state.generation_service.get(generation_id)
    selected = session.steps[offset : offset + limit]
    result = GenerationStepPage(
        generation_id=generation_id, total=len(session.steps), offset=offset, items=selected
    )
    return request.app.state.trace_serializer.response(result)


@route("/{generation_id}/steps/{step_index}", "GET", GenerationStep)
def step(generation_id: str, step_index: int, request: Request):
    session = request.app.state.generation_service.get(generation_id)
    if not 0 <= step_index < len(session.steps):
        from llm_lab.errors import ServiceError

        raise ServiceError("GENERATION_STEP_NOT_FOUND", "Generation step does not exist.", 404)
    return request.app.state.trace_serializer.response(session.steps[step_index])


@route("/{generation_id}/cancel", "POST", GenerationSession, status_code=202)
def cancel(generation_id: str, request: Request):
    result = request.app.state.generation_service.cancel(generation_id)
    return request.app.state.trace_serializer.response(result, status=202)


@route("/{generation_id}", "DELETE", None, status_code=204)
def delete(generation_id: str, request: Request):
    request.app.state.generation_service.delete(generation_id)
    return Response(status_code=204)
