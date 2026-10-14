"""Analysis routes orchestrate bounded jobs; scientific calculations live elsewhere."""

from functools import partial
from typing import Annotated

from fastapi import APIRouter, Path, Request

from llm_lab.api.traces import endpoint
from llm_lab.schemas.analysis import (
    AnalysisJob,
    AnalysisRequest,
    EvolutionRequest,
    PositionParameters,
)

router = APIRouter()
route = partial(endpoint, target=router)


@route("/{trace_id}/analyses", "POST", AnalysisJob, status_code=202)
def create_analysis(trace_id: str, body: AnalysisRequest, request: Request):
    result = request.app.state.analysis_service.create(trace_id, body)
    return request.app.state.trace_serializer.response(result, status=202)


@route("/{trace_id}/analyses/{analysis_id}", "GET", AnalysisJob)
def get_analysis(trace_id: str, analysis_id: str, request: Request):
    return request.app.state.trace_serializer.response(
        request.app.state.analysis_service.get(trace_id, analysis_id)
    )


@route("/{trace_id}/analyses/{analysis_id}", "DELETE", AnalysisJob)
def cancel_analysis(trace_id: str, analysis_id: str, request: Request):
    return request.app.state.trace_serializer.response(
        request.app.state.analysis_service.cancel(trace_id, analysis_id)
    )


@route("/{trace_id}/tokens/{token_index}/journey", "GET", AnalysisJob, status_code=202)
def token_journey(trace_id: str, token_index: Annotated[int, Path(ge=0)], request: Request):
    result = request.app.state.analysis_service.create(
        trace_id,
        EvolutionRequest(
            kind="token_evolution", parameters=PositionParameters(token_index=token_index)
        ),
    )
    return request.app.state.trace_serializer.response(result, status=202)
