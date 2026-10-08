"""Lazy retained-trace routes. Only POST creation submits model work."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response

from llm_lab.errors import ServiceError
from llm_lab.schemas.model import ErrorResponse
from llm_lab.schemas.trace_api import (
    AttentionHead,
    AttentionRelation,
    CancellationResponse,
    ComponentDescriptor,
    ComponentGroup,
    ComponentPage,
    CreateTraceRequest,
    LogitsResponse,
    Page,
    SliceRequest,
    StoreStatus,
    TensorSlice,
    TokenPage,
    TraceEnvelope,
)
from llm_lab.tracing.serialization import page, plain

router = APIRouter()
Limit = Annotated[int, Query(ge=1, le=200)]
Cursor = Annotated[str | None, Query(max_length=256)]
TopK = Annotated[int, Query(ge=1, le=100)]
BINARY_RESPONSE = {
    200: {
        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}
    }
}
ERRORS = {code: {"model": ErrorResponse} for code in (404, 409, 410, 413, 422, 429, 500, 503)}


def endpoint(path: str, method: str, model, **kwargs):
    responses = {**ERRORS, **kwargs.pop("responses", {})}

    def register(function):
        router.add_api_route(
            "/api/v1/traces" + path,
            function,
            methods=[method],
            response_model=model,
            tags=["traces"],
            responses=responses,
            **kwargs,
        )
        router.add_api_route(
            "/api/traces" + path,
            function,
            methods=[method],
            response_model=model,
            include_in_schema=False,
            **kwargs,
        )
        return function

    return register


def ready(snapshot, trace):
    if trace is None:
        raise ServiceError(
            "TRACE_NOT_READY",
            "Trace has no ready scientific result.",
            409,
            details={"trace_id": snapshot["trace_id"], "status": snapshot["status"]},
        )
    return trace


@endpoint("", "POST", TraceEnvelope, status_code=202, responses={201: {"model": TraceEnvelope}})
async def create_trace(body: CreateTraceRequest, request: Request, wait: bool = False):
    ack, job = request.app.state.trace_service.create(body)
    snapshot = await asyncio.shield(asyncio.wrap_future(ack))
    identity = snapshot["trace_id"]
    headers = {"Location": f"/api/v1/traces/{identity}"}
    serializer = request.app.state.trace_serializer
    if wait:
        await asyncio.shield(asyncio.wrap_future(job))
        with request.app.state.trace_store.acquire(identity) as (snapshot, trace):
            return serializer.response(
                serializer.envelope(snapshot, trace, body.display_top_k),
                status=201 if trace else 202,
                headers=headers,
            )
    return serializer.response(serializer.envelope(snapshot, None), status=202, headers=headers)


@endpoint("/status", "GET", StoreStatus)
def store_status(request: Request):
    settings = request.app.state.trace_settings
    return request.app.state.trace_serializer.response(
        {
            **request.app.state.trace_store.status(),
            **request.app.state.model_service.execution_metrics(),
            "max_elements": settings.trace_api_max_elements,
            "max_response_bytes": settings.trace_api_max_response_bytes,
        }
    )


@endpoint("", "GET", Page)
def list_traces(request: Request, limit: Limit = 50, cursor: Cursor = None):
    records = sorted(
        request.app.state.trace_store.listing(), key=lambda r: (r["created_at"], r["trace_id"])
    )
    return request.app.state.trace_serializer.response(page(records, limit, cursor))


@endpoint("/{trace_id}", "GET", TraceEnvelope)
def overview(trace_id: str, request: Request, display_top_k: TopK = 10):
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        return request.app.state.trace_serializer.response(
            request.app.state.trace_serializer.envelope(snapshot, trace, display_top_k)
        )


@endpoint(
    "/{trace_id}",
    "DELETE",
    CancellationResponse,
    status_code=202,
    responses={204: {"description": "Trace deleted; body empty"}},
)
def delete_trace(trace_id: str, request: Request):
    result = request.app.state.trace_store.delete(trace_id)
    return (
        request.app.state.trace_serializer.response(result, status=202)
        if result
        else Response(status_code=204)
    )


@endpoint("/{trace_id}/tokens", "GET", TokenPage)
def tokens(trace_id: str, request: Request, limit: Limit = 50, cursor: Cursor = None):
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        trace = ready(snapshot, trace)
        return request.app.state.trace_serializer.response(page(plain(trace.tokens), limit, cursor))


@endpoint("/{trace_id}/layers", "GET", Page)
def layers(trace_id: str, request: Request, limit: Limit = 50, cursor: Cursor = None):
    serializer = request.app.state.trace_serializer
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        trace = ready(snapshot, trace)
        return serializer.response(
            page(
                [serializer.layer_summary(trace, layer.layer_index) for layer in trace.layers],
                limit,
                cursor,
            )
        )


def group_response(trace_id, layer_index, request, kind):
    serializer = request.app.state.trace_serializer
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        return serializer.response(serializer.group(ready(snapshot, trace), layer_index, kind))


@endpoint("/{trace_id}/layers/{layer_index}", "GET", ComponentGroup)
def layer(trace_id: str, layer_index: int, request: Request):
    return group_response(trace_id, layer_index, request, "layer")


@endpoint("/{trace_id}/layers/{layer_index}/attention", "GET", ComponentGroup)
def attention(trace_id: str, layer_index: int, request: Request):
    return group_response(trace_id, layer_index, request, "attention")


@endpoint("/{trace_id}/layers/{layer_index}/mlp", "GET", ComponentGroup)
def mlp(trace_id: str, layer_index: int, request: Request):
    return group_response(trace_id, layer_index, request, "mlp")


@endpoint("/{trace_id}/layers/{layer_index}/residual", "GET", ComponentGroup)
def residual(trace_id: str, layer_index: int, request: Request):
    return group_response(trace_id, layer_index, request, "residual")


@endpoint("/{trace_id}/layers/{layer_index}/attention/heads/{head_index}", "GET", AttentionHead)
def head(
    trace_id: str,
    layer_index: int,
    head_index: int,
    request: Request,
    query_token_index: int | None = None,
    include_weights: bool = False,
    include_query: bool = False,
):
    serializer = request.app.state.trace_serializer
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        return serializer.response(
            serializer.head(
                ready(snapshot, trace),
                layer_index,
                head_index,
                query_token_index,
                include_weights,
                include_query,
            )
        )


@endpoint("/{trace_id}/logits", "GET", LogitsResponse)
def logits(
    trace_id: str, request: Request, display_top_k: TopK = 10, token_index: int | None = None
):
    serializer = request.app.state.trace_serializer
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        return serializer.response(
            serializer.logits(ready(snapshot, trace), display_top_k, token_index)
        )


@endpoint("/{trace_id}/components", "GET", ComponentPage)
def components(trace_id: str, request: Request, limit: Limit = 50, cursor: Cursor = None):
    serializer = request.app.state.trace_serializer
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        trace = ready(snapshot, trace)
        return serializer.response(
            page(
                [serializer.descriptor(d) for _, d in sorted(trace.descriptors.items())],
                limit,
                cursor,
            )
        )


@endpoint("/{trace_id}/components/{component_id}", "GET", ComponentDescriptor)
def component(trace_id: str, component_id: str, request: Request):
    serializer = request.app.state.trace_serializer
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        trace = ready(snapshot, trace)
        return serializer.response(serializer.descriptor(serializer.component(trace, component_id)))


@endpoint(
    "/{trace_id}/components/{component_id}/slice", "POST", TensorSlice, responses=BINARY_RESPONSE
)
def component_slice(trace_id: str, component_id: str, body: SliceRequest, request: Request):
    serializer = request.app.state.trace_serializer
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        return serializer.slice_response(ready(snapshot, trace), component_id, body)


@endpoint("/{trace_id}/tensors/{tensor_id}/slice", "POST", TensorSlice, responses=BINARY_RESPONSE)
def tensor_slice(trace_id: str, tensor_id: str, body: SliceRequest, request: Request):
    serializer = request.app.state.trace_serializer
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        trace = ready(snapshot, trace)
        matches = [d.component_id for d in trace.descriptors.values() if d.tensor_id == tensor_id]
        if not matches:
            raise ServiceError("COMPONENT_NOT_FOUND", "Unknown retained tensor ID.", 404)
        if body.component_id is None and len(matches) > 1:
            raise ServiceError(
                "INVALID_SLICE",
                "Aliased tensor ID requires component_id for semantic axes.",
                422,
                details={"component_ids": matches},
            )
        component_id = body.component_id or matches[0]
        if component_id not in matches:
            raise ServiceError("INVALID_SLICE", "Tensor/component references do not match.", 422)
        return serializer.slice_response(trace, component_id, body)


@endpoint(
    "/{trace_id}/layers/{layer_index}/attention/heads/{head_index}/relation",
    "GET",
    AttentionRelation,
)
def relation(
    trace_id: str,
    layer_index: int,
    head_index: int,
    request: Request,
    query_token_index: int,
    key_token_index: int,
):
    serializer = request.app.state.trace_serializer
    with request.app.state.trace_store.acquire(trace_id) as (snapshot, trace):
        trace = ready(snapshot, trace)
        head = serializer.head(trace, layer_index, head_index, query_token_index, False, False)
        if not 0 <= key_token_index < trace.metadata.sequence_length:
            raise ServiceError("INVALID_TOKEN", "Key token exceeds input bounds.", 422)
        attention = trace.layers[layer_index].attention
        selectors = {
            "batch": {"index": 0},
            "query_head": {"index": head_index},
            "query_token": {"index": query_token_index},
            "key_token": {"index": key_token_index},
        }
        observations = {
            name: serializer.slice(
                trace, getattr(attention, name), SliceRequest(selectors=selectors)
            )
            for name in (
                "similarity",
                "scaled_scores",
                "masked_scores",
                "weights",
                "weights_reconstructed",
            )
        }
        mask = serializer.slice(
            trace,
            "position.attention_mask",
            SliceRequest(
                selectors={
                    **{k: v for k, v in selectors.items() if k != "query_head"},
                    "mask_head": {"index": 0},
                }
            ),
        )
        value = mask["values"][0]
        return serializer.response(
            {
                "trace_id": trace_id,
                "layer_index": layer_index,
                "head_index": head_index,
                "kv_head_index": head["kv_head_index"],
                "query_token_index": query_token_index,
                "key_token_index": key_token_index,
                "scaling": attention.scaling,
                "masked": value is None or value < 0,
                "observations": observations,
                "normalization": "retained full key row; no route softmax or dot product",
            }
        )
