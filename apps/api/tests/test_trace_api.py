"""HTTP contract, retained CPU access, and actual pinned Qwen integration."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from llm_lab.config import Settings
from llm_lab.main import create_app
from llm_lab.schemas.trace_api import (
    AttentionHead,
    AttentionRelation,
    ComponentDescriptor,
    ComponentGroup,
    LogitsResponse,
    Page,
    StoreStatus,
    TensorSlice,
    TraceEnvelope,
)
from llm_lab.store.trace_store import TraceStore

PROMPT = "The capital of Germany is"
BODY = {"prompt": PROMPT, "capture": {"profile": "deep", "layers": [12]}}


def hooks(service):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in service._model.modules())


def test_trace_routes_exist_only_in_canonical_openapi(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/v1/traces" in paths
    assert "/api/traces" not in paths
    assert len([p for p in paths if p.startswith("/api/v1/traces")]) == 16
    assert "201" in paths["/api/v1/traces"]["post"]["responses"]
    assert "204" in paths["/api/v1/traces/{trace_id}"]["delete"]["responses"]


def test_store_status_does_not_require_model(client):
    for prefix in ("/api/traces", "/api/v1/traces"):
        body = client.get(prefix + "/status").json()
        StoreStatus.model_validate(body)
        assert body["trace_count"] == body["request_forward_passes"] == 0
        assert body["max_bytes"] == 512 * 1024**2
        assert body["ttl_seconds"] == 1800


def test_missing_trace_and_unready_model(client):
    response = client.get("/api/traces/00000000-0000-4000-8000-000000000001")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TRACE_NOT_FOUND"
    response = client.post("/api/traces", json=BODY)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": " "},
        {"prompt": PROMPT, "capture": {"profile": "invalid"}},
        {"prompt": PROMPT, "capture": {"layers": [-1]}},
        {"prompt": PROMPT, "capture": {"heads": [True]}},
        {"prompt": PROMPT, "capture": {"tokens": ["3"]}},
        {"prompt": PROMPT, "display_top_k": 101},
        {"prompt": PROMPT, "display_top_k": True},
        {"prompt": PROMPT, "capture": {"python_path": "model.layers"}},
        {"prompt": PROMPT, "temperature": 0.7},
    ],
)
def test_safe_capture_schema(client, body):
    assert client.post("/api/traces", json=body).status_code == 422


def test_delete_cors_and_collection_limits(client):
    response = client.options(
        "/api/traces/test",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "DELETE"},
    )
    assert response.status_code == 200
    assert client.get("/api/traces?limit=201").status_code == 422
    assert client.get("/api/traces?cursor=wrong!").status_code == 422


@pytest.fixture(scope="module")
def http_trace(real_service):
    initial_hooks = hooks(real_service)
    with TestClient(create_app(Settings(auto_load=False), real_service)) as client:
        forwards = real_service.execution_metrics()["request_forward_passes"]
        response = client.post("/api/traces?wait=true", json=BODY)
        assert response.status_code == 201, response.text
        envelope = response.json()
        assert real_service.execution_metrics()["request_forward_passes"] == forwards + 1
        assert hooks(real_service) == initial_hooks
        yield client, envelope, real_service


@pytest.mark.real_model
def test_real_creation_metadata_tokens_and_baseline(http_trace, real_result):
    client, body, service = http_trace
    TraceEnvelope.model_validate(body)
    assert UUID(body["trace_id"]).version == 4
    assert body["status"] == "ready"
    overview = body["overview"]
    assert overview["input"]["text"] == PROMPT
    assert [t["token_id"] for t in overview["input"]["tokens"]["items"]] == [
        785,
        6722,
        315,
        9856,
        374,
    ]
    assert overview["prediction"]["top_k"] == [
        c.model_dump() for c in real_result.response.prediction.top_k
    ]
    assert overview["model"]["revision"] == real_result.response.model.revision
    assert overview["available"]["layers"] == list(range(28))
    assert overview["available"]["attention_layers"] == overview["available"]["mlp_layers"] == [12]
    assert body["retained_bytes"] > 0
    assert len(str(body)) < 20000
    assert "values" not in str(body)
    StoreStatus.model_validate(client.get("/api/traces/status").json())


@pytest.mark.real_model
@pytest.mark.parametrize(
    "suffix,model",
    [
        ("", TraceEnvelope),
        ("/tokens", Page),
        ("/layers", Page),
        ("/layers/12", ComponentGroup),
        ("/layers/12/attention", ComponentGroup),
        ("/layers/12/mlp", ComponentGroup),
        ("/layers/12/residual", ComponentGroup),
        (
            "/layers/12/attention/heads/4?query_token_index=3&include_weights=true&include_query=true",
            AttentionHead,
        ),
        ("/logits?display_top_k=100", LogitsResponse),
        ("/components/layer[12].attention.weights_reconstructed", ComponentDescriptor),
        (
            "/layers/12/attention/heads/4/relation?query_token_index=3&key_token_index=4",
            AttentionRelation,
        ),
    ],
)
def test_real_lazy_retrieval_no_forward(http_trace, suffix, model):
    client, body, service = http_trace
    before = service.execution_metrics()["request_forward_passes"]
    response = client.get(f"/api/traces/{body['trace_id']}" + suffix)
    assert response.status_code == 200, response.text
    model.model_validate(response.json())
    assert service.execution_metrics()["request_forward_passes"] == before
    assert "NaN" not in response.text and "Infinity" not in response.text
    if model is AttentionHead:
        data = response.json()
        assert data["kv_head_index"] == 2
        assert data["weights"]["axes"] == ["batch", "query_head", "query_token", "key_token"]
        assert data["weights"]["shape"] == [1, 1, 1, 5]
        assert data["weights"]["provenance"] == "raw"
        assert data["query"]["provenance"] == "reconstructed"
    if model is AttentionRelation:
        assert response.json()["masked"] is True
        assert response.json()["observations"]["weights"]["values"] == [0.0]


@pytest.mark.real_model
@pytest.mark.parametrize(
    "component,selectors,shape",
    [
        ("layer[12].residual.before", {"batch": {"index": 0}, "token": {"index": 3}}, [1, 1, 1024]),
        (
            "layer[12].attention.query.rotary",
            {"query_head": {"index": 4}, "query_token": {"index": 3}},
            [1, 1, 1, 128],
        ),
        (
            "layer[12].attention.weights",
            {"query_head": {"index": 4}, "query_token": {"index": 3}},
            [1, 1, 1, 5],
        ),
        (
            "layer[12].mlp.gated",
            {"token": {"index": 3}, "intermediate": {"start": 0, "stop": 256}},
            [1, 1, 256],
        ),
        ("output.logits", {"vocabulary": {"start": 19846, "stop": 19847}}, [1, 1, 1]),
    ],
)
def test_real_bounded_slices_equal_retained_tensor(http_trace, component, selectors, shape):
    client, body, service = http_trace
    identity = body["trace_id"]
    before = service.execution_metrics()["request_forward_passes"]
    response = client.post(
        f"/api/traces/{identity}/components/{component}/slice", json={"selectors": selectors}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    TensorSlice.model_validate(data)
    assert data["shape"] == shape
    with client.app.state.trace_store.acquire(identity) as (_, trace):
        source = trace.tensor(component, copy=False)
        axis_selections = []
        for axis in trace.descriptors[component].axes:
            selected = selectors.get(axis)
            axis_selections.append(
                slice(None)
                if selected is None
                else slice(selected["index"], selected["index"] + 1)
                if "index" in selected
                else slice(selected["start"], selected["stop"])
            )
        assert data["values"] == source[tuple(axis_selections)].reshape(-1).tolist()
    assert service.execution_metrics()["request_forward_passes"] == before


@pytest.mark.real_model
@pytest.mark.parametrize(
    "suffix,status,code",
    [
        ("/layers/28", 422, "INVALID_LAYER"),
        ("/layers/-1", 422, "INVALID_LAYER"),
        ("/layers/15/attention", 409, "COMPONENT_NOT_CAPTURED"),
        ("/layers/12/attention/heads/20", 422, "INVALID_HEAD"),
        ("/layers/12/attention/heads/4?query_token_index=5", 422, "INVALID_TOKEN"),
        ("/components/layer[15].attention.weights", 409, "COMPONENT_NOT_CAPTURED"),
        ("/components/__dict__", 404, "COMPONENT_NOT_FOUND"),
        ("/logits?token_index=0", 409, "COMPONENT_NOT_CAPTURED"),
    ],
)
def test_real_errors_do_not_recapture(http_trace, suffix, status, code):
    client, body, service = http_trace
    before = service.execution_metrics()["request_forward_passes"]
    response = client.get(f"/api/traces/{body['trace_id']}" + suffix)
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert "Traceback" not in response.text
    assert service.execution_metrics()["request_forward_passes"] == before


@pytest.mark.real_model
@pytest.mark.parametrize(
    "capture,code",
    [
        ({"profile": "deep", "layers": [28]}, "INVALID_LAYER"),
        ({"profile": "attention", "layers": [12], "heads": [20]}, "INVALID_HEAD"),
        ({"profile": "deep", "layers": [12], "tokens": [5]}, "INVALID_TOKEN"),
        ({"profile": "deep", "layers": [12, 12]}, "INVALID_CAPTURE"),
        ({"profile": "deep", "layers": []}, "INVALID_CAPTURE"),
        ({"profile": "summary", "heads": [0]}, "INVALID_CAPTURE"),
    ],
)
def test_real_preflight_rejects_before_forward(http_trace, capture, code):
    client, _, service = http_trace
    before = service.execution_metrics()["request_forward_passes"]
    response = client.post("/api/traces", json={"prompt": PROMPT, "capture": capture})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == code
    assert service.execution_metrics()["request_forward_passes"] == before


@pytest.mark.real_model
def test_real_slice_limits_alias_disambiguation_and_pagination(http_trace):
    client, body, _ = http_trace
    base = f"/api/traces/{body['trace_id']}"
    response = client.post(base + "/components/output.logits/slice", json={})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "RESPONSE_TOO_LARGE"
    d = client.get(base + "/components/layer[12].attention.head_output").json()
    alias = base + f"/tensors/{d['tensor_id']}/slice"
    assert client.post(alias, json={}).status_code == 422
    response = client.post(
        alias, json={"component_id": d["component_id"], "selectors": {"query_head": {"index": 4}}}
    )
    assert response.status_code == 200
    first = client.get(base + "/components?limit=20").json()
    second = client.get(
        base + "/components", params={"limit": 20, "cursor": first["next_cursor"]}
    ).json()
    assert len(first["items"]) == len(second["items"]) == 20
    assert {d["component_id"] for d in first["items"]}.isdisjoint(
        d["component_id"] for d in second["items"]
    )


@pytest.mark.real_model
def test_real_concurrent_reads_and_creation_share_single_model_executor(http_trace, monkeypatch):
    client, body, service = http_trace
    active = maximum = 0
    lock = threading.Lock()
    original = service.adapter.forward

    def observe(*args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(service.adapter, "forward", observe)
    before = service.execution_metrics()["request_forward_passes"]
    baseline = service.infer(PROMPT)
    with ThreadPoolExecutor(max_workers=5) as pool:
        create = pool.submit(client.post, "/api/traces?wait=true", json={"prompt": PROMPT})
        reads = [
            pool.submit(client.get, f"/api/traces/{body['trace_id']}/layers/12") for _ in range(4)
        ]
        assert all(f.result(timeout=60).status_code == 200 for f in reads)
        result = create.result(timeout=180)
    assert result.status_code == 201, result.text
    assert baseline.result(timeout=180).prediction.top_k[0].token_id == 19846
    assert maximum == 1
    assert service.execution_metrics()["request_forward_passes"] == before + 2
    assert client.delete(f"/api/traces/{result.json()['trace_id']}").status_code == 204


@pytest.mark.real_model
def test_real_async_ack_poll_and_delete(http_trace):
    client, _, service = http_trace
    initial_hooks = hooks(service)
    response = client.post("/api/v1/traces", json={"prompt": PROMPT})
    assert response.status_code == 202, response.text
    ack = response.json()
    assert ack["overview"] is None and ack["status"] == "running"
    assert response.headers["Location"].endswith(ack["trace_id"])
    for _ in range(200):
        body = client.get(response.headers["Location"]).json()
        if body["status"] != "running":
            break
        time.sleep(0.05)
    assert body["status"] == "ready"
    assert hooks(service) == initial_hooks
    identity = ack["trace_id"]
    assert client.delete(f"/api/traces/{identity}").status_code == 204
    assert client.delete(f"/api/traces/{identity}").status_code == 204
    assert client.get(f"/api/traces/{identity}").status_code == 404


@pytest.mark.real_model
def test_real_failed_creation_safe_status_and_budget_rejections(http_trace, monkeypatch):
    _, _, service = http_trace
    tiny = Settings(auto_load=False, trace_max_bytes=1)
    with TestClient(create_app(tiny, service)) as client:
        response = client.post("/api/traces", json=BODY)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "CAPTURE_BUDGET_EXCEEDED"
    tiny = Settings(auto_load=False, trace_store_max_bytes=1)
    with TestClient(create_app(tiny, service)) as client:
        response = client.post("/api/traces", json=BODY)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "TRACE_STORE_BUDGET_EXCEEDED"

    def fail(*args):
        raise RuntimeError("Private injected diagnostic must never reach HTTP")

    monkeypatch.setattr(service, "_trace", fail)
    with TestClient(create_app(Settings(auto_load=False), service)) as client:
        response = client.post("/api/traces?wait=true", json=BODY)
        assert response.status_code == 202
        failed = response.json()
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "TRACE_CAPTURE_FAILED"
        assert "Private" not in response.text and "Traceback" not in response.text
        assert client.get(f"/api/traces/{failed['trace_id']}").status_code == 200
        assert client.get(f"/api/traces/{failed['trace_id']}/layers").status_code == 409
        assert client.get("/api/traces/status").json()["accounted_bytes"] == 0


@pytest.mark.real_model
def test_real_http_expiration_with_active_lease(http_trace):
    client, body, service = http_trace
    clock = [0.0]
    store = TraceStore(ttl_seconds=10, clock=lambda: clock[0])
    with client.app.state.trace_store.acquire(body["trace_id"]) as (_, trace):
        identity = store.reserve(trace.metadata.actual_retained_bytes)["trace_id"]
        store.commit(identity, trace)
    with TestClient(create_app(Settings(auto_load=False), service, store)) as ttl_client:
        with store.acquire(identity):
            clock[0] = 10
            response = ttl_client.get(f"/api/traces/{identity}")
            assert response.status_code == 410
            assert response.json()["error"]["code"] == "TRACE_EXPIRED"
            assert store.status()["retained_bytes"] > 0
        assert store.status()["retained_bytes"] == 0


@pytest.mark.real_model
def test_real_selected_coverage_preserves_full_keys_and_rejects_uncaptured_rows(http_trace):
    client, _, service = http_trace
    response = client.post(
        "/api/traces?wait=true",
        json={
            "prompt": PROMPT,
            "capture": {"profile": "attention", "layers": [12], "tokens": [3], "heads": [4]},
        },
    )
    assert response.status_code == 201, response.text
    identity = response.json()["trace_id"]
    base = f"/api/traces/{identity}"
    before = service.execution_metrics()["request_forward_passes"]
    head = client.get(
        base + "/layers/12/attention/heads/4?include_weights=true&query_token_index=3"
    ).json()
    assert head["kv_head_index"] == 2 and head["weights"]["shape"] == [1, 1, 1, 5]
    assert head["weights"]["coordinates"]["key_token"]["count"] == 5
    assert client.get(base + "/layers/12/attention/heads/3").status_code == 409
    assert (
        client.get(
            base + "/layers/12/attention/heads/4?query_token_index=2&include_weights=true"
        ).status_code
        == 409
    )
    assert client.get(base + "/layers/12/mlp").status_code == 409
    descriptor = client.get(base + "/components/layer[0].residual.before").json()
    assert (
        descriptor["availability"] == "summary_only"
        and descriptor["statistics"]["value_kind"] == "summary"
    )
    assert (
        client.post(base + "/components/layer[0].residual.before/slice", json={}).status_code == 409
    )
    assert service.execution_metrics()["request_forward_passes"] == before
    assert client.delete(base).status_code == 204


@pytest.mark.real_model
def test_real_running_http_cancellation_releases_reservation_and_hooks(http_trace, monkeypatch):
    client, _, service = http_trace
    entered, release = threading.Event(), threading.Event()
    original = service._trace

    def paused(*args):
        entered.set()
        assert release.wait(timeout=10)
        return original(*args)

    monkeypatch.setattr(service, "_trace", paused)
    initial_hooks = hooks(service)
    response = client.post("/api/traces", json={"prompt": PROMPT})
    identity = response.json()["trace_id"]
    assert entered.wait(timeout=5)
    try:
        cancellation = client.delete(f"/api/traces/{identity}")
        assert cancellation.status_code == 202
        assert cancellation.json()["cancellation_requested"] is True
        assert client.get(f"/api/traces/{identity}").json()["status"] == "cancelled"
    finally:
        release.set()
    # A barrier job on the same executor proves capture cleanup finished.
    service.infer(PROMPT).result(timeout=180)
    assert hooks(service) == initial_hooks
    status = client.get("/api/traces/status").json()
    assert status["reserved_bytes"] == 0
    assert client.get(f"/api/traces/{identity}").json()["retained_bytes"] == 0
    assert client.delete(f"/api/traces/{identity}").status_code == 204


@pytest.mark.real_model
def test_real_binary_slice_agrees_with_json_without_forward(http_trace):
    import json

    import torch

    client, body, service = http_trace
    path = f"/api/traces/{body['trace_id']}/components/layer[12].residual.before/slice"
    selection = {"selectors": {"token": {"index": 3}, "hidden": {"start": 0, "stop": 128}}}
    before = service.execution_metrics()["request_forward_passes"]
    expected = client.post(path, json=selection).json()
    response = client.post(path, json={**selection, "format": "binary"})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.content) == 128 * 4
    metadata = json.loads(response.headers["X-Tensor-Metadata"])
    assert (
        metadata["axes"] == expected["axes"] and metadata["coordinates"] == expected["coordinates"]
    )
    assert (
        torch.frombuffer(bytearray(response.content), dtype=torch.float32).tolist()
        == expected["values"]
    )
    assert service.execution_metrics()["request_forward_passes"] == before
