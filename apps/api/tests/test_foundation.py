import ast
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_lab.config import Settings
from llm_lab.errors import ServiceError
from llm_lab.main import create_app
from llm_lab.schemas.model import ModelStatus


def test_health_is_independent_of_model_readiness(client):
    for route in ["/api/health", "/api/v1/health"]:
        response = client.get(route)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "ready" not in response.json()
    assert client.get("/api/v1/status").json()["inference_ready"] is False


def test_status_schema_and_alias(client):
    status = ModelStatus.model_validate(client.get("/api/model/status").json())
    assert status.state == "unloaded"
    assert status.resolved_revision is None
    assert status.device is None
    assert status.detected_device in status.available_devices
    assert len(status.configured_revision) == 40
    assert client.get("/api/v1/models/qwen3-0.6b").json() == status.model_dump()


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_empty_prompt(client, text):
    response = client.post("/api/inference/next-token", json={"input": {"text": text}})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize("top_k", [0, -1, 101, 1.2, "10", True])
def test_invalid_top_k(client, top_k):
    response = client.post(
        "/api/inference/next-token", json={"input": {"text": "Hello"}, "top_k": top_k}
    )
    assert response.status_code == 422


def test_model_not_ready(client):
    response = client.post("/api/inference/next-token", json={"input": {"text": "Hello"}})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]


@pytest.mark.parametrize(
    "extra", [{"temperature": 0.7}, {"top_p": 0.8}, {"capture": {}}, {"messages": []}]
)
def test_future_features_are_rejected(client, extra):
    response = client.post("/api/inference/next-token", json={"input": {"text": "Hello"}, **extra})
    assert response.status_code == 422


def test_chat_and_special_token_insertion_rejected(client):
    for body in [{"mode": "chat", "text": "Hello"}, {"text": "Hello", "add_special_tokens": True}]:
        assert client.post("/api/v1/inference", json={"input": body}).status_code == 422


def test_baseline_routes_preserved_and_only_scoped_instrumentation():
    schema = create_app(Settings(auto_load=False)).openapi()
    assert set(schema["paths"]) >= {
        "/api/v1/health",
        "/api/v1/status",
        "/api/v1/models",
        "/api/v1/models/{model_id}",
        "/api/v1/models/{model_id}/load",
        "/api/v1/inference",
    }
    forbidden = {
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_backward_hook",
        "generate",
    }
    source = Path(__file__).parents[1] / "src" / "llm_lab"
    for file in source.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if file == source / "tracing" / "instrumentation.py":
                    assert node.func.attr not in {"generate", "register_backward_hook"}, file
                else:
                    assert node.func.attr not in forbidden, file


def test_cors_local_only(client):
    allowed = client.options(
        "/api/inference/next-token",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    denied = client.options(
        "/api/inference/next-token",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in denied.headers
    with pytest.raises(ValidationError):
        Settings(cors_origins=["*"])


def test_unknown_model(client):
    assert client.get("/api/v1/models/unknown").status_code == 404


def test_model_load_failure_is_safe(client, unloaded_service, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Injected load failure for lifecycle testing")

    monkeypatch.setattr("llm_lab.models.manager.AutoTokenizer.from_pretrained", fail)
    response = client.post("/api/v1/models/qwen3-0.6b/load", json={})
    assert response.status_code == 202
    with pytest.raises(ServiceError):
        unloaded_service.wait_until_ready()
    status = client.get("/api/model/status").json()
    assert status["state"] == "failed"
    assert status["error_code"] == "MODEL_LOAD_FAILED"
    assert "Injected" not in status["error"]
    assert client.get("/api/health").status_code == 200
    assert (
        client.post("/api/inference/next-token", json={"input": {"text": "Hello"}}).status_code
        == 503
    )


def test_health_and_status_respond_during_loading(client, unloaded_service, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def blocked_load(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        raise RuntimeError("Controlled load interruption")

    monkeypatch.setattr("llm_lab.models.manager.AutoTokenizer.from_pretrained", blocked_load)
    try:
        assert client.post("/api/v1/models/qwen3-0.6b/load", json={}).status_code == 202
        assert entered.wait(timeout=2)
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/model/status").json()["state"] == "loading"
        assert client.get("/api/v1/status").json()["inference_ready"] is False
    finally:
        release.set()
    with pytest.raises(ServiceError):
        unloaded_service.wait_until_ready()
