"""Stage 8 contract/lifecycle fixtures; numeric generation claims live in real-model audit."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from llm_lab.config import Settings
from llm_lab.errors import ServiceError
from llm_lab.generation.service import GenerationService
from llm_lab.schemas.generation import GenerationRequest, GenerationSampling
from llm_lab.schemas.inference import RawInput
from llm_lab.store.trace_store import TraceStore


class FakeAdapter:
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    device = "cpu"
    dtype = torch.float32

    def __init__(self, input_length=1, eos_token_id=3):
        self.tokenizer = SimpleNamespace(
            decode=lambda ids, **_: "".join(f"<{token}>" for token in ids),
            convert_ids_to_tokens=lambda token: f"token-{token}",
        )
        self.architecture = SimpleNamespace(
            vocabulary_size=4,
            hidden_size=8,
            intermediate_size=16,
            query_heads=2,
            config_digest="test-config",
            implementation_digest="test-implementation",
        )
        self.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=8),
            generation_config=SimpleNamespace(eos_token_id=eos_token_id),
        )
        self.input_length = input_length

    def tokenize(self, _text, **_):
        return {"input_ids": [list(range(self.input_length))]}


class FakeModel:
    def __init__(self, *, input_length=1, eos_token_id=3, fail_on_call=None, gate=None):
        self.settings = Settings(
            auto_load=False,
            generation_max_new_tokens=8,
            generation_max_prefix_tokens=8,
            generation_duration_seconds=10,
            generation_ttl_seconds=5,
        )
        self.adapter = FakeAdapter(input_length, eos_token_id)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._calls = []
        self._fail_on_call = fail_on_call
        self._gate = gate

    def _require_ready(self):
        return None

    def submit_trace_job(self, function, *args):
        return self._executor.submit(function, *args)

    def _compute(self, text, _top_k, _request_id, prefix):
        self._calls.append(list(prefix))
        call = len(self._calls)
        if self._gate:
            self._gate[0].set()
            self._gate[1].wait(timeout=5)
        if call == self._fail_on_call:
            raise RuntimeError("isolated synthetic failure")
        logits = torch.tensor([0.0, 0.0, 5.0, 0.0])
        if call == 1 and self.adapter.model.generation_config.eos_token_id == 0:
            logits = torch.tensor([5.0, 0.0, 0.0, 0.0])
        return SimpleNamespace(
            logits=logits,
            response=SimpleNamespace(runtime=SimpleNamespace(forward_duration_ms=1.0)),
        )

    def close(self):
        self._executor.shutdown(wait=True)


def fake_generation(*, model=None, ttl=5):
    model = model or FakeModel()
    settings = model.settings.model_copy(update={"generation_ttl_seconds": ttl})
    store = TraceStore(max_bytes=1_000_000, ttl_seconds=ttl, max_entries=32)
    service = GenerationService(model, store, settings)
    body = GenerationRequest(
        input=RawInput(text="raw prompt"),
        request_key="fake_request_1",
        max_new_tokens=2,
        capture_profile="none",
        retain_traces=0,
        sampling=GenerationSampling(mode="greedy"),
    )
    return model, store, service, body


def test_greedy_generation_freezes_non_sampling_defaults():
    request = GenerationRequest(
        input=RawInput(text="The capital is"),
        request_key="greedy_request_1",
        sampling=GenerationSampling(mode="greedy"),
    )
    assert request.sampling.mode == "greedy"
    assert (request.sampling.temperature, request.sampling.top_k, request.sampling.top_p) == (
        1,
        0,
        1,
    )
    assert request.sampling.seed is None
    assert request.use_cache is False


@pytest.mark.parametrize(
    "parameters",
    [
        {"mode": "sample"},
        {"mode": "sample", "seed": -1},
        {"mode": "greedy", "seed": 12},
        {"mode": "greedy", "temperature": 0.5},
        {"mode": "sample", "top_p": 0},
        {"mode": "sample", "temperature": 0},
        {"mode": "sample", "top_k": -1},
    ],
)
def test_invalid_sampling_settings_are_rejected(parameters):
    with pytest.raises(ValidationError):
        GenerationSampling(**parameters)


def test_generation_request_rejects_chat_and_cache_options():
    base = {
        "input": RawInput(text="A raw prompt"),
        "request_key": "stable_key_1",
        "sampling": GenerationSampling(mode="greedy"),
    }
    with pytest.raises(ValidationError):
        GenerationRequest(**base, use_cache=True)
    with pytest.raises(ValidationError):
        GenerationRequest(**base, chat_template="qwen")


def test_empty_generation_prompt_is_rejected_before_model_forward():
    model, store, service, body = fake_generation()
    try:
        with pytest.raises(ServiceError, match="non-whitespace") as rejected:
            invalid_input = RawInput.model_construct(
                mode="raw", text=" \n\t ", add_special_tokens=False
            )
            service.create(body.model_copy(update={"input": invalid_input}))
        assert rejected.value.status_code == 422
        assert model._calls == []
    finally:
        service.close()
        store.close()
        model.close()


def test_duplicate_generation_request_returns_the_same_session_without_extra_forwards():
    model, store, service, body = fake_generation()
    try:
        first = service.create(body)
        service._entries[first.generation_id].future.result(timeout=5)
        duplicate = service.create(body)
        assert duplicate.generation_id == first.generation_id
        result = service.get(first.generation_id)
        assert len(model._calls) == result.completed_steps == 2
        with pytest.raises(ServiceError, match="different settings") as conflict:
            service.create(body.model_copy(update={"max_new_tokens": 1}))
        assert conflict.value.status_code == 409
        assert len(model._calls) == 2
    finally:
        service.close()
        store.close()
        model.close()


def test_token_budget_and_prefix_length_are_rejected_before_forward():
    model, store, service, body = fake_generation()
    try:
        with pytest.raises(ServiceError, match="budget exceeds limit") as budget:
            service.create(body.model_copy(update={"max_new_tokens": 9}))
        assert budget.value.status_code == 422
        assert model._calls == []
    finally:
        service.close()
        store.close()
        model.close()

    long_model = FakeModel(input_length=9)
    long_model, store, service, body = fake_generation(model=long_model)
    try:
        with pytest.raises(ServiceError, match="prefix limit") as prefix:
            service.create(body)
        assert prefix.value.status_code == 413
        assert long_model._calls == []
    finally:
        service.close()
        store.close()
        long_model.close()


def test_context_limit_stops_before_an_additional_forward():
    model, store, service, body = fake_generation(model=FakeModel(input_length=8))
    try:
        session = service.create(body)
        service._entries[session.generation_id].future.result(timeout=5)
        result = service.get(session.generation_id)
        assert result.status == "stopped"
        assert result.stop_reason == "context_limit"
        assert result.completed_steps == 1
        assert result.attempted_forwards == 1
        assert len(model._calls) == 1
    finally:
        service.close()
        store.close()
        model.close()


def test_generation_appends_the_selected_id_once_and_stops_at_eos():
    model, store, service, body = fake_generation(model=FakeModel(eos_token_id=2))
    try:
        session = service.create(body)
        service._entries[session.generation_id].future.result(timeout=5)
        result = service.get(session.generation_id)
        assert result.status == "completed"
        assert result.stop_reason == "eos"
        assert result.selected_token_ids == [2]
        assert result.steps[0].prefix_token_ids == [0]
        assert result.steps[0].selected_position == 1
        assert model._calls == [[0]]
    finally:
        service.close()
        store.close()
        model.close()


def test_generation_uses_smallest_vocabulary_id_for_a_greedy_tie():
    model, store, service, body = fake_generation()
    model._compute = lambda text, k, identity, prefix: SimpleNamespace(
        logits=torch.tensor([5.0, 5.0, 0.0, 0.0]),
        response=SimpleNamespace(runtime=SimpleNamespace(forward_duration_ms=1.0)),
    )
    try:
        session = service.create(body)
        service._entries[session.generation_id].future.result(timeout=5)
        result = service.get(session.generation_id)
        assert result.steps[0].selected_token_id == 0
        assert result.steps[0].raw_argmax_token_id == 0
    finally:
        service.close()
        store.close()
        model.close()


def test_generation_records_exact_prefix_and_never_retokenizes_continuation():
    model, store, service, body = fake_generation(model=FakeModel(eos_token_id=3))
    try:
        session = service.create(body)
        service._entries[session.generation_id].future.result(timeout=5)
        result = service.get(session.generation_id)
        assert result.status == "completed"
        assert result.stop_reason == "max_new_tokens"
        assert model._calls == [[0], [0, 2]]
        assert [step.prefix_token_ids for step in result.steps] == [[0], [0, 2]]
        assert result.selected_token_ids == [2, 2]
        assert result.decoded_continuation == "<2><2>"
    finally:
        service.close()
        store.close()
        model.close()


def test_cancel_during_forward_discards_its_result_and_runs_no_next_step():
    started, release = threading.Event(), threading.Event()
    gated_model = FakeModel(eos_token_id=3, gate=(started, release))
    model, store, service, body = fake_generation(model=gated_model)
    try:
        session = service.create(body)
        assert started.wait(timeout=3)
        service.cancel(session.generation_id)
        release.set()
        service._entries[session.generation_id].future.result(timeout=5)
        result = service.get(session.generation_id)
        assert result.status == "cancelled"
        assert result.cancellation_requested
        assert result.attempted_forwards == 1
        assert result.completed_steps == 0
        assert model._calls == [[0]]
    finally:
        release.set()
        service.close()
        store.close()
        model.close()


def test_execution_failure_preserves_only_completed_steps():
    model, store, service, body = fake_generation(model=FakeModel(eos_token_id=3, fail_on_call=2))
    try:
        session = service.create(body)
        service._entries[session.generation_id].future.result(timeout=5)
        result = service.get(session.generation_id)
        assert result.status == "failed"
        assert result.stop_reason == "execution_error"
        assert result.error["code"] == "GENERATION_FAILED"
        assert result.attempted_forwards == 2
        assert result.completed_steps == 1
        assert result.selected_token_ids == [2]
    finally:
        service.close()
        store.close()
        model.close()


def test_session_expiry_hides_decisions_and_returns_gone():
    model, store, service, body = fake_generation(model=FakeModel(eos_token_id=2), ttl=0.05)
    try:
        session = service.create(body)
        service._entries[session.generation_id].future.result(timeout=5)
        time.sleep(0.06)
        with pytest.raises(ServiceError, match="Generation data is unavailable") as expired:
            service.get(session.generation_id)
        assert expired.value.status_code == 410
        assert service.status()["expiration_count"] == 1
    finally:
        service.close()
        store.close()
        model.close()
