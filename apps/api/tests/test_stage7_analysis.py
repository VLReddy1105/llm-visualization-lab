"""Stage 7 real captured tensors, independent module references and failure boundaries."""

import threading
import time
from dataclasses import replace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from stage7_validation import COMPONENTS, audit_capture
from test_stage6_analysis import done, insert

from llm_lab.analysis.block_internals import mlp_internals, residual_stream
from llm_lab.analysis.jobs import AnalysisService
from llm_lab.config import Settings
from llm_lab.errors import ServiceError
from llm_lab.main import create_app
from llm_lab.models.manager import ModelService
from llm_lab.schemas.analysis import (
    AnalysisJob,
    BlockParameters,
    MLPInternalsRequest,
    ResidualStreamRequest,
)
from llm_lab.store.trace_store import TraceStore
from llm_lab.tracing.capture_plan import CapturePlan


@pytest.fixture(scope="module", params=["Germany", "France"])
def block_capture(real_service, request):
    return audit_capture(real_service, f"The capital of {request.param} is")


def parameters(**changes):
    return BlockParameters(token_index=4, layer_index=12, **changes)


def test_real_original_observers_and_installed_module_references(block_capture):
    trace, evidence = block_capture
    assert evidence["passed"], {
        name: check for name, check in evidence["comparisons"].items() if not check["passed"]
    }
    assert len(evidence["comparisons"]) == 132
    assert evidence["explicit_capture_forwards"] == 1
    assert trace.metadata.actual_retained_bytes <= trace.metadata.estimated_bytes
    assert evidence["hook_cleanup_passed"]
    assert all(
        row["provenance"] == "directly captured module boundary" for row in evidence["coordinates"]
    )


@pytest.mark.parametrize("layer", [0, 12, 27])
@pytest.mark.parametrize("token", [0, 4])
def test_real_analysis_source_math_coordinates_and_no_forward(
    real_service, block_capture, layer, token
):
    trace, _ = block_capture
    params = BlockParameters(
        token_index=token,
        layer_index=layer,
        hidden_coordinate_index=19,
        intermediate_coordinate_index=3033,
    )
    before = real_service.execution_metrics()["request_forward_passes"]
    with (
        patch.object(real_service.adapter, "forward", side_effect=AssertionError("No forward")),
        patch.object(
            real_service.adapter.model, "forward", side_effect=AssertionError("No forward")
        ),
    ):
        residual = residual_stream(trace, params)
        mlp = mlp_internals(trace, params)
    assert real_service.execution_metrics()["request_forward_passes"] == before
    for result in (residual, mlp):
        assert result.source.trace_id == trace.metadata.trace_id
        assert result.source.token.token_index == token
        assert result.source.token.token_id == trace.metadata.token_ids[token]
        assert result.source.revision == trace.metadata.revision
        assert result.layer_index == layer
        assert all(r.metrics.passed for r in result.reconciliation)
        assert len(result.model_dump_json().encode()) < 64 * 1024
        for operation in result.operations:
            assert operation.provenance == "raw"
            values = trace.tensor(operation.component_id)[0, (0, 4).index(token)]
            assert operation.statistics.count == values.numel()
            assert operation.statistics.l2_norm == pytest.approx(
                float(values.double().square().sum().sqrt()), abs=1e-12, rel=1e-13
            )
            assert operation.output_shape == list(trace.descriptors[operation.component_id].shape)
            assert operation.selected_coordinate.value == float(
                values[operation.selected_coordinate.index]
            )
            expected = sorted(range(len(values)), key=lambda i: (-abs(float(values[i])), i))[:12]
            assert [c.index for c in operation.coordinates] == expected
    for comparison in residual.comparisons:
        if comparison.name == "block":
            assert comparison.update_provenance == "derived"
        else:
            assert comparison.update_provenance == "raw"
        for coordinate in comparison.coordinates + [comparison.selected_coordinate]:
            assert coordinate.signed_delta == coordinate.after - coordinate.before
    coordinate = mlp.selected_gate_coordinate
    assert coordinate.index == 3033
    assert coordinate.product == pytest.approx(coordinate.activated_gate * coordinate.up, abs=1e-5)
    assert mlp.intermediate_size == 3072
    assert next(o for o in mlp.operations if o.symbol == "m").selected_shape[-1] == 1024
    assert next(o for o in mlp.operations if o.symbol == "p").selected_shape[-1] == 3072


@pytest.mark.parametrize("ranking", ["positive", "negative", "absolute"])
def test_real_bounded_signed_coordinate_rankings(block_capture, ranking):
    trace, _ = block_capture
    result = mlp_internals(trace, parameters(ranking=ranking, top_k=7))
    for operation in result.operations:
        values = trace.tensor(operation.component_id)[0, 1]
        eligible = list(range(len(values)))
        if ranking == "positive":
            eligible = [i for i in eligible if values[i] > 0]
            expected = sorted(eligible, key=lambda i: (-float(values[i]), i))[:7]
        elif ranking == "negative":
            eligible = [i for i in eligible if values[i] < 0]
            expected = sorted(eligible, key=lambda i: (float(values[i]), i))[:7]
        else:
            expected = sorted(eligible, key=lambda i: (-abs(float(values[i])), i))[:7]
        assert [c.index for c in operation.coordinates] == expected


@pytest.mark.parametrize(
    "fields",
    [
        {"token_index": -1},
        {"token_index": True},
        {"token_index": "4"},
        {"layer_index": -1},
        {"layer_index": False},
        {"layer_index": float("nan")},
        {"top_k": 0},
        {"top_k": 65},
        {"top_k": True},
        {"ranking": "descending"},
        {"hidden_coordinate_index": -1},
        {"hidden_coordinate_index": 0.1},
        {"intermediate_coordinate_index": -1},
        {"intermediate_coordinate_index": float("inf")},
        {"unknown_field": 1},
    ],
)
def test_strict_invalid_block_parameters(fields):
    with pytest.raises(ValidationError):
        BlockParameters(**({"token_index": 4, "layer_index": 12} | fields))


@pytest.mark.parametrize(
    "changes",
    [
        {"token_index": 1},
        {"token_index": 999},
        {"layer_index": 1},
        {"layer_index": 28},
        {"hidden_coordinate_index": 1024},
        {"intermediate_coordinate_index": 3072},
    ],
)
def test_capture_and_coordinate_boundaries_fail_explicitly(block_capture, changes):
    trace, _ = block_capture
    params = BlockParameters(**({"token_index": 4, "layer_index": 12} | changes))
    for analyze in (residual_stream, mlp_internals):
        with pytest.raises(ServiceError):
            analyze(trace, params)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_tensors_fail_closed(block_capture, value):
    trace, _ = block_capture
    for symbol in ("r0", "a", "g", "u", "s", "p", "m", "r2"):
        tensors = dict(trace._tensors)
        component = f"layer[12].{COMPONENTS[symbol]}"
        tensors[component] = tensors[component].clone()
        tensors[component][0, 1, 0] = value
        invalid = replace(trace, _tensors=tensors)
        analyze = mlp_internals if symbol in {"g", "u", "s", "p"} else residual_stream
        with pytest.raises(ServiceError) as error:
            analyze(invalid, parameters())
        assert "NONFINITE" in error.value.code


def test_bad_reconciliation_and_inconsistent_token_identity(block_capture):
    trace, _ = block_capture
    tensors = dict(trace._tensors)
    tensors["layer[12].residual.after_mlp"] = tensors["layer[12].residual.after_mlp"].clone()
    tensors["layer[12].residual.after_mlp"][0, 1, 0] += 10
    with pytest.raises(ServiceError) as error:
        residual_stream(replace(trace, _tensors=tensors), parameters())
    assert "RECONCILIATION" in error.value.code
    tokens = list(trace.tokens)
    tokens[4] = replace(tokens[4], token_id=tokens[4].token_id + 1)
    with pytest.raises(ServiceError):
        mlp_internals(replace(trace, tokens=tuple(tokens)), parameters())


def test_stage7_cache_immutable_source_association_and_unloaded_analysis(block_capture):
    trace, _ = block_capture
    settings = Settings(auto_load=False)
    model, store = ModelService(settings), TraceStore()
    jobs = AnalysisService(model, store, settings)
    try:
        source = insert(store, trace)
        request = MLPInternalsRequest(kind="mlp_internals", parameters=parameters())
        first = done(jobs, source, jobs.create(source, request))
        assert first.status == "ready", first.error
        duplicate = jobs.create(source, request)
        assert duplicate.analysis_id == first.analysis_id and duplicate.cache_hit
        assert duplicate.method_version == "stage7.v1"
        first.result.operations[0].coordinates.clear()
        assert jobs.get(source, first.analysis_id).result.operations[0].coordinates
        other = insert(store, trace)
        with pytest.raises(ServiceError) as error:
            jobs.get(other, first.analysis_id)
        assert error.value.code == "ANALYSIS_NOT_FOUND"
        assert model.execution_metrics()["request_forward_passes"] == 0
        assert store.status()["active_leases"] == 0
        store.delete(source)
        with pytest.raises(ServiceError):
            jobs.get(source, first.analysis_id)
    finally:
        jobs.close()
        model.close()
        store.close()


def test_stage7_cancellation_expiry_lease_and_recovery(real_service, block_capture, monkeypatch):
    from llm_lab.analysis import jobs as jobs_module

    class Clock:
        value = time.time()

        def __call__(self):
            return self.value

    trace, _ = block_capture
    clock = Clock()
    settings = Settings(auto_load=False, analysis_ttl_seconds=10)
    store = TraceStore(ttl_seconds=20, clock=clock, wall_clock=clock)
    jobs = AnalysisService(real_service, store, settings, clock=clock, wall_clock=clock)
    entered, release = threading.Event(), threading.Event()
    original = jobs_module.mlp_internals

    def held(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    request = MLPInternalsRequest(kind="mlp_internals", parameters=parameters())
    try:
        identity = insert(store, trace)
        monkeypatch.setattr(jobs_module, "mlp_internals", held)
        first = jobs.create(identity, request)
        assert entered.wait(10)
        assert store.status()["active_leases"] == 1
        jobs.cancel(identity, first.analysis_id)
        release.set()
        jobs._executor.submit(lambda: None).result(10)
        assert jobs.get(identity, first.analysis_id).status == "cancelled"
        assert store.status()["active_leases"] == 0
        monkeypatch.setattr(jobs_module, "mlp_internals", original)
        second = done(jobs, identity, jobs.create(identity, request))
        assert second.status == "ready" and second.analysis_id != first.analysis_id
        clock.value += 9
        assert jobs.create(identity, request).expires_at == second.expires_at
        clock.value += 1
        with pytest.raises(ServiceError) as error:
            jobs.get(identity, second.analysis_id)
        assert error.value.code == "ANALYSIS_EXPIRED"
        entered.clear()
        release.clear()
        monkeypatch.setattr(jobs_module, "mlp_internals", held)
        third = jobs.create(identity, request)
        assert entered.wait(10)
        clock.value += 10
        store.sweep()
        assert store.status()["deferred_removals"] == 1
        release.set()
        jobs._executor.submit(lambda: None).result(10)
        assert store.status()["retained_bytes"] == 0
        with pytest.raises(ServiceError) as error:
            jobs.get(identity, third.analysis_id)
        assert error.value.code == "TRACE_EXPIRED"
    finally:
        release.set()
        jobs.close()
        store.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"analysis_work_max_bytes": 1024},
        {"analysis_cache_max_bytes": 1024},
        {"trace_api_max_response_bytes": 1024},
    ],
)
def test_stage7_resource_budgets_before_execution(real_service, block_capture, overrides):
    trace, _ = block_capture
    store = TraceStore()
    jobs = AnalysisService(real_service, store, Settings(auto_load=False, **overrides))
    try:
        identity = insert(store, trace)
        for kind, request_type in (
            ("residual_stream", ResidualStreamRequest),
            ("mlp_internals", MLPInternalsRequest),
        ):
            request = request_type(kind=kind, parameters=parameters(top_k=64))
            with pytest.raises(ServiceError):
                jobs.create(identity, request)
            assert not jobs._entries
        assert store.status()["active_leases"] == 0
    finally:
        jobs.close()
        store.close()


def test_stage7_http_missing_capture_explicit_capture_and_read_only_exploration(real_service):
    store = TraceStore()
    with TestClient(create_app(real_service.settings, real_service, store)) as client:
        prompt = "The capital of Germany is"
        summary = real_service.trace(prompt, CapturePlan()).result(180)
        source = insert(store, summary)
        before = real_service.execution_metrics()["request_forward_passes"]
        body = {"kind": "mlp_internals", "parameters": {"token_index": 4, "layer_index": 12}}
        missing = client.post(f"/api/v1/traces/{source}/analyses", json=body)
        assert missing.status_code == 409
        assert missing.json()["error"]["code"] == "COMPONENT_NOT_CAPTURED"
        assert real_service.execution_metrics()["request_forward_passes"] == before
        captured = client.post(
            "/api/v1/traces?wait=true",
            json={
                "prompt": prompt,
                "capture": {"profile": "mlp", "layers": [12], "tokens": [4]},
            },
        )
        assert captured.status_code == 201, captured.text
        detailed_source = captured.json()["trace_id"]
        assert detailed_source != source
        assert real_service.execution_metrics()["request_forward_passes"] == before + 1
        for kind in ("mlp_internals", "residual_stream"):
            body["kind"] = kind
            started = client.post(f"/api/v1/traces/{detailed_source}/analyses", json=body)
            assert started.status_code == 202, started.text
            ready = done(
                client.app.state.analysis_service,
                detailed_source,
                AnalysisJob.model_validate(started.json()),
            )
            assert ready.status == "ready", ready.error
            assert ready.result.source.trace_id == detailed_source
            assert ready.result.source.token.token_index == 4
            read = client.get(f"/api/v1/traces/{detailed_source}/analyses/{ready.analysis_id}")
            assert read.status_code == 200
        assert real_service.execution_metrics()["request_forward_passes"] == before + 1


def test_isolated_zero_near_zero_negative_algebra_and_stable_ties():
    import math

    import torch

    from llm_lab.analysis.block_internals import _comparison, rank_coordinates

    params = parameters()
    vectors = {"r0": torch.zeros(4), "a": torch.tensor([-2.0, 2.0, -1.0, 1.0])}
    vectors["r1"] = vectors["r0"] + vectors["a"]
    result = _comparison(params, vectors, "attention", "r0", "a", "r1")
    assert result["cosine_before_after"] is None
    assert result["update_to_residual_ratio"] is None
    assert set(result["undefined"]) == {"cosine_before_after", "update_to_residual_ratio"}
    assert rank_coordinates(vectors["a"], "absolute", 4) == [0, 1, 2, 3]
    assert rank_coordinates(vectors["a"], "positive", 4) == [1, 3]
    assert rank_coordinates(vectors["a"], "negative", 4) == [0, 2]
    assert rank_coordinates(torch.zeros(4), "negative", 4) == []
    vectors["r0"] = torch.tensor([1e-38, -1e-38, 0.0, 0.0])
    vectors["a"] = -2 * vectors["r0"]
    vectors["r1"] = vectors["r0"] + vectors["a"]
    result = _comparison(params, vectors, "attention", "r0", "a", "r1")
    assert result["cosine_before_after"] == pytest.approx(-1)
    assert result["update_to_residual_ratio"] == pytest.approx(2)
    assert math.isfinite(result["delta_norm"]) and result["delta_norm"] > 0
    assert not result["undefined"]


def test_unsupported_configuration_and_missing_reference_fail_explicitly(block_capture):
    trace, _ = block_capture
    invalid = replace(trace, metadata=replace(trace.metadata, revision="0" * 40))
    with pytest.raises(ServiceError) as error:
        residual_stream(invalid, parameters())
    assert error.value.code == "ANALYSIS_UNSUPPORTED"
    checks = dict(trace.metadata.validation)
    checks.pop("layer[12].mlp.gate_projection")
    old = replace(trace, metadata=replace(trace.metadata, validation=checks))
    with pytest.raises(ServiceError) as error:
        mlp_internals(old, parameters())
    assert error.value.code == "COMPONENT_NOT_CAPTURED"
    assert "recapture" in error.value.message


def test_store_identity_mismatch_is_rejected_before_cache_lookup(block_capture):
    trace, _ = block_capture
    settings = Settings(auto_load=False)
    model, store = ModelService(settings), TraceStore()
    jobs = AnalysisService(model, store, settings)
    try:
        identity = insert(store, trace)
        request = ResidualStreamRequest(kind="residual_stream", parameters=parameters())
        assert done(jobs, identity, jobs.create(identity, request)).status == "ready"
        # Deliberately corrupt a fixture's immutable-source association, not production data.
        store._entries[identity].trace = trace
        with pytest.raises(ServiceError) as error:
            jobs.create(identity, request)
        assert error.value.code == "SOURCE_IDENTITY_MISMATCH"
        assert store.status()["active_leases"] == 0
    finally:
        jobs.close()
        model.close()
        store.close()
