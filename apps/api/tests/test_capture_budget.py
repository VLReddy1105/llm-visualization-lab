"""Estimate-before-execution budget enforcement with bounded real captures."""

import pytest
from test_instrumentation_cleanup import hook_counts
from test_trace_equivalence import CAPITAL_PROMPT
from test_trace_equivalence import capital_trace as capital_trace

from llm_lab.tracing.capture_plan import CapturePlan

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="session")
def summary_trace(real_service):
    return real_service.trace(CAPITAL_PROMPT, CapturePlan()).result(timeout=180)


def test_summary_capture_is_small_and_estimate_bounds_actual_bytes(summary_trace, capital_trace):
    summary = summary_trace.metadata
    detailed = capital_trace.metadata
    assert 0 < summary.actual_retained_bytes <= summary.estimated_bytes
    assert summary.actual_retained_bytes < detailed.actual_retained_bytes
    assert detailed.actual_retained_bytes <= detailed.estimated_bytes
    assert summary.actual_retained_bytes < 16 * 1024 * 1024
    summary_only = summary_trace.descriptors["layer[12].residual.before"]
    assert summary_only.tensor_id is None
    assert summary_only.statistics is not None


def test_oversized_plan_rejected_before_hooks_or_model_forward(real_service, monkeypatch):
    before = hook_counts(real_service._model)
    executions = []
    registrations = []

    def prohibited_forward(*args, **kwargs):
        executions.append(True)
        raise AssertionError("Budget rejection must occur before model execution")

    def prohibited_registration(*args, **kwargs):
        registrations.append(True)
        raise AssertionError("Budget rejection must occur before instrumentation attachment")

    with monkeypatch.context() as patch:
        patch.setattr(real_service._model, "forward", prohibited_forward)
        for module in real_service._model.modules():
            patch.setattr(module, "register_forward_hook", prohibited_registration)
            patch.setattr(module, "register_forward_pre_hook", prohibited_registration)
        with pytest.raises(ValueError, match="(?i)budget|bytes|limit"):
            real_service.trace(
                CAPITAL_PROMPT,
                CapturePlan(profile="deep", layers=tuple(range(28)), max_bytes=1024),
            ).result(timeout=180)
    assert not executions
    assert not registrations
    assert hook_counts(real_service._model) == before


def test_memory_accounting_counts_unique_storage_once(capital_trace):
    unique = {}
    for component, descriptor in capital_trace.descriptors.items():
        if descriptor.tensor_id is not None:
            tensor = capital_trace.tensor(component, copy=False)
            unique.setdefault(descriptor.tensor_id, tensor.numel() * tensor.element_size())
    assert sum(unique.values()) == capital_trace.metadata.actual_retained_bytes
    layer_alias = capital_trace.descriptors["layer[27].output"]
    residual_alias = capital_trace.descriptors["layer[27].residual.after_mlp"]
    assert layer_alias.tensor_id == residual_alias.tensor_id


def test_detailed_statistics_only_capture_validates_without_retaining_raw_values(real_service):
    trace = real_service.trace(
        CAPITAL_PROMPT, CapturePlan(profile="deep", layers=(12,), raw_tensors=False)
    ).result(timeout=180)
    for component in (
        "layer[12].attention.query.projected",
        "layer[12].mlp.gated",
        "layer[12].residual.after_mlp",
    ):
        descriptor = trace.descriptors[component]
        assert descriptor.tensor_id is None
        assert descriptor.statistics is not None
        with pytest.raises(KeyError):
            trace.tensor(component)
    assert trace.metadata.validation["layer[12].residual.attention"]["passed"]
    assert trace.metadata.validation["layer[12].residual.mlp"]["passed"]
    assert trace.metadata.validation["layer[12].mlp.down_projection"]["passed"]
    assert (
        trace.tensor("output.logits").numel() == real_service.adapter.architecture.vocabulary_size
    )


def test_raw_capture_can_omit_statistics_without_omitting_scientific_values(real_service):
    trace = real_service.trace(
        CAPITAL_PROMPT, CapturePlan(profile="deep", layers=(12,), statistics=False)
    ).result(timeout=180)
    descriptor = trace.descriptors["layer[12].attention.query.projected"]
    assert descriptor.tensor_id is not None
    assert descriptor.statistics is None
    assert trace.tensor(descriptor.component_id).numel() > 0
    assert all(item.statistics is None for item in trace.descriptors.values())
    assert trace.metadata.validation["layer[12].residual.attention"]["passed"]
