"""Success/failure cleanup, CPU snapshot ownership and trace object lifetimes."""

import gc
import weakref

import pytest
import torch
from test_trace_equivalence import CAPITAL_PROMPT
from test_trace_equivalence import capital_trace as capital_trace

from llm_lab.tracing import engine
from llm_lab.tracing.capture_plan import CapturePlan
from llm_lab.tracing.collector import TraceCollector

pytestmark = pytest.mark.real_model


def hook_counts(model):
    return tuple(
        (len(module._forward_pre_hooks), len(module._forward_hooks)) for module in model.modules()
    )


def test_no_hook_growth_across_successful_trace_runs(real_service, real_result):
    before = hook_counts(real_service._model)
    for _ in range(3):
        trace = real_service.trace(
            CAPITAL_PROMPT, CapturePlan(profile="attention", layers=(12,))
        ).result(timeout=180)
        assert hook_counts(real_service._model) == before
        assert torch.equal(trace.tensor("output.logits").reshape(-1), real_result.logits)
    assert hook_counts(real_service._model) == before
    assert real_service._model.training is False


def test_collector_failure_removes_hooks_and_untraced_inference_recovers(
    real_service, real_result, monkeypatch
):
    before = hook_counts(real_service._model)
    observe = TraceCollector.observe
    captured_calls = []

    def fail_capture(self, *args, **kwargs):
        captured_calls.append(True)
        if len(captured_calls) == 10:
            raise RuntimeError("Intentional collector capture failure")
        return observe(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(TraceCollector, "observe", fail_capture)
        with pytest.raises(RuntimeError, match="Intentional collector capture failure"):
            real_service.trace(CAPITAL_PROMPT, CapturePlan(profile="deep", layers=(12,))).result(
                timeout=180
            )
    assert len(captured_calls) == 10
    assert hook_counts(real_service._model) == before
    recovered = real_service.infer_for_validation(CAPITAL_PROMPT, 10).result(timeout=180)
    assert torch.equal(recovered.logits, real_result.logits)


def test_inference_failure_removes_all_trace_hooks(real_service, real_result, monkeypatch):
    before = hook_counts(real_service._model)

    def fail_forward(*args, **kwargs):
        raise RuntimeError("Intentional traced model execution failure")

    with monkeypatch.context() as patch:
        patch.setattr(real_service._model, "forward", fail_forward)
        with pytest.raises(RuntimeError, match="Intentional traced model execution failure"):
            real_service.trace(CAPITAL_PROMPT, CapturePlan(profile="deep", layers=(12,))).result(
                timeout=180
            )
    assert hook_counts(real_service._model) == before
    recovered = real_service.infer_for_validation(CAPITAL_PROMPT, 1).result(timeout=180)
    assert torch.equal(recovered.logits, real_result.logits)


def test_retained_scientific_tensors_are_detached_cpu_snapshots(capital_trace):
    for component, descriptor in capital_trace.descriptors.items():
        if descriptor.tensor_id is None:
            continue
        tensor = capital_trace.tensor(component, copy=False)
        assert tensor.device.type == "cpu", component
        assert not tensor.requires_grad and tensor.grad_fn is None, component
        assert descriptor.storage_device == "cpu", component
        assert torch.device(descriptor.original_device).type in {"mps", "cpu", "cuda"}, component


def test_tensor_accessor_copy_cannot_mutate_trace(capital_trace):
    component = "layer[12].residual.before"
    original = capital_trace.tensor(component)
    mutable = capital_trace.tensor(component)
    mutable.zero_()
    assert torch.equal(original, capital_trace.tensor(component))
    assert not torch.equal(mutable, original)


def test_completed_trace_and_tensors_are_collectable_without_stale_service_references(real_service):
    future = real_service.trace(CAPITAL_PROMPT, CapturePlan(profile="deep", layers=(12,)))
    trace = future.result(timeout=180)
    snapshot = trace.tensor("layer[12].attention.query.projected", copy=False)
    tensor_reference = weakref.ref(snapshot)
    del snapshot, trace, future
    # A subsequent serialized inference also releases an executor's last work item.
    real_service.infer_for_validation(CAPITAL_PROMPT, 1).result(timeout=180)
    gc.collect()
    assert tensor_reference() is None


def test_failed_future_does_not_keep_partial_owned_snapshots(real_service, monkeypatch):
    observe = TraceCollector.observe
    references = []
    calls = []

    def capture_then_fail(self, *args, **kwargs):
        observe(self, *args, **kwargs)
        calls.append(True)
        references.extend(weakref.ref(tensor) for tensor in self.workspace.values())
        if len(calls) == 10:
            raise RuntimeError("Intentional failure after owned snapshots exist")

    with monkeypatch.context() as patch:
        patch.setattr(TraceCollector, "observe", capture_then_fail)
        future = real_service.trace(CAPITAL_PROMPT, CapturePlan(profile="deep", layers=(12,)))
        with pytest.raises(RuntimeError, match="Intentional failure after owned snapshots exist"):
            future.result(timeout=180)
    assert references and future.done()
    gc.collect()
    # Keep the failed Future alive: its stored error must not own capture data.
    assert all(reference() is None for reference in references)


def test_post_forward_validation_failure_releases_snapshots_and_temporary_outputs(
    real_service, real_result, monkeypatch
):
    before = hook_counts(real_service._model)
    observe = TraceCollector.observe
    compute = real_service._compute
    snapshot_references = []
    source_references = []
    output_references = []
    validation_calls = []

    def record_snapshots(self, spec, tensor, **kwargs):
        observe(self, spec, tensor, **kwargs)
        snapshot_references.extend(weakref.ref(value) for value in self.workspace.values())
        if tensor.device.type == "cpu":
            source_references.append(weakref.ref(tensor))

    def record_outputs(*args, **kwargs):
        result = compute(*args, **kwargs)
        output_references.extend((weakref.ref(result.logits), weakref.ref(result.probabilities)))
        return result

    def fail_after_reconstruction(adapter, plan, sequence_length, tensors):
        validation_calls.append(True)
        assert sequence_length == 5 and plan.detailed_layers == (12,)
        assert all(value.device.type == "cpu" for value in tensors.values())
        assert "layer[12].attention.weights_reconstructed" in tensors
        assert "output.logits" in tensors and "output.probabilities" in tensors
        assert hook_counts(adapter.model) == before
        raise RuntimeError("Intentional scientific validation failure after forward")

    with monkeypatch.context() as patch:
        patch.setattr(TraceCollector, "observe", record_snapshots)
        patch.setattr(real_service, "_compute", record_outputs)
        patch.setattr(engine, "validate_detailed_capture", fail_after_reconstruction)
        future = real_service.trace(CAPITAL_PROMPT, CapturePlan(profile="deep", layers=(12,)))
        with pytest.raises(RuntimeError, match="Intentional scientific validation failure"):
            future.result(timeout=180)
    assert validation_calls == [True]
    assert snapshot_references and source_references and len(output_references) == 2
    assert future.done()
    assert hook_counts(real_service._model) == before
    gc.collect()
    # Keeping the failed Future live exercises retained exception tracebacks,
    # including post-forward local output and reconstruction variables.
    assert all(reference() is None for reference in snapshot_references)
    assert all(reference() is None for reference in source_references)
    assert all(reference() is None for reference in output_references)
    recovered = real_service.infer_for_validation(CAPITAL_PROMPT, 10).result(timeout=180)
    assert torch.equal(recovered.logits, real_result.logits)
