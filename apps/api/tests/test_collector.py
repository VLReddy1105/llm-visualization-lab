"""Tensor ownership and instrumentation scaffolding, without model-result mocks."""

import math

import pytest
import torch

from llm_lab.models.adapters.base import HookTarget, Observation
from llm_lab.schemas.components import ComponentSpec, ValueKind
from llm_lab.tracing.capture_plan import CapturePlan
from llm_lab.tracing.collector import TraceCollector
from llm_lab.tracing.instrumentation import TraceSession


def residual_spec(component="layer[0].residual.before", **kwargs):
    return ComponentSpec(component, ("batch", "token", "hidden"), (1, 2, 3), **kwargs)


def test_snapshot_is_owned_even_when_source_already_lives_on_cpu():
    collector = TraceCollector(CapturePlan(profile="overview"), 4096)
    source = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3).requires_grad_()
    expected = source.detach().clone()
    spec = residual_spec(layer_index=0)
    collector.observe(spec, source)
    with torch.no_grad():
        source.add_(100)
    snapshot = collector.tensors[spec.component_id]
    assert torch.equal(snapshot, expected)
    assert snapshot.data_ptr() != source.data_ptr()
    assert not snapshot.requires_grad and snapshot.grad_fn is None
    assert snapshot.device.type == "cpu"
    descriptor = collector.descriptors[spec.component_id]
    assert descriptor.original_device == descriptor.storage_device == "cpu"
    assert collector.actual_retained_bytes == source.numel() * source.element_size()


def test_statistics_use_population_standard_deviation_and_report_nonfinite_values():
    collector = TraceCollector(CapturePlan(profile="overview"), 4096)
    values = torch.tensor([1.0, 2.0, 3.0, float("nan"), float("inf"), -float("inf")])
    spec = residual_spec(layer_index=0)
    collector.observe(spec, values.reshape(spec.logical_shape))
    statistics = collector.descriptors[spec.component_id].statistics
    assert statistics.total_count == 6
    assert statistics.finite_count == statistics.nonfinite_count == 3
    assert statistics.masked_count == 0
    assert statistics.min == 1.0 and statistics.max == 3.0
    assert statistics.mean == 2.0
    assert statistics.std == pytest.approx(math.sqrt(2 / 3), rel=1e-6)
    assert statistics.l2_norm == pytest.approx(math.sqrt(14), rel=1e-6)
    assert statistics.accumulation_dtype == "float32"
    assert "correction=0" in statistics.reduction_scope
    assert any("3 non-finite" in warning for warning in collector.warnings)


def test_causal_mask_sentinel_does_not_pollute_score_or_mask_statistics():
    collector = TraceCollector(CapturePlan(profile="attention", layers=(0,)), 4096)
    sentinel = torch.finfo(torch.float32).min
    axes = ("batch", "query_head", "query_position", "key_position")
    mask = torch.tensor([[[[0.0, sentinel], [0.0, 0.0]]]])
    mask_spec = ComponentSpec("position.attention_mask", axes, (1, 1, 2, 2))
    collector.observe(mask_spec, mask)
    scores = torch.tensor([[[[2.0, sentinel], [4.0, 8.0]]]])
    score_spec = ComponentSpec(
        "layer[0].attention.masked_scores", axes, (1, 1, 2, 2), layer_index=0
    )
    collector.observe(score_spec, scores)
    score_statistics = collector.descriptors[score_spec.component_id].statistics
    assert score_statistics.total_count == 4
    assert score_statistics.masked_count == 1
    assert score_statistics.nonfinite_count == 0
    assert score_statistics.finite_count == 3
    assert score_statistics.min == 2.0 and score_statistics.max == 8.0
    assert score_statistics.mean == pytest.approx(14 / 3, rel=1e-6)
    expected = torch.tensor([2.0, 4.0, 8.0])
    assert score_statistics.std == pytest.approx(float(expected.std(correction=0)), rel=1e-6)
    mask_statistics = collector.descriptors[mask_spec.component_id].statistics
    assert mask_statistics.masked_count == 1
    assert mask_statistics.mean == mask_statistics.std == 0.0


def test_selected_query_mask_statistics_use_absolute_query_coordinates():
    collector = TraceCollector(
        CapturePlan(profile="attention", layers=(0,), token_indices=(1,)), 4096
    )
    sentinel = torch.finfo(torch.float32).min
    axes = ("batch", "query_head", "query_position", "key_position")
    mask = torch.tensor([[[[0.0, sentinel, sentinel], [0.0, 0.0, sentinel], [0.0, 0.0, 0.0]]]])
    mask_spec = ComponentSpec("position.attention_mask", axes, (1, 1, 3, 3))
    collector.observe(mask_spec, mask)
    spec = ComponentSpec(
        "layer[0].attention.masked_scores",
        axes,
        (1, 2, 3, 3),
        layer_index=0,
        coordinates=((0,), (1,), (1,), (0, 1, 2)),
    )
    collector.observe(spec, torch.tensor([[[[2.0, 4.0, sentinel]]]]), selected=True)
    statistics = collector.descriptors[spec.component_id].statistics
    assert statistics.masked_count == 1
    assert statistics.finite_count == 2 and statistics.total_count == 3
    assert statistics.mean == 3.0 and statistics.std == 1.0


def test_verified_aliases_share_storage_and_are_counted_once():
    collector = TraceCollector(CapturePlan(profile="overview"), 4096)
    values = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    original = residual_spec("layer[0].residual.after_mlp", layer_index=0)
    alias = residual_spec(
        "layer[0].output",
        layer_index=0,
        sources=(original.component_id,),
        method="verified decoder-output alias",
    )
    collector.observe(original, values)
    collector.observe(alias, values.clone())
    assert collector.tensors[original.component_id] is collector.tensors[alias.component_id]
    assert (
        collector.descriptors[original.component_id].tensor_id
        == collector.descriptors[alias.component_id].tensor_id
    )
    assert collector.actual_retained_bytes == values.numel() * values.element_size()


def test_equal_non_alias_values_have_independent_owned_storage():
    collector = TraceCollector(CapturePlan(profile="overview"), 4096)
    values = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    original = residual_spec(layer_index=0)
    other = residual_spec(
        "layer[0].attention.output", layer_index=0, sources=(original.component_id,)
    )
    collector.observe(original, values)
    collector.observe(other, values)
    assert collector.tensors[original.component_id] is not collector.tensors[other.component_id]
    assert collector.actual_retained_bytes == 2 * values.numel() * values.element_size()


def test_verified_concatenation_alias_can_change_axes_without_copying_owned_storage():
    collector = TraceCollector(CapturePlan(profile="overview"), 4096)
    values = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    concatenated = ComponentSpec(
        "layer[0].attention.concatenated",
        ("batch", "token", "head_feature"),
        (1, 2, 6),
        layer_index=0,
        value_kind=ValueKind.RAW,
    )
    head_output = ComponentSpec(
        "layer[0].attention.head_output",
        ("batch", "query_head", "query_token", "head_dimension"),
        (1, 2, 2, 3),
        layer_index=0,
        value_kind=ValueKind.RAW,
        sources=(concatenated.component_id,),
        method="verified head concatenation alias",
    )
    collector.observe(concatenated, values)
    collector.observe(head_output, values.reshape(1, 2, 2, 3).transpose(1, 2))
    original = collector.tensors[concatenated.component_id]
    aliased = collector.tensors[head_output.component_id]
    assert torch.equal(aliased, original.reshape(1, 2, 2, 3).transpose(1, 2))
    assert aliased.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert (
        collector.descriptors[concatenated.component_id].tensor_id
        == collector.descriptors[head_output.component_id].tensor_id
    )
    assert collector.actual_retained_bytes == values.numel() * values.element_size()


def test_summary_only_capture_keeps_statistics_without_raw_tensor_substitute():
    collector = TraceCollector(CapturePlan(), 4096)
    spec = residual_spec(layer_index=0)
    collector.observe(spec, torch.arange(6, dtype=torch.float32).reshape(spec.logical_shape))
    descriptor = collector.descriptors[spec.component_id]
    assert descriptor.tensor_id is None
    assert descriptor.availability == "summary_only"
    assert descriptor.statistics is not None
    assert spec.component_id not in collector.tensors
    assert collector.actual_retained_bytes == 0
    with pytest.raises(KeyError):
        collector.tensors[spec.component_id]


def test_partial_hook_registration_failure_removes_already_registered_hooks(monkeypatch):
    first, failing = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    spec = ComponentSpec("example.output", ("batch", "hidden"), (1, 3))
    collector = TraceCollector(CapturePlan(profile="overview"), 4096)

    def observation(_args, _kwargs, output):
        return (Observation(spec, output),)

    targets = tuple(
        HookTarget(module, "forward", (spec,), observation) for module in (first, failing)
    )
    session = TraceSession(targets, collector)

    def fail_registration(*args, **kwargs):
        assert len(first._forward_hooks) == 1
        raise RuntimeError("Intentional registration failure")

    monkeypatch.setattr(failing, "register_forward_hook", fail_registration)
    with pytest.raises(RuntimeError, match="Intentional registration failure"):
        with session:
            pytest.fail("Session cannot enter after a registration failure")
    assert not first._forward_hooks and not failing._forward_hooks
    assert session.handles == []
    assert session.targets == () and session.collector is None
    assert collector.actual_retained_bytes == 0
