"""Selection validation uses structural metadata, never invented model outputs."""

from dataclasses import FrozenInstanceError

import pytest

from llm_lab.schemas.trace import ArchitectureMetadata
from llm_lab.tracing.capture_plan import CapturePlan


@pytest.fixture
def architecture():
    return ArchitectureMetadata(
        layers=28,
        hidden_size=1024,
        intermediate_size=3072,
        query_heads=16,
        kv_heads=8,
        head_dim=128,
        vocabulary_size=151936,
        rms_norm_epsilon=1e-6,
        rope_theta=1_000_000,
        tied_embeddings=True,
        attention_dropout=0.0,
        attention_backend="eager",
        config_digest="selection-test-structure",
        implementation_digest="selection-test-structure",
        adapter_id="qwen3",
        adapter_version="1.0.0",
    )


def test_summary_defaults_are_lightweight_and_resolve_all_tokens(architecture):
    requested = CapturePlan()
    resolved = requested.resolve(architecture, 5)
    assert resolved.profile == "summary"
    assert resolved.token_indices == (0, 1, 2, 3, 4)
    assert not resolved.effective_raw
    assert not resolved.captures_attention
    assert not resolved.captures_mlp
    assert resolved.query_heads is None
    assert resolved.detailed_layers == ()
    assert resolved.max_bytes == 256 * 1024 * 1024
    assert requested.token_indices is None
    assert resolved.resolve(architecture, 5) == resolved


@pytest.mark.parametrize(
    ("profile", "attention", "mlp"),
    [("layer", True, True), ("attention", True, False), ("mlp", False, True), ("deep", True, True)],
)
def test_detailed_profiles_retain_explicit_scopes(architecture, profile, attention, mlp):
    plan = CapturePlan(profile=profile, layers=[12, 4], token_indices=[3, 1])
    resolved = plan.resolve(architecture, 5)
    assert resolved.layers == (4, 12)
    assert resolved.detailed_layers == (4, 12)
    assert resolved.token_indices == (1, 3)
    assert resolved.captures_attention is attention
    assert resolved.captures_mlp is mlp
    assert resolved.effective_raw
    assert resolved.query_heads == (tuple(range(16)) if attention else None)


def test_overview_and_stats_only_detail(architecture):
    assert CapturePlan(profile="overview").resolve(architecture, 5).effective_raw
    plan = CapturePlan(profile="deep", layers=(12,), raw_tensors=False).resolve(architecture, 5)
    assert not plan.effective_raw
    assert plan.statistics


def test_selected_head_indices_are_query_heads(architecture):
    plan = CapturePlan(profile="attention", layers=(12,), query_heads=(15, 8))
    resolved = plan.resolve(architecture, 5)
    assert resolved.query_heads == (8, 15)
    assert resolved.query_heads[-1] >= architecture.kv_heads


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "future"},
        {"profile": "layer"},
        {"profile": "attention"},
        {"profile": "mlp"},
        {"profile": "deep"},
        {"layers": (0,)},
        {"profile": "overview", "layers": (12,)},
        {"raw_tensors": True},
        {"query_heads": (0,)},
        {"profile": "mlp", "layers": (12,), "query_heads": (0,)},
        {"token_indices": ()},
        {"profile": "layer", "layers": (12,), "query_heads": ()},
        {"token_indices": (1, 1)},
        {"profile": "layer", "layers": (12, 12)},
        {"token_indices": (-1,)},
        {"max_bytes": 0},
        {"max_bytes": -1},
    ],
)
def test_invalid_scope_or_budget_rejected(kwargs):
    with pytest.raises(ValueError):
        CapturePlan(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layers": (True,)},
        {"token_indices": (False,)},
        {"token_indices": (1.0,)},
        {"token_indices": ("1",)},
        {"token_indices": "1"},
        {"token_indices": {1}},
        {"raw_tensors": 1},
        {"statistics": 1},
        {"max_bytes": True},
        {"max_bytes": 1.0},
    ],
)
def test_no_boolean_or_coerced_numeric_selectors(kwargs):
    with pytest.raises(TypeError):
        CapturePlan(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "layer", "layers": (28,)},
        {"token_indices": (5,)},
        {"profile": "attention", "layers": (0,), "query_heads": (16,)},
    ],
)
def test_actual_architecture_bounds_checked_before_execution(architecture, kwargs):
    with pytest.raises(ValueError):
        CapturePlan(**kwargs).resolve(architecture, 5)


@pytest.mark.parametrize("sequence_length", [0, -1])
def test_nonpositive_sequence_length_rejected(architecture, sequence_length):
    with pytest.raises(ValueError):
        CapturePlan().resolve(architecture, sequence_length)


@pytest.mark.parametrize("sequence_length", [True, 1.0, "1"])
def test_sequence_length_is_strict_integer(architecture, sequence_length):
    with pytest.raises(TypeError):
        CapturePlan().resolve(architecture, sequence_length)


def test_plan_is_frozen():
    plan = CapturePlan()
    with pytest.raises(FrozenInstanceError):
        plan.max_bytes = 1
