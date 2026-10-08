"""Config-only adapter contracts; these do not claim real-model numerical evidence."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from llm_lab.models.adapters import Qwen3Adapter, create_adapter
from llm_lab.schemas.components import ValueKind
from llm_lab.tracing.capture_plan import CapturePlan


@pytest.fixture
def structural_model():
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "eager"
    # Construct only meta shapes. No weights are downloaded, retained or run.
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    return model.eval()


@pytest.fixture
def structural_adapter(structural_model):
    return Qwen3Adapter(structural_model, None, "0" * 40, "meta")


def test_architecture_uses_explicit_head_dimension_and_actual_modules(structural_adapter):
    arch = structural_adapter.architecture
    assert arch.layers == 2
    assert arch.hidden_size == 16
    assert arch.query_heads == 4 and arch.kv_heads == 2
    assert arch.head_dim == 8
    assert arch.head_dim != arch.hidden_size // arch.query_heads
    assert arch.intermediate_size == 32 and arch.vocabulary_size == 64
    assert arch.tied_embeddings
    assert arch.attention_backend == "eager"
    assert len(arch.implementation_digest) == len(arch.config_digest) == 64


def test_selected_queries_keep_compact_kv_and_full_key_context(structural_adapter):
    plan = CapturePlan(profile="attention", layers=(1,), token_indices=(3,), query_heads=(3,))
    specs = {spec.component_id: spec for spec in structural_adapter.component_specs(plan, 5)}
    query = specs["layer[1].attention.query.normalized"]
    key = specs["layer[1].attention.key.normalized"]
    value = specs["layer[1].attention.value"]
    weights = specs["layer[1].attention.weights"]
    assert query.logical_shape == (1, 4, 5, 8)
    assert query.coordinates[1:3] == ((3,), (3,))
    assert key.logical_shape == value.logical_shape == (1, 2, 5, 8)
    assert key.axes[1] == value.axes[1] == "kv_head"
    assert key.coordinates[1:3] == ((1,), (0, 1, 2, 3, 4))
    assert weights.coordinates[1:4] == ((3,), (3,), (0, 1, 2, 3, 4))
    assert "layer[0].attention.query.normalized" not in specs


def test_raw_observation_and_functional_reconstruction_have_distinct_provenance(structural_adapter):
    specs = {
        spec.component_id: spec
        for spec in structural_adapter.component_specs(CapturePlan(profile="deep", layers=(0,)), 5)
    }
    assert specs["layer[0].attention.weights"].value_kind == ValueKind.RAW
    assert specs["layer[0].mlp.activated_gate"].value_kind == ValueKind.RAW
    assert specs["layer[0].mlp.gated"].value_kind == ValueKind.RAW
    assert specs["layer[0].attention.query.rotary"].value_kind == ValueKind.RECONSTRUCTED
    assert specs["layer[0].attention.weights_reconstructed"].value_kind == ValueKind.RECONSTRUCTED
    assert specs["output.probabilities"].value_kind == ValueKind.DERIVED
    assert specs["layer[0].attention.query.rotary"].sources == (
        "layer[0].attention.query.normalized",
        "position.rotary_cos",
        "position.rotary_sin",
    )


def test_target_discovery_installs_no_hooks_or_wrappers(structural_adapter):
    before = tuple(
        (len(module._forward_hooks), len(module._forward_pre_hooks))
        for module in structural_adapter.model.modules()
    )
    targets = structural_adapter.hook_targets(CapturePlan(profile="deep", layers=(0, 1)), 5)
    assert targets
    assert all(target.specs and callable(target.observations) for target in targets)
    assert (
        tuple(
            (len(module._forward_hooks), len(module._forward_pre_hooks))
            for module in structural_adapter.model.modules()
        )
        == before
    )
    assert not getattr(structural_adapter.model.model, "_output_capturing_hooks_installed", False)
    assert len({spec.component_id for target in targets for spec in target.specs}) == sum(
        len(target.specs) for target in targets
    )


@pytest.mark.parametrize("mode", ["training", "backend", "hf_capture"])
def test_unsupported_capture_modes_fail_without_mutating_model(structural_adapter, mode):
    model = structural_adapter.model
    if mode == "training":
        model.train()
    elif mode == "backend":
        model.config._attn_implementation = "sdpa"
    else:
        model.model._output_capturing_hooks_installed = True
    flags = (model.training, model.config._attn_implementation)
    with pytest.raises(ValueError):
        structural_adapter.hook_targets(CapturePlan(), 5)
    assert (model.training, model.config._attn_implementation) == flags


def test_registry_rejects_unavailable_family():
    model = SimpleNamespace(config=SimpleNamespace(model_type="unvalidated_family"))
    with pytest.raises(ValueError, match="No scientific trace adapter"):
        create_adapter(model, None, "0" * 40, "cpu")


def test_last_position_output_retention_does_not_claim_full_sequence_logits(structural_adapter):
    specs = {
        spec.component_id: spec
        for spec in structural_adapter.component_specs(CapturePlan(profile="overview"), 5)
    }
    assert specs["output.logits"].logical_shape == (1, 1, 64)
    assert specs["output.lm_head_input"].logical_shape == (1, 1, 16)
    assert specs["output.normalized"].logical_shape == (1, 5, 16)
    assert "absolute input position 4" in specs["output.logits"].method
    assert specs["output.residual_final"].sources == ("layer[1].residual.after_mlp",)
