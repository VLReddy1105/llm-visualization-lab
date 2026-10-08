"""Configuration-derived semantic dimensions, GQA and retained coordinates."""

import pytest
import torch
from test_trace_equivalence import CAPITAL_PROMPT
from test_trace_equivalence import capital_trace as capital_trace

from llm_lab.tracing.capture_plan import CapturePlan

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="session")
def selected_attention_trace(real_service):
    return real_service.trace(
        CAPITAL_PROMPT,
        CapturePlan(profile="attention", layers=(12,), token_indices=(3,), query_heads=(3,)),
    ).result(timeout=180)


def test_adapter_architecture_matches_loaded_configuration(real_service):
    architecture = real_service.adapter.architecture
    config = real_service._model.config
    assert architecture.layers == config.num_hidden_layers == 28
    assert architecture.hidden_size == config.hidden_size == 1024
    assert architecture.intermediate_size == config.intermediate_size == 3072
    assert architecture.query_heads == config.num_attention_heads == 16
    assert architecture.kv_heads == config.num_key_value_heads == 8
    assert architecture.head_dim == config.head_dim == 128
    assert architecture.vocabulary_size == config.vocab_size == 151936
    # Qwen3's concatenated head width is twice the residual width here.
    assert architecture.query_heads * architecture.head_dim != architecture.hidden_size


def test_semantic_shapes_cover_real_layer_dataflow(real_service, capital_trace):
    architecture = real_service.adapter.architecture
    length = capital_trace.metadata.sequence_length
    residual = (1, length, architecture.hidden_size)
    queries = (1, architecture.query_heads, length, architecture.head_dim)
    keys_values = (1, architecture.kv_heads, length, architecture.head_dim)
    intermediate = (1, length, architecture.intermediate_size)
    expected = {
        "embedding.output": residual,
        "output.residual_final": residual,
        "output.normalized": residual,
        "output.lm_head_input": (1, 1, architecture.hidden_size),
        "output.logits": (1, 1, architecture.vocabulary_size),
    }
    for layer in (0, 12, 27):
        prefix = f"layer[{layer}]."
        expected.update(
            {
                prefix + "residual.before": residual,
                prefix + "attention.input_normalized": residual,
                prefix + "attention.query.projected": queries,
                prefix + "attention.query.normalized": queries,
                prefix + "attention.query.rotary": queries,
                prefix + "attention.key.projected": keys_values,
                prefix + "attention.key.normalized": keys_values,
                prefix + "attention.key.rotary": keys_values,
                prefix + "attention.value": keys_values,
                prefix + "attention.head_output": queries,
                prefix + "attention.output": residual,
                prefix + "attention.weights": (1, architecture.query_heads, length, length),
                prefix + "residual.after_attention": residual,
                prefix + "mlp.gate": intermediate,
                prefix + "mlp.up": intermediate,
                prefix + "mlp.gated": intermediate,
                prefix + "mlp.output": residual,
                prefix + "residual.after_mlp": residual,
                prefix + "output": residual,
            }
        )
    for component, shape in expected.items():
        descriptor = capital_trace.descriptors[component]
        assert tuple(descriptor.spec.logical_shape) == shape, component
        assert tuple(descriptor.storage_shape) == shape, component
        assert tuple(capital_trace.tensor(component).shape) == shape, component
        assert len(descriptor.spec.axes) == len(shape), component
        assert len(set(descriptor.spec.axes)) == len(shape), component


def test_head_layout_matches_independently_observed_native_module_coordinates(
    real_service, capital_trace
):
    """A correct shape alone cannot detect a token/head transpose mistake."""
    layer = real_service._model.model.layers[12]
    attention = layer.self_attn
    native = {}
    handles = []

    def record(name):
        def observe(_module, _inputs, output):
            value = output[1] if name == "weights" else output
            native[name] = value.detach().cpu().clone()

        return observe

    try:
        for name, module in (
            ("query.projected", attention.q_proj),
            ("key.projected", attention.k_proj),
            ("value", attention.v_proj),
            ("query.normalized", attention.q_norm),
            ("key.normalized", attention.k_norm),
            ("weights", attention),
        ):
            handles.append(module.register_forward_hook(record(name)))
        real_service.reference_baseline(CAPITAL_PROMPT).result(timeout=180)
    finally:
        for handle in handles:
            handle.remove()
    architecture = real_service.adapter.architecture
    for name, original in native.items():
        if name == "weights":
            expected = original
        elif name.endswith("normalized"):
            expected = original.transpose(1, 2)
        else:
            heads = architecture.query_heads if name.startswith("query") else architecture.kv_heads
            expected = original.reshape(1, capital_trace.metadata.sequence_length, heads, -1)
            expected = expected.transpose(1, 2)
        assert torch.equal(capital_trace.tensor(f"layer[12].attention.{name}"), expected), name


def test_gqa_keeps_distinct_compact_head_axes(real_service, capital_trace):
    architecture = real_service.adapter.architecture
    layer = next(layer for layer in capital_trace.layers if layer.layer_index == 12)
    mapping = layer.attention.query_to_kv_head
    assert tuple(mapping) == tuple(
        head // (architecture.query_heads // architecture.kv_heads)
        for head in range(architecture.query_heads)
    )
    assert mapping[3] == 1
    query = capital_trace.descriptors["layer[12].attention.query.projected"]
    key = capital_trace.descriptors["layer[12].attention.key.projected"]
    value = capital_trace.descriptors["layer[12].attention.value"]
    assert query.spec.axes[1] == "query_head"
    assert key.spec.axes[1] == value.spec.axes[1] == "kv_head"
    assert key.storage_shape[1] == value.storage_shape[1] == architecture.kv_heads


def test_actual_head_concatenation_has_verified_layout_and_shared_full_head_storage(
    real_service, capital_trace, selected_attention_trace
):
    architecture = real_service.adapter.architecture
    component = "layer[12].attention.concatenated"
    heads_component = "layer[12].attention.head_output"
    descriptor = capital_trace.descriptors[component]
    concatenated = capital_trace.tensor(component, copy=False)
    head_output = capital_trace.tensor(heads_component, copy=False)
    shape = (
        1,
        capital_trace.metadata.sequence_length,
        architecture.query_heads * architecture.head_dim,
    )
    assert descriptor.spec.axes == ("batch", "token", "head_feature")
    assert concatenated.shape == descriptor.storage_shape == shape
    expected = head_output.transpose(1, 2).reshape(shape)
    assert torch.equal(concatenated, expected)
    assert descriptor.tensor_id == capital_trace.descriptors[heads_component].tensor_id
    assert concatenated.untyped_storage().data_ptr() == head_output.untyped_storage().data_ptr()
    selected = selected_attention_trace.descriptors[component]
    assert selected.storage_shape == (1, 1, architecture.query_heads * architecture.head_dim)
    assert selected.spec.coordinates[1] == (3,)
    assert selected.spec.coordinates[2] == tuple(range(shape[2]))
    assert torch.equal(selected_attention_trace.tensor(component), concatenated[:, 3:4, :])
    assert selected_attention_trace.tensor(heads_component).shape == (
        1,
        1,
        1,
        architecture.head_dim,
    )


def test_selected_attention_retains_original_coordinates_and_full_key_row(
    capital_trace, selected_attention_trace
):
    selected = selected_attention_trace
    prefix = "layer[12].attention."
    for suffix in ("query.projected", "query.normalized", "query.rotary", "head_output"):
        component = prefix + suffix
        expected = capital_trace.tensor(component)[:, 3:4, 3:4, :]
        assert torch.equal(selected.tensor(component), expected)
        descriptor = selected.descriptors[component]
        assert descriptor.spec.coordinates[1] == (3,)
        assert descriptor.spec.coordinates[2] == (3,)
    for suffix in ("key.projected", "key.normalized", "key.rotary", "value"):
        component = prefix + suffix
        expected = capital_trace.tensor(component)[:, 1:2, :, :]
        assert torch.equal(selected.tensor(component), expected)
        descriptor = selected.descriptors[component]
        assert descriptor.spec.coordinates[1] == (1,)
        assert descriptor.storage_shape[2] == capital_trace.metadata.sequence_length
    for suffix in ("weights", "weights_reconstructed"):
        component = prefix + suffix
        weights = selected.tensor(component)
        assert weights.shape == (1, 1, 1, capital_trace.metadata.sequence_length)
        reference = capital_trace.tensor(component)[:, 3:4, 3:4, :]
        if suffix == "weights":
            assert torch.equal(weights, reference)
        else:
            torch.testing.assert_close(weights, reference, atol=1e-5, rtol=1e-4)
        assert abs(float(weights.sum()) - 1.0) < 1e-5
        assert float(weights[..., 4]) == 0.0
    assert selected.tensor("position.ids").tolist() == [[0, 1, 2, 3, 4]]


def test_q_norm_rope_and_final_norm_remain_distinct_stages(capital_trace):
    for operand in ("query", "key"):
        projected = capital_trace.tensor(f"layer[12].attention.{operand}.projected")
        normalized = capital_trace.tensor(f"layer[12].attention.{operand}.normalized")
        rotary = capital_trace.tensor(f"layer[12].attention.{operand}.rotary")
        assert not torch.equal(projected, normalized)
        assert not torch.equal(normalized[:, :, 1:, :], rotary[:, :, 1:, :])
        assert (
            capital_trace.descriptors[f"layer[12].attention.{operand}.projected"].spec.value_kind
            == "raw"
        )
        assert (
            capital_trace.descriptors[f"layer[12].attention.{operand}.normalized"].spec.value_kind
            == "raw"
        )
        assert (
            capital_trace.descriptors[f"layer[12].attention.{operand}.rotary"].spec.value_kind
            == "reconstructed"
        )
    residual = capital_trace.tensor("output.residual_final")
    normalized = capital_trace.tensor("output.normalized")
    assert not torch.equal(residual, normalized)
    assert torch.equal(normalized[:, -1:, :], capital_trace.tensor("output.lm_head_input"))
