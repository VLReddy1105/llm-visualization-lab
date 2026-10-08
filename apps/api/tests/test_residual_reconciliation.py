"""Actual checkpoint reconciliation, native MLP projection and attention maths."""

import pytest
import torch
from test_trace_equivalence import capital_trace as capital_trace
from torch.nn import functional as F

pytestmark = pytest.mark.real_model


@pytest.mark.parametrize("layer", (0, 12, 27))
def test_both_residual_additions_and_actual_layer_output(capital_trace, layer):
    prefix = f"layer[{layer}]."
    before = capital_trace.tensor(prefix + "residual.before")
    attention = capital_trace.tensor(prefix + "attention.output")
    after_attention = capital_trace.tensor(prefix + "residual.after_attention")
    mlp = capital_trace.tensor(prefix + "mlp.output")
    after_mlp = capital_trace.tensor(prefix + "residual.after_mlp")
    torch.testing.assert_close(before + attention, after_attention, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(after_attention + mlp, after_mlp, atol=1e-5, rtol=1e-4)
    assert torch.equal(after_mlp, capital_trace.tensor(prefix + "output"))
    if layer == 27:
        assert torch.equal(after_mlp, capital_trace.tensor("output.residual_final"))
    for check in ("residual.attention", "residual.mlp", "layer_output"):
        metrics = capital_trace.metadata.validation[prefix + check]
        assert metrics["passed"]
        assert metrics["max_absolute_error"] >= 0
        assert metrics["mean_absolute_error"] >= 0
        assert metrics["relative_l2_error"] >= 0


@pytest.mark.parametrize("layer", (0, 12, 27))
def test_real_gated_mlp_operands_and_native_down_projection(real_service, capital_trace, layer):
    prefix = f"layer[{layer}].mlp."
    gate = capital_trace.tensor(prefix + "gate")
    up = capital_trace.tensor(prefix + "up")
    activated = capital_trace.tensor(prefix + "activated_gate")
    gated = capital_trace.tensor(prefix + "gated")
    actual_update = capital_trace.tensor(prefix + "output")
    torch.testing.assert_close(activated, F.silu(gate), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(gated, F.silu(gate) * up, atol=1e-5, rtol=1e-4)
    reconstructed = real_service.adapter.native_mlp_projection(layer, gated)
    torch.testing.assert_close(reconstructed, actual_update, atol=1e-5, rtol=1e-4)
    for check in ("gated", "down_projection"):
        assert capital_trace.metadata.validation[prefix + check]["passed"]
    assert capital_trace.descriptors[prefix + "gated"].spec.value_kind == "raw"
    assert capital_trace.descriptors[prefix + "activated_gate"].spec.value_kind == "raw"


@pytest.mark.parametrize("layer", (0, 12, 27))
def test_reconstructed_attention_matches_actual_eager_weights_and_head_output(
    real_service, capital_trace, layer
):
    prefix = f"layer[{layer}].attention."
    weights = capital_trace.tensor(prefix + "weights")
    reconstructed = capital_trace.tensor(prefix + "weights_reconstructed")
    torch.testing.assert_close(reconstructed, weights, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        capital_trace.tensor(prefix + "head_output_reconstructed"),
        capital_trace.tensor(prefix + "head_output"),
        atol=1e-5,
        rtol=1e-4,
    )
    assert capital_trace.descriptors[prefix + "weights"].spec.value_kind == "raw"
    assert (
        capital_trace.descriptors[prefix + "weights_reconstructed"].spec.value_kind
        == "reconstructed"
    )
    assert torch.all(weights >= 0)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-5, rtol=0
    )
    future_keys = torch.triu(torch.ones(weights.shape[-2:], dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights[..., future_keys]) == 0
    # Independent CPU calculation checks RoPE stages, actual GQA mapping,
    # scale, causality and the full-key-row softmax denominator.
    query = capital_trace.tensor(prefix + "query.rotary")
    key = capital_trace.tensor(prefix + "key.rotary")
    layer_trace = next(item for item in capital_trace.layers if item.layer_index == layer)
    mapping = torch.tensor(layer_trace.attention.query_to_kv_head)
    expanded_key = key.index_select(1, mapping)
    scores = torch.matmul(query, expanded_key.transpose(-2, -1))
    scores = scores * (real_service.adapter.architecture.head_dim**-0.5)
    scores = scores.masked_fill(future_keys, torch.finfo(scores.dtype).min)
    independent = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    torch.testing.assert_close(independent, weights, atol=1e-5, rtol=1e-4)
    for check in ("weights", "head_output", "row_normalization", "causal_mask"):
        assert capital_trace.metadata.validation[prefix + check]["passed"]
