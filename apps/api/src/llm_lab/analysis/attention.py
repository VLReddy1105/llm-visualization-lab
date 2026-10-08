"""Bounded attention reconstructions; these are never labeled raw observations."""

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from llm_lab.models.adapters.base import BaseModelAdapter
from llm_lab.tracing.capture_plan import CapturePlan


def reconstruct_attention(
    adapter: BaseModelAdapter,
    layer_index: int,
    plan: CapturePlan,
    sequence_length: int,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reconstruct selected rows using every key and compact mapped KV operands.

    Inputs are already selected, owned CPU snapshots, not full model outputs.
    Native-device arithmetic preserves the installed backend's cast/order. All
    results return detached to CPU; temporary expanded operands are not retained.
    The caller submits these against DERIVED/RECONSTRUCTED ComponentSpecs.
    """
    plan = plan.resolve(adapter.architecture, sequence_length)
    if not plan.captures_attention or layer_index not in plan.detailed_layers:
        raise ValueError("Attention reconstruction requires a selected detailed attention layer")
    assert plan.token_indices is not None and plan.query_heads is not None
    prefix = f"layer[{layer_index}].attention."
    required = (
        prefix + "query.normalized",
        prefix + "key.normalized",
        prefix + "value",
        "position.rotary_cos",
        "position.rotary_sin",
        "position.attention_mask",
    )
    missing = tuple(name for name in required if name not in tensors)
    if missing:
        raise ValueError(f"Attention reconstruction lacks retained prerequisites: {missing}")
    for name in required:
        value = tensors[name]
        if value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None:
            raise ValueError("Attention reconstruction requires detached CPU-owned operands")
    query, key, value = (tensors[name] for name in required[:3])
    cos, sin, mask = (tensors[name] for name in required[3:])
    arch = adapter.architecture
    expected_query = (1, len(plan.query_heads), len(plan.token_indices), arch.head_dim)
    groups = arch.query_heads // arch.kv_heads
    mapped_heads = tuple(head // groups for head in plan.query_heads)
    compact_heads = tuple(sorted(set(mapped_heads)))
    expected_kv = (1, len(compact_heads), sequence_length, arch.head_dim)
    if query.shape != expected_query or key.shape != expected_kv or value.shape != expected_kv:
        raise ValueError("Selected Q/K/V operands do not match semantic GQA coverage")
    if cos.shape != (1, sequence_length, arch.head_dim) or sin.shape != cos.shape:
        raise ValueError("Attention reconstruction requires rotary operands for all key positions")
    if mask.shape != (1, 1, len(plan.token_indices), sequence_length):
        raise ValueError("Attention reconstruction requires the actual selected full-key mask")
    if not mask.is_floating_point():
        raise ValueError("The validated eager path requires its actual floating additive mask")
    with torch.inference_mode():
        positions = torch.tensor(plan.token_indices, dtype=torch.long)
        query_rotary = adapter.native_rope(
            query, cos.index_select(1, positions), sin.index_select(1, positions)
        )
        key_rotary = adapter.native_rope(key, cos, sin)
        query_native = query_rotary.to(device=adapter.device, dtype=adapter.dtype)
        key_native = key_rotary.to(device=adapter.device, dtype=adapter.dtype)
        value_native = value.to(device=adapter.device, dtype=adapter.dtype)
        # repeat_kv's ordering maps h -> floor(h/groups). Select only required
        # repeated operands transiently instead of storing expanded K/V arrays.
        mapping = torch.tensor(
            tuple(compact_heads.index(head) for head in mapped_heads),
            dtype=torch.long,
            device=adapter.device,
        )
        expanded_key = key_native.index_select(1, mapping)
        expanded_value = value_native.index_select(1, mapping)
        similarity = torch.matmul(query_native, expanded_key.transpose(2, 3))
        scaled = similarity * adapter.attention_scaling(layer_index)
        masked = scaled + mask.to(device=adapter.device, dtype=adapter.dtype)
        weights = F.softmax(masked, dim=-1, dtype=torch.float32).to(query_native.dtype)
        weights = F.dropout(weights, p=0.0, training=False)
        head_output = torch.matmul(weights, expanded_value)
        return {
            prefix + "query.rotary": query_rotary,
            prefix + "key.rotary": key_rotary,
            prefix + "similarity": similarity.detach().cpu(),
            prefix + "scaled_scores": scaled.detach().cpu(),
            prefix + "masked_scores": masked.detach().cpu(),
            prefix + "weights_reconstructed": weights.detach().cpu(),
            prefix + "head_output_reconstructed": head_output.detach().cpu(),
        }
