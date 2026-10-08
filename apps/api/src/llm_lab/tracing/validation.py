"""Scalar scientific gates for bounded, retained internal observations."""

from collections.abc import Mapping

import torch

from llm_lab.models.adapters.base import BaseModelAdapter
from llm_lab.tracing.capture_plan import CapturePlan


def compare_tensors(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict:
    """Evaluate native values with float64 audit reductions, without widening the gate."""
    if actual.shape != reference.shape:
        return {
            "passed": False,
            "reason": "shape_mismatch",
            "actual_shape": tuple(actual.shape),
            "reference_shape": tuple(reference.shape),
            "absolute_tolerance": atol,
            "relative_tolerance": rtol,
        }
    with torch.inference_mode():
        actual_audit = actual.detach().cpu().to(torch.float64)
        reference_audit = reference.detach().cpu().to(torch.float64)
        finite = bool(torch.isfinite(actual_audit).all() and torch.isfinite(reference_audit).all())
        if not finite:
            return {
                "passed": False,
                "reason": "nonfinite_operand",
                "absolute_tolerance": atol,
                "relative_tolerance": rtol,
            }
        error = torch.abs(actual_audit - reference_audit)
        passed = bool(torch.all(error <= atol + rtol * reference_audit.abs()))
        reference_norm = torch.linalg.vector_norm(reference_audit)
        return {
            "passed": passed,
            "bitwise_equal": bool(torch.equal(actual.detach().cpu(), reference.detach().cpu())),
            "absolute_tolerance": atol,
            "relative_tolerance": rtol,
            "max_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()),
            "relative_l2_error": float(
                torch.linalg.vector_norm(actual_audit - reference_audit)
                / torch.clamp(reference_norm, min=1e-12)
            ),
            "relative_l2_epsilon": 1e-12,
            "comparison_dtype": "float64 audit of original values",
        }


def validate_detailed_capture(
    adapter: BaseModelAdapter,
    plan: CapturePlan,
    sequence_length: int,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, dict]:
    """Validate actual selected boundaries without installing observers or rerunning a model."""
    plan = plan.resolve(adapter.architecture, sequence_length)
    atol, rtol, row_atol = {
        torch.float32: (1e-5, 1e-4, 1e-5),
        torch.float16: (2e-3, 1e-2, 2e-3),
        torch.bfloat16: (1e-2, 5e-2, 1e-2),
    }.get(adapter.dtype, (1e-5, 1e-4, 1e-5))
    records: dict[str, dict] = {}

    def check(name, actual, reference):
        records[name] = compare_tensors(actual, reference, atol=atol, rtol=rtol)

    def prerequisites(name, *ids):
        missing = tuple(component for component in ids if component not in tensors)
        if missing:
            records[name] = {
                "status": "skipped",
                "reason": "required raw coverage not retained",
                "missing_components": missing,
            }
            return False
        return True

    for layer in plan.detailed_layers:
        prefix = f"layer[{layer}]."
        if plan.captures_attention:
            for stage, source, target in (
                ("attention", prefix + "residual.before", prefix + "attention.input_normalized"),
                (
                    "query",
                    prefix + "attention.query.projected",
                    prefix + "attention.query.normalized",
                ),
                ("key", prefix + "attention.key.projected", prefix + "attention.key.normalized"),
            ):
                name = prefix + (
                    stage + "_norm" if stage in {"query", "key"} else stage + ".rmsnorm"
                )
                if prerequisites(name, source, target):
                    check(name, tensors[target], adapter.native_norm(stage, tensors[source], layer))
            for suffix, raw, reconstructed in (
                ("attention.weights", "attention.weights", "attention.weights_reconstructed"),
                (
                    "attention.head_output",
                    "attention.head_output",
                    "attention.head_output_reconstructed",
                ),
            ):
                name = prefix + suffix
                if prerequisites(name, prefix + raw, prefix + reconstructed):
                    check(name, tensors[prefix + raw], tensors[prefix + reconstructed])
            weights_id = prefix + "attention.weights"
            if prerequisites(prefix + "attention.row_normalization", weights_id):
                weights = tensors[weights_id].detach().cpu()
                row_error = torch.abs(weights.to(torch.float64).sum(-1) - 1.0)
                records[prefix + "attention.row_normalization"] = {
                    "passed": bool(torch.isfinite(weights).all())
                    and bool((weights >= 0).all())
                    and bool((row_error <= row_atol).all()),
                    "max_absolute_error": float(row_error.max()),
                    "absolute_tolerance": row_atol,
                    "scope": "full key rows for selected query heads and absolute query positions",
                }
                assert plan.token_indices is not None
                query_positions = torch.tensor(plan.token_indices).view(1, 1, -1, 1)
                key_positions = torch.arange(sequence_length).view(1, 1, 1, -1)
                future = (key_positions > query_positions).expand_as(weights)
                masked_weights = weights.masked_select(future)
                records[prefix + "attention.causal_mask"] = {
                    "passed": bool((masked_weights == 0).all()),
                    "masked_count": masked_weights.numel(),
                    "max_masked_weight": float(masked_weights.max())
                    if masked_weights.numel()
                    else 0.0,
                    "absolute_tolerance": 0.0,
                }
            mask_name = prefix + "attention.actual_mask"
            if prerequisites(mask_name, "position.attention_mask"):
                mask = tensors["position.attention_mask"]
                assert plan.token_indices is not None
                query_positions = torch.tensor(plan.token_indices).view(1, 1, -1, 1)
                key_positions = torch.arange(sequence_length).view(1, 1, 1, -1)
                future = (key_positions > query_positions).expand_as(mask)
                minimum = torch.finfo(mask.dtype).min
                records[mask_name] = {
                    "passed": bool(torch.all(mask.masked_select(future) == minimum))
                    and bool(torch.all(mask.masked_select(~future) == 0)),
                    "mask_sentinel": float(minimum),
                    "mask_kind": "actual unpadded causal additive mask with finite dtype minimum",
                }
        if plan.captures_mlp:
            norm_name = prefix + "mlp.rmsnorm"
            if prerequisites(
                norm_name, prefix + "residual.after_attention", prefix + "mlp.input_normalized"
            ):
                check(
                    norm_name,
                    tensors[prefix + "mlp.input_normalized"],
                    adapter.native_norm("mlp", tensors[prefix + "residual.after_attention"], layer),
                )
            activation_name = prefix + "mlp.activation"
            if prerequisites(activation_name, prefix + "mlp.gate", prefix + "mlp.activated_gate"):
                check(
                    activation_name,
                    tensors[prefix + "mlp.activated_gate"],
                    adapter.native_mlp_activation(layer, tensors[prefix + "mlp.gate"]),
                )
            gating_name = prefix + "mlp.gated"
            if prerequisites(
                gating_name, prefix + "mlp.activated_gate", prefix + "mlp.up", prefix + "mlp.gated"
            ):
                with torch.inference_mode():
                    product = (
                        tensors[prefix + "mlp.activated_gate"].to(
                            device=adapter.device, dtype=adapter.dtype
                        )
                        * tensors[prefix + "mlp.up"].to(device=adapter.device, dtype=adapter.dtype)
                    ).cpu()
                check(gating_name, tensors[prefix + "mlp.gated"], product)
            projection_name = prefix + "mlp.down_projection"
            if prerequisites(projection_name, prefix + "mlp.gated", prefix + "mlp.output"):
                check(
                    projection_name,
                    tensors[prefix + "mlp.output"],
                    adapter.native_mlp_projection(layer, tensors[prefix + "mlp.gated"]),
                )
    if prerequisites("output.final_rmsnorm", "output.residual_final", "output.normalized"):
        check(
            "output.final_rmsnorm",
            tensors["output.normalized"],
            adapter.native_norm("final", tensors["output.residual_final"]),
        )
    assert plan.token_indices is not None
    if sequence_length - 1 in plan.token_indices:
        if prerequisites("output.lm_head", "output.lm_head_input", "output.logits"):
            check(
                "output.lm_head",
                tensors["output.logits"],
                adapter.native_lm_head(tensors["output.lm_head_input"]),
            )
        if prerequisites("output.lm_head_input", "output.normalized", "output.lm_head_input"):
            offset = plan.token_indices.index(sequence_length - 1)
            check(
                "output.lm_head_input",
                tensors["output.lm_head_input"],
                tensors["output.normalized"][:, offset : offset + 1],
            )
    else:
        records["output.lm_head"] = {
            "status": "skipped",
            "reason": "selected normalized positions exclude the last input position",
        }
        records["output.lm_head_input"] = {
            "status": "skipped",
            "reason": "selected normalized positions exclude the last input position",
        }
    return records
