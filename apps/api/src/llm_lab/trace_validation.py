"""Run real, bounded Stage 2 captures and report metadata without tensor arrays."""

import argparse
import json
import platform
import resource
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import transformers

from llm_lab.config import HF_MODEL_ID, MODEL_ID, Settings
from llm_lab.models.manager import InferenceResult, ModelService
from llm_lab.schemas.trace import TraceResult
from llm_lab.tracing.capture_plan import CapturePlan

PROMPTS = (
    "The capital of Germany is",
    "After the rain, Berlin feels quiet.",
)
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-4
PROBABILITY_ATOL = 1e-6
PROBABILITY_RTOL = 1e-4


def _plain(value: Any) -> Any:
    """Convert bounded metadata only; scientific tensor serialization is excluded."""
    if isinstance(value, torch.Tensor):
        raise TypeError("Stage 2 validation artifacts must not serialize tensor arrays")
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _plain(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _descriptor_metadata(descriptor) -> dict:
    """Compact validation-only coverage metadata, independent of any future wire format."""
    record = {
        item.name: _plain(getattr(descriptor, item.name))
        for item in fields(descriptor)
        if item.name != "spec"
    }
    spec = descriptor.spec
    metadata = {
        item.name: _plain(getattr(spec, item.name))
        for item in fields(spec)
        if item.name != "coordinates"
    }
    metadata["coordinates"] = (
        None
        if spec.coordinates is None
        else [
            {"axis": axis, "selection": "all", "size": size}
            if indices == tuple(range(size))
            else {"axis": axis, "selection": "indices", "indices": list(indices)}
            for axis, size, indices in zip(
                spec.axes, spec.logical_shape, spec.coordinates, strict=True
            )
        ]
    )
    record["spec"] = metadata
    return record


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    delta = actual.double() - expected.double()
    absolute = delta.abs()
    reference_norm = torch.linalg.vector_norm(expected.double()).item()
    delta_norm = torch.linalg.vector_norm(delta).item()
    return {
        "shape_match": True,
        "max_absolute_error": absolute.max().item(),
        "mean_absolute_error": absolute.mean().item(),
        "relative_l2_error": delta_norm / reference_norm if reference_norm else None,
        "relative_error_definition": (
            "L2(actual-reference)/L2(reference); null when denominator is zero"
        ),
        "bitwise_equal": torch.equal(actual, expected),
    }


def _hook_count(adapter) -> int:
    return sum(
        len(module._forward_pre_hooks) + len(module._forward_hooks)
        for module in adapter.model.modules()
    )


def _memory_snapshot(device: str) -> dict:
    # macOS reports bytes; Linux reports KiB. This is process-lifetime high water,
    # including resident model weights, not an isolated trace peak allocation.
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "process_lifetime_peak_rss_bytes": int(
            maximum if sys.platform == "darwin" else maximum * 1024
        ),
        "rss_scope": "process lifetime, includes model load and all preceding captures",
    }
    if device == "mps" and torch.backends.mps.is_available():
        result["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return result


def _plans(architecture, sequence_length: int) -> tuple[tuple[str, CapturePlan], ...]:
    middle = min(12, architecture.layers - 1)
    detailed = tuple(sorted({0, middle, architecture.layers - 1}))
    return (
        ("summary", CapturePlan()),
        ("deep_selected_layers", CapturePlan(profile="deep", layers=detailed)),
        ("layer_12", CapturePlan(profile="layer", layers=(middle,))),
        (
            "attention_selected_token_heads",
            CapturePlan(
                profile="attention",
                layers=(middle,),
                token_indices=(sequence_length - 1,),
                query_heads=tuple(sorted({0, architecture.query_heads - 1})),
            ),
        ),
        ("mlp_statistics_only", CapturePlan(profile="mlp", layers=(middle,), raw_tensors=False)),
        ("overview", CapturePlan(profile="overview")),
    )


def _tensor_properties(trace: TraceResult) -> dict:
    retained = [
        trace.tensor(name, copy=False)
        for name, descriptor in trace.descriptors.items()
        if descriptor.tensor_id is not None and descriptor.availability == "captured"
    ]
    return {
        "retained_tensor_components": len(retained),
        "unique_tensor_ids": len(
            {
                descriptor.tensor_id
                for descriptor in trace.descriptors.values()
                if descriptor.tensor_id
            }
        ),
        "all_retained_tensors_on_cpu": all(tensor.device.type == "cpu" for tensor in retained),
        "all_retained_tensors_detached": all(
            not tensor.requires_grad and tensor.grad_fn is None for tensor in retained
        ),
        "retained_component_counts_by_kind": {
            kind: sum(
                descriptor.spec.value_kind.value == kind and descriptor.tensor_id is not None
                for descriptor in trace.descriptors.values()
            )
            for kind in ("raw", "derived", "reconstructed", "summary")
        },
    }


def _failed_validation_checks(value: Any, prefix: str = "") -> list[str]:
    """Honor every nested scientific gate, rather than only output equality."""
    failed = []
    if isinstance(value, Mapping):
        if value.get("passed") is False:
            failed.append(prefix or "scientific_validation")
        for key, item in value.items():
            if isinstance(item, (Mapping, list, tuple)):
                failed.extend(_failed_validation_checks(item, f"{prefix}.{key}".strip(".")))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            failed.extend(_failed_validation_checks(item, f"{prefix}[{index}]"))
    return failed


def _trace_record(
    trace: TraceResult, baseline: InferenceResult, hooks_before: int, hooks_after: int
) -> dict:
    logits = trace.tensor(trace.output.logits, copy=False)[0, -1].float()
    probabilities = trace.tensor(trace.output.probabilities, copy=False)[0, -1].float()
    logits_error = _error(logits, baseline.logits)
    probability_error = _error(probabilities, baseline.probabilities)
    baseline_ids = [candidate.token_id for candidate in baseline.response.prediction.top_k]
    actual_ids = [candidate.token_id for candidate in trace.output.candidates]
    properties = _tensor_properties(trace)
    failed_scientific_checks = _failed_validation_checks(trace.metadata.validation)
    checks = {
        "exact_token_ids": trace.metadata.token_ids
        == tuple(token.token_id for token in baseline.response.input.tokens),
        "exact_raw_prompt": trace.metadata.prompt == baseline.response.input.text,
        "logits_shape": tuple(trace.descriptors[trace.output.logits].storage_shape)
        == tuple(baseline.response.prediction.logits_shape),
        "logits_within_tolerance": torch.allclose(
            logits, baseline.logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL
        ),
        "probabilities_within_tolerance": torch.allclose(
            probabilities,
            baseline.probabilities,
            atol=PROBABILITY_ATOL,
            rtol=PROBABILITY_RTOL,
        ),
        "top_10_prediction_ids": actual_ids == baseline_ids,
        "full_vocabulary_probability_normalization": abs(probabilities.double().sum().item() - 1)
        < 1e-6,
        "hooks_cleaned_up": hooks_after == hooks_before,
        "retained_bytes_within_estimate": trace.metadata.actual_retained_bytes
        <= trace.metadata.estimated_bytes,
        "retained_bytes_within_requested_budget": trace.metadata.actual_retained_bytes
        <= trace.metadata.capture_plan.max_bytes,
        "cpu_snapshots": properties["all_retained_tensors_on_cpu"],
        "no_autograd_graphs": properties["all_retained_tensors_detached"],
        "all_recorded_scientific_gates": not failed_scientific_checks,
    }
    # The artifacts retain descriptor/statistics evidence for the representative
    # layer and global boundaries. They never retain numerical tensor arrays.
    manifest = {
        name: _descriptor_metadata(descriptor)
        for name, descriptor in trace.descriptors.items()
        if name.startswith(("layer[12].", "embedding.", "output.", "position."))
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "trace_id": trace.metadata.trace_id,
        "capture_plan": _plain(trace.metadata.capture_plan),
        "sequence_length": trace.metadata.sequence_length,
        "component_descriptor_count": len(trace.descriptors),
        **properties,
        "estimated_retained_bytes": trace.metadata.estimated_bytes,
        "actual_retained_bytes": trace.metadata.actual_retained_bytes,
        "transient_estimated_bytes": trace.metadata.transient_estimated_bytes,
        "duration_ms": trace.metadata.duration_ms,
        "forward_duration_ms": trace.metadata.forward_duration_ms,
        "baseline_forward_duration_ms": baseline.response.runtime.forward_duration_ms,
        "baseline_inference_duration_ms": baseline.response.runtime.inference_duration_ms,
        "forward_overhead_ms_single_measurement": (
            trace.metadata.forward_duration_ms - baseline.response.runtime.forward_duration_ms
        ),
        "total_overhead_ms_single_measurement": (
            trace.metadata.duration_ms - baseline.response.runtime.inference_duration_ms
        ),
        "hooks_before": hooks_before,
        "hooks_after": hooks_after,
        "checks": checks,
        "baseline_logit_comparison": logits_error,
        "baseline_probability_comparison": probability_error,
        "top_predictions": _plain(trace.output.candidates),
        "omitted_probability_mass": trace.output.omitted_probability_mass,
        "prediction_position": trace.output.prediction_position,
        "next_position": trace.output.next_position,
        "scientific_validation": _plain(trace.metadata.validation),
        "failed_scientific_checks": failed_scientific_checks,
        "warnings": _plain(trace.metadata.warnings),
        "runtime": _plain(trace.metadata.runtime),
        "components": manifest,
    }


def _run(service: ModelService) -> dict:
    # Imported here so rendering an existing metadata artifact does not require
    # the execution engine or cause a model load.
    from llm_lab.tracing.engine import estimate_capture

    ready = service.status()
    adapter = service.adapter
    architecture = adapter.architecture
    hooks_initial = _hook_count(adapter)
    report = {
        "status": "PASS",
        "recorded_at": datetime.now(UTC).isoformat(),
        "scope": "Stage 2 real run-scoped model instrumentation; no trace persistence or HTTP",
        "coordinate_encoding": (
            "Validation metadata only: null means full logical coverage; selected components "
            "record each axis as selection=all with size, or selection=indices "
            "with absolute indices."
        ),
        "model": {
            "model_id": MODEL_ID,
            "hf_model_id": HF_MODEL_ID,
            "revision": ready.resolved_revision,
            "tokenizer_revision": ready.resolved_revision,
            "device": ready.device,
            "dtype": ready.dtype,
            "load_duration_ms": ready.load_duration_ms,
            "architecture": _plain(architecture),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
        },
        "numerical_tolerance": {
            "logit_absolute": LOGIT_ATOL,
            "logit_relative": LOGIT_RTOL,
            "probability_absolute": PROBABILITY_ATOL,
            "probability_relative": PROBABILITY_RTOL,
            "criterion": "abs(actual-reference) <= absolute + relative*abs(reference)",
            "comparison_scope": (
                "same resident weights, same device/dtype/backend and exact raw input"
            ),
        },
        "hooks_initial": hooks_initial,
        "memory_after_model_load": _memory_snapshot(adapter.device),
        "prompt_runs": [],
        "automated_test_results": (
            "Run pytest separately; this command executes real validation gates."
        ),
    }
    for prompt in PROMPTS:
        before = service.infer_for_validation(prompt, 10).result(timeout=600)
        direct = service.reference_baseline(prompt).result(timeout=600)
        sequence_length = before.response.input.token_count
        direct_check = {
            "logits": _error(before.logits, direct.logits),
            "probabilities": _error(before.probabilities, direct.probabilities),
            "exact_token_ids": direct.token_ids
            == [token.token_id for token in before.response.input.tokens],
            "logits_within_tolerance": torch.allclose(
                before.logits, direct.logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL
            ),
            "probabilities_within_tolerance": torch.allclose(
                before.probabilities,
                direct.probabilities,
                atol=PROBABILITY_ATOL,
                rtol=PROBABILITY_RTOL,
            ),
            "top_10_ids_equal": before.logits.topk(10).indices.equal(
                direct.logits.topk(10).indices
            ),
            "reference_method": (
                "independent direct HF all-position forward; final position selected afterward"
            ),
        }
        prompt_record = {
            "prompt": prompt,
            "input_mode": "raw",
            "add_special_tokens": False,
            "token_ids": [token.token_id for token in before.response.input.tokens],
            "tokens": [token.model_dump(mode="json") for token in before.response.input.tokens],
            "sequence_length": sequence_length,
            "untraced_logits_shape": before.response.prediction.logits_shape,
            "untraced_forward_duration_ms": before.response.runtime.forward_duration_ms,
            "untraced_inference_duration_ms": before.response.runtime.inference_duration_ms,
            "stage1_vs_independent_direct_hf": direct_check,
            "untraced_top_predictions": [
                candidate.model_dump(mode="json") for candidate in before.response.prediction.top_k
            ],
            "traces": {},
        }
        if not all(
            direct_check[key]
            for key in (
                "exact_token_ids",
                "logits_within_tolerance",
                "probabilities_within_tolerance",
                "top_10_ids_equal",
            )
        ):
            report["status"] = "FAIL"
        for name, plan in _plans(architecture, sequence_length):
            hooks_before = _hook_count(adapter)
            memory_before = _memory_snapshot(adapter.device)
            trace = service.trace(prompt, plan, top_k=10).result(timeout=600)
            hooks_after = _hook_count(adapter)
            record = _trace_record(trace, before, hooks_before, hooks_after)
            record["memory_before"] = memory_before
            record["memory_after"] = _memory_snapshot(adapter.device)
            prompt_record["traces"][name] = record
            if record["status"] != "PASS":
                report["status"] = "FAIL"
            del trace
        after = service.infer_for_validation(prompt, 10).result(timeout=600)
        repeated = _error(after.logits, before.logits)
        repeated["forward_duration_ms"] = after.response.runtime.forward_duration_ms
        repeated["inference_duration_ms"] = after.response.runtime.inference_duration_ms
        repeated["token_ids_equal"] = [token.token_id for token in after.response.input.tokens] == [
            token.token_id for token in before.response.input.tokens
        ]
        repeated["top_10_ids_equal"] = [
            candidate.token_id for candidate in after.response.prediction.top_k
        ] == [candidate.token_id for candidate in before.response.prediction.top_k]
        repeated["within_tolerance"] = torch.allclose(
            after.logits, before.logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL
        )
        prompt_record["untraced_before_vs_after"] = repeated
        all_layers_plan = CapturePlan(profile="deep", layers=tuple(range(architecture.layers)))
        prompt_record["all_layer_deep_estimate"] = _plain(
            estimate_capture(adapter, all_layers_plan, sequence_length)
        )
        if not all(
            repeated[key] for key in ("token_ids_equal", "top_10_ids_equal", "within_tolerance")
        ):
            report["status"] = "FAIL"
        report["prompt_runs"].append(prompt_record)
    report["hooks_final"] = _hook_count(adapter)
    report["hooks_equal_initial"] = report["hooks_final"] == hooks_initial
    report["memory_after_all_captures"] = _memory_snapshot(adapter.device)
    if not report["hooks_equal_initial"]:
        report["status"] = "FAIL"
    return report


def _table(rows: list[tuple], columns: tuple[str, ...]) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        + ["| " + " | ".join(cell(item) for item in row) + " |" for row in rows]
    )


def render_report(report: Mapping[str, Any]) -> str:
    """Render a reviewable report from measured, bounded metadata only."""
    model = report.get("model", {})
    architecture = model.get("architecture", {})
    lines = [
        "# Stage 2 validation report",
        "",
        f"**Status: {report.get('status', 'FAIL')}.** "
        f"Recorded {report.get('recorded_at', 'unavailable')}.",
        "",
        str(report.get("scope", "Real model validation")),
        "",
        "## Execution identity",
        "",
        _table(
            [
                (key, model.get(key, "unavailable"))
                for key in (
                    "model_id",
                    "hf_model_id",
                    "revision",
                    "tokenizer_revision",
                    "device",
                    "dtype",
                    "load_duration_ms",
                )
            ],
            ("Property", "Measured value"),
        ),
        "",
        _table(list(report.get("software", {}).items()), ("Software", "Version / environment")),
        "",
        "## Installed architecture and adapter",
        "",
        _table(list(architecture.items()), ("Property", "Actual config / implementation value")),
        "",
        "`ModelService` serializes loading, inference and tracing on its existing executor. "
        "`TraceEngine` validates the plan and budget, then a run-scoped `TraceSession` centrally "
        "registers module observers. `TraceCollector` receives semantic observations through "
        "`BaseModelAdapter`; Qwen paths remain inside `Qwen3Adapter`. Observers return no "
        "replacement outputs. Hooks are removed in `finally`, including failed inference. "
        "Final tensors are detached owned CPU snapshots in an immutable internal `TraceResult`.",
        "",
        "`BaseModelAdapter` declares architecture, hook targets, semantic components, unchanged "
        "tokenization/forward, trace eligibility and native reconstruction helpers. "
        "`Qwen3Adapter` checks actual module widths, per-head RMSNorm, compact GQA, RoPE, "
        "SiLU gated MLP and embedding/head tying against the installed model/config. "
        "It preserves the existing eager backend, evaluation mode, device and dtype.",
        "",
        "## Capture policy and provenance",
        "",
        "`CapturePlan` supports summary, overview, selected layer, attention, MLP and deep "
        "profiles, explicit layer/query-head/token bounds, optional raw retention, statistics "
        "and a 256 MiB default byte budget. Summary retains global final outputs and scalar "
        "internal statistics; selected queries preserve complete mapped key contexts. "
        "Oversized requests fail before inference rather than truncating scientific tensors.",
        "",
        "RAW values are actual module-boundary observations, including projections, "
        "normalizations, residual checkpoints, eager attention weights, MLP activation/product "
        "and final outputs when supported by this installed release. DERIVED output "
        "probabilities use the trusted Stage 1 full-vocabulary normalization. Unscaled, scaled "
        "and masked scores are DERIVED. RECONSTRUCTED RoPE Q/K, softmax weights and head "
        "mixing preserve named sources and are checked against observed eager weights/head output. "
        "SUMMARY "
        "statistics record their finite population, masked/nonfinite counts, accumulation "
        "dtype and population standard-deviation definition.",
        "",
        report.get("coordinate_encoding", "Full coordinate coverage is recorded in descriptors."),
        "",
        "## Numerical acceptance criterion",
        "",
        "```json",
        json.dumps(_plain(report.get("numerical_tolerance", {})), indent=2),
        "```",
    ]
    for index, run in enumerate(report.get("prompt_runs", []), 1):
        lines.extend(
            [
                "",
                f"## Prompt {index}: measured traces",
                "",
                f"Exact raw prompt: `{run['prompt']}`. "
                f"Sequence length: **{run['sequence_length']}**.",
                "",
                "Untraced Stage 1 forward: "
                f"`{run.get('untraced_forward_duration_ms', 'unavailable')}` ms; "
                "total inference: "
                f"`{run.get('untraced_inference_duration_ms', 'unavailable')}` ms. "
                "Capture overhead differences are recorded per profile. These are single runs, "
                "with allocator/cache/scheduling effects, not a controlled latency benchmark.",
                "",
                _table(
                    [
                        (
                            token["token_index"],
                            token["token_id"],
                            token["tokenizer_piece"],
                            json.dumps(token["decoded_token"], ensure_ascii=False),
                        )
                        for token in run.get("tokens", [])
                    ],
                    ("Index", "ID", "Native piece", "Decoded token"),
                ),
                "",
                _table(
                    [
                        (
                            name,
                            trace["status"],
                            trace["retained_tensor_components"],
                            trace["actual_retained_bytes"],
                            trace["estimated_retained_bytes"],
                            trace["transient_estimated_bytes"],
                            round(trace["duration_ms"], 3),
                            trace["baseline_logit_comparison"].get(
                                "max_absolute_error", "shape mismatch"
                            ),
                            trace["baseline_logit_comparison"].get(
                                "mean_absolute_error", "shape mismatch"
                            ),
                        )
                        for name, trace in run["traces"].items()
                    ],
                    (
                        "Capture",
                        "Gate",
                        "Tensor components",
                        "Retained bytes",
                        "Estimated retained",
                        "Transient estimate",
                        "Duration ms",
                        "Max logit error",
                        "Mean logit error",
                    ),
                ),
                "",
                "All-layer deep metadata-only estimate:",
                "",
                "```json",
                json.dumps(_plain(run.get("all_layer_deep_estimate", {})), indent=2),
                "```",
                "",
                "Untraced before/after neutrality:",
                "",
                "```json",
                json.dumps(_plain(run.get("untraced_before_vs_after", {})), indent=2),
                "```",
                "",
                "Stage 1 baseline versus independent direct Hugging Face forward:",
                "",
                "```json",
                json.dumps(_plain(run.get("stage1_vs_independent_direct_hf", {})), indent=2),
                "```",
            ]
        )
        detail = run["traces"].get("layer_12", {})
        components = detail.get("components", {})
        if components:
            selected = [
                (name, item)
                for name, item in components.items()
                if name.startswith(("embedding.", "output.", "layer[12]."))
            ]
            lines.extend(
                [
                    "",
                    "### Layer 12 semantic shapes and provenance",
                    "",
                    _table(
                        [
                            (
                                name,
                                descriptor["spec"]["logical_shape"],
                                descriptor["storage_shape"],
                                ", ".join(descriptor["spec"]["axes"]),
                                descriptor["spec"]["value_kind"],
                                descriptor["availability"],
                            )
                            for name, descriptor in selected
                        ],
                        (
                            "Component",
                            "Logical shape",
                            "Retained shape",
                            "Semantic axes",
                            "Kind",
                            "Availability",
                        ),
                    ),
                    "",
                    "Residual, layer-output, gated-MLP and attention reconstruction validation:",
                    "",
                    _table(
                        [
                            (
                                name,
                                check.get("passed", check.get("status", "unavailable")),
                                check.get("max_absolute_error", "—"),
                                check.get("mean_absolute_error", "—"),
                                check.get("relative_l2_error", "—"),
                                check.get("bitwise_equal", "—"),
                            )
                            for name, check in detail.get("scientific_validation", {}).items()
                            if name.startswith(
                                ("layer[0].", "layer[12].", "layer[27].", "output.", "hook_cleanup")
                            )
                        ],
                        (
                            "Check",
                            "Pass / status",
                            "Max absolute",
                            "Mean absolute",
                            "Relative L2",
                            "Bitwise",
                        ),
                    ),
                    "",
                    "Actual Layer 12 statistics and retained coordinate coverage are in the "
                    "machine-readable artifact; no complete tensor arrays are stored here.",
                    "",
                    _table(
                        [
                            (
                                candidate["rank"],
                                json.dumps(candidate["decoded_token"], ensure_ascii=False),
                                candidate["token_id"],
                                candidate["raw_logit"],
                                candidate["probability"],
                            )
                            for candidate in detail.get("top_predictions", [])
                        ],
                        (
                            "Rank",
                            "Decoded candidate",
                            "ID",
                            "Raw logit",
                            "Full-vocabulary probability",
                        ),
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "## Hook lifecycle, memory and automated evidence",
            "",
            f"Initial hooks: `{report.get('hooks_initial', 'unavailable')}`; final hooks: "
            f"`{report.get('hooks_final', 'unavailable')}`; equality: "
            f"`{report.get('hooks_equal_initial', 'unavailable')}`. Each capture also records "
            "its own before/after count and detached CPU storage checks.",
            "",
            "Process peak RSS includes model loading and all earlier operations. MPS driver "
            "allocation includes allocator cache; it is not a measure of retained trace tensors. "
            "Trace-retained bytes count unique owned storage, separate from the conservative "
            "transient/workspace estimate.",
            "",
            "```json",
            json.dumps(_plain(report.get("memory_after_all_captures", {})), indent=2),
            "```",
            "",
            "Automated test outcomes:",
            "",
            "```json",
            json.dumps(_plain(report.get("automated_test_results", {})), indent=2),
            "```",
        ]
    )
    if "additional_device_validation" in report:
        lines.extend(
            [
                "",
                "Additional device validation:",
                "",
                "```json",
                json.dumps(_plain(report["additional_device_validation"]), indent=2),
                "```",
            ]
        )
    if "files" in report:
        lines.extend(
            [
                "",
                "## Exact file inventory",
                "",
                "```json",
                json.dumps(_plain(report["files"]), indent=2),
                "```",
            ]
        )
    limitations = report.get(
        "limitations",
        [
            "Validation is for the pinned installed Qwen3 implementation, eager attention, "
            "batch one, raw unpadded input and cache-disabled inference.",
            "Post-RoPE Q/K and attention mathematics are explicitly reconstructed; "
            "they are not relabeled RAW observations.",
            "Attention weights describe value routing, not causal attribution or "
            "a complete explanation of reasoning.",
            "No intermediate vocabulary logits, generation, fused-backend compatibility "
            "or cross-device bitwise equivalence is claimed.",
            "Peak RSS is a process-lifetime measurement; conservative workspace estimates "
            "are not measured instantaneous device peaks.",
        ],
    )
    lines.extend(
        ["", "## Scientific and implementation limits", ""] + [f"- {item}" for item in limitations]
    )
    if "error" in report:
        lines.extend(
            [
                "",
                "Recorded failure:",
                "",
                "```json",
                json.dumps(_plain(report["error"]), indent=2),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Stage 3 boundary",
            "",
            "Stage 3 will add bounded trace persistence, expiration and leases, trace-creation "
            "and retrieval HTTP contracts, component/layer/attention slice endpoints, explicit "
            "serialization and generated client types. It will serve this validated semantic "
            "data without rerunning inference on reads. No Stage 3 API, store, expiration "
            "behavior, serialization or frontend trace consumption has been implemented in "
            "Stage 2. Visualization screens remain later work.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--render-from", type=Path, help="Render existing metadata without model execution"
    )
    args = parser.parse_args()
    if args.render_from:
        if args.report is None:
            parser.error("--render-from requires --report")
        write_report(json.loads(args.render_from.read_text()), args.report)
        print(f"Rendered validation metadata report: {args.report}")
        return
    if args.output is None:
        parser.error("Real validation requires --output")
    service = ModelService(Settings(auto_load=False, device=args.device))
    try:
        service.start_loading()
        service.wait_until_ready()
        report = _run(service)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "recorded_at": datetime.now(UTC).isoformat(),
            "scope": "Stage 2 real validation interrupted before completion",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    finally:
        service.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_plain(report), indent=2, ensure_ascii=False) + "\n")
    if args.report:
        write_report(report, args.report)
    # Human-readable summary omits complete descriptors/statistics and tensors.
    print(f"Stage 2 real validation: {report['status']}")
    if "model" in report:
        model = report["model"]
        print(
            f"{model['hf_model_id']} @ {model['revision']} — {model['device']} / {model['dtype']}"
        )
        for run in report["prompt_runs"]:
            print(f"Prompt: {run['prompt']!r} ({run['sequence_length']} tokens)")
            for name, trace in run["traces"].items():
                error = trace["baseline_logit_comparison"].get("max_absolute_error")
                print(
                    f"  {name}: {trace['status']}; "
                    f"{trace['actual_retained_bytes']} retained bytes; "
                    f"{trace['duration_ms']:.3f} ms; max logit error {error}"
                )
        print(f"Hook cleanup: {report['hooks_initial']} → {report['hooks_final']}")
    else:
        print(json.dumps(report["error"]))
    print(f"Metadata artifact: {args.output}")
    if args.report:
        print(f"Validation report: {args.report}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
