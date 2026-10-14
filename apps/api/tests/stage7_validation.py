"""Reproducible Stage 7 real-Qwen audit; references never enter production analyses.

Run with ``python tests/stage7_validation.py --device cpu --output PATH``.
Independent observers retain only three full five-token blocks during the explicit
validation capture. They are removed before direct installed-module references.
No second transformer forward is used for either the references or exploration.
"""

import argparse
import inspect
import json
import platform
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from transformers.models.qwen3 import modeling_qwen3

from llm_lab.config import MODEL_REVISION, Settings
from llm_lab.models.manager import ModelService
from llm_lab.tracing.capture_plan import CapturePlan
from llm_lab.tracing.instrumentation import hook_counts
from llm_lab.tracing.serialization import plain
from llm_lab.tracing.validation import compare_tensors

LAYERS = (0, 12, 27)
TOKENS = (0, 4)
COMPONENTS = {
    "r0": "residual.before",
    "n_attn": "attention.input_normalized",
    "a": "attention.output",
    "r1": "residual.after_attention",
    "n_mlp": "mlp.input_normalized",
    "g": "mlp.gate",
    "u": "mlp.up",
    "s": "mlp.activated_gate",
    "p": "mlp.gated",
    "m": "mlp.output",
    "r2": "residual.after_mlp",
}


def audit_capture(service, prompt):
    """Capture once and compare with separately observed/native module references."""
    adapter = service.adapter
    observations, layer_inputs, attention_inputs, handles = {}, {}, {}, []
    hooks_before = hook_counts(adapter.model)

    def observe(layer, symbol):
        def hook(_module, _args, output):
            observations.setdefault((layer, symbol), output.detach().cpu().clone())

        return hook

    def observe_input(layer, symbol, record=None):
        def hook(_module, args, kwargs):
            values = args[0] if args else kwargs["hidden_states"]
            observations.setdefault((layer, symbol), values.detach().cpu().clone())
            if record is not None and layer not in record:
                record[layer] = (args, dict(kwargs))

        return hook

    try:
        for index in LAYERS:
            layer = adapter.model.model.layers[index]
            handles.append(
                layer.register_forward_pre_hook(
                    observe_input(index, "r0", layer_inputs), with_kwargs=True
                )
            )
            handles.append(
                layer.self_attn.register_forward_pre_hook(
                    observe_input(index, "n_attn", attention_inputs), with_kwargs=True
                )
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    observe_input(index, "r1"), with_kwargs=True
                )
            )
            handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    observe_input(index, "p"), with_kwargs=True
                )
            )
            for module, symbol in (
                (layer.self_attn.o_proj, "a"),
                (layer.post_attention_layernorm, "n_mlp"),
                (layer.mlp.gate_proj, "g"),
                (layer.mlp.up_proj, "u"),
                (layer.mlp.act_fn, "s"),
                (layer.mlp.down_proj, "m"),
                (layer, "r2"),
            ):
                handles.append(module.register_forward_hook(observe(index, symbol)))
        before = service.execution_metrics()["request_forward_passes"]
        trace = service.trace(
            prompt, CapturePlan(profile="mlp", layers=LAYERS, token_indices=TOKENS), 10
        ).result(180)
    finally:
        for handle in reversed(handles):
            handle.remove()

    def references():
        result = {}
        with torch.inference_mode():
            for index in LAYERS:
                layer = adapter.model.model.layers[index]

                def observed(symbol, layer_index=index):
                    return observations[layer_index, symbol].to(adapter.device)

                n_attn = layer.input_layernorm.forward(observed("r0"))
                a_args, a_kwargs = attention_inputs[index]
                a_kwargs = dict(a_kwargs)
                if a_args:
                    a_args = (n_attn, *a_args[1:])
                else:
                    a_kwargs["hidden_states"] = n_attn
                attention = layer.self_attn.forward(*a_args, **a_kwargs)[0]
                r1 = observed("r0") + observed("a")
                n_mlp = layer.post_attention_layernorm.forward(observed("r1"))
                gate = layer.mlp.gate_proj.forward(observed("n_mlp"))
                up = layer.mlp.up_proj.forward(observed("n_mlp"))
                activated = layer.mlp.act_fn.forward(observed("g"))
                product = observed("s") * observed("u")
                down = layer.mlp.down_proj.forward(observed("p"))
                r2 = observed("r1") + observed("m")
                args, kwargs = layer_inputs[index]
                complete = layer.forward(*args, **kwargs)
                result[index] = {
                    "n_attn": n_attn.cpu(),
                    "a": attention.cpu(),
                    "r1": r1.cpu(),
                    "n_mlp": n_mlp.cpu(),
                    "g": gate.cpu(),
                    "u": up.cpu(),
                    "s": activated.cpu(),
                    "p": product.cpu(),
                    "m": down.cpu(),
                    "r2": r2.cpu(),
                    "complete_layer": complete.cpu(),
                }
        return result

    native = service.submit_analysis_job(references).result(180)
    checks, coordinates = {}, []
    for layer in LAYERS:
        for token in TOKENS:
            key = f"layer[{layer}].token[{token}]"
            for symbol, suffix in COMPONENTS.items():
                component = f"layer[{layer}].{suffix}"
                descriptor = trace.descriptors[component]
                assert descriptor.coordinates[1] == TOKENS
                assert descriptor.dtype == descriptor.storage_dtype == "float32"
                assert descriptor.spec.value_kind == "raw"
                actual = trace.tensor(component)[0, TOKENS.index(token)]
                checks[f"{key}.{symbol}.independent_observation"] = compare_tensors(
                    actual, observations[layer, symbol][0, token]
                )
                if symbol in native[layer]:
                    checks[f"{key}.{symbol}.installed_module_reference"] = compare_tensors(
                        actual, native[layer][symbol][0, token]
                    )
                coordinates.append(
                    {
                        "source_trace_id": trace.metadata.trace_id,
                        "layer_index": layer,
                        "token_index": token,
                        "token_id": trace.tokens[token].token_id,
                        "symbol": symbol,
                        "component_id": component,
                        "logical_shape": descriptor.logical_shape,
                        "storage_shape": descriptor.storage_shape,
                        "coordinate_0": float(actual[0]),
                        "dtype": descriptor.dtype,
                        "device": descriptor.original_device,
                        "provenance": "directly captured module boundary",
                    }
                )
            checks[f"{key}.complete_layer.installed_module_reference"] = compare_tensors(
                trace.tensor(f"layer[{layer}].residual.after_mlp")[0, TOKENS.index(token)],
                native[layer]["complete_layer"][0, token],
            )
    hooks_after = hook_counts(adapter.model)
    after = service.execution_metrics()["request_forward_passes"]
    return trace, {
        "prompt": prompt,
        "source_trace_id": trace.metadata.trace_id,
        "tokens": [plain(trace.tokens[t]) for t in TOKENS],
        "capture": plain(trace.metadata.capture_plan),
        "actual_retained_bytes": trace.metadata.actual_retained_bytes,
        "estimated_retained_bytes": trace.metadata.estimated_bytes,
        "transient_estimated_bytes": trace.metadata.transient_estimated_bytes,
        "capture_duration_ms": trace.metadata.duration_ms,
        "request_forward_passes_before": before,
        "request_forward_passes_after": after,
        "explicit_capture_forwards": after - before,
        "independent_reference_method": (
            "Separate original-forward observers, removed before direct full-sequence calls "
            "to installed Qwen RMSNorm, self-attention, gate/up/SiLU/down and decoder block. "
            "References reuse the original causal mask and rotary positions. "
            "No second complete-model forward is executed."
        ),
        "reference_operation_replays": len(LAYERS) * 9,
        "hooks_before": hooks_before,
        "hooks_after": hooks_after,
        "hook_cleanup_passed": hooks_before == hooks_after,
        "comparisons": checks,
        "coordinates": coordinates,
        "passed": all(check["passed"] for check in checks.values())
        and hooks_before == hooks_after
        and after == before + 1,
    }


def audit_analyses(service, trace):
    """Exercise real retained-data jobs, cache reuse and bounded response payloads."""
    from llm_lab.analysis.jobs import AnalysisService
    from llm_lab.schemas.analysis import (
        BlockParameters,
        MLPInternalsRequest,
        ResidualStreamRequest,
    )
    from llm_lab.store.trace_store import TraceStore

    store = TraceStore()
    identity = store.reserve(trace.metadata.actual_retained_bytes)["trace_id"]
    trace = replace(trace, metadata=replace(trace.metadata, trace_id=identity))
    store.commit(identity, trace)
    jobs = AnalysisService(service, store, service.settings)
    before = service.execution_metrics()["request_forward_passes"]
    records = []
    try:
        with (
            patch.object(
                service.adapter, "forward", side_effect=AssertionError("Unexpected model forward")
            ),
            patch.object(
                service.adapter.model, "forward", side_effect=AssertionError("Unexpected forward")
            ),
        ):
            for layer in LAYERS:
                for token in TOKENS:
                    for kind, request_type in (
                        ("residual_stream", ResidualStreamRequest),
                        ("mlp_internals", MLPInternalsRequest),
                    ):
                        request = request_type(
                            kind=kind,
                            parameters=BlockParameters(token_index=token, layer_index=layer),
                        )
                        start = time.perf_counter()
                        first = jobs.create(identity, request)
                        deadline = time.monotonic() + 60
                        while time.monotonic() < deadline:
                            ready = jobs.get(identity, first.analysis_id)
                            if ready.status not in {"queued", "running"}:
                                break
                            time.sleep(0.005)
                        assert ready.status == "ready", ready.error
                        hits = [jobs.create(identity, request) for _ in range(5)]
                        records.append(
                            {
                                "analysis_id": first.analysis_id,
                                "source_trace_id": identity,
                                "kind": kind,
                                "layer_index": layer,
                                "token_index": token,
                                "elapsed_ms": (time.perf_counter() - start) * 1000,
                                "retained_bytes": ready.retained_bytes,
                                "response_bytes": len(ready.model_dump_json().encode()),
                                "verified_cache_hits": sum(
                                    h.cache_hit and h.analysis_id == first.analysis_id for h in hits
                                ),
                                "result_source_matches": ready.result.source.trace_id == identity,
                                "result": ready.result.model_dump(),
                            }
                        )
        after = service.execution_metrics()["request_forward_passes"]
        return {
            "source_trace_id": identity,
            "request_forward_passes_before": before,
            "request_forward_passes_after": after,
            "additional_exploration_forwards": after - before,
            "active_leases_after": store.status()["active_leases"],
            "analysis_cache_bytes": jobs._accounted(),
            "source_retained_bytes": store.status()["retained_bytes"],
            "jobs": records,
            "passed": before == after
            and all(r["verified_cache_hits"] == 5 and r["result_source_matches"] for r in records),
        }
    finally:
        jobs.close()
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "FAIL",
        "recorded_at": datetime.now(UTC).isoformat(),
        "device_requested": args.device,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "tolerance_policy": {
            "atol": 1e-5,
            "rtol": 1e-4,
            "gate": "componentwise abs(actual-reference) <= atol + rtol*abs(reference)",
            "audit_dtype": "float64",
        },
        "installed_source": {
            name: inspect.getsource(getattr(modeling_qwen3, name).forward)
            for name in ("Qwen3DecoderLayer", "Qwen3MLP", "Qwen3RMSNorm")
        },
        "prompts": [],
    }
    service = ModelService(Settings(auto_load=False, device=args.device))
    try:
        service.start_loading()
        service.wait_until_ready()
        assert service.adapter.revision == MODEL_REVISION
        report["model"] = service.status().model_dump()
        for country in ("Germany", "France"):
            trace, capture = audit_capture(service, f"The capital of {country} is")
            capture["analyses"] = audit_analyses(service, trace)
            report["prompts"].append(capture)
        report["status"] = (
            "PASS"
            if all(p["passed"] and p["analyses"]["passed"] for p in report["prompts"])
            else "FAIL"
        )
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        service.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Stage 7 {args.device}: {report['status']} — {args.output}")
    if report["status"] != "PASS":
        print(report.get("error", "One or more numerical checks failed"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
