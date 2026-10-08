"""One original forward, scoped observations, bounded capture, then scientific checks."""

import math
import platform
import time
import traceback
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from uuid import uuid4

import torch
import transformers

from llm_lab.analysis.attention import reconstruct_attention
from llm_lab.config import HF_MODEL_ID, MODEL_ID
from llm_lab.errors import ServiceError
from llm_lab.schemas.trace import (
    AttentionTrace,
    EmbeddingTrace,
    LayerTrace,
    MLPTrace,
    OutputCandidate,
    OutputTrace,
    ResidualTrace,
    TokenTrace,
    TraceMetadata,
    TraceResult,
)
from llm_lab.tracing.capture_plan import CapturePlan
from llm_lab.tracing.collector import TraceCollector, should_retain, storage_shape
from llm_lab.tracing.instrumentation import TraceSession, hook_counts
from llm_lab.tracing.validation import compare_tensors, validate_detailed_capture


@dataclass(frozen=True, slots=True)
class BudgetEstimate:
    retained_bytes: int
    transient_bytes: int
    peak_bytes: int


def estimate_capture(adapter, plan, sequence_length):
    """Conservative tensor-storage upper bound; no model execution or allocations.

    Model weights/backend workspaces are separate. Allow sixteen extra copies of
    selected operands for native reconstruction, float64 audits and reduction scratch.
    Exact semantic aliases deliberately make this estimate conservative.
    """
    plan = plan.resolve(adapter.architecture, sequence_length)
    retained = globals_size = 0
    layers = {}
    for spec in adapter.component_specs(plan, sequence_length):
        itemsize = (
            8
            if spec.component_id == "position.ids"
            else (4 if spec.component_id == "output.probabilities" else adapter.dtype.itemsize)
        )
        size = math.prod(storage_shape(spec)) * itemsize
        if should_retain(spec, plan):
            retained += size
        if spec.layer_index is None:
            globals_size += size
        else:
            layers[spec.layer_index] = layers.get(spec.layer_index, 0) + size
    detailed = sum(layers[index] for index in plan.detailed_layers)
    base_peak = max(
        (size for index, size in layers.items() if index not in plan.detailed_layers), default=0
    )
    # No full-layer detailed workspace survives for layers outside the plan.
    scratch_factor = max(
        16, 4 * (adapter.architecture.query_heads // adapter.architecture.kv_heads)
    )
    transient = scratch_factor * (globals_size + detailed + base_peak)
    return BudgetEstimate(retained, transient, retained + transient)


def _now():
    return datetime.now(UTC).isoformat()


def _layers(plan, adapter, descriptors):
    architecture = adapter.architecture
    result = []
    for index in range(architecture.layers):
        prefix = f"layer[{index}]."
        residual = ResidualTrace(
            *(
                prefix + suffix
                for suffix in (
                    "residual.before",
                    "attention.output",
                    "residual.after_attention",
                    "mlp.output",
                    "residual.after_mlp",
                )
            )
        )
        attention = mlp = None
        if index in plan.detailed_layers and plan.captures_attention:
            names = {
                "query_projected": "query.projected",
                "key_projected": "key.projected",
                "query_normalized": "query.normalized",
                "key_normalized": "key.normalized",
                "query_rotary": "query.rotary",
                "key_rotary": "key.rotary",
            }
            values = {}
            for field in fields(AttentionTrace):
                component = prefix + "attention." + names.get(field.name, field.name)
                if component in descriptors:
                    values[field.name] = component
            values["mask"] = "position.attention_mask"
            values["query_to_kv_head"] = tuple(
                head // (architecture.query_heads // architecture.kv_heads)
                for head in plan.query_heads
            )
            values["scaling"] = adapter.attention_scaling(index)
            attention = AttentionTrace(index, **values)
        if index in plan.detailed_layers and plan.captures_mlp:
            values = {}
            for field in fields(MLPTrace):
                component = prefix + "mlp." + field.name
                if component in descriptors:
                    values[field.name] = component
            mlp = MLPTrace(index, **values)
        result.append(LayerTrace(index, residual, attention, mlp, prefix + "output"))
    return tuple(result)


class TraceEngine:
    def __init__(self, adapter, settings):
        self.adapter = adapter
        self.settings = settings

    def run(self, text, plan, top_k, compute):
        started, created_at = time.perf_counter(), _now()
        plan = plan or CapturePlan()
        if not isinstance(plan, CapturePlan):
            raise TypeError("plan must be a CapturePlan")
        if not text.strip():
            raise ServiceError("EMPTY_PROMPT", "Enter a non-empty prompt.", 422)
        if len(text) > self.settings.max_prompt_characters:
            raise ServiceError("INPUT_LIMIT_EXCEEDED", "Prompt exceeds the character limit.", 413)
        encoded = self.adapter.tokenize(
            text, add_special_tokens=False, return_tensors="pt", truncation=False
        )
        token_ids = tuple(encoded["input_ids"][0].tolist())
        n = len(token_ids)
        if not n or n > self.settings.max_input_tokens:
            raise ServiceError("INPUT_LIMIT_EXCEEDED", "Prompt exceeds the token limit.", 413)
        del encoded
        plan = plan.resolve(self.adapter.architecture, n)
        estimate = estimate_capture(self.adapter, plan, n)
        budget = min(plan.max_bytes, self.settings.trace_max_bytes)
        if estimate.peak_bytes > budget:
            raise ValueError(
                f"Capture budget exceeded: retained estimate {estimate.retained_bytes} bytes; "
                f"peak extra tensor estimate {estimate.peak_bytes} bytes; limit {budget}. "
                "Select fewer layers, query heads or tokens, or omit raw storage."
            )
        specs = self.adapter.component_specs(plan, n)
        spec_by_id = {spec.component_id: spec for spec in specs}
        validation = {}

        def validate_layer(index, tensors):
            prefix = f"layer[{index}]."
            validation[prefix + "residual.attention"] = compare_tensors(
                tensors[prefix + "residual.before"] + tensors[prefix + "attention.output"],
                tensors[prefix + "residual.after_attention"],
            )
            validation[prefix + "residual.mlp"] = compare_tensors(
                tensors[prefix + "residual.after_attention"] + tensors[prefix + "mlp.output"],
                tensors[prefix + "residual.after_mlp"],
            )
            validation[prefix + "layer_output"] = compare_tensors(
                tensors[prefix + "residual.after_mlp"],
                tensors[prefix + "output"],
            )

        collector = TraceCollector(plan, budget, on_layer=validate_layer)
        before = hook_counts(self.adapter.model)
        result = reconstructed = tensor = trace = None
        try:
            with TraceSession(self.adapter.hook_targets(plan, n), collector):
                result = compute(text, top_k, str(uuid4()))
            after = hook_counts(self.adapter.model)
            if before != after:
                raise RuntimeError("Trace instrumentation did not restore original hook counts")
            if tuple(token.token_id for token in result.response.input.tokens) != token_ids:
                raise RuntimeError("Trace preflight and original forward tokenization disagree")
            for index in plan.detailed_layers if plan.captures_attention else ():
                reconstructed = reconstruct_attention(
                    self.adapter, index, plan, n, collector.workspace
                )
                for component, tensor in reconstructed.items():
                    collector.observe(
                        spec_by_id[component],
                        tensor,
                        selected=True,
                        original_device=self.adapter.device,
                    )
                reconstructed.clear()
            collector.observe(
                spec_by_id["output.probabilities"],
                result.probabilities.reshape(1, 1, -1),
                original_device="cpu",
            )
            validation.update(validate_detailed_capture(self.adapter, plan, n, collector.workspace))
            validation["hook_cleanup"] = {
                "passed": before == after,
                "hooks_before": sum(sum(row) for row in before),
                "hooks_after": sum(sum(row) for row in after),
            }
            collector.missing(
                specs, str(self.adapter.dtype).removeprefix("torch."), self.adapter.device
            )
            failed = [name for name, check in validation.items() if check.get("passed") is False]
            if failed:
                collector.warnings.append("Scientific reconciliation failed: " + ", ".join(failed))
            response = result.response
            metadata = TraceMetadata(
                trace_id=response.request_id,
                created_at=created_at,
                completed_at=_now(),
                model_id=MODEL_ID,
                hf_model_id=HF_MODEL_ID,
                revision=self.adapter.revision,
                tokenizer_revision=response.model.tokenizer_revision,
                device=self.adapter.device,
                dtype=str(self.adapter.dtype).removeprefix("torch."),
                prompt=text,
                token_ids=token_ids,
                pieces=tuple(token.tokenizer_piece for token in response.input.tokens),
                sequence_length=n,
                architecture=self.adapter.architecture,
                capture_plan=plan,
                estimated_bytes=estimate.retained_bytes,
                actual_retained_bytes=collector.actual_retained_bytes,
                transient_estimated_bytes=estimate.transient_bytes,
                duration_ms=(time.perf_counter() - started) * 1000,
                forward_duration_ms=response.runtime.forward_duration_ms,
                warnings=tuple(collector.warnings),
                validation=validation,
                runtime={
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "transformers_version": transformers.__version__,
                    "platform": platform.platform(),
                    "tracing_enabled": True,
                    "attention_backend": self.adapter.architecture.attention_backend,
                    "eval_mode": not self.adapter.model.training,
                    "use_cache": False,
                    "output_hidden_states": False,
                    "output_attentions": False,
                    "logits_to_keep": 1,
                    "forward_count": 1,
                    "sampling": False,
                    "snapshot_device": "cpu",
                    "peak_extra_tensor_estimated_bytes": estimate.peak_bytes,
                    "budget_bytes": budget,
                    "unrequested_detail": "unavailable; omitted by capture profile",
                    "statistics": "float32, finite unmasked retained coordinates, population std",
                },
            )
            output = OutputTrace(
                prediction_position=n - 1,
                next_position=n,
                candidates=tuple(
                    OutputCandidate(**candidate.model_dump())
                    for candidate in response.prediction.top_k
                ),
                omitted_probability_mass=response.prediction.omitted_probability_mass,
            )
            trace = TraceResult(
                metadata=metadata,
                descriptors=collector.descriptors,
                layers=_layers(plan, self.adapter, collector.descriptors),
                embedding=EmbeddingTrace(),
                output=output,
                tokens=tuple(TokenTrace(**token.model_dump()) for token in response.input.tokens),
                _tensors=collector.tensors,
            )
            collector.clear()
            return trace
        except BaseException as exc:
            collector.clear()
            # An executing frame cannot be cleared by clear_frames. Failed
            # Futures keep this traceback, so drop completed forward/derived
            # values here as well as clearing the collector's dictionaries.
            del result, reconstructed, tensor, trace
            # The trusted Stage1 path wraps safe API errors. Internal scientific
            # callers receive the original failure after hooks/CPU work are gone.
            original = exc.__cause__ if isinstance(exc, ServiceError) and exc.__cause__ else exc
            traceback.clear_frames(original.__traceback__)
            if original is not exc:
                raise original.with_traceback(None) from None
            raise
