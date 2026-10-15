"""Session metadata is bounded separately; original tensors use the existing TraceStore.

One session occupies the resident executor for its bounded loop. Reads, cancellation and
expiry use an independent lock. A native forward finishes safely; cancelled/expired work
is discarded before step publication. No saved-data route submits inference.
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from uuid import uuid4

import torch

from llm_lab.analysis.distributions import PROBABILITY_POLICY, RANK_POLICY, distribution
from llm_lab.analysis.sampling import process_logits
from llm_lab.config import HF_MODEL_ID, MODEL_ID
from llm_lab.errors import ServiceError
from llm_lab.schemas.generation import (
    GenerationCandidate,
    GenerationRequest,
    GenerationSession,
    GenerationStep,
)
from llm_lab.store.trace_store import utc_timestamp
from llm_lab.tracing.capture_plan import CapturePlan
from llm_lab.tracing.engine import TraceEngine, estimate_capture

logger = logging.getLogger(__name__)
TERMINAL = {"completed", "stopped", "cancelled", "failed"}


@dataclass
class Entry:
    session: GenerationSession
    request: GenerationRequest
    fingerprint: str
    deadline: float
    created: float
    reserved: int
    working: bool = True
    removed: bool = False
    future: object = None


class GenerationService:
    def __init__(self, model, store, settings, clock=time.monotonic, wall_clock=time.time):
        self.model, self.store, self.settings = model, store, settings
        self._clock, self._wall = clock, wall_clock
        self._lock = threading.RLock()
        self._entries = OrderedDict()
        self._keys = {}
        self._tombstones = OrderedDict()
        self._closed = False
        self._evictions = self._expirations = 0

    def _delete_sources(self, entry):
        for step in entry.session.steps:
            if step.source_trace_id:
                try:
                    self.store.delete(step.source_trace_id)
                except ServiceError:
                    pass

    def _remove(self, identity, reason):
        entry = self._entries[identity]
        entry.removed = True
        self._delete_sources(entry)
        self._tombstones[identity] = reason
        while len(self._tombstones) > 128:
            self._tombstones.popitem(last=False)
        if not entry.working:
            self._entries.pop(identity)
            self._keys.pop(entry.request.request_key, None)

    def _sweep(self):
        for identity, entry in list(self._entries.items()):
            if not entry.removed and self._clock() >= entry.deadline:
                self._expirations += 1
                self._remove(identity, "expired")

    def sweep(self):
        with self._lock:
            self._sweep()

    def _lookup(self, identity):
        self._sweep()
        entry = self._entries.get(identity)
        if not entry or entry.removed:
            reason = self._tombstones.get(identity)
            raise ServiceError(
                "GENERATION_" + (reason.upper() if reason else "NOT_FOUND"),
                "Generation data is unavailable.",
                410 if reason else 404,
            )
        self._entries.move_to_end(identity)
        return entry

    def create(self, body):
        self.model._require_ready()
        digest = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        with self._lock:
            self._sweep()
            if self._closed:
                raise ServiceError("EXECUTION_UNAVAILABLE", "Generation service is closed.", 503)
            existing = self._keys.get(body.request_key)
            if existing:
                entry = self._lookup(existing)
                if entry.fingerprint != digest:
                    raise ServiceError(
                        "DUPLICATE_REQUEST", "Request key has different settings.", 409
                    )
                return self._snapshot(entry)
            if body.max_new_tokens > self.settings.generation_max_new_tokens:
                raise ServiceError("GENERATION_TOKEN_LIMIT", "New-token budget exceeds limit.", 422)
            if body.retain_traces > self.settings.generation_max_traces:
                raise ServiceError("GENERATION_TRACE_LIMIT", "Trace retention exceeds limit.", 422)
            if (
                sum(e.working for e in self._entries.values())
                >= self.settings.generation_max_inflight
            ):
                raise ServiceError("GENERATION_QUEUE_FULL", "Too many active generations.", 429)
            # Tokenizer work is bounded and weight-free. No decode/retokenize at later steps.
            adapter = self.model.adapter
            if not body.input.text.strip():
                raise ServiceError("INVALID_INPUT", "Prompt must contain non-whitespace text.", 422)
            if len(body.input.text) > self.settings.max_prompt_characters:
                raise ServiceError("INPUT_LIMIT_EXCEEDED", "Prompt exceeds character limit.", 413)
            encoded_ids = adapter.tokenize(
                body.input.text, add_special_tokens=False, truncation=False
            )["input_ids"]
            if isinstance(encoded_ids, torch.Tensor):
                ids = encoded_ids[0].tolist()
            elif encoded_ids and isinstance(encoded_ids[0], (list, tuple)):
                ids = list(encoded_ids[0])
            else:
                ids = list(encoded_ids)
            prefix_limit = min(
                self.settings.generation_max_prefix_tokens,
                self.settings.max_input_tokens,
                adapter.model.config.max_position_embeddings,
            )
            if not ids or len(ids) > prefix_limit:
                raise ServiceError(
                    "INPUT_LIMIT_EXCEEDED",
                    "Prompt exceeds prefix limit.",
                    413,
                    details={"token_count": len(ids), "max_prefix_tokens": prefix_limit},
                )
            if body.sampling.top_k > adapter.architecture.vocabulary_size:
                raise ServiceError("INVALID_TOP_K", "Top-k exceeds vocabulary.", 422)
            reserve = self.settings.generation_session_max_bytes * 4
            while (
                len(self._entries) >= self.settings.generation_max_sessions
                or sum(e.reserved for e in self._entries.values()) + reserve
                > self.settings.generation_state_max_bytes
            ):
                victim = next((i for i, e in self._entries.items() if not e.working), None)
                if victim is None:
                    raise ServiceError(
                        "GENERATION_RESOURCE_LIMIT", "No session capacity available.", 503
                    )
                self._evictions += 1
                self._remove(victim, "evicted")
            identity = str(uuid4())
            decoded = self._decode(ids)
            now = self._wall()
            session = GenerationSession(
                generation_id=identity,
                request_key=body.request_key,
                original_prompt=body.input.text,
                original_input_ids=ids,
                original_decoded_prompt=decoded,
                model_id=MODEL_ID,
                hf_model_id=HF_MODEL_ID,
                model_revision=adapter.revision,
                tokenizer_revision=adapter.revision,
                tokenizer_identity=type(adapter.tokenizer).__name__,
                device=adapter.device,
                dtype=str(adapter.dtype).removeprefix("torch."),
                execution_configuration={
                    "use_cache": False,
                    "attention_backend": "eager",
                    "eval_mode": True,
                    "input_mode": "raw",
                    "add_special_tokens": False,
                    "logits_to_keep": 1,
                    "position_policy": "absolute 0..prefix_length-1",
                    "config_digest": adapter.architecture.config_digest,
                    "implementation_digest": adapter.architecture.implementation_digest,
                },
                sampling=body.sampling,
                started_at=utc_timestamp(now),
                status="queued",
                max_new_tokens=body.max_new_tokens,
                decoded_full_sequence=decoded,
                expires_at=utc_timestamp(now + self.settings.generation_ttl_seconds),
                budgets={
                    "max_prefix_tokens": prefix_limit,
                    "max_step_summaries": body.max_new_tokens,
                    "max_retained_traces": body.retain_traces,
                    "max_result_bytes": self.settings.generation_session_max_bytes,
                    "reserved_state_bytes": reserve,
                    "max_trace_bytes": self.settings.generation_trace_max_bytes,
                    "max_execution_seconds": self.settings.generation_duration_seconds,
                    "scratch_budget_bytes": self.settings.generation_scratch_max_bytes,
                    "scratch_estimated_bytes": self._scratch_estimate(adapter, len(ids)),
                },
            )
            entry = Entry(
                session,
                body.model_copy(deep=True),
                digest,
                self._clock() + self.settings.generation_ttl_seconds,
                self._clock(),
                reserve,
            )
            self._entries[identity], self._keys[body.request_key] = entry, identity
            try:
                entry.future = self.model.submit_trace_job(self._run, identity)
            except BaseException:
                self._entries.pop(identity)
                self._keys.pop(body.request_key)
                raise
            return self._snapshot(entry)

    def _decode(self, ids):
        return self.model.adapter.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    @staticmethod
    def _scratch_estimate(adapter, sequence_length):
        architecture = adapter.architecture
        vocabulary = architecture.vocabulary_size * 80
        feed_forward = sequence_length * architecture.intermediate_size * 12
        residual = sequence_length * architecture.hidden_size * 12
        eager_attention = 4 * architecture.query_heads * sequence_length**2 * 4
        return vocabulary + feed_forward + residual + eager_attention

    def _snapshot(self, entry):
        session = entry.session.model_copy(deep=True)
        retained = 0
        available = 0
        for step in session.steps:
            if step.source_trace_id:
                try:
                    snapshot = self.store.snapshot(step.source_trace_id)
                    retained += snapshot["retained_bytes"]
                    available += snapshot["status"] == "ready"
                except ServiceError as exc:
                    # Original identity remains; availability is not a substitute execution.
                    step.source_capture_coverage["current_availability"] = exc.code
        session.trace_retention_summary = {
            "original_traces": sum(s.source_trace_id is not None for s in session.steps),
            "available_traces": available,
            "retained_bytes": retained,
        }
        session.result_bytes = len(session.model_dump_json().encode())
        session.accounted_state_bytes = entry.reserved
        return session

    def get(self, identity):
        with self._lock:
            return self._snapshot(self._lookup(identity))

    def cancel(self, identity):
        with self._lock:
            entry = self._lookup(identity)
            if entry.session.status not in TERMINAL:
                entry.session.cancellation_requested = True
                if entry.session.status == "queued":
                    entry.session.status = "cancelled"
                    entry.session.stop_reason = "cancelled"
                    entry.session.completed_at = utc_timestamp(self._wall())
            return self._snapshot(entry)

    def delete(self, identity):
        with self._lock:
            self._sweep()
            entry = self._entries.get(identity)
            if entry is None:
                # DELETE is idempotent for known terminal tombstones.
                if identity in self._tombstones:
                    return
                raise ServiceError("GENERATION_NOT_FOUND", "Generation data does not exist.", 404)
            if entry.working:
                entry.session.cancellation_requested = True
                return
            self._remove(identity, "deleted")

    def _halt(self, entry):
        if entry.removed or self._clock() >= entry.deadline:
            return "session_expired"
        if self._closed or entry.session.cancellation_requested:
            return "cancelled"
        if self._clock() - entry.created >= self.settings.generation_duration_seconds:
            return "timeout"
        return None

    def _finish(self, entry, reason, error=None):
        session = entry.session
        session.stop_reason = reason
        if error:
            session.status = "failed"
        elif reason == "cancelled":
            session.status = "cancelled"
        elif reason in {"eos", "max_new_tokens"}:
            session.status = "completed"
        else:
            session.status = "stopped"
        session.completed_at = utc_timestamp(self._wall())
        session.session_duration_ms = (self._clock() - entry.created) * 1000
        session.active_forward_execution_id = None
        session.error = error

    def _capture(self, entry, prefix, execution_id):
        body = entry.request
        captured = sum(s.source_trace_id is not None for s in entry.session.steps)
        held = entry.session.trace_retention_summary.get("retained_bytes", 0)
        reservation = None
        unavailable = None
        if body.capture_profile != "none" and captured < body.retain_traces:
            plan = CapturePlan(
                profile=body.capture_profile,
                layers=(body.capture_layer,) if body.capture_profile == "deep" else (),
                token_indices=(len(prefix) - 1,),
                retain_base_states=True,
                max_bytes=self.settings.trace_max_bytes,
            )
            estimate = estimate_capture(self.model.adapter, plan, len(prefix))
            if estimate.peak_bytes > self.settings.trace_max_bytes:
                unavailable = "capture execution estimate exceeds budget"
            elif held + estimate.retained_bytes > self.settings.generation_trace_max_bytes:
                unavailable = "per-session detailed trace byte limit"
            else:
                try:
                    reservation = self.store.reserve(estimate.retained_bytes)
                except ServiceError as exc:
                    unavailable = exc.code
        else:
            unavailable = (
                "capture disabled" if body.capture_profile == "none" else "trace count limit"
            )
        text = self._decode(prefix)
        if reservation:
            identity = reservation["trace_id"]
            try:

                def compute(_text, top_k, _request_id):
                    return self.model._compute(text, top_k, execution_id, prefix)

                trace = TraceEngine(self.model.adapter, self.settings).run(
                    text, plan, 100, compute, token_ids_override=prefix
                )
                trace = replace(
                    trace,
                    metadata=replace(
                        trace.metadata,
                        trace_id=identity,
                        runtime={
                            **trace.metadata.runtime,
                            "generation_id": entry.session.generation_id,
                            "generation_step": len(entry.session.steps),
                            "forward_execution_id": execution_id,
                            "exact_input_ids": True,
                        },
                    ),
                )
                return (
                    trace.tensor("output.logits", copy=False).reshape(-1),
                    trace,
                    identity,
                    None,
                    trace.metadata.forward_duration_ms,
                )
            except BaseException:
                self.store.fail(identity, {"code": "GENERATION_FORWARD_FAILED"})
                self.store.delete(identity)
                raise
        result = self.model._compute(text, 100, execution_id, prefix)
        return result.logits, None, None, unavailable, result.response.runtime.forward_duration_ms

    def _decision(
        self, entry, prefix, logits, execution_id, trace, identity, unavailable, ms, start, rng
    ):
        params = entry.request.sampling
        raw, raw_order, raw_ranks = distribution(logits)
        argmax = int(raw_order[0])
        if params.mode == "greedy":
            processed = logits.detach().cpu().double().reshape(-1)
            adjusted, order, ranks = raw, raw_order, raw_ranks
            selected = argmax
            adjusted_probability = None
            adjusted_entropy = None
        else:
            processed = process_logits(logits, params)
            adjusted, order, ranks = distribution(processed)
            selected = int(torch.multinomial(adjusted.double(), 1, generator=rng))
            adjusted_probability = float(adjusted[selected])
            adjusted_entropy = float(
                -(adjusted.double() * adjusted.double().clamp_min(1e-300).log()).sum()
            )
        eligible = torch.isfinite(processed)
        top = order[: min(params.display_top_k, int(eligible.sum()))].tolist()
        candidates = []
        # Include both selected and raw argmax even when outside displayed top candidates.
        for t in dict.fromkeys(top + [selected, argmax]):
            candidates.append(
                GenerationCandidate(
                    token_id=t,
                    tokenizer_piece=self.model.adapter.tokenizer.convert_ids_to_tokens(t),
                    decoded_in_isolation=self._decode([t]),
                    raw_logit=float(logits[t]),
                    model_probability=float(raw[t]),
                    processed_logit=float(processed[t]) if eligible[t] else None,
                    adjusted_probability=(float(adjusted[t]) if params.mode == "sample" else None),
                    raw_rank=int(raw_ranks[t]),
                    adjusted_rank=int(ranks[t]) if params.mode == "sample" else None,
                    eligible=bool(eligible[t]),
                )
            )
        full = self._decode(prefix + [selected])
        prior = self._decode(prefix)
        delta = full[len(prior) :] if full.startswith(prior) else None
        generated = entry.session.selected_token_ids + [selected]
        coverage = (
            {
                "profile": trace.metadata.capture_plan.profile,
                "layers": list(trace.metadata.capture_plan.layers),
                "tokens": list(trace.metadata.capture_plan.token_indices),
                "retain_base_states": True,
            }
            if trace
            else {"profile": "none"}
        )
        index = len(entry.session.steps)
        return GenerationStep(
            generation_id=entry.session.generation_id,
            step_id=f"{entry.session.generation_id}:{index}",
            step_index=index,
            prefix_token_ids=prefix,
            prefix_sha256=hashlib.sha256(
                json.dumps(prefix, separators=(",", ":")).encode()
            ).hexdigest(),
            decoded_prefix=prior,
            decision_position=len(prefix) - 1,
            selected_position=len(prefix),
            vocabulary_size=len(logits),
            selected_token_id=selected,
            tokenizer_piece=self.model.adapter.tokenizer.convert_ids_to_tokens(selected),
            selected_display_text=delta,
            display_note=(
                "Contextual full-sequence decode suffix; null if prior Unicode changed. "
                "Token pieces are not concatenated."
            ),
            decoded_continuation=self._decode(generated),
            decoded_full_sequence=full,
            selected_raw_logit=float(logits[selected]),
            selected_model_probability=float(raw[selected]),
            selected_adjusted_probability=adjusted_probability,
            raw_argmax_token_id=argmax,
            candidates=candidates,
            distribution_statistics={
                "eligible_count": int(eligible.sum()),
                "raw_probability_sum": float(raw.double().sum()),
                "adjusted_probability_sum": (
                    float(adjusted.double().sum()) if params.mode == "sample" else None
                ),
                "raw_entropy_nats": float(
                    -(raw.double() * raw.double().clamp_min(1e-300).log()).sum()
                ),
                "adjusted_entropy_nats": adjusted_entropy,
                "displayed_adjusted_mass": (
                    float(adjusted[top].double().sum()) if params.mode == "sample" else None
                ),
            },
            sampling=params.model_copy(deep=True),
            source_trace_id=identity,
            source_capture_coverage=coverage,
            capture_unavailable_reason=unavailable,
            forward_execution_id=execution_id,
            forward_duration_ms=ms,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            provenance={
                "model_revision": self.model.adapter.revision,
                "tokenizer_revision": self.model.adapter.revision,
                "device": self.model.adapter.device,
                "dtype": str(self.model.adapter.dtype).removeprefix("torch."),
                "use_cache": False,
                "forward_count": 1,
                "input_policy": "exact prefix IDs",
                "rank_policy": RANK_POLICY,
                "probability_policy": PROBABILITY_POLICY,
                "sampling_policy": (
                    "raw-logit argmax; no sampling warpers or adjusted distribution"
                    if params.mode == "greedy"
                    else (
                        "Stage6 temperature -> top-k -> top-p; CPU float64 processors, "
                        "float32 probability then float64 multinomial weights"
                    )
                ),
                "rng_policy": "dedicated CPU torch.Generator; no global RNG mutation",
                "transformers_default_processors": "none; no inherited penalties",
                "raw_logits_retained": bool(trace),
            },
        )

    def _run(self, identity):
        entry = self._entries[identity]
        rng = torch.Generator(device="cpu")
        if entry.request.sampling.seed is not None:
            rng.manual_seed(entry.request.sampling.seed)
        try:
            with self._lock:
                reason = self._halt(entry)
                if reason:
                    self._finish(entry, reason)
                    return
                entry.session.status = "running"
            while True:
                with self._lock:
                    reason = self._halt(entry)
                    prefix = entry.session.original_input_ids + entry.session.selected_token_ids
                    if not reason and len(entry.session.steps) >= entry.request.max_new_tokens:
                        reason = "max_new_tokens"
                    if not reason and len(prefix) > entry.session.budgets["max_prefix_tokens"]:
                        reason = "context_limit"
                    scratch = self._scratch_estimate(self.model.adapter, len(prefix))
                    if not reason and scratch > self.settings.generation_scratch_max_bytes:
                        reason = "resource_budget"
                    # Reserve enough encoded space before starting the next forward.
                    if (
                        not reason
                        and len(entry.session.model_dump_json().encode()) + 24000 + len(prefix) * 16
                        > self.settings.generation_session_max_bytes
                    ):
                        reason = "resource_budget"
                    if reason:
                        self._finish(entry, reason)
                        return
                    execution_id = str(uuid4())
                    entry.session.active_forward_execution_id = execution_id
                    entry.session.attempted_forwards += 1
                start = time.perf_counter()
                trace = source = None
                try:
                    logits, trace, source, unavailable, ms = self._capture(
                        entry, prefix, execution_id
                    )
                    step = self._decision(
                        entry,
                        prefix,
                        logits,
                        execution_id,
                        trace,
                        source,
                        unavailable,
                        ms,
                        start,
                        rng,
                    )
                    with self._lock:
                        reason = self._halt(entry)
                        if reason:
                            if source:
                                self.store.fail(source, {})
                                self.store.delete(source)
                            self._finish(entry, reason)
                            return
                        if (
                            len(entry.session.model_dump_json().encode())
                            + len(step.model_dump_json().encode())
                            + 2048
                            > self.settings.generation_session_max_bytes
                        ):
                            if source:
                                self.store.fail(source, {})
                                self.store.delete(source)
                            self._finish(entry, "resource_budget")
                            return
                        if source:
                            self.store.commit(source, trace)
                            # Deleted sources must not appear as retained science.
                            snapshot = self.store.snapshot(source)
                            if snapshot["status"] != "ready":
                                step = step.model_copy(
                                    update={
                                        "source_trace_id": None,
                                        "capture_unavailable_reason": "original trace cancelled",
                                    }
                                )
                            else:
                                entry.session.trace_retention_summary["retained_bytes"] = (
                                    entry.session.trace_retention_summary.get("retained_bytes", 0)
                                    + snapshot["retained_bytes"]
                                )
                        entry.session.steps.append(step)
                        entry.session.selected_token_ids.append(step.selected_token_id)
                        entry.session.completed_steps += 1
                        entry.session.decoded_continuation = step.decoded_continuation
                        entry.session.decoded_full_sequence = step.decoded_full_sequence
                        entry.session.active_forward_execution_id = None
                        eos = self.model.adapter.model.generation_config.eos_token_id
                        eos = [eos] if isinstance(eos, int) else eos or []
                        if step.selected_token_id in eos:
                            self._finish(entry, "eos")
                            return
                except BaseException:
                    if source:
                        try:
                            self.store.fail(source, {"code": "GENERATION_EXECUTION_ERROR"})
                            self.store.delete(source)
                        except (ServiceError, KeyError):
                            pass
                    raise
                finally:
                    trace = logits = step = None
        except Exception as exc:
            logger.exception("Generation execution failed")
            with self._lock:
                self._finish(
                    entry,
                    "execution_error",
                    {
                        "code": getattr(exc, "code", "GENERATION_FAILED"),
                        "message": "Generation failed; completed steps are preserved.",
                    },
                )
        finally:
            with self._lock:
                entry.working = False
                entry.future = None
                if entry.removed:
                    self._delete_sources(entry)
                    self._entries.pop(identity, None)
                    self._keys.pop(entry.request.request_key, None)

    def status(self):
        with self._lock:
            self._sweep()
            return {
                "session_count": len(self._entries),
                "inflight": sum(e.working for e in self._entries.values()),
                "reserved_state_bytes": sum(e.reserved for e in self._entries.values()),
                "result_bytes": sum(
                    len(e.session.model_dump_json().encode()) for e in self._entries.values()
                ),
                "eviction_count": self._evictions,
                "expiration_count": self._expirations,
            }

    def close(self):
        with self._lock:
            self._closed = True
            for identity in list(self._entries):
                self._remove(identity, "expired")
