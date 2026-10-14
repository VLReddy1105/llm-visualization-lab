"""Bounded summary-only jobs, source leases and absolute expiry; no tensor cache."""

import json
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from llm_lab.analysis.block_internals import (
    mlp_internals,
    require_block_capture,
    residual_stream,
)
from llm_lab.analysis.logit_lens import logit_lens, require_matching_adapter
from llm_lab.analysis.sampling import sampling_preview
from llm_lab.analysis.token_journey import checkpoints, evolution, source_metadata, vector
from llm_lab.errors import ServiceError
from llm_lab.schemas.analysis import BLOCK_METHOD_VERSION, METHOD_VERSION, AnalysisJob
from llm_lab.store.trace_store import utc_timestamp

logger = logging.getLogger(__name__)
BLOCK_KINDS = {"residual_stream", "mlp_internals"}


@dataclass
class JobEntry:
    envelope: AnalysisJob
    key: str
    deadline: float
    working: bool = True
    result_bytes: bytes | None = None


class AnalysisService:
    """One CPU worker; resident readouts also acquire the existing model executor.

    Each admitted job reserves one max-response slot. Terminal entries retain only
    encoded JSON summaries. Source tensors remain charged to the leased TraceStore.
    """

    def __init__(self, model, store, settings, clock=time.monotonic, wall_clock=time.time):
        self.model, self.store, self.settings = model, store, settings
        self._clock, self._wall = clock, wall_clock
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, JobEntry] = OrderedDict()
        self._keys = {}
        self._removed: OrderedDict[str, str] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-analysis")
        self._closed = False

    def _remove(self, identity, reason):
        entry = self._entries.pop(identity)
        if self._keys.get(entry.key) == identity:
            self._keys.pop(entry.key)
        self._removed[identity] = reason
        while len(self._removed) > self.settings.analysis_cache_max_entries * 2:
            self._removed.popitem(last=False)

    def sweep(self):
        with self._lock:
            for identity, entry in list(self._entries.items()):
                if entry.working:
                    continue
                try:
                    self.store.snapshot(entry.envelope.trace_id)
                    valid = self._clock() < entry.deadline
                except ServiceError:
                    valid = False
                if not valid:
                    self._remove(identity, "expired")

    def _accounted(self):
        return sum(
            self.settings.trace_api_max_response_bytes if e.working else len(e.result_bytes or b"")
            for e in self._entries.values()
        )

    def _make_room(self):
        self.sweep()
        reserve = self.settings.trace_api_max_response_bytes
        victims, released = [], 0
        for identity, entry in self._entries.items():
            if (
                self._accounted() - released + reserve <= self.settings.analysis_cache_max_bytes
                and len(self._entries) - len(victims) < self.settings.analysis_cache_max_entries
            ):
                break
            if not entry.working:
                victims.append(identity)
                released += len(entry.result_bytes or b"")
        if (
            self._accounted() - released + reserve > self.settings.analysis_cache_max_bytes
            or len(self._entries) - len(victims) >= self.settings.analysis_cache_max_entries
        ):
            raise ServiceError(
                "ANALYSIS_BUDGET_EXCEEDED", "No analysis cache capacity is available.", 503
            )
        for identity in victims:
            self._remove(identity, "evicted")

    def _preflight(self, trace, request):
        p, a = request.parameters, trace.metadata.architecture
        if request.kind in BLOCK_KINDS:
            # All retained vectors stay charged to TraceStore. Allow float64 copies,
            # sorting indices and native/audit reconstruction scratch for one block.
            estimated = (7 * a.hidden_size + 4 * a.intermediate_size) * 8 * 12
            if estimated > self.settings.analysis_work_max_bytes:
                raise ServiceError(
                    "ANALYSIS_BUDGET_EXCEEDED", "Block analysis exceeds scratch budget.", 413
                )
            if 24000 + p.top_k * 2400 > self.settings.trace_api_max_response_bytes:
                raise ServiceError(
                    "RESPONSE_TOO_LARGE", "Block coordinates exceed response budget.", 413
                )
            require_block_capture(trace, p)
            return
        source_metadata(trace, p.token_index)
        if request.kind != "sampling_preview":
            for state in checkpoints(trace):
                vector(trace, state.component_id, p.token_index)
        if request.kind == "logit_lens":
            require_matching_adapter(trace, self.model.adapter)
            if any(t >= a.vocabulary_size for t in p.candidate_ids):
                raise ServiceError("INVALID_CANDIDATE", "Candidate ID exceeds vocabulary.", 422)
        if request.kind == "sampling_preview":
            if p.token_index != trace.output.prediction_position:
                raise ServiceError(
                    "COMPONENT_NOT_CAPTURED", "Sampling requires saved final-position logits.", 409
                )
            if p.top_k > a.vocabulary_size:
                raise ServiceError("INVALID_TOP_K", "Sampling top-k exceeds vocabulary.", 422)
        # Conservative per-job tensor scratch bound. No depth × vocabulary storage.
        estimated = (a.layers + 2) * a.hidden_size * 16 + a.vocabulary_size * 96
        if estimated > self.settings.analysis_work_max_bytes:
            raise ServiceError(
                "ANALYSIS_BUDGET_EXCEEDED", "Analysis exceeds tensor scratch budget.", 413
            )
        if (
            request.kind == "token_evolution"
            and (a.layers + 2) ** 2 * 220 > self.settings.trace_api_max_response_bytes
        ):
            raise ServiceError(
                "RESPONSE_TOO_LARGE", "Depth comparisons exceed response budget.", 413
            )

    def create(self, trace_id, request):
        with self._lock, self.store.acquire(trace_id) as (snapshot, trace):
            if self._closed:
                raise ServiceError("EXECUTION_UNAVAILABLE", "Analysis service is closed.", 503)
            if trace is None:
                raise ServiceError("TRACE_NOT_READY", "Analysis requires a ready trace.", 409)
            if trace.metadata.trace_id != trace_id:
                raise ServiceError(
                    "SOURCE_IDENTITY_MISMATCH", "Stored source identity disagrees.", 409
                )
            version = BLOCK_METHOD_VERSION if request.kind in BLOCK_KINDS else METHOD_VERSION
            key = f"{trace_id}:{version}:{request.model_dump_json()}"
            self.sweep()
            existing = self._keys.get(key)
            if existing and self._entries[existing].envelope.status in {
                "queued",
                "running",
                "ready",
            }:
                return self.get(trace_id, existing, cache_hit=True)
            self._preflight(trace, request)
            if sum(e.working for e in self._entries.values()) >= self.settings.queue_capacity:
                raise ServiceError("QUEUE_FULL", "Analysis queue is full.", 429, retryable=True)
            self._make_room()
            identity, now = str(uuid4()), self._wall()
            expiry = min(
                now + self.settings.analysis_ttl_seconds,
                datetime.fromisoformat(snapshot["expires_at"]).timestamp(),
            )
            envelope = AnalysisJob(
                analysis_id=identity,
                trace_id=trace_id,
                method_version=version,
                request=request,
                status="queued",
                completed_checkpoints=0,
                total_checkpoints=1
                if request.kind == "sampling_preview" or request.kind in BLOCK_KINDS
                else trace.metadata.architecture.layers + 2,
                created_at=utc_timestamp(now),
                expires_at=utc_timestamp(expiry),
                retained_bytes=0,
            )
            entry = JobEntry(envelope, key, self._clock() + max(0, expiry - now))
            self._entries[identity], self._keys[key] = entry, identity
            try:
                self._executor.submit(self._run, identity)
            except Exception:
                self._remove(identity, "deleted")
                raise
            return envelope.model_copy(deep=True)

    def _progress(self, identity, completed, total):
        with self._lock:
            entry = self._entries[identity]
            if self._closed or entry.envelope.status == "cancelled":
                raise ServiceError("ANALYSIS_CANCELLED", "Analysis was cancelled.", 409)
            if self._clock() >= entry.deadline:
                raise ServiceError(
                    "ANALYSIS_EXPIRED", "Analysis reached its absolute deadline.", 410
                )
            self.store.snapshot(entry.envelope.trace_id)
            entry.envelope.status = "running"
            entry.envelope.completed_checkpoints, entry.envelope.total_checkpoints = (
                completed,
                total,
            )

    def _run(self, identity):
        entry = self._entries[identity]
        try:

            def progress(done, total):
                self._progress(identity, done, total)

            progress(0, entry.envelope.total_checkpoints)
            with self.store.acquire(entry.envelope.trace_id) as (_, trace):
                if trace is None:
                    raise ServiceError("TRACE_NOT_READY", "Source is unavailable.", 409)
                if trace.metadata.trace_id != entry.envelope.trace_id:
                    raise ServiceError(
                        "SOURCE_IDENTITY_MISMATCH", "Stored source identity disagrees.", 409
                    )
                request = entry.envelope.request
                if request.kind == "token_evolution":
                    result = evolution(trace, request.parameters.token_index, progress)
                elif request.kind == "sampling_preview":
                    result = sampling_preview(trace, request.parameters)
                elif request.kind == "residual_stream":
                    result = residual_stream(trace, request.parameters, progress)
                elif request.kind == "mlp_internals":
                    result = mlp_internals(trace, request.parameters, progress)
                else:
                    # Resolve/check the adapter again on its serialized executor.
                    def readout():
                        return logit_lens(trace, request.parameters, self.model.adapter, progress)

                    result = self.model.submit_analysis_job(readout).result()
                encoded = result.model_dump_json().encode()
                if len(encoded) + 8192 > self.settings.trace_api_max_response_bytes:
                    raise ServiceError(
                        "RESPONSE_TOO_LARGE", "Analysis summary exceeds response budget.", 413
                    )
                with self._lock:
                    progress(entry.envelope.total_checkpoints, entry.envelope.total_checkpoints)
                    entry.result_bytes = encoded
                    entry.envelope.retained_bytes = len(encoded)
                    entry.envelope.status = "ready"
        except Exception as exc:
            with self._lock:
                if entry.envelope.status != "cancelled":
                    entry.envelope.status = "failed"
                    if isinstance(exc, ServiceError):
                        error = {
                            "code": exc.code,
                            "message": exc.message,
                            "details": exc.details,
                            "retryable": exc.retryable,
                        }
                    else:
                        logger.exception("Analysis failed")
                        error = {
                            "code": "ANALYSIS_FAILED",
                            "message": "Analysis failed; check backend logs.",
                        }
                    from llm_lab.schemas.analysis import AnalysisError

                    entry.envelope.error = AnalysisError(**error)
        finally:
            with self._lock:
                entry.working = False

    def get(self, trace_id, identity, cache_hit=False):
        with self._lock:
            self.store.snapshot(trace_id)
            self.sweep()
            entry = self._entries.get(identity)
            if entry is None:
                reason = self._removed.get(identity)
                raise ServiceError(
                    "ANALYSIS_EXPIRED"
                    if reason in {"expired", "evicted"}
                    else "ANALYSIS_NOT_FOUND",
                    "Analysis is unavailable; request a new saved-data analysis.",
                    410 if reason in {"expired", "evicted"} else 404,
                )
            if entry.envelope.trace_id != trace_id:
                raise ServiceError(
                    "ANALYSIS_NOT_FOUND", "Analysis does not belong to this trace.", 404
                )
            if self._clock() >= entry.deadline:
                raise ServiceError("ANALYSIS_EXPIRED", "Analysis reached its deadline.", 410)
            self._entries.move_to_end(identity)
            data = entry.envelope.model_dump()
            data["cache_hit"] = cache_hit
            if entry.result_bytes is not None:
                data["result"] = json.loads(entry.result_bytes)
            return AnalysisJob.model_validate(data)

    def cancel(self, trace_id, identity):
        with self._lock:
            self.get(trace_id, identity)
            entry = self._entries[identity]
            entry.envelope.status = "cancelled"
            entry.result_bytes = None
            entry.envelope.retained_bytes = 0
            return self.get(trace_id, identity)

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)
        with self._lock:
            self._entries.clear()
            self._keys.clear()
            self._removed.clear()
