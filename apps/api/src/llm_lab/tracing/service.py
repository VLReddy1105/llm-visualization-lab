"""Trace jobs share the resident model's bounded, single-worker execution queue."""

import logging
from concurrent.futures import Future
from dataclasses import replace

from llm_lab.config import Settings
from llm_lab.errors import ServiceError
from llm_lab.models.manager import ModelService
from llm_lab.schemas.trace_api import CreateTraceRequest
from llm_lab.store.trace_store import TraceStore
from llm_lab.tracing.capture_plan import CapturePlan
from llm_lab.tracing.engine import estimate_capture

logger = logging.getLogger(__name__)


class TraceService:
    def __init__(self, model: ModelService, store: TraceStore, settings: Settings):
        self.model, self.store, self.settings = model, store, settings

    def create(self, body: CreateTraceRequest) -> tuple[Future, Future]:
        acknowledged: Future = Future()
        job = self.model.submit_trace_job(self._run, body, acknowledged)

        def finish(future):
            if not acknowledged.done():
                acknowledged.set_exception(
                    ServiceError("EXECUTION_UNAVAILABLE", "Trace execution could not start.", 503)
                )

        job.add_done_callback(finish)
        return acknowledged, job

    def _preflight(self, body: CreateTraceRequest):
        if len(body.prompt) > self.settings.max_prompt_characters:
            raise ServiceError("INPUT_LIMIT_EXCEEDED", "Prompt exceeds character limit.", 413)
        adapter = self.model.adapter
        encoded = adapter.tokenize(body.prompt, add_special_tokens=False, truncation=False)
        n = len(encoded["input_ids"])
        if not 1 <= n <= self.settings.max_input_tokens:
            raise ServiceError(
                "INPUT_LIMIT_EXCEEDED",
                "Prompt exceeds token limit.",
                413,
                details={"token_count": n, "max_input_tokens": self.settings.max_input_tokens},
            )
        selections = (
            (body.capture.layers, adapter.architecture.layers, "INVALID_LAYER"),
            (body.capture.tokens or [], n, "INVALID_TOKEN"),
            (body.capture.heads or [], adapter.architecture.query_heads, "INVALID_HEAD"),
        )
        for indices, bound, code in selections:
            if any(i >= bound for i in indices):
                raise ServiceError(
                    code,
                    "Capture selection exceeds model/input bounds.",
                    422,
                    details={"upper_bound_exclusive": bound},
                )
        try:
            plan = CapturePlan(
                profile=body.capture.profile,
                layers=tuple(body.capture.layers),
                token_indices=tuple(body.capture.tokens) if body.capture.tokens else None,
                query_heads=tuple(body.capture.heads) if body.capture.heads else None,
                max_bytes=self.settings.trace_max_bytes,
            ).resolve(adapter.architecture, n)
        except (ValueError, TypeError) as exc:
            raise ServiceError("INVALID_CAPTURE", str(exc), 422) from exc
        estimate = estimate_capture(adapter, plan, n)
        if estimate.peak_bytes > self.settings.trace_max_bytes:
            raise ServiceError(
                "CAPTURE_BUDGET_EXCEEDED",
                "Capture exceeds the execution memory budget.",
                413,
                details={
                    "estimated_peak_bytes": estimate.peak_bytes,
                    "max_bytes": self.settings.trace_max_bytes,
                    "hint": "Select fewer layers, tokens or heads, or use summary.",
                },
            )
        return plan, estimate

    def _run(self, body: CreateTraceRequest, acknowledged: Future) -> None:
        identity = None
        try:
            plan, estimate = self._preflight(body)
            snapshot = self.store.reserve(estimate.retained_bytes)
            identity = snapshot["trace_id"]
            acknowledged.set_result(snapshot)
            if self.store.snapshot(identity)["status"] == "cancelled":
                self.store.fail(identity, {})
                return
            # Retain up to 100 already-ranked candidates, so all bounded logits
            # GETs use captured tokens/probabilities without consulting a tokenizer.
            trace = self.model._trace(body.prompt, plan, 100)
            trace = replace(trace, metadata=replace(trace.metadata, trace_id=identity))
            self.store.commit(identity, trace)
        except Exception as exc:
            if isinstance(exc, ServiceError):
                error = ServiceError(
                    exc.code,
                    exc.message,
                    exc.status_code,
                    retryable=exc.retryable,
                    details=exc.details,
                )
            else:
                logger.exception("Trace creation failed")
                error = ServiceError(
                    "TRACE_CAPTURE_FAILED", "Trace capture failed; check backend logs.", 500
                )
            if identity is not None:
                self.store.fail(
                    identity,
                    {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details,
                        "retryable": error.retryable,
                    },
                )
            if not acknowledged.done():
                acknowledged.set_exception(error)
