import gc
import logging
import platform
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from uuid import uuid4

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_lab.config import HF_MODEL_ID, Settings
from llm_lab.errors import ServiceError
from llm_lab.models.adapters import create_adapter
from llm_lab.models.baseline import BaselineResult, direct_hf_baseline
from llm_lab.models.devices import DeviceSelection, select_device, synchronize
from llm_lab.schemas.inference import (
    Candidate,
    InputResult,
    NextTokenResponse,
    Prediction,
    TokenRecord,
)
from llm_lab.schemas.model import ModelMetadata, ModelStatus, RuntimeMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    response: NextTokenResponse
    logits: torch.Tensor
    probabilities: torch.Tensor


class ModelService:
    """Own one resident model and a bounded single-thread execution queue.

    Routes submit work; only the executor loads or runs model weights. Status
    snapshots use a separate lock and remain available during those operations.
    InferenceResult contains final-position output only and is never stored.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        detected = select_device("auto")
        self._status = ModelStatus(
            state="unloaded",
            configured_revision=settings.model_revision,
            detected_device=detected.device,
            available_devices=list(detected.available),
            max_input_tokens=settings.max_input_tokens,
            max_prompt_characters=settings.max_prompt_characters,
            queue_capacity=settings.queue_capacity,
        )
        self._state_lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(settings.queue_capacity)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-model")
        self._model = None
        self._tokenizer = None
        self._adapter = None
        self._selection: DeviceSelection | None = None
        self._load_future: Future | None = None
        self._load_options: tuple[str, str] | None = None
        self._closed = False
        self._request_forward_passes = 0

    def status(self) -> ModelStatus:
        with self._state_lock:
            return self._status.model_copy(deep=True)

    def _update(self, **changes) -> None:
        with self._state_lock:
            for name, value in changes.items():
                setattr(self._status, name, value)

    def _submit(self, function, *args) -> Future:
        with self._state_lock:
            if self._closed:
                raise ServiceError("EXECUTION_UNAVAILABLE", "Model service is closed.", 503)
            if not self._slots.acquire(blocking=False):
                raise ServiceError(
                    "QUEUE_FULL", "Inference queue is full. Try again shortly.", 429, retryable=True
                )
            self._status.pending_operations += 1

            def run():
                self._update(execution_busy=True)
                try:
                    return function(*args)
                finally:
                    self._update(execution_busy=False)

            try:
                future = self._executor.submit(run)
            except Exception:
                self._status.pending_operations -= 1
                self._slots.release()
                raise

            def finished(_future):
                if not _future.cancelled():
                    exception = _future.exception()
                    seen = set()
                    pending = [exception] if exception is not None else []
                    while pending:
                        failure = pending.pop()
                        if id(failure) in seen:
                            continue
                        seen.add(id(failure))
                        if failure.__traceback__ is not None:
                            origin = "".join(
                                traceback.format_list(
                                    traceback.extract_tb(failure.__traceback__, limit=-8)
                                )
                            )
                            failure.add_note(
                                "Worker origin (frame references released):\n" + origin
                            )
                        # Now all model/trace frames have unwound. A retained
                        # failed Future must not retain their weights/snapshots.
                        traceback.clear_frames(failure.__traceback__)
                        # CPython 3.11 cleared frames can still own closure
                        # cells. Drop the worker traceback itself, preserving
                        # exception type, message and causal chain.
                        failure.__traceback__ = None
                        pending.extend(
                            linked
                            for linked in (failure.__cause__, failure.__context__)
                            if linked is not None
                        )
                with self._state_lock:
                    self._status.pending_operations -= 1
                    self._slots.release()

            future.add_done_callback(finished)
            return future

    def start_loading(self, device: str | None = None, revision: str | None = None) -> ModelStatus:
        selection = select_device(device or self.settings.device)
        resolved_revision = revision or self.settings.model_revision
        options = (selection.device, resolved_revision)
        with self._state_lock:
            if self._status.state in {"loading", "ready"}:
                if self._load_options != options:
                    raise ServiceError(
                        "MODEL_BUSY",
                        "Restart the service to change the resident model settings.",
                        409,
                    )
                return self.status()
            self._load_options = options
            self._update(
                state="loading",
                device=selection.device,
                dtype=str(selection.dtype).removeprefix("torch."),
                dtype_reason=selection.reason,
                loading_phase="loading_tokenizer",
                error=None,
                error_code=None,
                resolved_revision=None,
                load_duration_ms=None,
            )
            try:
                self._load_future = self._submit(self._load, selection, resolved_revision)
            except ServiceError:
                self._update(state="unloaded", loading_phase=None)
                raise
        return self.status()

    def wait_until_ready(self, timeout: float = 1200) -> None:
        if self._load_future is None:
            self.start_loading()
        assert self._load_future is not None
        self._load_future.result(timeout=timeout)

    def _load(self, selection: DeviceSelection, revision: str) -> None:
        started = time.perf_counter()
        model = tokenizer = result = encoded = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                HF_MODEL_ID,
                revision=revision,
                trust_remote_code=False,
                token=False,
                cache_dir=self.settings.cache_dir,
            )
            self._update(loading_phase="loading_weights")
            model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_ID,
                revision=revision,
                trust_remote_code=False,
                token=False,
                cache_dir=self.settings.cache_dir,
                dtype=selection.dtype,
                attn_implementation="eager",
                use_safetensors=True,
            )
            self._update(loading_phase="moving_to_device")
            model.to(selection.device)
            model.eval()
            self._update(loading_phase="smoke_check")
            encoded = tokenizer("Hello", add_special_tokens=False, return_tensors="pt")
            with torch.inference_mode():
                result = model(
                    **{key: value.to(selection.device) for key, value in encoded.items()},
                    logits_to_keep=1,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
                if not bool(torch.isfinite(result.logits).all().item()):
                    raise RuntimeError("Model smoke check returned non-finite logits")
            synchronize(selection.device)
            self._model, self._tokenizer, self._selection = model, tokenizer, selection
            self._adapter = create_adapter(model, tokenizer, revision, selection.device)
            self._update(
                state="ready",
                resolved_revision=revision,
                loading_phase=None,
                load_duration_ms=(time.perf_counter() - started) * 1000,
            )
            logger.info("Model ready on %s using %s", selection.device, selection.dtype)
        except Exception as exc:
            logger.exception("Model loading failed")
            code = exc.code if isinstance(exc, ServiceError) else "MODEL_LOAD_FAILED"
            self._model = self._tokenizer = self._selection = self._adapter = None
            # A failed Future retains its exception. Drop partially loaded
            # weights and clear unwound frames before retaining the safe error.
            model = tokenizer = result = encoded = None
            traceback.clear_frames(exc.__traceback__)
            self._update(
                state="failed",
                loading_phase=None,
                error_code=code,
                error="Qwen could not be loaded. Check backend logs, network access and memory.",
                load_duration_ms=(time.perf_counter() - started) * 1000,
            )
            gc.collect()
            raise ServiceError(
                code, self.status().error or "Model loading failed.", 503, retryable=True
            ) from exc

    def _require_ready(self) -> None:
        state = self.status()
        if state.state == "failed":
            raise ServiceError(
                "MODEL_LOAD_FAILED", state.error or "Model loading failed.", 503, retryable=True
            )
        if state.state != "ready":
            raise ServiceError(
                "MODEL_NOT_READY",
                "Model is not ready. Wait for loading or load it.",
                409,
                retryable=True,
                details={"state": state.state},
            )

    def infer(self, text: str, top_k: int = 10, request_id: str | None = None) -> Future:
        self._require_ready()
        return self._submit(self._response_only, text, top_k, request_id or str(uuid4()))

    def infer_for_validation(self, text: str, top_k: int = 10) -> Future:
        """Output-only computation for real-model tests; no internals or persistent store."""
        self._require_ready()
        return self._submit(self._compute, text, top_k, str(uuid4()))

    @property
    def adapter(self):
        self._require_ready()
        return self._adapter

    def trace(self, text: str, plan=None, top_k: int = 10) -> Future:
        """In-process Stage2 result, serialized with ordinary inference; no store."""
        self._require_ready()
        return self._submit(self._trace, text, plan, top_k)

    def submit_trace_job(self, function, *args) -> Future:
        """Stage 3 orchestration uses the same executor and queue admission."""
        self._require_ready()
        return self._submit(function, *args)

    def submit_analysis_job(self, function, *args) -> Future:
        """Serialize saved-state readout with loading/tracing; never calls forward."""
        self._require_ready()
        return self._submit(function, *args)

    def execution_metrics(self) -> dict:
        with self._state_lock:
            return {
                "request_forward_passes": self._request_forward_passes,
                "counter_scope": (
                    "adapter.forward attempts from inference/trace/generation; "
                    "excludes load warmup and independent validation reference"
                ),
            }

    def _trace(self, text, plan, top_k):
        from llm_lab.tracing.engine import TraceEngine

        return TraceEngine(self.adapter, self.settings).run(text, plan, top_k, self._compute)

    def reference_baseline(self, text: str) -> Future:
        self._require_ready()
        return self._submit(self._reference, text)

    def _reference(self, text: str) -> BaselineResult:
        assert self._selection is not None
        return direct_hf_baseline(self._model, self._tokenizer, text, self._selection.device)

    def _response_only(self, text: str, top_k: int, request_id: str) -> NextTokenResponse:
        return self._compute(text, top_k, request_id).response

    def _compute(
        self, text: str, top_k: int, request_id: str, token_ids_override=None
    ) -> InferenceResult:
        self._require_ready()
        if token_ids_override is None and not text.strip():
            raise ServiceError("EMPTY_PROMPT", "Enter a non-empty prompt.", 422)
        if not 1 <= top_k <= 100 or isinstance(top_k, bool):
            raise ServiceError("INVALID_TOP_K", "top_k must be an integer between 1 and 100.", 422)
        if len(text) > self.settings.max_prompt_characters:
            raise ServiceError("INPUT_LIMIT_EXCEEDED", "Prompt exceeds the character limit.", 413)
        assert self._selection is not None
        started = time.perf_counter()
        try:
            if token_ids_override is None:
                encoded = self._adapter.tokenize(
                    text,
                    add_special_tokens=False,
                    return_tensors="pt",
                    truncation=False,
                    return_offsets_mapping=True,
                )
                token_ids = encoded["input_ids"][0].tolist()
                offsets = encoded.pop("offset_mapping")[0].tolist()
            else:
                token_ids = list(token_ids_override)
                if any(
                    type(t) is not int or not 0 <= t < self._model.config.vocab_size
                    for t in token_ids
                ):
                    raise ServiceError("INVALID_TOKEN", "Prefix has invalid vocabulary IDs.", 422)
                encoded = {
                    "input_ids": torch.tensor([token_ids], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
                }
                # Exact-ID forwards have no character-offset mapping; -1 means unavailable.
                offsets = [None] * len(token_ids)
            if not token_ids or len(token_ids) > self.settings.max_input_tokens:
                raise ServiceError(
                    "INPUT_LIMIT_EXCEEDED",
                    "Prompt exceeds token limit.",
                    413,
                    details={
                        "token_count": len(token_ids),
                        "max_input_tokens": self.settings.max_input_tokens,
                    },
                )
            inputs = {key: value.to(self._selection.device) for key, value in encoded.items()}
            synchronize(self._selection.device)
            forward_started = time.perf_counter()
            with torch.inference_mode():
                with self._state_lock:
                    self._request_forward_passes += 1
                output = self._adapter.forward(
                    **inputs,
                    logits_to_keep=1,
                )
                logits_device = output.logits[0, -1].float()
                if not bool(torch.isfinite(logits_device).all().item()):
                    raise RuntimeError("Inference returned non-finite logits")
                synchronize(self._selection.device)
                forward_ms = (time.perf_counter() - forward_started) * 1000
                # One bounded final-position copy. Softmax is over the full vocabulary.
                logits = logits_device.cpu()
                # Float32 CPU softmax exceeded the normalization gate on this
                # runtime. Promote only this bounded vocabulary normalization,
                # then expose float32 probabilities; model logits are unchanged.
                probabilities = torch.softmax(logits, dim=-1, dtype=torch.float64).float()
                _, ids = torch.topk(logits, k=top_k, sorted=True)
                selected_ids = ids.tolist()
            special_ids = set(self._tokenizer.all_special_ids)
            tokens = [
                TokenRecord(
                    token_index=index,
                    token_id=token_id,
                    tokenizer_piece=self._tokenizer.convert_ids_to_tokens(token_id),
                    decoded_token=self._decode(token_id),
                    position_id=index,
                    is_special=token_id in special_ids,
                    character_start=offsets[index][0] if offsets[index] is not None else None,
                    character_end=offsets[index][1] if offsets[index] is not None else None,
                )
                for index, token_id in enumerate(token_ids)
            ]
            candidates = [
                Candidate(
                    rank=rank,
                    token_id=token_id,
                    tokenizer_piece=self._tokenizer.convert_ids_to_tokens(token_id),
                    decoded_token=self._decode(token_id),
                    raw_logit=float(logits[token_id]),
                    probability=float(probabilities[token_id]),
                )
                for rank, token_id in enumerate(selected_ids, start=1)
            ]
            status = self.status()
            assert status.resolved_revision is not None and status.load_duration_ms is not None
            response = NextTokenResponse(
                request_id=request_id,
                model=ModelMetadata(
                    revision=status.resolved_revision,
                    tokenizer_revision=status.resolved_revision,
                    device=self._selection.device,
                    dtype=str(self._selection.dtype).split(".")[-1],
                    vocabulary_size=self._model.config.vocab_size,
                    layer_count=self._model.config.num_hidden_layers,
                    hidden_size=self._model.config.hidden_size,
                ),
                input=InputResult(
                    text=text,
                    effective_text=text,
                    token_count=len(token_ids),
                    tokens=tokens,
                    decoded_text=self._tokenizer.decode(
                        token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                ),
                prediction=Prediction(
                    position=len(token_ids) - 1,
                    next_position=len(token_ids),
                    logits_shape=list(output.logits.shape),
                    vocabulary_size=len(logits),
                    top_k=candidates,
                    omitted_probability_mass=max(0.0, 1.0 - sum(c.probability for c in candidates)),
                    argmax_token_id=selected_ids[0],
                ),
                runtime=RuntimeMetadata(
                    python_version=platform.python_version(),
                    torch_version=torch.__version__,
                    transformers_version=transformers.__version__,
                    load_duration_ms=status.load_duration_ms,
                    forward_duration_ms=forward_ms,
                    inference_duration_ms=(time.perf_counter() - started) * 1000,
                ),
            )
            return InferenceResult(response, logits, probabilities)
        except ServiceError:
            raise
        except Exception as exc:
            logger.exception("Inference failed for request %s", request_id)
            raise ServiceError(
                "INFERENCE_FAILED",
                "Inference failed. Check the backend logs.",
                500,
                retryable=False,
            ) from exc

    def _decode(self, token_id: int) -> str:
        return self._tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._load_future = None
        self._model = self._tokenizer = self._selection = self._adapter = None
        self._update(state="unloaded", device=None, dtype=None, resolved_revision=None)
        gc.collect()
