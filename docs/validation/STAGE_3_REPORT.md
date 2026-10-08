# Stage 3 validation report

**Stage 3: PASS.** Recorded 2026-10-08T08:41:04.592145+00:00. Stage 1 and Stage 2 remain validated; Stage 4 has not begun.

## 1. Scope and result

The server captures once with the existing resident Qwen3 adapter/Trace Engine, retains the immutable CPU TraceResult, and serves lazy bounded reads. No replacement model forward, new attention/MLP/residual mathematics, visualization, external persistence or new frontend library was introduced. The real HTTP gate and both 224-test device suites pass.

## 2. Exact created and modified files

Created (23):

- `apps/api/src/llm_lab/api/traces.py`
- `apps/api/src/llm_lab/schemas/trace_api.py`
- `apps/api/src/llm_lab/store/__init__.py`
- `apps/api/src/llm_lab/store/trace_store.py`
- `apps/api/src/llm_lab/tracing/serialization.py`
- `apps/api/src/llm_lab/tracing/service.py`
- `apps/api/src/llm_lab/trace_http_validation.py`
- `apps/api/tests/test_trace_api.py`
- `apps/api/tests/test_trace_serialization.py`
- `apps/api/tests/test_trace_store.py`
- `docs/validation/STAGE_3_REPORT.md`
- `docs/validation/stage3-create-trace.json`
- `docs/validation/stage3-trace-overview.json`
- `docs/validation/stage3-layer12.json`
- `docs/validation/stage3-attention-head.json`
- `docs/validation/stage3-residual.json`
- `docs/validation/stage3-mlp.json`
- `docs/validation/stage3-logits.json`
- `docs/validation/stage3-tensor-slice.json`
- `docs/validation/stage3-http-validation.json`
- `docs/validation/stage3-tests-mps.xml`
- `docs/validation/stage3-tests-cpu.xml`
- `docs/validation/stage3-checks.json`

Modified (13):

- `README.md`
- `docs/API_CONTRACT.md`
- `docs/SYSTEM_ARCHITECTURE.md`
- `docs/IMPLEMENTATION_STAGES.md`
- `apps/api/.env.example`
- `apps/api/src/llm_lab/config.py`
- `apps/api/src/llm_lab/main.py`
- `apps/api/src/llm_lab/models/manager.py`
- `apps/api/src/llm_lab/export_openapi.py`
- `apps/api/tests/test_foundation.py`
- `apps/web/src/lib/api/client.ts`
- `packages/shared/openapi.json`
- `packages/shared/src/generated/api.ts`

Generated caches/build outputs are excluded. There is no Git repository in this workspace, so no commit/PR was created. Stage 2 adapter, engine, collector, instrumentation, analysis and scientific validation source files remain unchanged. The model manager changes add only shared job admission and a forward counter. The Stage 2 foundation test's old exact route inventory now asserts predecessor inclusion while retaining its hook/generation guard.

## 3–7. Store architecture, TTL, memory, eviction and leases

`TraceStore` is one process-local OrderedDict/RLock owner. UUID4 entries move from running to ready/failed/cancelled. Metadata contains timestamps, status, model/revision/sequence/profile/detail-layer information, unique retained bytes and reservation bytes. Ready tensors are detached CPU storage from Stage 2; descriptors retain semantic aliases.

Default absolute TTL is **1,800 seconds** after completion, configurable by `LLM_LAB_TRACE_TTL_SECONDS`. Access updates LRU/last access without extending expiry. Lookup/reservation sweeps and the configurable **15-second** idle sweep reclaim expired records. Failed/cancelled records also receive completion TTL. Running work is bounded by the existing one-running/two-queued admission policy.

Default store cap is **536,870,912 bytes (512 MiB)** through `LLM_LAB_TRACE_STORE_MAX_BYTES`. Preflight reserves Stage 2's conservative retained-byte estimate before forward execution. Commit releases the estimate and charges its actual unique-storage measurement; aliases do not double-charge. Execution's separate conservative peak cap remains 256 MiB. Model weights, Python metadata, allocator caches and total RSS are outside this tensor-store cap; metadata is independently bounded to **1,024** entries.

Capacity eviction follows oldest-last-access among terminal unleased entries. Jobs and leased data cannot be evicted. If candidates cannot satisfy both byte and entry limits, creation returns 503 without unnecessarily evicting viable results. A capture never becomes a partially stored trace. An unexpected estimate overrun fails closed and releases its reservation.

Acquire/release increments leases under the RLock, then serialization reads immutable CPU tensors outside it. Expiration/deletion denies new readers immediately but keeps existing readers' data alive until release. Deferred storage remains charged to the budget. Cleanup and lease release are safe after serializer exceptions and shutdown. Removed-ID tombstones are capped at 2,048 and at 24 hours; expired/evicted IDs return 410 while remembered, deleted IDs return 404.

## 8. Complete endpoint inventory

All 18 operations below also exist under `/api/traces` through the same handlers; only `/api/v1` appears in OpenAPI. Original health/model/inference aliases remain unchanged.

| Method | Canonical endpoint |
| --- | --- |
| POST | `/api/v1/traces` |
| GET | `/api/v1/traces` |
| GET | `/api/v1/traces/status` |
| GET | `/api/v1/traces/{trace_id}` |
| DELETE | `/api/v1/traces/{trace_id}` |
| GET | `/api/v1/traces/{trace_id}/tokens` |
| GET | `/api/v1/traces/{trace_id}/layers` |
| GET | `/api/v1/traces/{trace_id}/layers/{layer_index}` |
| GET | `/api/v1/traces/{trace_id}/layers/{layer_index}/attention` |
| GET | `/api/v1/traces/{trace_id}/layers/{layer_index}/attention/heads/{head_index}` |
| GET | `/api/v1/traces/{trace_id}/layers/{layer_index}/attention/heads/{head_index}/relation` |
| GET | `/api/v1/traces/{trace_id}/layers/{layer_index}/mlp` |
| GET | `/api/v1/traces/{trace_id}/layers/{layer_index}/residual` |
| GET | `/api/v1/traces/{trace_id}/logits` |
| GET | `/api/v1/traces/{trace_id}/components` |
| GET | `/api/v1/traces/{trace_id}/components/{component_id}` |
| POST | `/api/v1/traces/{trace_id}/components/{component_id}/slice` |
| POST | `/api/v1/traces/{trace_id}/tensors/{tensor_id}/slice` |

## 9. Capture request schema

```json
{"prompt":"The capital of Germany is","capture":{"profile":"deep","layers":[12]} ,"display_top_k":10}
```

Public capture accepts only `profile` (`summary/overview/layer/attention/mlp/deep`), unique nonnegative integer `layers`, optional `tokens` and query `heads`. Summary/overview omit layer selections; detailed profiles require explicit layers. Only attention-capable profiles accept heads. Real tokenization and architectural bounds are checked before hooks/forward. No arbitrary Python path, sampling, chat, storage-dtype or internal raw/budget controls are exposed. Detailed plans retain selected-layer internals and all-layer base summaries; `overview` retains base raw states everywhere. Selected queries retain full key context.

## 10. Creation response schema

Default POST acknowledges after worker preflight/reservation with **202**, a Location and a compact envelope (`trace_id`, schema version, running status, timestamps, retained/reserved bytes, stored metadata, null overview/error). Queue waiting precedes allocation of the ID. A browser disconnect does not cancel it. `?wait=true` returns **201** with a ready overview, or 202 with a failed/cancelled envelope. GET uses the same envelope. The reserved queued state is not currently externally exposed before preflight.

Ready overview contains model/tokenizer/architecture metadata, raw input and first token page, resolved capture plan, scalar byte/timing summary, bounded predictions, actual layer/detail/component availability, provenance counts, existing validation counts/warnings/runtime, and lazy links. No raw arrays are embedded. Top 100 native candidates are retained during capture so all allowed logits GET counts use original tokens/probabilities without tokenizer access.

## 11. Component descriptor schema

`component_id`, `tensor_id`, `axes`, `logical_shape`, `storage_shape`, absolute `coverage`, observed/storage dtype/device, numel/reference bytes, provenance, sources, method, availability/raw availability/reason, and typed statistics. Statistics include finite/nonfinite/masked counts, nullable scalar reductions, original scope/accumulation dtype, their own SUMMARY classification, finite flag and undefined fields. Reference byte sums can include aliases; trace/store totals are unique storage.

## 12–13. Tensor slicing and response policy

Named selectors are either strict `index` or half-open `start/stop/step`; axes must match the registered descriptor. Token/head coordinates are absolute. The output `prediction_position` singleton uses local 0, explicitly mapped to the final input position by Stage 2 metadata. Omitted axes request all retained coverage. Fixed axes remain length 1. Unretained coordinates fail rather than using a storage offset or zero. Ambiguous aliased tensor IDs require their semantic `component_id`.

JSON contains preserved axes/logical shape, returned shape, compact coordinate maps, selectors, dtypes, source/provenance/method and flat row-major values. Maximum **65,536 elements** is checked before selection/conversion; maximum **1,048,576 encoded bytes** is checked before response delivery. Head response value counts are combined. Limits are configurable, never truncation policies.

Binary slices use the same selection/element guard and response byte limit, little-endian float32 and a maximum 8 KiB `X-Tensor-Metadata` header. Low-precision floats widen exactly; integer tensors use JSON to avoid ID rounding. JSON nonfinite values become null with flat index/kind flags; binary uses IEEE-754 encodings. Native mask flags are obtained from retained mask data; scores are never replaced with zero. Tests compare binary and JSON to the same stored values.

## 14. Error schema/status mapping

Errors have `error.code/message/details/retryable/request_id`; the header repeats request identity. Worker failure envelopes hold a safe error and remain readable at 200; detail requests are 409. No Python stack traces or cache paths are serialized.

| HTTP | Codes |
| --- | --- |
| 404 | TRACE_NOT_FOUND, COMPONENT_NOT_FOUND; deleted IDs |
| 409 | MODEL_NOT_READY (Stage 1 compatibility), TRACE_NOT_READY, COMPONENT_NOT_CAPTURED |
| 410 | TRACE_EXPIRED, TRACE_EVICTED |
| 413 | INPUT_LIMIT_EXCEEDED, CAPTURE_BUDGET_EXCEEDED, RESPONSE_TOO_LARGE |
| 422 | INVALID_REQUEST, INVALID_CAPTURE, INVALID_LAYER, INVALID_HEAD, INVALID_TOKEN, INVALID_SLICE, INVALID_CURSOR |
| 429 | QUEUE_FULL, with Retry-After |
| 503 | MODEL_LOAD_FAILED, EXECUTION_UNAVAILABLE, TRACE_STORE_BUDGET_EXCEEDED |
| 500 | INFERENCE_FAILED, TRACE_CAPTURE_FAILED |

Malformed JSON uses the predecessor's structured 422 mapping. Uncaptured components include retained availability and an explicit new-capture hint, with zero automatic forwards. Unknown semantic attributes never resolve. Active DELETE is 202 cancellation; terminal/repeated deletion is 204 while remembered, then GET 404. Running cancellation discards output at a safe boundary and releases reservation; it does not claim to interrupt GPU kernels.

## 15. Provenance preservation

Every descriptor/slice carries the original lower-case Stage 2 enum (`raw`, `derived`, `reconstructed`, `summary`), sources and method. Native weights remain RAW; post-RoPE Q/K and reconstructed weights/head mix remain RECONSTRUCTED; similarity/scaled/masked scores and final probabilities remain DERIVED. Statistic reductions independently remain SUMMARY. The real trace advertises {'raw': 191, 'reconstructed': 4, 'derived': 4}. Its 100 existing Stage 2 gates passed, with no failed gate. Routes never rerun their calculations.

## 16–19. Actual capital trace

Prompt: **The capital of Germany is**. Raw tokens are `[785, 6722, 315, 9856, 374]`, sequence length 5, all 28 layer summaries and detailed attention/MLP/residual Layer 12.

Trace ID: **`16ae9419-b56d-4849-a100-a069a2a4c8c0`**. The validation deleted it intentionally; the saved artifacts are the evidence, not a live retained record.

Model: Qwen/Qwen3-0.6B at immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`, MPS float32/eager/eval/cache disabled. Architecture remains 28 layers, hidden 1024, intermediate 3072, 16 query heads, 8 KV heads, head dimension 128 and vocabulary 151,936. Model and tokenizer revisions/digests are recorded in the real response.

Top prediction is token **19846**, native piece `ĠBerlin`, decoded ` Berlin`, raw logit **16.52149200439453**, full-vocabulary probability **0.47704339027404785**. The captured top 10 exactly equal the ordinary Stage 1 response, including logits/probabilities/pieces/ranks.

Creation HTTP time: **378.615 ms**; internal Trace Engine duration: **224.217 ms**; original forward duration: **161.481 ms**. Actual retained/store bytes: **1,970,124**. Reservation estimate was **2,031,564**, separately estimated transient instrumentation bytes **34,471,104**. After commit reserved bytes were zero; after DELETE stored/accounted bytes returned to zero.

## 20–22. Actual retrieval latency and payload evidence

| Request | Client HTTP elapsed ms | Uncompressed body bytes |
| --- | ---: | ---: |
| Create, wait=true | 378.615 | 5,644 |
| Overview | 26.218 | 5,644 |
| Layer 12 | 3.512 | 35,234 |
| Head 4, query token 3; weights and reconstructed Q | 3.128 | 12,925 |
| Residual token 3, hidden dimensions 0:128 (JSON) | 1.831 | 3,223 |
| Same residual slice (binary) | 6.327 | 512 |

These are single actual loopback requests using urllib to a running uvicorn process, not percentile/throughput benchmarks. Files are pretty-printed evidence; their filesystem sizes differ from compact transmitted body bytes. The attention head example reads query head 4 → KV head 2, one 5-key row and one 128-dimensional reconstructed Q vector. The residual slice reads token 3, dimensions 0:128. Full matrices/logits are absent.

## 23. Zero retrieval-forward proof

The resident adapter counter changed **1 → 2** during explicit trace creation, then remained **2** through overview/tokens/layers/layer/attention/head/MLP/residual/logits/component/relation/JSON/binary slices and uncaptured/oversize errors. DELETE/repeated DELETE/post-delete failure also left it at 2. Counter scope is full request adapter-forward attempts; load warmup and independent test reference are excluded. GET routes have no model/tokenizer access, and slice POSTs share the same read-only CPU path. Actual counters and all requests are in `stage3-http-validation.json`.

## 24. Baseline comparison and predecessor preservation

Actual HTTP Stage 1 top 10 and Stage 3 top 10 are identical. Both device suites also retain all original bitwise same-path Stage 1/Stage 2 logits/probabilities gates and independent HF reference tolerances. Original residual/MLP/attention reconstructions and hook cleanup pass. No tolerance, precision, attention implementation or model weights were changed.

## 25–28. Expiration, eviction, lease/concurrency and deletion results

PASS: fake-clock expiry denies access exactly at absolute TTL, with reads unable to renew it, safe background/synchronous sweeps, bounded tombstones and eventual 404. A real retained trace is exercised over HTTP with 410 while an existing lease still holds charged bytes.

PASS: LRU order responds to access; reservations evict only eligible records; over-budget requests preserve existing results/counters; active reservations/leases cannot be reclaimed; an unexpected capture overrun fails closed. Failed/cancelled records release reserved bytes.

PASS: eight concurrent leased readers survive insertion/deletion, final release reclaims storage, serializer failures release leases, and shutdown denies new reservations while retaining leased data safely. Real concurrent HTTP reads plus ordinary inference/trace creation observe a maximum of one simultaneous resident adapter forward. Selected query head/token coverage retains all keys and rejects omitted heads/rows without recapture.

PASS: actual terminal DELETE and repeat DELETE return 204; GET afterwards is structured 404, and actual store bytes return to zero. Real running HTTP cancellation returns 202, leaves a cancelled status, drops the completed result, releases reservation and leaks no hooks.

## 29–30. Automated MPS and CPU results

| Suite | Passed | Failed/errors/skipped | Duration |
| --- | ---: | --- | ---: |
| Full pinned Qwen MPS float32 | 224 | 0 / 0 / 0 | 22.456 s |
| Full pinned Qwen CPU float32 | 224 | 0 / 0 / 0 | 19.627 s |

Totals include the original 137 Stage 1/2 tests and 87 added Stage 3 cases. Of 224, 134 are foundation/storage/transport tests and 90 use real Qwen. Synthetic tensors test only transport/ownership, never substitute scientific inference. MPS and CPU processes ran serially. One existing Starlette/httpx TestClient deprecation warning remains; no failure or new dependency workaround was introduced. CUDA and low precision remain untested.

Reproduce from `apps/api`:

```sh
LLM_LAB_DEVICE=mps .venv/bin/pytest -q --real-model --junitxml=../../docs/validation/stage3-tests-mps.xml
LLM_LAB_DEVICE=cpu .venv/bin/pytest -q --real-model --junitxml=../../docs/validation/stage3-tests-cpu.xml
```

For actual network evidence, start uvicorn from `apps/api`, then run `.venv/bin/python -m llm_lab.trace_http_validation` in another terminal. The command records all requested bounded artifacts and does not overwrite Stage 1/2 reports.

## 31–32. Contract and code quality results

PASS: Pydantic OpenAPI export matches the checked-in snapshot; generated TypeScript check passes; Ruff check and format check pass; TypeScript no-emit, ESLint and Next.js webpack production build pass. Structured public schemas cover capture/envelope/model/input/statistics/descriptor/head/relation/logits/slice/status/token and component pages. Variable scientific validation records/structures remain dictionaries with existing Stage 2 names.

Typed client methods cover all trace operations, explicit JSON/binary slices, structured error details and 204 deletion. Mutations have no automatic retry. Binary methods return ArrayBuffer plus metadata; JSON methods restrict their format so return types cannot silently mismatch. No frontend page/component or dependency lockfile changed.

## 33. Known limitations

Local, single-process/worker, in-memory retention only; no persistence, authentication, pinning or snapshot pagination for a changing trace list. Byte budgeting covers Stage 2 owned tensor storage/reservations, not total RSS or Python metadata. Active native execution cannot be interrupted mid-kernel. There are no queued IDs before worker preflight, streaming events, ETags, chat templates, arbitrary-position output projections or candidate text decoding on GET. Binary transport is float32 for floating tensors only and is not native tensor export. Scientific validation metadata is retained evidence, not a new independent API scientific audit. Real timing evidence is one short prompt/selected layer; no large-context load test, CUDA or low-precision validation is claimed.

## 34. Stage 4 boundary

Stage 4 will build the horizontal Model Journey UI for input/tokens/embedding/28 layers/final norm/head/logits/prediction, using these real summaries and lazy endpoints. It will add global token/layer/component selection, pan/zoom/fit and keyboard navigation, breadcrumbs, an accessible context/inspector shell, stale-request handling and capture/provenance/expiry states. Detailed attention heatmaps, residual/MLP explorer graphics, token journey/logit evolution, generation and geometry remain their later assigned stages. **No Stage 4 implementation was started.**
