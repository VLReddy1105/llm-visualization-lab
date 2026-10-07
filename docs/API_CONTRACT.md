# API contract

Status: Stage 1 health, model lifecycle and raw next-token inference are implemented. Later-stage endpoints below remain design only. Base path: `/api/v1`. Trace schema planned: `1.0.0`. Pydantic models generate the implemented OpenAPI and TypeScript types. Examples below describe contracts and lifecycle states; they contain no invented scientific measurements.

Stage 1 also exposes compatibility aliases `GET /api/health`, `GET /api/model/status` and `POST /api/inference/next-token`. These aliases share canonical handlers and response schemas and are omitted from OpenAPI to avoid duplicate generated operations. `/api/v1/status` returns `inference_ready` plus the full model status; it has no trace-store fields yet. Model loading reports `loading_tokenizer`, `loading_weights`, `moving_to_device` and `smoke_check`; states remain `unloaded`, `loading`, `ready`, `failed`.

## General rules

JSON uses snake_case and ISO 8601 UTC timestamps. IDs are opaque UUIDs; `model_id` is a registry slug such as `qwen3-0.6b`, distinct from the Hugging Face ID. All tensor axes and index conventions follow [Trace design](TRACE_DESIGN.md). Unknown request fields are rejected. HTTP statuses and a typed error code communicate distinct conditions.

Stage 1 provides health/model loading and a minimal synchronous inference result. Stage 3 introduces asynchronous traces and lazy raw data. Derived-analysis and generation endpoints become available only in their assigned later stages. Schema examples are not promises that future endpoints are present in Stage 1.

Long inference runs are submitted to a bounded serialized executor; status endpoints stay responsive. Trace creation acknowledges with 202 and a Location header. Once ready, GET returns an overview, never a huge tensor dump. No retrieval request performs model inference. Model-dependent derived analysis can execute vocabulary projections from saved states but requires the exact matching resident weights; it never silently reloads or changes a trace.

## Endpoint inventory and ownership

| Endpoint | Response / purpose | Stage |
| --- | --- | --- |
| `GET /health` | Process liveness, API/schema versions | 1 |
| `GET /status` | Readiness, effective device/dtype, model state, queue/store budgets | 1; store fields added 3 |
| `GET /models` | Registry, load status, known versus verified capabilities | 1 |
| `GET /models/{model_id}` | Metadata, revisions, load state, error/capabilities | 1 |
| `POST /models/{model_id}/load` | 202 asynchronous load acknowledgment | 1 |
| `POST /inference` | 200 tokenization and actual last-position output summary; no persistent trace | 1 |
| `POST /traces` | 202 capture request acknowledgment | 3 |
| `GET /traces/{trace_id}` | 200 status and ready overview | 3 |
| `DELETE /traces/{trace_id}` | 202 active-job cancellation; 204 terminal-record deletion | 3 |
| `GET /traces/{trace_id}/tokens` | Paginated token metadata | 3 |
| `GET /traces/{trace_id}/layers` | Ordered paginated scalar summaries | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}` | Layer topology, summaries, component descriptors | 3 |
| `GET /traces/{trace_id}/components` | Paginated semantic component descriptors | 3 |
| `GET /traces/{trace_id}/components/{component_id}` | One descriptor; URL-encode brackets/dots as needed | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/attention` | Head counts/mapping, coverage, descriptors; no matrix | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/attention/heads/{head_index}` | Selected query head's summary and mapped KV index | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/attention/heads/{head_index}/relation` | Selected query/key relation, operands, full-row denominator and provenance | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/mlp` | Gated/non-gated topology, summaries, tensor descriptors | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/residual` | Checkpoint/update summaries and references | 3 |
| `GET /traces/{trace_id}/logits` | Bounded top candidates or explicit vocabulary IDs for a prediction position | 3 |
| `POST /traces/{trace_id}/tensors/{tensor_id}/slice` | Read-only bounded tensor retrieval with structured selectors | 3 |
| `GET /traces/{trace_id}/tokens/{token_index}/journey` | All-depth saved scalar summaries for one position | 6 |
| `POST /traces/{trace_id}/analyses` | 202 parameterized derived analysis | 6; kinds expanded later |
| `GET /traces/{trace_id}/analyses/{analysis_id}` | Analysis status, provenance, bounded result/pages | 6 |
| `POST /generations` | 202 bounded autoregressive run | 8 |
| `GET /generations/{generation_id}` | Status/settings/termination and paginated step mapping | 8 |
| `GET /generations/{generation_id}/steps/{generation_step}` | Decision record and associated trace ID | 8 |
| `POST /generations/{generation_id}/cancel` | 202 step-boundary cancellation | 8 |
| `POST /interventions` | 202 matched baseline/intervention experiment | 10 |
| `GET /interventions/{intervention_id}` | Status, specification, trace links, effect analysis | 10 |

Do not implement speculative endpoints ahead of their stage. Future trace export/import, persistent pinning, model unloading, and standalone recipe management require separate scoped contracts.

## Health, readiness, and model loading

`GET /health` returns 200 if the API process can respond; unloaded weights do not make it unhealthy. `GET /status` also returns 200 for a working service even if inference is not ready; clients inspect `inference_ready` and `model_state`. It includes allowed devices, selected device/dtype, queue occupancy/limit, execution state, trace-store byte budget, and active runtime limits once present.

Model states are `unloaded`, `loading`, `ready`, or `failed`. Loading reports its current phase (`resolving`, `downloading`, `initializing`, `moving_to_device`, `smoke_check`) and any real available progress; do not invent a percentage. A failure returns a safe diagnostic and retryability. `GET /models` never downloads weights.

Example load request:

```json
{
  "device": "auto",
  "dtype_policy": "scientific_default",
  "revision": null
}
```

This body is for `POST /models/qwen3-0.6b/load`. Null revision means resolve the configured revision once and record its immutable hash. A Stage 1 lock/config must then pin it. Allowed explicit device strings are `cpu`, `mps`, and `cuda:0`; configuration defines any later CUDA indices. Loading the same already-ready model/settings returns 200 with its status. A repeated identical in-progress load returns 202 referencing the existing operation. An incompatible load while execution is active returns 409 `MODEL_BUSY`.

State-shape example for an unloaded model:

```json
{
  "model_id": "qwen3-0.6b",
  "hf_model_id": "Qwen/Qwen3-0.6B",
  "state": "unloaded",
  "resolved_revision": null,
  "device": null,
  "dtype": null,
  "loading_phase": null,
  "capabilities_verified": false,
  "error": null
}
```

Inference requests require a ready model; otherwise return 409 `MODEL_NOT_READY` (503 `MODEL_LOAD_FAILED` after a failed load). They do not trigger loading. Stage 1's `POST /inference` accepts `model_id`, `input`, and bounded `top_k` (default 10, integers 1–100), following the explicit Stage 1 request. This is display count, with the same semantics as the originally planned `display_top_k`; later trace APIs retain their planned name. It returns model/runtime/input metadata, real tokens, last-position logits/probabilities for displayed candidates, omitted probability mass, and argmax preview. It does not claim internal tracing or sampled generation. The minimal frontend renders these actual results as readable text.

Implemented request example:

```json
{"model_id":"qwen3-0.6b","input":{"mode":"raw","text":"The capital of Germany is","add_special_tokens":false},"top_k":10}
```

The response contains `input.tokens` (indices, IDs, native pieces, decoded pieces, positions, special-token flags and character offsets), `prediction.top_k` (ranks, IDs, pieces, raw logits and full-vocabulary probabilities), final/next positions, omitted probability mass and runtime provenance. Stage 1 requests only last-position logits, reporting `[1, 1, V]`. The normalization policy is CPU float64 accumulation with float32 returned probabilities, separately recorded from model dtype. Unknown options, chat and special-token insertion are rejected. The generated [OpenAPI snapshot](../packages/shared/openapi.json) defines exact implemented field names.

## Create a trace

Example `POST /traces` request:

```json
{
  "model_id": "qwen3-0.6b",
  "input": {
    "mode": "raw",
    "text": "The capital of Germany is",
    "add_special_tokens": false
  },
  "capture": {
    "profile": "attention",
    "layers": [4],
    "query_heads": [3],
    "query_token_indices": "all",
    "storage_precision": "native"
  },
  "output": {"display_top_k": 10},
  "parent_trace_id": null
}
```

Attention/deep `layers` selects added detail; the overview's residual boundary capture still spans every layer by default. Request semantic details only, never Python paths. `query_heads` means query heads, not KV heads. Capturing a query row includes all its allowed keys. Head indices/sequence positions are validated after real tokenization and before forward execution.

`input.mode=chat` uses `messages` instead of `text` and declares template controls (`add_generation_prompt` and an adapter-supported thinking setting). Raw/chat inputs are mutually exclusive. Stage 1 raw-only support is allowed; unsupported chat requests fail explicitly until the documented stage adds them. Effective text/tokens and the applied template policy are returned.

Prefill uses cache disabled and the validated eager backend; device/dtype are those of the resident model. Advanced capture requests are preflighted against estimated bytes and capabilities. If they cannot fit, return a typed error and a narrower suggested capture. Input token limits are checked after tokenization; no implicit truncation.

Example 202 acknowledgment (`Location: /api/v1/traces/00000000-0000-4000-8000-000000000001`):

```json
{
  "trace_id": "00000000-0000-4000-8000-000000000001",
  "schema_version": "1.0.0",
  "status": "queued",
  "created_at": "2026-10-07T18:00:00Z",
  "completed_at": null,
  "expires_at": null,
  "overview": null,
  "error": null
}
```

UUIDs/times are illustrative contract values. Pending GET status uses the same envelope and an actual progress phase; it contains no provisional scientific result. A client polls with bounded backoff (initially about 500 ms while running, slower in background); the effective strategy is tuned in Stage 3. Stage 8 can add events if polling becomes insufficient.

Trace states are `queued → running → ready|failed|cancelled`. A ready record's `overview` is non-null. Its typed fields are:

| Overview field | Shape / meaning |
| --- | --- |
| `model`, `tokenizer`, `runtime` | IDs/revisions, architecture dimensions, actual software/backend/device/dtype settings |
| `input` | User/effective text metadata, token count, input mode/template policy |
| `tokens` | Bounded first page of TokenRecords; cursor when needed |
| `topology` | Semantic nodes/operations/edges, capability metadata |
| `layers` | Bounded first page of scalar LayerSummaries; cursor when needed |
| `components` | Bounded core descriptor list and manifest link |
| `output` | Prediction position, top candidate summary, omitted mass; chosen token nullable |
| `capture` | Requested/effective coverage, provenance and precision policy |
| `storage` | Retained unique bytes and current limits |
| `validation` | Lightweight invariant status; reference to compatibility/audit reports if present |
| `lineage` | Parent/generation/intervention association |

Each candidate summary contains `candidate_token_id`, tokenizer piece/display text, raw logit, full-vocabulary base probability, and rank. Actual numeric values are filled only by inference. `validation` must never label a trace scientifically validated solely because schema parsing passed.

## Lazy component and tensor retrieval

Overview-profile traces can honestly expose unsupported or uncaptured detail. Example descriptor fragment for `GET /traces/{id}/layers/4/attention/heads/3`:

```json
{
  "layer_index": 4,
  "head_index": 3,
  "kv_head_index": 1,
  "weights": {
    "component_id": "layer[4].attention.weights",
    "availability": "not_captured",
    "unavailable_reason": "not_requested",
    "tensor_id": null,
    "statistics": null
  }
}
```

The indices here demonstrate the verified query/KV mapping, not captured values. Descriptor GET succeeds with 200 for an existing component even when it is unavailable; trying to retrieve its values returns 409. A genuinely unsupported component can have an explicit descriptor; a nonexistent semantic component returns 404 `COMPONENT_NOT_FOUND`.

Layer/MLP/residual summaries accept an optional `token_index` and default to the final input position; their reduction scope is returned. Head summaries accept `query_token_index` where relevant. The relation endpoint requires `query_token_index` and `key_token_index`; it returns selected Q/K references, mapped KV head, dot product, actual scale/mask status, scaled/masked score, full-row normalization metadata, captured weight when available, and separately labeled reconstructed weight when requested/derivable. It uses retained operands and exact mask semantics only. A masked relation reports that status explicitly. Backend derivations power the mathematical inspector; the frontend does not reimplement attention.

Use a tensor reference returned by the manifest. Example read-only slice body for `POST /traces/{id}/tensors/{tensor_id}/slice`:

```json
{
  "selectors": {
    "batch": {"index": 0},
    "query_head": {"index": 3},
    "query_token": {"index": 4},
    "key_token": {"start": 0, "stop": 5, "step": 1}
  },
  "format": "json"
}
```

Named selectors use absolute original-axis coordinates, half-open ranges, positive steps, and either index or range per axis. Omitted axes request all retained coverage on that axis; they do not request unretained logical coordinates. Unknown axes, negative/out-of-range indices, invalid ranges, or unretained subsets are errors. Head/KV selectors apply only to the named tensor's axes.

A successful JSON slice returns `trace_id`, `tensor_id`, source component/provenance, logical and returned shape, remaining axes, fixed index selectors, absolute coordinate maps, observation/storage/wire dtypes, and a flat row-major `values` array. No example scientific values are supplied in this design. JSON number encoding widens native low-precision values as necessary and reports its wire dtype. Non-finite entries use JSON null with a parallel `nonfinite` list of flat index and kind (`nan`, `positive_infinity`, `negative_infinity`); masked score entries additionally carry mask status. Never turn a masked/null entry into zero.

`format=binary` returns contiguous little-endian float32 bytes with `Content-Type: application/octet-stream`. Headers include returned shape/axes, fixed selectors/coordinate map, original/storage/wire dtype, trace/tensor IDs, and content length; the matching descriptor supplies full provenance. Binary widening preserves the numeric value of native float16/bfloat16, but it is not a byte-for-byte native tensor export. Selectors must keep coordinate headers bounded. Imported/exported native tensor files are a separate future feature.

Default slice limits are 65,536 returned elements and 1 MiB uncompressed JSON; enforce whichever is reached first. Return 413 with limits and narrowing advice, never silently decimate or truncate. Binary is also bounded; binary transport does not authorize unbounded tensors. A slice request leases its source record until serialization completes and returns a stable immutable result.

Collection endpoints use `limit` (default 50, maximum 200) and an opaque `cursor`; responses have `items`, `next_cursor`, and total count when cheaply available. Trace overviews provide bounded first pages. Layer/token/head ordering is explicit and stable. Use ETag/conditional GET for immutable ready resources where useful; request keys include selectors so incompatible slices cannot collide.

## Logits, distributions, and analyses

`GET /traces/{id}/logits?token_index=4&display_top_k=10` returns retained output summaries for that prediction position. `candidate_token_ids` is a bounded list alternative; both query modes return base probabilities with their full-vocabulary normalization, not a candidate-only softmax. Defaults select the final input position. `display_top_k` is 1–100; it does not apply a sampling filter. Missing position logits return 409 unless an explicitly requested output-projection analysis is available.

Analysis kinds and stage ownership are `logit_lens` and `sampling_preview` (6), `pca`/`umap` (9), and `intervention_comparison` (10). Analyses are queued/read like traces, keyed by source IDs and full parameters; they cannot mutate source records. Example sampling-preview request:

```json
{
  "kind": "sampling_preview",
  "parameters": {
    "token_index": 4,
    "mode": "sample",
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.8,
    "display_top_k": 10
  }
}
```

These are analysis settings, not asserted predictions. Preview returns a bounded table of raw/base/processed values, filter eligibility, omitted mass, processor order, and method provenance; it does not draw a token. Greedy settings require canonical temperature 1, top_k 0, top_p 1 and use explicit mode `greedy`. Sampled settings require temperature > 0, top_k between 0 and vocabulary size, and top_p in (0,1]. Invalid combinations return 422.

Logit Lens requests specify checkpoint stages/layers, position, candidate IDs, `final_norm_then_lm_head` recipe, and requested metrics. PCA/UMAP requests declare fit population and reproducible parameters. Responses carry fit/method version and source references. Expensive output pages have their own pagination; they never include original high-dimensional arrays by default. Expired sources cannot be revived by analysis.

## Generation and intervention extension contracts

Stage 8 `POST /generations` accepts the same model/input/capture definitions, `max_new_tokens` (initial default 8, hard limit 32), frozen sampling settings, seed for sampled mode, and cache mode `disabled` initially. Return generation ID and queued status. Decision records contain zero-based step, context length, source trace ID, actual chosen vocabulary ID/piece, settings/RNG provenance, and termination reason (`eos`, `length_limit`, `cancelled`, `budget_limit`, or `failed`). Generation settings cannot edit an existing run; branch with a new request. Cache-enabled mode remains rejected until its validation gate passes.

Stage 10 intervention requests specify an existing compatible baseline or a new paired execution recipe, semantic target, position/head/dimension scope, intervention kind, and optional patch-source trace/component selectors. Initial head ablation targets query-head weighted-value output before output projection. Validate revisions, topology, dtype/shape, source availability, and paired budgets. Return experiment ID, baseline/intervention trace IDs when ready, controlled differences, and provenance. Incompatible source/target patches return 422; expired sources return 410.

## Errors and cancellation

Every error has `error.code`, a safe `message`, `details`, `retryable`, and `request_id`. Never return stack traces, raw tensor dumps, cache paths, or credentials. Example:

```json
{
  "error": {
    "code": "TRACE_EXPIRED",
    "message": "This trace has expired. Create a new trace to inspect the computation again.",
    "details": {"trace_id": "00000000-0000-4000-8000-000000000001"},
    "retryable": false,
    "request_id": "00000000-0000-4000-8000-000000000002"
  }
}
```

| HTTP | Codes / conditions |
| --- | --- |
| 404 | `TRACE_NOT_FOUND`, `MODEL_NOT_FOUND`, `COMPONENT_NOT_FOUND`, `ANALYSIS_NOT_FOUND`; invalid resource identity |
| 409 | `MODEL_NOT_READY`, `MODEL_BUSY`, `TRACE_NOT_READY`, `TRACE_FAILED`, `TRACE_CANCELLED`, `TENSOR_NOT_CAPTURED`, `CAPABILITY_UNSUPPORTED`, `MODEL_REVISION_NOT_RESIDENT` |
| 410 | `TRACE_EXPIRED`, `TRACE_EVICTED`, `TRACE_DELETED`; known removed record |
| 413 | `INPUT_LIMIT_EXCEEDED`, `CAPTURE_BUDGET_EXCEEDED`, `SLICE_LIMIT_EXCEEDED` |
| 422 | `INVALID_SELECTOR`, `INDEX_OUT_OF_RANGE`, `INVALID_SAMPLING_SETTINGS`, `PATCH_INCOMPATIBLE`; typed request validation |
| 429 | `QUEUE_FULL`, `STORE_CAPACITY_EXCEEDED`; include Retry-After when meaningful |
| 503 | `DEVICE_UNAVAILABLE`, `MODEL_LOAD_FAILED`, `EXECUTION_UNAVAILABLE`; working HTTP process cannot execute |
| 500 | `INFERENCE_FAILED`, `TRACE_CAPTURE_FAILED`; unexpected execution failure, safe diagnostic |

The worker records runtime errors in the trace envelope; GET status still returns 200 so clients can inspect failure. Requesting its scientific detail returns the corresponding 409. `DELETE` on a queued/running job returns 202 with `trace_id`, current status, and `cancellation_requested: true`; queued jobs become cancelled immediately and running jobs clean up at a safe boundary. The cancelled status record remains readable until its status TTL. `DELETE` on a terminal record returns 204, removes its data subject to active leases, and creates a deleted tombstone. No partial cancelled execution becomes a ready trace. Disconnecting a browser does not cancel server execution. Cancellation does not claim to interrupt an arbitrary GPU kernel.

## Expiration and retention

Ready traces expire 30 minutes after completion by default. Failed/cancelled status records have the same bounded status TTL but hold no raw tensors. Budget pressure may evict an unleased/unpinned ready trace sooner using oldest-last-access order; overview includes `retention_policy: ttl_or_capacity`. Reads update eviction recency without extending absolute expiry. Store tombstones for 24 hours after removal to return 410; afterwards, or after process restart with in-memory storage, an ID may return 404.

An active request lease delays physical removal, while new reads after expiry fail. Analyses inherit source expiry and leases during computation; analysis metadata must not keep tensors alive indefinitely. Active generation reserves capacity and stops on budget pressure; old historical steps can expire and show their unavailable state. There is no persistent pin endpoint initially.

Browser caches invalidate on 410, cancel dependent requests, and show expiry/eviction distinctly. They may retain already-rendered values only with an explicit stale label; no promise of further inspection. Saving a request recipe allows a new execution, not recovery of exact expired tensors. Local process restart discards traces.

## Operational limits and reproducibility

Initial settings come from [System architecture](SYSTEM_ARCHITECTURE.md#capture-memory-and-latency-budget) and are exposed by `/status`. Queue capacity initially allows one running execution and two queued jobs; Stage 2 measurements can revise this explicit configuration. Model weights and transient GPU/CPU execution allocations are accounted separately from retained-store bytes.

Bind locally by default, allow only configured local frontend origins, restrict model IDs to the adapter registry, and keep prompts/tensors out of default logs. Request IDs, phases, durations, byte counts, and safe errors support debugging. Remote multi-user deployment would require its own authentication/isolation scope.

Clients must not automatically retry trace/generation mutations, which could create duplicate real executions. Retrying reads and identical analysis lookups is safe. A later idempotency mechanism needs an explicit request fingerprint and contract. Schema/API version compatibility, immutable revisions, capture recipes, sampler settings, and method parameters are part of any future reproducible experiment export.
