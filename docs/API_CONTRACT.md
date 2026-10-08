# API contract

Status: Stages 1–3 are implemented. Trace Store, lazy trace APIs and typed client are validated on CPU and Apple MPS. Endpoints assigned to Stages 6–10 remain future designs. Base path: `/api/v1`; `/api/traces` is a shared-handler alias. Trace schema: `1.0.0`. Pydantic models generate the implemented OpenAPI and TypeScript types. Examples below describe contracts and lifecycle states; they contain no invented scientific measurements.

Stage 1 also exposes compatibility aliases `GET /api/health`, `GET /api/model/status` and `POST /api/inference/next-token`. These aliases share canonical handlers and response schemas and are omitted from OpenAPI to avoid duplicate generated operations. `/api/v1/status` returns `inference_ready` plus the full model status; it has no trace-store fields yet. Model loading reports `loading_tokenizer`, `loading_weights`, `moving_to_device` and `smoke_check`; states remain `unloaded`, `loading`, `ready`, `failed`.

## General rules

JSON uses snake_case and ISO 8601 UTC timestamps. IDs are opaque UUIDs; `model_id` is a registry slug such as `qwen3-0.6b`, distinct from the Hugging Face ID. All tensor axes and index conventions follow [Trace design](TRACE_DESIGN.md). Unknown request fields are rejected. HTTP statuses and a typed error code communicate distinct conditions.

Stage 1 provides health/model loading and a minimal synchronous inference result. Stage 3 provides asynchronous traces and lazy retained data, with an optional synchronous wait. Derived-analysis and generation endpoints become available only in their assigned later stages. Schema examples are not promises that future endpoints are present in Stage 1.

Long inference runs are submitted to a bounded serialized executor; status endpoints stay responsive. Trace creation acknowledges with 202 and a Location header. Once ready, GET returns an overview, never a huge tensor dump. No retrieval request performs model inference. Model-dependent derived analysis can execute vocabulary projections from saved states but requires the exact matching resident weights; it never silently reloads or changes a trace.

## Endpoint inventory and ownership

| Endpoint | Response / purpose | Stage |
| --- | --- | --- |
| `GET /health` | Process liveness, API/schema versions | 1 |
| `GET /status` | Readiness, effective device/dtype, model state, queue occupancy | 1 |
| `GET /models` | Registry, load status, known versus verified capabilities | 1 |
| `GET /models/{model_id}` | Metadata, revisions, load state, error/capabilities | 1 |
| `POST /models/{model_id}/load` | 202 asynchronous load acknowledgment | 1 |
| `POST /inference` | 200 tokenization and actual last-position output summary; no persistent trace | 1 |
| `POST /traces` | 202 after preflight/reservation; `?wait=true` returns 201 ready overview | 3 |
| `GET /traces` | Bounded active record metadata | 3 |
| `GET /traces/status` | Store counters, limits, request-forward counter | 3 |
| `GET /traces/{trace_id}` | 200 status and ready overview | 3 |
| `DELETE /traces/{trace_id}` | 202 active-job cancellation; 204 terminal-record deletion | 3 |
| `GET /traces/{trace_id}/tokens` | Paginated token metadata | 3 |
| `GET /traces/{trace_id}/layers` | Ordered paginated scalar summaries | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}` | Existing semantic layer structure, summaries, component descriptors | 3 |
| `GET /traces/{trace_id}/components` | Paginated semantic component descriptors | 3 |
| `GET /traces/{trace_id}/components/{component_id}` | One descriptor; URL-encode brackets/dots as needed | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/attention` | Head counts/mapping, coverage, descriptors; no matrix | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/attention/heads/{head_index}` | Selected query head's summary and mapped KV index | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/attention/heads/{head_index}/relation` | Retained similarity, scaled/masked scores, weights, native mask and provenance | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/mlp` | Gated/non-gated topology, summaries, tensor descriptors | 3 |
| `GET /traces/{trace_id}/layers/{layer_index}/residual` | Checkpoint/update summaries and references | 3 |
| `GET /traces/{trace_id}/logits` | Bounded top candidates for final prediction position; arbitrary vocabulary values through slices | 3 |
| `POST /traces/{trace_id}/components/{component_id}/slice` | Bounded component slice, JSON or binary | 3 |
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

Implemented request (`POST /api/v1/traces`, or `/api/traces`):

```json
{
  "prompt": "The capital of Germany is",
  "capture": {"profile": "deep", "layers": [12], "tokens": [0,1,2,3,4], "heads": [4]},
  "display_top_k": 10
}
```

`prompt` is raw text, with special-token insertion disabled and no implicit truncation. `capture` defaults to `summary`; `display_top_k` defaults to 10, strictly integer 1–100. Unknown fields, chat templates, sampling, Python paths and storage precision controls are rejected. The API does not expose internal `raw_tensors`, `statistics` or budget overrides.

Profiles preserve the existing Stage 2 semantics:

| Profile | Coverage |
| --- | --- |
| `summary` | All-layer statistics; final-position logits/probabilities retained |
| `overview` | Base raw states across every layer, plus statistics |
| `layer`, `deep` | Detailed attention and MLP in explicit `layers`; base summaries across all layers |
| `attention` | Detailed attention in explicit `layers`; base summaries across all layers |
| `mlp` | Detailed MLP in explicit `layers`; base summaries across all layers |

Detailed profiles require nonempty unique `layers`. Summary/overview omit `layers` and `heads`. Optional `tokens` select absolute input positions; optional `heads` select **query** heads and require an attention-capable profile. Omitted selections mean complete coverage for applicable axes. The real tokenizer determines token bounds before any observer/forward. Token selection narrows residual/MLP/query coverage; captured K/V and attention rows retain complete key context. Qwen3 has 16 query heads, 8 KV heads and 128 dimensions per head. Mapping comes from the stored adapter-produced `query_to_kv_head`.

Creation shares the existing resident model, tokenizer, semaphore and single worker with ordinary inference. Preflight checks input, selectors, Stage 2's execution peak estimate, and an atomic store reservation. Rejection returns a structured error without a forward. The default response is **202**, `Location: /api/v1/traces/{uuid}`, with status `running`, timestamps, reservation bytes and null `overview`. A queued executor request receives its identity only after preflight reaches the worker; no provisional science is returned. Browser disconnect does not cancel capture. Poll the Location with bounded backoff (about 500 ms initially). `?wait=true` waits for the same job and returns **201** with the ready overview, or 202 with a failed/cancelled envelope.

The envelope contains `trace_id`, `schema_version`, `status`, `created_at`, `last_access_at`, `completed_at`, `expires_at`, `retained_bytes`, `reserved_bytes`, compact stored `metadata`, `overview` and safe `error`. Observable states are `running → ready|failed|cancelled`; `queued` is reserved in the schema. Completion establishes absolute expiry. Ready overview contains:

| Field | Implemented contents |
| --- | --- |
| `model` | Model/tokenizer revisions, actual device/dtype and architecture/adapter digests |
| `input` | Raw text, sequence length and first paginated token page |
| `capture` | Resolved original Stage 2 plan and coverage |
| `summary` | 28-layer count, unique retained/estimated/transient bytes, capture/forward durations |
| `prediction` | Bounded original candidates, full-vocabulary probabilities and omitted mass |
| `available` | All layer indices, detailed attention/MLP layers, descriptor/raw counts, manifest URL |
| `provenance` | Counts by original value kind |
| `validation` | Existing Stage 2 gate counts and failure names; no new HTTP computation |
| `warnings`, `runtime`, `links` | Existing capture warnings/runtime, lazy endpoint links |

No raw tensor arrays, graph UI, analysis or lineage are included. API capture retains the original top 100 token candidates so logits GETs need neither weights nor tokenizer calls; the requested display count only controls presentation.

## Lazy component and tensor retrieval

Tokens preserve original IDs, native pieces, decoded text, absolute positions, special flags, character offsets and user origin. Layers expose structural references and captured statistics across all 28 layers. Layer/attention/MLP/residual endpoints contain descriptors and existing validation records, without default vectors or matrices. Residual statistics state their original reduction scope; a selected-token vector requires a slice. `referenced_bytes` per layer may include aliases; only store/trace totals are unique physical storage.

A component descriptor contains `component_id`, `tensor_id`, semantic `axes`, full `logical_shape`, retained `storage_shape`, absolute `coverage`, observation/storage dtype/device, `numel`, `byte_size`, `provenance`, `sources`, `method`, `availability`, `raw_available`, `unavailable_reason`, and `statistics`. Statistics keep their own `value_kind: summary`, finite/nonfinite/masked counts, population reduction scope, accumulation dtype, scalar fields, `finite`, and `undefined_fields`. Component provenance remains exactly `raw`, `derived`, `reconstructed`, or `summary`. Summary-only descriptor GET succeeds; its raw slice returns 409. Known detail omitted from the plan returns `COMPONENT_NOT_CAPTURED` with available retained components and an explicit new-capture hint. Unknown semantic names return 404. There is no automatic recapture.

Canonical component IDs use the existing syntax, e.g. `layer[12].residual.before` and `layer[12].attention.query.rotary`. URL-encode brackets. The manifest advertises actual descriptor IDs; arbitrary attribute paths never resolve.

Head GET defaults to metadata and mapped KV index. `include_weights=true` opts into the retained native attention weights; `include_query=true` opts into separately labeled reconstructed post-RoPE Q. `query_token_index` narrows that absolute row; omission includes retained query rows only. All arrays remain bounded. The relation endpoint requires `query_token_index` and `key_token_index` and reads scalar slices of already-retained similarity/scaled/masked scores/native weights/reconstructed weights and native mask. It does not perform a dot product, rebuild a mask or normalize a row.

Structured read-only slice example:

```json
{
  "component_id": "layer[12].residual.before",
  "selectors": {
    "batch": {"index": 0},
    "token": {"index": 3},
    "hidden": {"start": 0, "stop": 128, "step": 1}
  },
  "format": "json"
}
```

POST this to `/traces/{id}/components/{component_id}/slice` or `/traces/{id}/tensors/{tensor_id}/slice`. Tensor IDs can alias components with different axes (head output versus concatenation); an ambiguous tensor ID **requires** `component_id`, matched against the manifest. Component routes need no duplicate body ID. Indices are strict nonnegative integers, ranges are nonempty half-open with positive steps, unknown axes are rejected, and token/head selectors refer to original absolute coordinates. The last-position `prediction_position` singleton uses local index 0, mapped to the final input position by the Stage 2 method/trace output metadata. Omitted axes mean all **retained** coverage. Unretained coordinates are 409, never substituted. Fixed-index dimensions are retained with size 1, preserving all semantic axes.

JSON response contains source IDs/provenance/method, logical/returned shape, axes, selectors, compact coordinate maps (`count,start,stop,step` or `indices`), observation/storage/wire dtypes, flat row-major `values`, `nonfinite`, and `masked_flat_indices`. Finite masked-score minima stay unchanged; mask flags come from the retained native mask. Nonfinite numbers become JSON null, with flat index and `nan`, `positive_infinity` or `negative_infinity` in `nonfinite`. Nonfinite/undefined statistic fields are null and explicitly flagged. No invalid JSON NaN/Infinity tokens are emitted.

`format=binary` returns contiguous little-endian float32 with `application/octet-stream`. `X-Tensor-Metadata` carries shape, axes, coordinate maps, selectors, IDs, source provenance and original/storage/wire dtype (maximum 8 KiB). Float16/bfloat16 widening preserves values. IEEE-754 nonfinite values remain in the binary data; mask information is available through its registered component. Integer tensors use JSON, avoiding silent ID rounding. Browser CORS exposes the metadata and Location headers. The typed client returns an ArrayBuffer plus metadata.

Both transports enforce **65,536 elements** before tensor selection/value conversion and **1 MiB encoded response bytes**, configurable via `LLM_LAB_TRACE_API_MAX_ELEMENTS` and `LLM_LAB_TRACE_API_MAX_RESPONSE_BYTES`. Head responses enforce the combined requested value count. JSON bytes are checked after strict UTF-8 encoding; oversize requests return 413 `RESPONSE_TOO_LARGE`, with limits and narrowing advice. Never truncate. Serialization finishes under a CPU read lease and returns owned bytes before releasing it.

Collection endpoints use `limit` (default 50, maximum 200) and opaque offset `cursor`, returning `items,total,next_cursor`. Tokens/layers/components are immutable and ordered; trace listing is ordered by creation time/UUID and reflects the active set, so concurrent deletion/expiry can shift its pages. No snapshot-pagination guarantee or ETag is claimed.

## Logits, distributions, and analyses

`GET /traces/{id}/logits?display_top_k=10` returns final-position descriptors and 1–100 original ranked candidates, with full-vocabulary probabilities and descriptive omitted mass. Optional `token_index` must be the captured final prediction position; another valid input position returns 409, out-of-bounds is 422. Arbitrary vocabulary values use explicit bounded logits/probability slices. No all-position LM-head projection, candidate-only softmax or sampling runs here.

The following analyses remain future scope. Analysis kinds and stage ownership are `logit_lens` and `sampling_preview` (6), `pca`/`umap` (9), and `intervention_comparison` (10). Analyses are queued/read like traces, keyed by source IDs and full parameters; they cannot mutate source records. Example sampling-preview request:

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

| HTTP | Implemented trace conditions |
| --- | --- |
| 404 | `TRACE_NOT_FOUND`, `COMPONENT_NOT_FOUND`; includes deleted IDs |
| 409 | `MODEL_NOT_READY` (preserved Stage 1 mapping), `TRACE_NOT_READY`, `COMPONENT_NOT_CAPTURED` |
| 410 | `TRACE_EXPIRED`, `TRACE_EVICTED` while bounded tombstones remain |
| 413 | `INPUT_LIMIT_EXCEEDED`, `CAPTURE_BUDGET_EXCEEDED`, `RESPONSE_TOO_LARGE` |
| 422 | `INVALID_REQUEST` for schema validation; `INVALID_CAPTURE`, `INVALID_LAYER`, `INVALID_HEAD`, `INVALID_TOKEN`, `INVALID_SLICE`, `INVALID_CURSOR` |
| 429 | `QUEUE_FULL`, with Retry-After |
| 503 | `MODEL_LOAD_FAILED`, `EXECUTION_UNAVAILABLE`, `TRACE_STORE_BUDGET_EXCEEDED` |
| 500 | `INFERENCE_FAILED`, `TRACE_CAPTURE_FAILED` unexpected execution errors |

Malformed JSON also uses the existing structured 422 `INVALID_REQUEST` schema. A worker failure after acknowledgment becomes a safe failed envelope; status GET remains 200 and detail reads return 409. The error contains code, message, details, retryable and request ID; failure envelopes omit the per-read request ID but response headers retain it. No Python stack, weights or paths reach HTTP.

DELETE of a running job returns 202 `{trace_id,status:"cancelled",cancellation_requested:true}`. Cancellation prevents storage of its result, releases reservations on safe completion, and leaves a bounded cancelled status record. It does not interrupt a native kernel. Terminal DELETE returns 204, marks inaccessible immediately, and reclaims data after readers release. Repeat DELETE returns 204 while its deleted tombstone remains; subsequent GET returns 404.

## Expiration and retention

`LLM_LAB_TRACE_TTL_SECONDS=1800` is absolute after completion for ready/failed/cancelled records. Reads update last access and LRU order without renewing TTL. Lookup and reservation sweep expiry synchronously; a 15-second configurable background sweep also cleans idle records. All mutations/counters use an RLock. A lease increments under that lock; CPU serialization occurs outside it. Expiry/deletion deny new reads immediately and defer physical removal while leased. Deferred bytes remain charged.

`LLM_LAB_TRACE_STORE_MAX_BYTES=536870912` covers unique Stage 2 CPU storage plus outstanding retained-byte reservations. Reservations occur before the existing forward; completion replaces conservative estimates with actual alias-deduplicated bytes. Eligible oldest-last-access terminal records are evicted under capacity pressure; active jobs/readers cannot be evicted. If no sufficient eligible capacity exists, return 503. Rejected reservations do not discard otherwise viable traces. Per-trace execution peak (`LLM_LAB_TRACE_MAX_BYTES`, default 256 MiB), model weights and allocator/RSS are separate budgets. The store does not claim a total-process RSS bound.

Metadata is bounded by 1,024 entries; up to 2,048 removed-ID tombstones live at most 24 hours. Expired/evicted access returns 410 while remembered, then 404. Deleted access returns 404. Restart discards all traces. Store shutdown defers leased/running reclamation safely. No persistence, pinning, Redis or database is implemented.

`GET /traces/status` exposes active count, actual retained/reserved/accounted bytes, limits/TTL, evictions/expirations, active leases/deferred removals/tombstones, and cumulative full request-forward attempts. The latter excludes the load warmup and independent test reference; it is incremented immediately before the resident adapter forward. Retrieval has no tokenizer/weight access and never increments it.

## Operational limits and reproducibility

Initial settings come from [System architecture](SYSTEM_ARCHITECTURE.md#capture-memory-and-latency-budget) and are exposed by `/traces/status` for retention/payload settings and `/status` for model readiness. Queue capacity initially allows one running execution and two queued jobs; Stage 2 measurements can revise this explicit configuration. Model weights and transient GPU/CPU execution allocations are accounted separately from retained-store bytes.

Bind locally by default, allow only configured local frontend origins, restrict model IDs to the adapter registry, and keep prompts/tensors out of default logs. Request IDs, phases, durations, byte counts, and safe errors support debugging. Remote multi-user deployment would require its own authentication/isolation scope.

Clients must not automatically retry trace/generation mutations, which could create duplicate real executions. Retrying reads and identical analysis lookups is safe. A later idempotency mechanism needs an explicit request fingerprint and contract. Schema/API version compatibility, immutable revisions, capture recipes, sampler settings, and method parameters are part of any future reproducible experiment export.
