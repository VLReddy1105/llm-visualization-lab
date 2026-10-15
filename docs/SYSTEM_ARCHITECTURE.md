# System architecture

Status: Stage 0 blueprint with Stages 1–7 implemented. The [Stage 3 report](validation/STAGE_3_REPORT.md) records the bounded trace store and lazy HTTP contract. Stages 4–5 add the horizontal Model Journey and Layer Microscope; the [Stage 6 report](validation/STAGE_6_REPORT.md) records saved-state evolution, diagnostic Logit Lens and sampling preview. Stage 2 capture infrastructure remains intact. Later-stage sections remain architectural plans.

## Decisions and boundaries

Use the requested Next.js / React / TypeScript / Tailwind CSS frontend and Python / FastAPI / PyTorch / Transformers / Pydantic backend. Model execution stays in Python. The browser renders semantic traces and changes selection; scientific transformations and sampling distributions are computed server-side.

The most important boundary is the **versioned semantic trace contract between model-specific execution and all visualizations**. A Trace Engine captures once, a Trace Store retains immutable CPU data, and every view asks for bounded portions of the same record. The frontend never uses Python module names or assumes a particular head width, layer count, normalization, or residual topology.

Begin as a local single-user application, with one resident model, one inference executor, and one ASGI worker. Keep inference off the asynchronous HTTP event loop. A serialized executor holds a model execution lock across loading, tracing, instrumentation, model-weight-dependent analyses, and interventions. This prevents overlapping hooks from contaminating runs and prevents model replacement during a vocabulary projection. Multiple web-server workers would each load weights and have separate in-memory stores; they are deliberately excluded until a shared execution/storage architecture exists.

## A. Complete system architecture

```mermaid
flowchart LR
  subgraph Browser[Browser: Next.js workspace]
    Controls[Prompt and generation controls]
    Selection[Zustand selection state]
    Queries[TanStack Query API cache]
    Views[Model Journey and microscope views]
    Controls --> Queries
    Selection --> Views
    Queries --> Views
    Views --> Selection
  end
  subgraph Backend[Python: FastAPI]
    API[Versioned API and Pydantic contracts]
    Manager[Model manager and serialized executor]
    Engine[Trace Engine and capture planner]
    Adapter[Model Adapter registry]
    Model[PyTorch and Transformers model]
    Collector[TraceCollector]
    Store[Trace Store: CPU tensors and metadata]
    Analysis[Derived analysis and sampling service]
    API --> Manager
    Manager --> Engine
    Engine --> Adapter
    Adapter --> Model
    Model --> Collector
    Collector --> Store
    API --> Store
    Store --> Analysis
    Analysis --> API
  end
  Queries <-->|HTTP JSON and bounded tensor slices| API
  Contract[OpenAPI schema and generated TypeScript types]
  API -.-> Contract
  Contract -.-> Queries
```

Arrows show responsibility and data movement, not sequential execution time. All scientific views share `trace_id` and component descriptors.

## Recommended future layout

```text
/
├── apps/
│   ├── web/
│   │   ├── src/app/                       # Next.js shell and routes
│   │   ├── src/features/                  # journey, layer, attention, etc.
│   │   ├── src/state/selection.ts         # linked selection and viewport
│   │   ├── src/lib/api/                   # typed client and query keys
│   │   └── tests/                         # interactions and browser checks
│   └── api/
│       ├── pyproject.toml
│       ├── src/llm_lab/
│       │   ├── main.py
│       │   ├── api/                       # v1 routes and error mapping
│       │   ├── schemas/                   # Pydantic contracts, authoritative
│       │   ├── models/                    # manager, devices, adapters/
│       │   ├── tracing/                   # engine, collector, capture plans
│       │   ├── storage/                   # trace store, leases, expiration
│       │   ├── analysis/                  # metrics, lens, projections
│       │   ├── generation/                # step executor and sampler
│       │   └── interventions/             # later controlled modifications
│       └── tests/                         # numerical, contract, lifecycle
├── packages/shared/
│   ├── openapi.json                       # generated API snapshot
│   └── src/generated/                    # generated TypeScript types
├── docs/                                  # current blueprint
└── README.md
```

This separates Python and JavaScript toolchains while sharing the contract, not runtime model code. Pydantic/OpenAPI is authoritative; do not hand-maintain a second TypeScript schema. A minimal npm workspace can manage `apps/web` and `packages/shared`; Python has its own environment and lockfile. No Nx/Turborepo, Docker, database, distributed queue, or chart collection is needed initially.

## Frontend responsibilities and libraries

| Choice | Purpose | Adoption boundary |
| --- | --- | --- |
| Next.js, React, TypeScript | Workspace shell and typed components | Stage 1 |
| Tailwind CSS | Consistent white-dominant layout and typography | Stage 1 |
| Zustand | One linked selection and viewport store | Stage 4 |
| TanStack Query | Future visualization polling/cache integration; Stage 3 exposes a typed fetch client only | Stage 4 |
| SVG / Canvas | Accessible architecture diagrams; dense heatmap rendering | Stages 4–5 |
| OpenAPI type generator | Prevent Python/TypeScript contract drift | Stage 1, expanded Stage 3 |
| scikit-learn; umap-learn | Server-side PCA; later UMAP | Stage 9 only |

React Flow is optional if graph editing/layout becomes materially complex; the fixed horizontal journey starts with custom SVG and DOM labels. Add D3 utilities or Plotly only for a specific demonstrated need. Add Three.js / React Three Fiber only for an approved later 3D scope. Avoid mounting a large visualization bundle for basic navigation.

Server state contains trace records, tensor descriptors, model status, and analysis results. Client state contains token/layer/head/component/candidate/step selection, visualization mode, viewport, and inspector depth. Never put large tensor arrays in Zustand or duplicate API data in it. Components query by trace identity and slice parameters.

## Backend responsibilities

- **API layer:** validate bounds, expose status, return summaries, resolve availability, enforce payload limits, and map structured errors. It does not know model module paths.
- **Model manager:** resolve revisions, own loading state, device/dtype and resident weights, execution lock, and baseline inference.
- **Model Adapter:** map modules to semantic operations, declare capabilities/topology, prepare model-specific instrumentation, and execute through the real model.
- **Trace Engine:** validate a capture plan, reserve memory, establish a run context, execute the adapter, finalize an immutable trace, and guarantee cleanup.
- **TraceCollector:** snapshot requested values and summaries without changing the forward computation; canonicalize axes and provenance.
- **Trace Store:** hold metadata and detached CPU tensors, deduplicate aliases, account bytes, expire records, and provide request leases.
- **Analysis service:** compute named metrics, vocabulary projections, sampling distributions, and later geometry/intervention comparisons from available data.

Adapter responsibilities can be expressed as `metadata`, `load_model`, `tokenize`, `forward`, `component_map`, `capabilities`, `prepare_instrumentation`, and `generate_step`. If an adapter exposes `trace`, it delegates orchestration to the Trace Engine; there must be no second per-view inference pipeline. `prepare_instrumentation` returns a context manager with run-scoped handles and cleanup.

## Verified Qwen3 structure

Structural inspection used official sources on 2026-10-07. The checkpoint's [configuration](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json) reports:

| Property | Value |
| --- | --- |
| Decoder layers | 28, indexed 0–27 |
| Residual width `D` | 1,024 |
| Query heads `Hq`; KV heads `Hkv` | 16; 8 |
| Head width `d` | 128 |
| MLP intermediate width `I` | 3,072 |
| Vocabulary size `V` | 151,936 |
| Activation | SiLU |
| RMSNorm epsilon | 1e-6 |
| RoPE theta | 1,000,000 |
| Attention bias / dropout | false / 0.0 |
| Tied input/output embeddings | true |
| Sliding window | disabled |

The [model card](https://huggingface.co/Qwen/Qwen3-0.6B) lists a 32,768-token context, while config `max_position_embeddings` is 40,960. Preserve both facts in metadata rather than declaring either to be a proven tracing limit. Local capture limits below are much smaller. The card identifies Qwen3 support from Transformers 4.51.0 onward and describes chat-template thinking controls. Stage 1 must pin a currently compatible release and exact model/tokenizer revision; the checkpoint's historical `transformers_version` is not a dependency lock.

The [versioned Qwen3 implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py) defines sequential pre-normalized attention and gated MLP updates:

```text
r_in → RMSNorm → Q/K/V projections
Q/K → per-head RMSNorm → RoPE
Q,K,V → grouped-query causal attention → concatenation → output projection = attention update
r_mid = r_in + attention update
r_mid → RMSNorm → SiLU(gate projection) × up projection → down projection = MLP update
r_out = r_mid + MLP update
last r_out → final RMSNorm → LM head → logits
```

The [current implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py) uses different output-capture and cache plumbing. The adapter must inspect and test the installed release rather than copy signatures from a historical tag. Neither source is vendored or executed during Stage 0.

### Adapter mapping and capture points

All paths in this table are Python adapter concerns. The canonical vocabulary is in [Trace design](TRACE_DESIGN.md).

| Qwen module / observation | Semantic component | Strategy |
| --- | --- | --- |
| `model.embed_tokens` output | `embedding.output` | Forward hook |
| `model.layers[i]` input/output | `layer[i].residual.before` / `.after_mlp` | Pre/forward hooks; handle tuple/tensor release differences |
| `input_layernorm` output | `layer[i].attention.input_normalized` | Forward hook |
| `self_attn.q_proj`, `.k_proj`, `.v_proj` outputs | `.attention.query.projected`, `.key.projected`, `.value` | Hooks and canonical reshape |
| `self_attn.q_norm`, `.k_norm` outputs | `.attention.query.normalized`, `.key.normalized` | Hooks; retain named head axes |
| Actual attention-function inputs | `.attention.query.rotary`, `.key.rotary`; mask, scale | Scoped observation wrapper for functional boundary |
| Actual attention-function output weights | `.attention.weights` | Eager output or scoped wrapper; capability-dependent |
| `self_attn.o_proj` input/output | `.attention.head_output` / `.concatenated` / `.output` | Pre/forward hooks, alias views |
| `post_attention_layernorm` input/output | `.residual.after_attention` / `.mlp.input_normalized` | Pre/forward hooks |
| `mlp.gate_proj`, `.up_proj` outputs | `.mlp.gate`, `.up` | Forward hooks |
| `mlp.down_proj` input/output | `.mlp.gated` / `.mlp.output` | Pre/forward hooks |
| SiLU result before product | `.mlp.activated_gate` | Functional observation if supported; otherwise a labeled derivation |
| `model.norm` input/output | `output.residual_final` / `output.normalized` | Hooks; separate pre-final-norm state |
| `lm_head` output | `output.logits` | Model output or forward hook; requested positions only |

Do not infer `d = D/Hq`: here `Hq*d` is 2,048, so query projection and concatenated head output are wider than the residual stream. GQA query head `h` uses KV head `floor(h/2)` for this configuration. K/V stay stored in their compact KV-head form. Q/K normalization precedes rotation; V has neither operation. RoPE is an attention positional transformation, not an additive token embedding.

Scores, masking, softmax, residual additions, and SiLU may be functional operations without ordinary module hooks. Capture actual boundaries where supported; otherwise derive from captured operands and label the method. Ordinary hooks alone do not expose every intermediate. A reconstructed attention matrix from a fused run is a derivation, not its captured attention weights. [Attention backend documentation](https://huggingface.co/docs/transformers/en/attention_interface) is a release-specific implementation reference.

## B. Inference and tracing data flow

```mermaid
sequenceDiagram
  participant UI as Workspace
  participant API as Trace API
  participant E as Trace Engine
  participant A as Model Adapter
  participant C as TraceCollector
  participant S as Trace Store
  UI->>API: POST traces with prompt and capture plan
  API->>E: Validate, reserve budget, enqueue
  API-->>UI: 202 trace_id and queued status
  E->>A: Acquire execution lock and attach observers
  A->>A: Real model forward in inference mode
  A->>C: Requested observations and provenance
  C->>C: Detach, snapshot to CPU, summarize, deduplicate
  A-->>E: Real output logits and runtime metadata
  E->>A: Remove observers and restore instrumentation
  E->>S: Atomically commit immutable ready trace
  UI->>API: GET trace overview
  API->>S: Read metadata under lease
  API-->>UI: Tokens, summaries, descriptors, output summary
  UI->>API: GET selected component slice
  API->>S: Read retained bounded tensor slice
  API-->>UI: Values, axes, precision, availability
```

Failures remove observers and release reservations before publishing a failed record. A slice read never initiates model inference. Selecting a component not retained offers a new capture request referencing the previous trace, without modifying it.

## C. Frontend interaction flow

```mermaid
flowchart TD
  Run[Run prompt] --> Pending[Poll trace status]
  Pending --> Ready[Load immutable overview]
  Ready --> Map[Model Journey]
  Map --> Select[Update global selection]
  Select --> Validate[Validate selection against topology and coverage]
  Validate --> Mode[Resolve context panel and microscope depth]
  Mode --> Key[Query key: trace, component, axes, analysis parameters]
  Key --> Cached{Retained data available?}
  Cached -->|yes| Fetch[Fetch bounded detail or cached result]
  Cached -->|no| Missing[Explain missing data and offer new capture]
  Fetch --> Linked[Render synchronized views and provenance]
  Linked --> Select
  Step[Generation step selection] --> Ready
```

`selectedToken` is an input position, `selectedCandidateToken` is a vocabulary ID, and neither is a token string. Generation selection resolves to a trace before dependent queries run. [Visualization workflow](VISUALIZATION_WORKFLOW.md) specifies reset and persistence rules.

## Device and execution policy

Auto selection tries available CUDA, then Apple MPS, then CPU; an explicit device choice overrides auto. CPU and MPS begin with float32 for a conservative scientific baseline. CUDA can use bfloat16 after capability checks, with a float32 reference path; float16 requires its own validation profile. MPS lower precision is enabled only after an explicit capability smoke test and numerical checks. Record effective choices and fallback reasons.

Use `eval()` and `torch.inference_mode()`, batch size 1, eager attention for the initial scientific trace path, and full-prefix cache-disabled generation in Stage 8. Disable compilation, quantization, remote model code, and kernel substitutions in the validated baseline. Fast fused attention and KV-cache generation can later become separately validated modes. Switching backends changes execution provenance and requires a separate baseline comparison. A failed run never silently becomes a successful trace on another device.

Stage 1 implementation note: final-vocabulary softmax accumulates on CPU in float64 before returning float32 probabilities. The measured CPU float32-only normalization exceeded the scientific accuracy gate; the correction and exact discrepancy are recorded in [Scientific validation](SCIENTIFIC_VALIDATION.md). This output calculation does not change model execution or logits.

## Capture, memory, and latency budget

Lazy retrieval reduces network and browser work; it does not make backend capture free. Estimate retained bytes from each unique tensor's shape and storage dtype, plus transient attention scores, copy buffers, model weights, and allocator headroom. Deduplicate layer output / next layer input and view aliases. Do not retain autograd graphs, model outputs, or device tensors inside the Trace Store.

For full attention weights, bytes = `L * Hq * N² * bytes_per_element`, batch 1. Using the verified configuration yields these **capacity estimates, not measured benchmarks**:

| Sequence length | All-layer attention at 2 bytes | At 4 bytes |
| --- | --- | --- |
| 128 | 14 MiB | 28 MiB |
| 512 | 224 MiB | 448 MiB |
| 1,024 | 896 MiB | 1.75 GiB |
| 2,048 | 3.5 GiB | 7 GiB |

Eager execution may materialize full scores even when only selected heads are retained. Retention selection alone does not reduce its peak N² allocation. Avoid requesting all `output_attentions` when per-layer streaming capture suffices. Full-position logits cost `N * V * bytes`; at N=512 and float32 this is about 296.75 MiB. Keep last-position vocabulary logits by default and derive other positions on demand from retained final normalized states while the matching weights are available. Ordinary inference should request only necessary logits where the installed release permits it.

Implemented configurable limits: 128 input tokens by default (configuration ceiling 512), 256 MiB conservative per-capture extra tensor peak, 512 MiB aggregate retained/reserved trace bytes, 30-minute absolute TTL, 65,536 elements per slice, and 1 MiB encoded response. These bound capture/store/transport responsibilities, not total process RSS or hardware throughput. Capture plans exceeding estimated budgets fail before execution with a smaller-plan suggestion. Do not silently truncate inputs or lower capture coverage.

Stage 2 applies the existing prompt limit and a 256 MiB default capture cap (`LLM_LAB_TRACE_MAX_BYTES`, configurable up to 1 GiB). It rejects a conservative extra-tensor peak estimate before registering any observer: estimated retained snapshots plus temporary capture/reconstruction/statistics work. The temporary estimate allows sixteen copies of selected operands, increased for a larger GQA grouping ratio. This includes float64 audit scratch and alias overestimation; it is a bound for instrumentation tensors, not model weights, the backend allocator or exact process RSS. Actual retained bytes count unique owned CPU storage. Summary releases per-layer temporary snapshots immediately after residual reconciliation. Selected detailed states survive only through reconstruction and validation, then the run releases its workspace. Stage 3 adds aggregate retained-byte reservation/accounting and TTL without changing this capture calculation.

Detailed Q/K/V variants and four MLP intermediates scale with sequence length and layers and can exceed attention storage for short prompts. GPU-to-CPU copies can synchronize execution; repeated tiny hooks add latency. Record capture overhead and peak accelerator/host memory separately. A browser must virtualize long layer/token lists and use Canvas for dense heatmaps rather than one React element per cell.

## Trace Store lifecycle and extensibility

Implemented modules are `store/trace_store.py`, `tracing/service.py`, `tracing/serialization.py`, `schemas/trace_api.py`, and `api/traces.py`. `TraceService` submits preflight/reservation/capture to the existing one-thread `ModelService` executor. It consumes the unchanged Stage 2 engine, assigns a UUID4 metadata identity and commits an immutable TraceResult. It never loads another model. API capture keeps 100 original candidates to serve bounded output reads without tokenizer access. A full-request forward counter makes lazy behavior auditable.

`TraceStore` protects entry maps, LRU order, tombstones, leases and byte counters with an RLock. It reserves conservative retained bytes before forward execution, then charges Stage 2's actual unique CPU storage at commit. Jobs and leased records cannot be capacity-evicted. Eligible terminal records are reclaimed oldest-last-access first, or creation fails if no adequate capacity exists. Capacity failures do not silently trim a capture. Model weights, Python metadata and transient execution allocations are accounted separately.

A lease is acquired under the store lock; CPU serialization runs outside it. Expiry/deletion denies new readers immediately; an existing reader retains its data until release, and deferred bytes remain charged. TTL is absolute from completion and reads only update eviction recency. Synchronous lookup/reservation sweeps plus a 15-second background sweep reclaim idle records. Active-job DELETE marks cancellation, with safe completion discarding its result. Status records and tombstones are bounded (1,024 entries; 2,048 tombstones, at most 24 hours). Restart discards storage; run one uvicorn worker.

The dedicated serializer preserves observed axes/coverage, provenance/method/sources and Stage 2 statistics/checks. Scalar or vector retrieval performs only selection/presentation; no attention, MLP, residual or LM-head recomputation lives in the routes. JSON uses null plus nonfinite flags, with exact encoded-byte checks. Binary slices widen floating values to bounded little-endian float32 and return compact metadata headers; integer observations use JSON. Component aliases require semantic disambiguation when addressing a shared tensor ID. Defaults and endpoints are specified in [API contract](API_CONTRACT.md).

The actual capital trace retained 1,970,124 unique bytes. HTTP instrumentation recorded one capture forward and zero extra retrieval forwards; all stored bytes returned to zero after DELETE. Both devices pass the 224-test Stage 1–3 suite. The typed client and generated contracts are ready for Stage 4; no visualization, topology renderer, persistence or generation pinning is implemented.

Later disk storage can implement the same interface using tensor blobs and a metadata index. Export/import requires a schema version, checksums, provenance, safe non-executable data formats, and import bounds. A recipe can recreate an expired trace but cannot restore its exact values without saved tensor data. Model-dependent derivations require the matching weights to remain resident or be explicitly loaded; store access alone must never trigger loading.

Additional adapters provide a semantic operation graph with ordered operations, named tensors, optional branches, residual topology, and capabilities. This accommodates LayerNorm, parallel residuals, gated/non-gated MLPs, sliding attention, or later MoE nodes without pretending every architecture is Qwen. Cross-model token/coordinate identities are incompatible by default. SAE feature views require separately identified feature models and evidence; placeholders are not activations.

## Principal risks and resolution gates

| Risk | Resolution before dependent features |
| --- | --- |
| Release-specific instrumentation and fused functional boundaries | Inspect pinned source; compare every observer path with baseline in Stage 2 |
| Capture mutates outputs, masks, cache, or module state | Serialized execution, observer-only hooks, exception cleanup, parity gates |
| Hidden-state tuple obscures final pre-normalized residual | Capture layer boundaries and final-norm input explicitly |
| GQA or head-width assumptions corrupt values | Named-axis assertions and configuration-derived dimensions |
| N² scores, large logits, deep linear captures | Preflight reservation, sequence limits, streaming copies, capture profiles |
| In-memory data disappears or concurrent reads race expiration | Typed unavailable/expired states and request leases |
| Interpretation overclaims | Provenance badges, formula context, human claim review |

The [validation plan](SCIENTIFIC_VALIDATION.md) defines experimental evidence. The [stages](IMPLEMENTATION_STAGES.md) specify when each gate must pass.

## Stage 6 analysis integration

`analysis/token_journey.py`, `logit_lens.py`, `sampling.py`, `distributions.py` and `jobs.py` extend the existing scientific layer. `schemas/analysis.py` is the generated-contract authority and `api/analyses.py` uses the trace routes' registration/alias conventions and bounded serializer. The original trace engine, capture plans, immutable tensor storage and lease implementation remain unchanged.

Overview already captures all required depth states. Selected absolute coordinates are resolved from descriptors before tensor access. Representation metrics and sampling run on a bounded single analysis worker; readout projections additionally submit to `ModelService.submit_analysis_job`, sharing the existing one-thread executor with trace capture and loading. `native_norm("final", ...)` and `native_lm_head(...)` reuse resident modules. No analysis loads weights or calls adapter.forward, model.forward or generate. The adapter head helper now invokes the resident head's own forward method rather than expressing its linear operation separately.

The summary-only analysis cache reserves encoded-response capacity before work, deduplicates canonical source/version/parameter requests, acquires existing source leases, checks source TTL during progress and before publication, and limits both entries and bytes. It does not duplicate the trace tensor store or model weights. Detailed budgets and cancellation/expiry semantics are in the Stage 6 API section.

The existing inspector now offers Token Journey, Representation Evolution, Logit Lens, Candidate Evolution and Sampling Lab. TanStack Query owns job acknowledgements/results; Zustand holds selected token/checkpoint/layer/candidate, pinned IDs, applied sampling settings and explicit-capture associations. Horizontal checkpoint navigation, SVG chart points and the Model Journey selection are linked. Missing retained data requires an explicit overview capture, with late-acknowledgement isolation and trace-qualified microscope navigation. Stage 7 extends this same inspector and job infrastructure with residual/MLP exploration.

## Stage 7 integration

`analysis/block_internals.py` extends the existing scientific layer with selected-block residual and MLP diagnostics. The existing `AnalysisService`, Pydantic discriminated contract, cache, worker and TraceStore leases are reused. The MLP capture profile adds one selected input-RMSNorm observation and capture-time gate/up projection reference checks. No tensor database, execution queue, model instance or tracing architecture is added.

`components/residual/BlockWorkspace.tsx`, `BlockFlow.tsx` and `BlockDiagnostics.tsx` extend the contextual Inspector with Residual Stream and MLP Explorer. The existing capture hook supports explicit selected-layer/token MLP captures and stores UUID associations in Zustand; TanStack Query owns bounded jobs. Model Journey, token strip, Microscope and Stage 6 selection remain linked. Source identity, numerical reconciliation and undefined diagnostic states are visible; exploration never initiates a model forward. Independent audit tooling lives under backend tests so production observer registration remains confined to `tracing/instrumentation.py`.

## Stage 8 integration

`generation/service.py` owns bounded session metadata while the existing `TraceStore` owns detailed source tensors. It tokenizes the original raw prompt once, then submits a bounded cache-disabled loop to `ModelService.submit_trace_job`; each forward consumes exact prefix IDs and returns one decision. Optional `TraceEngine` capture is attached to that same forward, never a second pass. Session execution IDs and UUID source trace IDs are distinct, and each trace records generation ID, step index, absolute decision position and exact input IDs. Retained summaries are byte-bounded and contain only candidate subsets and scalar distribution statistics, not full-vocabulary histories. Cancellation/expiry stop before the next forward and discard an unpublished in-flight result.

`GenerationWorkspace.tsx` shares the raw prompt field and workspace mode tabs. It renders saved steps, full-sequence decodes, raw/adjusted candidate values, provenance, and explicit trace availability. Selecting a detailed step opens the existing Journey, Attention, Residual or MLP inspector with that step's original source ID and decision coordinate. TanStack Query polls only the current session; GET, step selection and all historical inspectors remain inference-free. Capture is capped at four source traces, with base residual states retained across layers to satisfy the current inspector coverage. KV caching and recapture are out of scope.
