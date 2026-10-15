# LLM Visualization Lab

An interactive microscope for understanding how a real transformer processes text: tokens, embeddings, attention, residual updates, MLP transformations, vocabulary logits, and next-token generation.

**Current status: Stage 8 — PASS (2026-10-14).** Users can run bounded greedy or seeded sampled generation, inspect each real token decision, and open its original captured forward in existing scientific inspectors. Independent Qwen CPU and MPS audits pass; 359 backend tests pass on each device, 77 frontend tests pass, and 11 production Chrome regressions pass (with separately passing Stage 8 session/source expiry and offline checks). Exploration adds zero model forwards. See the [Stage 8 validation report](docs/validation/STAGE_8_REPORT.md) for detailed evidence, measurements and limitations.

The initial model is **Qwen/Qwen3-0.6B**, pinned to `c1899de289a04d12100db370d81485cdf75e47ca`. The implemented stack is Next.js / React / TypeScript / Tailwind CSS, Zustand, TanStack Query and Python / FastAPI / PyTorch / Hugging Face Transformers / Pydantic. Stage 2 supplies model adapters and scientific capture; Stage 3 retains these results and exposes bounded lazy access. Stage 4 consumes those contracts. Stage 5 adds bounded-slice statistics and a retained-output comparison endpoint. Stage 6 adds bounded, versioned analysis jobs using existing source leases and the resident model executor; the trace engine and store remain intact.

The central rule is **trace once, visualize many ways**. Every scientific value must originate from real model computation or a named, reproducible transformation of retained data. Missing tensors remain unavailable. Attention patterns, intermediate vocabulary projections, and representation geometry must carry appropriate interpretation limits.

## Blueprint

| Document | Contents |
| --- | --- |
| [Project vision](docs/PROJECT_VISION.md) | Research purpose, principles, scientific fidelity, success criteria |
| [System architecture](docs/SYSTEM_ARCHITECTURE.md) | Subsystems, three Mermaid diagrams, Qwen3 mapping, device and memory strategy |
| [Trace design](docs/TRACE_DESIGN.md) | Semantic vocabulary, tensor provenance, capture profiles, generation semantics |
| [Visualization workflow](docs/VISUALIZATION_WORKFLOW.md) | Linked selection, semantic zoom, all visualization modes and interpretation limits |
| [API contract](docs/API_CONTRACT.md) | Versioned endpoints, examples, lazy retrieval, loading, errors, expiration |
| [Implementation stages](docs/IMPLEMENTATION_STAGES.md) | Exact scope, modules, verification, and completion gates for Stages 1–11 |
| [Scientific validation](docs/SCIENTIFIC_VALIDATION.md) | Baseline equivalence, shape checks, numerical reconciliation, claim review |

Start with the vision, architecture, and trace design. The API and visualization documents consume those definitions; implementation stages specify when each planned feature becomes executable.

The unchanged [conceptual PDF](docs/reference/LLM_Visualization_Conceptual_Architecture.pdf) is retained locally. See the [Stage 1 report](docs/validation/STAGE_1_REPORT.md), [MPS smoke evidence](docs/validation/stage1-smoke.json), and [CPU smoke evidence](docs/validation/stage1-smoke-cpu.json) for measured results.

The [Stage 2 report](docs/validation/STAGE_2_REPORT.md), [MPS trace evidence](docs/validation/stage2-validation.json), and [CPU trace evidence](docs/validation/stage2-validation-cpu.json) record actual semantic shapes, provenance, reconstruction errors, cleanup, memory and timing. Validation uses two short raw prompts and layers 0, 12 and 27. CPU's independent all-position LM-head reference differs slightly within the existing tolerance; same-path tracing remains bitwise identical on both devices.

The [Stage 3 report](docs/validation/STAGE_3_REPORT.md) records the exact file inventory, lifecycle/API contract and all requested completion points. [Actual HTTP evidence](docs/validation/stage3-http-validation.json) proves one capture and zero retrieval forwards. The recorded capital trace retains 1,970,124 unique bytes; Layer 12 attention/MLP detail accompanies all-layer summaries.

The [Stage 4 report](docs/validation/STAGE_4_REPORT.md) records the rendering decision, exact file inventory, all 31 completion points, tests and real-browser screenshots. The default UI uses the existing `overview` capture: all-layer checkpoints, embeddings, final output and logits, without deep attention/MLP tensors. The capital example displays five tokens, 28 layers and the real next-token candidate ` Berlin` at 47.70%. This is an argmax preview, not an autoregressive generation step. See the [completed journey](docs/validation/stage4-model-journey.png) and [Layer 12 inspector](docs/validation/stage4-layer12.png).

Select a token or model stage to inspect its retained data. Wheel/trackpad scrolling, background dragging, arrow/Home/End keys, a layer jump selector, Fit Journey and Reset navigate the horizontal rail. Token selection persists across inspectors. Prompt edits keep the previous trace visible until explicit execution. Expiry, deletion and missing components are explicit; deeper data never triggers automatic recapture.

The [Stage 5 report](docs/validation/STAGE_5_REPORT.md) contains all 38 completion points, the exact file inventory, API additions, scientific evidence, tests and screenshots. From Layer 12, choose Attention and explicitly **Capture Layer 12 Attention**. This creates a separate execution for the same raw prompt and retains all query heads for that layer. Returning to the layer reuses the detailed trace; head, token and cell changes never run the model. Trace Details distinguishes both executions and compares their complete final outputs.

The [attention explorer](docs/validation/stage5-attention-heatmap.png) and [Q/K/V inspector](docs/validation/stage5-qkv-inspector.png) show actual reconstructed weights and backend statistics. Query head 4 maps to KV head 2; the Germany example’s `" is" → " Germany"` relation has weight **0.4407791793346405**. Masked cells are hatched and cannot select a valid relation. The matrix renders at most 16 × 16 cells, retaining full-row normalization; longer sequences use scrolling and window controls. Attention describes internal routing, not causal attribution. Detailed residual and MLP exploration is now implemented in Stage 7.

The [Stage 6 report](docs/validation/STAGE_6_REPORT.md) records the complete file inventory, numerical examples, unchanged output tolerances, cache measurements and CPU/MPS/browser evidence. Open Token Journey, Representation Evolution, Logit Lens, Candidate Evolution or Sampling Lab from the inspector. A horizontal keyboard-accessible timeline and clickable charts share token/layer selection with the Microscope. Missing vectors offer explicit overview recapture. Sampling settings operate on retained final-position logits without drawing a token or running another transformer forward.

The [candidate evolution view](docs/validation/stage6-germany-candidates.png) projects through the actual resident final normalization and LM head over all 151,936 vocabulary entries. Intermediate projections are diagnostics, not evidence of internal beliefs or causal reasoning. The Germany final projection's maximum absolute logit error is **3.814697265625e-6**, within the established tolerance; France is bitwise equal. [Performance evidence](docs/validation/stage6-performance.json) verifies 120 identical cache hits and no additional model forwards. Analysis results default to a 16 MiB / 128-entry cache with a 600-second absolute lifetime capped by the source trace; each response remains below 1 MiB.

## Local setup

Prerequisites: Python 3.11, uv, Node 22 and npm. Tested versions are recorded in the report. JavaScript uses npm workspaces and `package-lock.json`; Python uses its own `pyproject.toml`, `uv.lock` and virtual environment.

From the repository root, install the frontend:

```sh
npm ci
```

For a first setup only, create local configuration files (these already exist in the current workspace):

```sh
cp apps/web/.env.example apps/web/.env.local
cp apps/api/.env.example apps/api/.env
```

Install the backend:

```sh
cd apps/api
UV_CACHE_DIR=../../.cache/uv uv sync --frozen --python python3.11
```

Terminal 1, from the repository root:

```sh
cd apps/api
.venv/bin/uvicorn llm_lab.main:app --host 127.0.0.1 --port 8000
```

Terminal 2, from the repository root:

```sh
npm run dev
```

Open http://127.0.0.1:3000. The API documentation is at http://127.0.0.1:8000/docs. Initial loading downloads public model files into the ignored Hugging Face cache; later starts reuse them. Status stays responsive while loading. Use one backend worker to avoid duplicating resident weights.

Auto device priority is CUDA → MPS → CPU. CPU/MPS use float32; CUDA uses bfloat16 only with native support, otherwise float32. An unavailable explicitly requested device fails instead of silently changing devices. Output normalization uses CPU float64 softmax followed by float32 probabilities because the observed float32 CPU normalization failed its accuracy gate; model execution and raw logits remain unchanged. See the scientific validation record for the measured discrepancy.

Raw input is preserved exactly, with no chat template, special-token insertion, sampling or generation. `top_k` defaults to 10 and accepts integers 1–100; it controls display count only. Limits default to 128 tokens and 8,192 characters, with rejection rather than truncation. Local CORS permits only localhost and 127.0.0.1 origins on port 3000. Trace options remain excluded from the inference HTTP contract.

The requested `/api/health`, `/api/model/status`, and `/api/inference/next-token` routes are aliases for the canonical Stage 0 `/api/v1` contract. Health means process liveness; inspect model status for readiness. Generated TypeScript types are derived from the implemented Pydantic/OpenAPI schema.

## Validation commands

From `apps/api`:

```sh
.venv/bin/pytest -q --real-model
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m llm_lab.smoke --output ../../docs/validation/stage1-smoke.json
.venv/bin/python -m llm_lab.trace_validation --device mps --output ../../docs/validation/stage2-validation.json --report ../../docs/validation/STAGE_2_REPORT.md
.venv/bin/python -m llm_lab.export_openapi ../../packages/shared/openapi.json --check
```

`pytest -q` alone runs foundation tests and explicitly skips real-model tests. `--real-model` runs the actual pinned Qwen integration gates; no fake numerical fixture substitutes for them. The baseline utility uses an independent direct HF forward with all-position logits, then selects the last position.

For CPU validation, use `LLM_LAB_DEVICE=cpu .venv/bin/pytest -q --real-model` and pass `--device cpu` to the trace validation command. Run real model processes serially to avoid duplicating resident weights. `ModelService.trace(text, CapturePlan(...))` returns a Future containing an in-process `TraceResult`; summary is the default, and detailed profiles require selected layers. `LLM_LAB_TRACE_MAX_BYTES` defaults to 256 MiB and limits a conservative extra-tensor peak estimate before hooks attach. Actual snapshots are owned CPU tensors; Stage 3 stores them for bounded HTTP access. Persistence remains future work.

From the repository root:

```sh
npm run check:types
npm run typecheck
npm run lint
npm test
npm run build
```

After changing implemented schemas, export OpenAPI without `--check`, then run `npm run generate:types`. Next.js development/build uses supported webpack mode because Turbopack's CSS subprocess port was blocked in this environment. A production preview can run with `npm run start --workspace @llm-lab/web` after building.

With the real backend and frontend running, verify the UI in installed Google Chrome:

```sh
npm run test:browser
```

The normal browser run covers real Qwen execution, selection, navigation, changed-prompt output, deletion, mobile layout and an axe accessibility scan. It saves screenshots and JSON evidence under `docs/validation/`. Two fault scenarios run separately because they require different backend availability:

```sh
# Stop the backend for this scenario; keep the frontend running.
LLM_LAB_BROWSER_FAULT_MODE=offline npm run test:browser --workspace @llm-lab/web -- lifecycle.spec.ts -g unavailable

# Start the backend with LLM_LAB_TRACE_TTL_SECONDS=5 for this scenario.
LLM_LAB_BROWSER_FAULT_MODE=expiry npm run test:browser --workspace @llm-lab/web -- lifecycle.spec.ts -g expiry
```

Restore the normal backend command afterward. Isolated frontend tests mock HTTP using recorded real responses; production and browser validation use the actual API with no scientific fallback fixtures.

## Trace API

The server supports `/api/v1/traces` and shared `/api/traces` aliases. Default creation returns 202 after preflight and a Location to poll. To wait for a compact ready overview:

```sh
curl -X POST 'http://127.0.0.1:8000/api/traces?wait=true' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of Germany is","capture":{"profile":"deep","layers":[12]}}'
```

Use the returned UUID for GET overview, `/tokens`, `/layers`, `/layers/12`, `/layers/12/attention`, `/layers/12/attention/heads/4`, `/layers/12/mlp`, `/layers/12/residual`, `/logits`, and `/components`. Head GET is metadata-only by default; `include_weights=true&query_token_index=3` requests one retained row. Tensor values use POST `/components/{component_id}/slice` or `/tensors/{tensor_id}/slice`, with structured semantic selectors. DELETE releases a trace. `/api/traces/status` reports actual retained/reserved bytes and request-forward counts. See [API contract](docs/API_CONTRACT.md) for the complete inventory and schema.

Defaults: 30-minute absolute TTL, 512 MiB retained-store budget, eligible LRU eviction, 65,536 slice elements and 1 MiB response bytes. Configure `LLM_LAB_TRACE_TTL_SECONDS`, `LLM_LAB_TRACE_STORE_MAX_BYTES`, `LLM_LAB_TRACE_STORE_MAX_ENTRIES`, `LLM_LAB_TRACE_API_MAX_ELEMENTS`, `LLM_LAB_TRACE_API_MAX_RESPONSE_BYTES`, and `LLM_LAB_TRACE_SWEEP_SECONDS`; examples are in `apps/api/.env.example`. GETs read retained CPU data without executing or locking the model. Missing detail returns an explicit conflict; a new capture is an explicit POST.

With uvicorn running, reproduce the bounded actual HTTP artifacts from `apps/api`:

```sh
.venv/bin/python -m llm_lab.trace_http_validation
```

The generated typed API client supports trace and generation creation, polling, metadata/step pages, head/relation reads, JSON/binary slices and deletion. TanStack Query owns server payloads; Zustand owns scientific selection and viewport state. In-memory sessions and traces disappear on restart; the API is for one local process/worker. Attention Explorer, Layer Microscope, Token Journey, Logit Lens, Residual/MLP inspection and the Generation Journey are implemented.

To reproduce the Stage 5 real-model browser checks with the local backend and production frontend running:

```sh
LLM_LAB_BROWSER_STAGE=stage5 npm run test:browser -- attention.spec.ts
```

This refreshes only Stage 5 evidence. Isolated attention tests replay recorded real HTTP responses from `docs/validation/stage5-http-fixtures.json`; production code does not import those fixtures.

## Stage 7 inspection and reproduction

Select an input token and a layer, then choose **Residual Stream** or **MLP Explorer** in the contextual Inspector. **Capture Layer L Token T Details** explicitly runs the original prompt once with the existing narrow MLP profile. The new source UUID supplies every displayed value; operation, coordinate, ranking, token and layer changes read retained data. Missing coverage requires another explicit capture. Coordinate ranks and update magnitudes are descriptive measurements.

The residual flow exposes both real branch updates and additions, signed changes, ratios/cosines with undefined-zero handling, and recorded numerical errors. The horizontal MLP pathway branches gate/up from one learned-normalized input, shows SiLU and product coordinates, maps the 3,072-dimensional product to the 1,024-dimensional update, and shows its residual addition. Responses contain at most 64 ranked coordinates per operation. All arithmetic is in Python.

From `apps/api`, reproduce the independent numerical audits:

```sh
LLM_LAB_DEVICE=cpu .venv/bin/pytest -q --real-model
LLM_LAB_DEVICE=mps .venv/bin/pytest -q --real-model
.venv/bin/python tests/stage7_validation.py --device cpu --output ../../docs/validation/stage7-scientific-cpu.json
.venv/bin/python tests/stage7_validation.py --device mps --output ../../docs/validation/stage7-scientific-mps.json
```

With the production frontend and real backend running, use these root commands:

```sh
LLM_LAB_BROWSER_STAGE=stage7 npm run test:browser --workspace @llm-lab/web -- residual.spec.ts
LLM_LAB_BROWSER_STAGE=stage7-regression LLM_LAB_BROWSER_OUTPUT_DIR=../../docs/validation/stage7-regression npm run test:browser --workspace @llm-lab/web -- journey.spec.ts attention.spec.ts evolution.spec.ts
```

The normal Stage 7 run conditionally skips the separately executed expiry scenario. To reproduce it, start the backend temporarily with `LLM_LAB_ANALYSIS_TTL_SECONDS=3 LLM_LAB_TRACE_TTL_SECONDS=12`, then run `LLM_LAB_BROWSER_STAGE=stage7 LLM_LAB_BROWSER_FAULT_MODE=analysis-expiry npm run test:browser --workspace @llm-lab/web -- residual.spec.ts -g 'analysis expiry'`. Restore normal TTL settings afterward. The HTTP resource benchmark (`.venv/bin/python tests/stage7_http_validation.py` from `apps/api`) requires the same live backend and source UUIDs from the normal browser run.
