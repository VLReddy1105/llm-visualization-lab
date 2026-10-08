# LLM Visualization Lab

An interactive microscope for understanding how a real transformer processes text: tokens, embeddings, attention, residual updates, MLP transformations, vocabulary logits, and next-token generation.

**Current status: Stage 2 — PASS.** Stage 1 remains the trusted inference path. Python model adapters, scoped internal capture and scientific reconciliation pass real CPU and Apple MPS checks with bitwise-identical tracing-off/on logits and probabilities. Both device suites pass all 137 automated tests.

The initial model is **Qwen/Qwen3-0.6B**, pinned to `c1899de289a04d12100db370d81485cdf75e47ca`. The implemented stack is Next.js / React / TypeScript / Tailwind CSS and Python / FastAPI / PyTorch / Hugging Face Transformers / Pydantic. Stage 2 adds model adapters and in-process traces. A server-side Trace Store remains later-stage work.

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

For CPU validation, use `LLM_LAB_DEVICE=cpu .venv/bin/pytest -q --real-model` and pass `--device cpu` to the trace validation command. Run real model processes serially to avoid duplicating resident weights. `ModelService.trace(text, CapturePlan(...))` returns a Future containing an in-process `TraceResult`; summary is the default, and detailed profiles require selected layers. `LLM_LAB_TRACE_MAX_BYTES` defaults to 256 MiB and limits a conservative extra-tensor peak estimate before hooks attach. Actual snapshots are owned CPU tensors; no HTTP trace serialization or persistence is present.

From the repository root:

```sh
npm run check:types
npm run typecheck
npm run lint
npm run build
```

After changing implemented schemas, export OpenAPI without `--check`, then run `npm run generate:types`. Next.js development/build uses supported webpack mode because Turbopack's CSS subprocess port was blocked in this environment. A production preview can run with `npm run start --workspace @llm-lab/web` after building.

Stage 2 is Python instrumentation only. Trace HTTP routes, persistence, serialization and frontend consumption require explicit Stage 3 approval.
