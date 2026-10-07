# Stage 1 completion report

**Status: PASS.** Recorded 2026-10-07T22:24:44.208263+00:00 UTC; local date 2026-10-08 (Europe/Berlin).

The browser → Next.js → FastAPI → pinned Qwen path was exercised with real raw tokenization, one forward pass, final-position logits, full-vocabulary normalization and displayed top candidates. Stage 2 has not started. No TraceCollector, internal capture, custom hooks, trace store, attention visualization, generation or fabricated prediction fixtures exist in production code.

## Runtime and pinned dependencies

| Item | Tested version |
| --- | --- |
| Python / platform | 3.11.15 / Darwin arm64 |
| Node / npm / uv | 22.23.1 / 10.9.8 / 0.11.7 |
| PyTorch / Transformers | 2.14.1 / 5.19.0 |
| FastAPI / Pydantic / pydantic-settings | 0.142.2 / 2.13.5 / 2.15.0 |
| Uvicorn | 0.54.0 |
| HF Hub / tokenizers / safetensors | 1.33.0 / 0.23.2 / 0.8.0 |
| Next.js / React / ReactDOM | 16.4.0 / 19.3.0 / 19.3.0 |
| TypeScript / Tailwind CSS | 5.9.3 / 4.3.3 |
| ESLint / typescript-eslint / react-hooks plugin | 10.12.0 / 8.71.1 / 7.1.1 |
| openapi-typescript | 7.13.0 |
| pytest / Ruff / httpx | 9.1.1 / 0.16.10 / 0.28.1 |

JavaScript and Python have independent manifests and lockfiles. The frontend uses npm workspaces; there is no additional monorepo orchestrator.

## Model and execution

- HF ID: `Qwen/Qwen3-0.6B`; registry ID: `qwen3-0.6b`.
- Model and tokenizer revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Auto detection outside the sandbox: MPS and CPU available; MPS selected. CUDA was unavailable and is untested.
- Both real CPU and MPS runs use float32, eval mode, inference mode, eager attention and disabled KV cache.
- Model dimensions: 28 layers, width 1,024, vocabulary 151,936. Only final output logits are returned, shape `[1, 1, 151936]`.
- Remote arbitrary code disabled; safetensors required; public model retrieval does not need credentials.
- Prompt limit: 128 tokens / 8,192 characters by default; reject rather than truncate. `top_k` defaults to 10, bounded to integer 1–100 and controls display count only.

## Measured timings

For exactly `The capital of Germany is`:

| Run | Cached load incl. warmup (ms) | Synchronized forward (ms) | Total inference (ms) |
| --- | --- | --- | --- |
| Live HTTP, MPS | 9859.388 | 809.783 | 818.576 |
| Verified smoke, MPS | 29172.832 | 1295.317 | 1376.363 |
| Verified smoke, CPU | 3641.374 | 152.503 | 157.514 |

These are individual observations, not benchmarks. Cached loading excludes the initial download. Concurrent model processes caused substantial resource contention during some checks; use one backend worker. The final browser request displayed 1,020.6 ms total and 932.5 ms forward. HTTP requests and standalone smoke runs have distinct timings and request IDs.

## Exact raw tokenization

No chat template, special-token insertion or input modification. Input and decoded full text both equal the exact supplied sentence. Token indices and positions are 0–4; position 4 predicts position 5. Character offsets are half-open.

| Index / position | Decoded string | Native piece | Token ID | Character span |
| --- | --- | --- | --- | --- |
| 0 | `"The"` | `The` | 785 | 0:3 |
| 1 | `" capital"` | `Ġcapital` | 6722 | 3:11 |
| 2 | `" of"` | `Ġof` | 315 | 11:14 |
| 3 | `" Germany"` | `ĠGermany` | 9856 | 14:22 |
| 4 | `" is"` | `Ġis` | 374 | 22:25 |

Quoted strings preserve leading spaces; `Ġ` is the tokenizer-native whitespace marker.

## Real top 10 predictions

Probability below is a fraction, normalized over all 151,936 logits. Values come from the live HTTP response and exactly match the verified MPS smoke response.

| Rank | Decoded string | ID | Probability | Raw logit |
| --- | --- | --- | --- | --- |
| 1 | `" Berlin"` | 19846 | 0.4770433903 | 16.5214920044 |
| 2 | `" located"` | 7407 | 0.0715774298 | 14.6246643066 |
| 3 | `" ______"` | 32671 | 0.0464569442 | 14.1924104691 |
| 4 | `" ____"` | 30743 | 0.0232905932 | 13.5019340515 |
| 5 | `" in"` | 304 | 0.0151218390 | 13.0700244904 |
| 6 | `" "` | 220 | 0.0150193637 | 13.0632247925 |
| 7 | `" ("` | 320 | 0.0144463778 | 13.0243282318 |
| 8 | `" a"` | 264 | 0.0140006142 | 12.9929857254 |
| 9 | `" one"` | 825 | 0.0131504657 | 12.9303417206 |
| 10 | `" the"` | 279 | 0.0129384408 | 12.9140872955 |

Omitted probability mass: `0.2969545414671302`. The candidate table is not renormalized.

## Baseline and numerical evidence

The reusable `models/baseline.py` runs a direct HF forward on the same resident weights with all-position logits, then selects the last position. It does not call the service's inference computation. `smoke.py` compares every vocabulary logit, real token IDs, top candidates, full probabilities and deterministic repeats. Float32 elementwise criterion: `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`.

- MPS: maximum baseline logit error `0.0`; repeat error `0.0`; bitwise repeat equality true; full float32 probability sum `1.0000001192092896`.
- CPU: maximum baseline logit error `2.47955322265625e-05`; repeat error `0.0`; bitwise repeat equality true; full probability sum `1.0`.
- CPU/MPS are separate compatibility checks; no cross-device equality is claimed.
- A concrete Stage 0 implementation correction was required: CPU float32 softmax summed to `1.0001225397008613` under a float64 audit, with maximum error `5.845935186632456e-05` against float64 softmax. Only the bounded final-vocabulary softmax is now computed in CPU float64, then cast to float32. Raw logits and model execution remain unchanged. Tolerances were not relaxed. An independent max-shifted exp/reduction test checks the distribution, and both normalization dtypes are returned as provenance.

## Verification results

| Check | Result |
| --- | --- |
| Backend complete suite, real MPS | **30 passed**, 1 test-client deprecation warning, 10.46 s |
| Backend complete suite, real CPU | **30 passed**, same warning, 162.05 s during concurrent server loading |
| Independent real smoke, CPU and MPS | **PASS** |
| Ruff lint / format check | **PASS**, 21 Python files formatted |
| Production frontend build | **PASS**, Next.js webpack build |
| TypeScript | **PASS**, `tsc --noEmit` |
| Frontend lint | **PASS**, ESLint |
| OpenAPI snapshot parity | **PASS** |
| Generated TypeScript parity | **PASS**, `npm run check:types` |
| npm dependency audit | **0 vulnerabilities** |
| Live HTTP vs verified MPS smoke | **Exact token / prediction equality** |
| Browser end-to-end | **PASS**, Ready / Connected, five real tokens and ten predictions |
| Browser error states | Empty prompt and stopped API show understandable errors |
| Final browser error/warning logs | Empty |
| Stage boundary | No trace routes, custom hooks or generation calls in production source |

Foundation checks include liveness without weights, truthful loading/failure states, responsive status during loading, strict request validation, local CORS and safe errors. Real-model tests include tokenizer fidelity, finite full logits, normalization and sorted top-k, independent baseline parity, deterministic repeats, whitespace/Unicode preservation, token limits, actual HTTP schema and recovery after an injected inference failure. Failure injection tests simulate errors only, never numerical model outputs.

## Evidence files

- [Verified MPS smoke](stage1-smoke.json)
- [Verified CPU smoke](stage1-smoke-cpu.json)
- [Live HTTP response](stage1-http.json)
- [Browser screenshot](stage1-browser.jpg)
- [Unchanged conceptual reference](../reference/LLM_Visualization_Conceptual_Architecture.pdf)

The reference PDF is byte-identical to the source in Downloads. SHA256: `69506f1068f5d6a42a60d85c2f7c4d77f46946cd647370b8ae019214f91870e9`.

## Final project tree and exact file inventory

`[+]` means created in Stage 1; `[~]` means an existing Stage 0 file modified. Unmarked files are unchanged Stage 0 documents. This accounts for **49 new source/document/evidence files** and **5 modified files**. Ignored dependency installs, model weights and caches are excluded.

```text
llm interpreter/
├── apps/
│   ├── api/
│   │   ├── src/
│   │   │   └── llm_lab/
│   │   │       ├── api/
│   │   │       │   ├── __init__.py [+]
│   │   │       │   ├── dependencies.py [+]
│   │   │       │   ├── health.py [+]
│   │   │       │   ├── inference.py [+]
│   │   │       │   └── models.py [+]
│   │   │       ├── models/
│   │   │       │   ├── __init__.py [+]
│   │   │       │   ├── baseline.py [+]
│   │   │       │   ├── devices.py [+]
│   │   │       │   └── manager.py [+]
│   │   │       ├── schemas/
│   │   │       │   ├── __init__.py [+]
│   │   │       │   ├── inference.py [+]
│   │   │       │   └── model.py [+]
│   │   │       ├── __init__.py [+]
│   │   │       ├── config.py [+]
│   │   │       ├── errors.py [+]
│   │   │       ├── export_openapi.py [+]
│   │   │       ├── main.py [+]
│   │   │       └── smoke.py [+]
│   │   ├── tests/
│   │   │   ├── conftest.py [+]
│   │   │   ├── test_foundation.py [+]
│   │   │   └── test_real_inference.py [+]
│   │   ├── .env.example [+]
│   │   ├── .python-version [+]
│   │   ├── pyproject.toml [+]
│   │   └── uv.lock [+]
│   └── web/
│       ├── src/
│       │   ├── app/
│       │   │   ├── globals.css [+]
│       │   │   ├── layout.tsx [+]
│       │   │   └── page.tsx [+]
│       │   └── lib/
│       │       └── api/
│       │           └── client.ts [+]
│       ├── .env.example [+]
│       ├── eslint.config.mjs [+]
│       ├── next-env.d.ts [+]
│       ├── next.config.ts [+]
│       ├── package.json [+]
│       ├── postcss.config.mjs [+]
│       └── tsconfig.json [+]
├── docs/
│   ├── reference/
│   │   └── LLM_Visualization_Conceptual_Architecture.pdf [+]
│   ├── validation/
│   │   ├── STAGE_1_REPORT.md [+]
│   │   ├── stage1-browser.jpg [+]
│   │   ├── stage1-http.json [+]
│   │   ├── stage1-smoke-cpu.json [+]
│   │   └── stage1-smoke.json [+]
│   ├── API_CONTRACT.md [~]
│   ├── IMPLEMENTATION_STAGES.md [~]
│   ├── PROJECT_VISION.md
│   ├── SCIENTIFIC_VALIDATION.md [~]
│   ├── SYSTEM_ARCHITECTURE.md [~]
│   ├── TRACE_DESIGN.md
│   └── VISUALIZATION_WORKFLOW.md
├── packages/
│   └── shared/
│       ├── src/
│       │   └── generated/
│       │       └── api.ts [+]
│       ├── openapi.json [+]
│       └── package.json [+]
├── .gitignore [+]
├── .nvmrc [+]
├── README.md [~]
├── package-lock.json [+]
└── package.json [+]
```

Local ignored configuration files also created: `apps/api/.env`, `apps/web/.env.local`. Generated runtime directories: `apps/api/.venv`, root `node_modules`, `.cache`, Python/test/Ruff caches and `apps/web/.next`. These are not source deliverables. No Git repository was present, so no commit or remote publication was performed; `.gitignore` is ready for initialization.

The folder remains `llm interpreter`: renaming the active workspace would move it outside its configured writable root. The user explicitly allowed leaving it unchanged when unsafe.

## Exact local startup commands

Dependencies and local configuration are already installed in this workspace. First-time/reproducible setup commands are in the root README.

Terminal 1:

```sh
cd "/Users/likhithreddy/Documents/llm interpreter/apps/api"
.venv/bin/uvicorn llm_lab.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```sh
cd "/Users/likhithreddy/Documents/llm interpreter"
npm run dev
```

Open http://127.0.0.1:3000. Both services were left running after final validation; stop those instances before starting replacements on the same ports. The frontend currently runs the verified production build.

## Warnings and limitations

- Starlette emits one deprecation warning for its supported current `httpx` test-client fallback; tests pass. Production API serving is unaffected.
- Turbopack CSS processing tried to bind a subprocess port and was denied, including the permission retry. Supported webpack development/build is configured and passes.
- CUDA/bfloat16 were not available for validation; no GPU-general compatibility claim is made.
- Public HF downloads can emit an unauthenticated rate-limit advisory. The initial network download needed permission in this environment; later runs use the ignored cache.
- Timings vary with warmup and resource contention. No memory benchmark or production deployment was requested.
- Local-only development configuration is intentional. No authentication, database or container infrastructure was added.

## Stage 2 next, pending explicit approval

Stage 2 will introduce BaseModelAdapter/registry/Qwen3Adapter and inspect the pinned installed implementation's real module and functional boundaries. It will add run-scoped instrumentation and a bounded semantic in-process trace: embeddings; residual and normalization checkpoints; attention/MLP updates; requested Q/K/V, eager attention weights, gate/up/product values; axes, mask/position/scale metadata and captured-versus-derived provenance. Capture profiles, memory budgets, owned CPU snapshots and finally cleanup will be verified.

Scientific gates will compare tracing off/on against this trusted baseline, check config-derived shapes/GQA mapping, attention and residual/MLP reconciliation, observer neutrality and cleanup after failure. Stage 2 will record capture overhead and supported device/dtype compatibility. Trace HTTP APIs/storage, model journey UI and advanced visualization remain later stages. **No Stage 2 code has been implemented.**
