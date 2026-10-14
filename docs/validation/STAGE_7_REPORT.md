# Stage 7 — Residual Stream and MLP Interpretability

## 1. Overall result

**PASS — completed 2026-10-14 (Europe/Berlin).** Stage 7 is implemented and validated on the pinned real Qwen3-0.6B using CPU and Apple MPS float32 eager execution. All **340 backend tests pass on each device**, **73 frontend tests pass**, and **10 real production Chrome checks pass** across normal Stage 7, predecessor views, unavailable-backend and expiry runs. TypeScript, ESLint, Ruff, production build and generated-contract checks pass. No requested CPU/MPS or browser gate remains untested.

The normal Stage 7 browser command reports two passes and one conditional skip: the expiry case is deliberately run separately against a real short-TTL backend, where it passes. CUDA, other dtypes/models, exhaustive screen-reader coverage and total-process/accelerator peak memory are outside this validation and are explicitly untested below. Stage 8 remains unimplemented.

## 2. Baseline, preservation and exact scope

Implementation began on `main`, HEAD `631475b1bd929bc0b231efe9acd28ecef4563a03`, with extensive uncommitted Stage 5–6 work. The [initial branch/status/inventory](stage7-initial-state.json) and [baseline hashes](stage7-baseline-hashes.json) were recorded before Stage 7 edits. The [architectural assessment](STAGE_7_ARCHITECTURAL_ASSESSMENT.md) records the inspected integration plan. No existing file was deleted; the final branch/HEAD remain unchanged. No commits, pushes, merges or deployments were performed, and dependency manifests/lockfiles are unchanged.

A regression-runner preservation defect was discovered: the existing Stage 6 browser test hardcoded its evidence directory and refreshed 16 previous artifacts despite the supplied separate-directory setting. All 16 originals were recovered from pre-existing Git objects **only after an exact SHA-256 match to the initial working tree**, and restored byte-for-byte. New regression artifacts were moved to `stage7-regression/`. The runner now honors `LLM_LAB_BROWSER_OUTPUT_DIR`. The [recovery map](stage7-recovery-map.json) records matching object IDs and verification; every predecessor validation artifact matches its baseline again. This recovery did not reset files, change Git HEAD/index, or create a commit. Next.js generated type-path churn was likewise restored only after verifying the recorded initial hash.

The appendix and [scope audit](stage7-scope-audit.json) distinguish files changed by Stage 7 from earlier uncommitted Stage 5–6 files. The inventory is relative to the initial working tree, not only Git HEAD.

## 3. Existing modules reused

The implementation extends `Qwen3Adapter`, `TraceSession`, `TraceCollector`, `TraceEngine`, capture plans, native normalization/projection helpers and `compare_tensors`. `TraceStore` continues to own immutable CPU tensors, alias accounting, reservations, leases, TTL, tombstones and LRU eviction. `AnalysisService` keeps its existing worker, canonical deduplication, encoded-summary cache, source-liveness checks, cancellation and publication policy. Existing trace and analysis routes are reused; no tensor database, model instance, tracing architecture or execution queue was added.

The frontend reuses Inspector, Model Journey, Layer Microscope, token strip, TanStack Query, Zustand, `useAnalysis` and the established explicit-capture hook. It adds three small residual/MLP components and no dependency. Backend independent observers are confined to test tooling; production observer registration remains centralized in `tracing/instrumentation.py`, as enforced by the unchanged foundation test.

## 4. Installed mathematics and capture design

The inspected installed source is preserved in [stage7-installed-qwen3-source.txt](stage7-installed-qwen3-source.txt), with versions/digest in [runtime metadata](stage7-runtime.json). Model/tokenizer revision is `c1899de289a04d12100db370d81485cdf75e47ca`; Transformers 5.19.0, PyTorch 2.14.1, Python 3.11.15, FastAPI 0.142.2, Pydantic 2.13.5, Next.js 16.4.0 and React 19.3.0 remain installed. Qwen has 28 layers, hidden width 1024, intermediate width 3072, and learned RMSNorm epsilon 1e-6.

```text
r0 = incoming residual
n_attn = input_layernorm(r0)
a = self_attention(n_attn), including o_proj
r1 = r0 + a
n_mlp = post_attention_layernorm(r1)
g = gate_proj(n_mlp)
u = up_proj(n_mlp)
s = SiLU(g) = g * sigmoid(g)
p = s * u                      [elementwise]
m = down_proj(p)
r2 = r1 + m
```

All eleven operands are directly observed during the one explicit original forward. Existing hooks already supplied r0/a/r1/m/r2 and selected MLP internals, including the actual activation-module SiLU output and actual down-projection input. Stage 7 adds the input-RMSNorm output to the narrow MLP profile, with no QKV or attention-matrix retention. Learned norms and projections are the installed modules; no convenient RMSNorm approximation is used.

The UI requests `profile: mlp`, one selected layer and one absolute token position through the existing capture endpoint. A new capture has its own UUID and immutable coverage. It never merges another execution's tensors. Independent learned RMSNorm/gate/up/down reference checks are computed during explicit capture validation after observers are removed, and only scalar comparison records are retained. Older captures lacking required operands/reference checks require explicit recapture. `mps` manager metadata and `mps:0` tensor metadata are preserved and validated as equivalent device placement.

## 5. Residual stream calculations

`block_internals.py` resolves absolute token coordinates and verifies shape, axes, dtype, original device, raw provenance, source token identity and revision before computation. Python computes float64 L2 norms, after-minus-before delta magnitude, signed coordinate deltas, before/after cosine and update-to-incoming-residual norm ratios. Zero denominators return null with reasons; near-zero finite inputs remain measurable. Nonfinite scientific operands fail explicitly.

The attention and MLP updates are **actual directly captured branch outputs**, not subtraction reconstructions. The combined block update `a+m` is separately DERIVED; signed delta is after-minus-before and is also DERIVED. The backend returns bounded before/update/after coordinate rows. Magnitude and coordinate ranks do not establish causal importance or named concepts.

For final browser sources at Layer 12, token 4 (` is`, ID 374), these are measured MPS observations:

| Prompt | ‖r0‖₂ | ‖a‖₂ | ‖r1‖₂ | ‖m‖₂ | ‖r2‖₂ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Germany | 43.68595819 | 8.030234036 | 44.2408147 | 12.35228757 | 48.60543163 |
| France | 43.71581218 | 7.572075942 | 43.83832948 | 12.45824421 | 48.42867771 |

## 6. MLP calculations and coordinates

The horizontal diagram branches gate and up from the same learned-normalized input, applies **SiLU rather than sigmoid alone**, multiplies the 3072-coordinate operands elementwise, and maps the product through the down projection to the 1024-coordinate MLP update. The final addition includes the selected captured r1/m/r2 values. Every selectable operation shows its formula, full input/output shapes, selected-vector width, exact selected coordinate, backend summary statistics, dtype/device and capture provenance.

Server-side ranking supports 1–64 largest absolute, positive or negative coordinates. Sign filters are strict; an empty eligible set is explicit. Stable ordering breaks exact ties by ascending original coordinate. Hidden and intermediate coordinate namespaces have separate bounds. The selected gate row is independently checked against exact stored slices in the browser.

Measured Layer 12, token 4, intermediate coordinate 17:

| Prompt | Raw gate | SiLU(gate) | Up | Product | Float64 scalar multiplication audit error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Germany | 1.76534235477 | 1.50738608837 | 0.0842253267765 | 0.126960083842 | -2.02915018122e-09 |
| France | 1.69824063778 | 1.43553102016 | 0.000351831316948 | 0.00050506478874 | 1.93960403294e-11 |

The scalar multiplication audit uses widened captured float32 operands; its tiny rounding difference is distinct from native float32 product reconciliation. No tensor value or neuron meaning is invented.

## 7. Reconciliation and independent references

Both analyses check r1 ≈ r0+a, r2 ≈ r1+m, complete-block r2 ≈ (r0+a)+m, SiLU and product from saved operands. They additionally expose the original independent learned input/MLP normalization and gate/up/down projection checks with their **actual retained-token scope**. Independent reference and saved-operand provenance are visible in the table and raw disclosure.

The existing float32 gate is unchanged: `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`. Errors are audited in float64; results include maximum/mean absolute error, relative L2 error, bitwise equality and tolerances. Relative L2 retains the declared 1e-12 reference-norm floor; this floor is not used to manufacture cosine or update ratios. Failed reconciliation returns an explicit error.

The standalone audit separately observes all eleven original-forward tensors, removes observers, then directly replays installed RMSNorm, self-attention, gate/up/activation/down and complete decoder-layer methods using the original mask/rotary inputs. It validates both prompts at positions **0 and 4**, layers **0, 12 and 27**, with one original complete-model capture per prompt. Independent references are test-only and never substitute for displayed captured values.

| Device / prompt | Independent comparisons | Largest independent max absolute / relative L2 error | Residual additions | Saved-data SiLU largest max error | Result |
| --- | ---: | ---: | --- | ---: | --- |
| CPU / Germany | 132 | 0 / 0 | Bitwise equal | 0 | PASS |
| CPU / France | 132 | 0 / 0 | Bitwise equal | 0 | PASS |
| MPS / Germany | 132 | 0 / 0 | Bitwise equal | 4.76837158203e-07 | PASS |
| MPS / France | 132 | 0 / 0 | Bitwise equal | 4.76837158203e-07 | PASS |

All **528 independent comparisons** pass. The MPS-to-CPU saved-operand SiLU audit has small implementation rounding differences within the unchanged gate; this is not a cross-device whole-model equality claim. [CPU evidence](stage7-scientific-cpu.json) and [MPS evidence](stage7-scientific-mps.json) preserve every error, source coordinate, shape and hook-cleanup observation.

## 8. CPU/MPS and backend regression results

| Gate | Actual result | Evidence |
| --- | --- | --- |
| Full pinned-Qwen CPU suite | 340 passed, 30.38 s | [CPU log](stage7-cpu-tests.txt) |
| Full pinned-Qwen MPS suite | 340 passed, 30.52 s | [MPS log](stage7-mps-tests.txt) |
| Standalone CPU and MPS audits | PASS on both | [CPU](stage7-scientific-cpu.json), [MPS](stage7-scientific-mps.json) |
| Frontend unit/regression suite | 73 passed | [Frontend log](stage7-frontend-tests.txt) |
| TypeScript / ESLint | PASS / PASS | [Types](stage7-typecheck.txt), [Lint](stage7-lint.txt) |
| Ruff lint / format | PASS / PASS | [Ruff](stage7-ruff.txt) |
| Production webpack build | PASS | [Build](stage7-build.txt) |
| OpenAPI / generated TypeScript drift | PASS / PASS | [OpenAPI](stage7-openapi-check.txt), [Generated types](stage7-contract-checks.txt) |

All predecessor 269 backend cases and 68 frontend cases remain passing; 71 backend and five frontend cases were added. Coverage includes residual and MLP references, attention/QKV/GQA and observer neutrality/cleanup, trace serialization/leases, Logit Lens and Sampling Lab parity, absolute coordinates, cache identity and copy immutability, cancellation, TTL/deferred release, deletion, source mismatch, missing captures/checks, failed reconciliation, unsupported configuration, nonfinite operands, near-zero/negative algebra and deterministic tie cases. Synthetic values are confined to isolated algebra/failure tests. One pre-existing Starlette/httpx TestClient deprecation warning remains; no dependency upgrade was made.

Early exploratory checks detected equivalent MPS device naming, an independent-audit script outside the centralized-observer boundary, and browser selector assumptions. Those were corrected; test tooling was moved under `tests/`, controls were clarified, and final passing logs above supersede those attempts. Tolerances were never widened.

## 9. API contracts, lifecycle and errors

Pydantic defines `residual_stream` and `mlp_internals` requests/results in the existing analysis schema. OpenAPI and generated TypeScript were regenerated through the project's existing commands. The new job method is `stage7.v1`; Stage 6 remains `stage6.v1`. Source-qualified jobs carry exact immutable request/result associations. Analyses run from retained CPU data with unloaded resident weights as well as loaded weights.

The same source leases, one analysis worker, queue capacity, JSON summary cache, TTL capped by source lifetime and cancellation checks apply. Publication rechecks source liveness. Cancelled/failed results are not reused; queued/running/ready identical requests deduplicate. Source/job mismatch, invalid token/layer/coordinate, unsupported configuration, missing capture, expired source, nonfinite values, failed reconciliation and resource exhaustion all produce explicit errors. No scientific fallback or automatic recapture exists. Actual deletion is reported through the existing store's `TRACE_NOT_FOUND` policy.

## 10. Resources, cache and no-extra-forward evidence

Defaults remain 16 MiB / 128 analysis entries, queue capacity 3, 600-second absolute analysis TTL capped by source expiry, 32 MiB estimated scratch and 1 MiB encoded response limit. Active jobs reserve one response slot; terminal entries retain encoded summaries only. Stage 7 scratch preflight for this model is **1,867,776 bytes**, and coordinate-response admission is conservatively estimated. Source tensors remain charged to the existing store.

A real selected Layer 12/token 4 MLP capture retains **1,314,856 bytes**, versus **3,559,464 bytes** for the browser overview. The three-layer/two-token independent audit retains **1,699,880 bytes**, below its **1,740,840-byte** retained estimate; capture transient estimate is **28,639,872 bytes**. These are retained counts and conservative tensor estimates, not measured total-process or accelerator peak memory.

Each device's standalone audits verify 24 ready analyses and 120 cache hits across both prompts; both report zero analysis forwards and zero active leases afterward. The real HTTP benchmark runs 12 maximum-coordinate (64) cases and **240 additional verified cache hits**, with stable source bytes, identical IDs/results and no forwards. Largest complete HTTP response: **63,491 bytes**, below 1 MiB. [Full HTTP measurements](stage7-http-performance.json) include every timing and limit case.

| Prompt / kind / ranking | Uncached ms | Cached p50 / p95 ms | Max HTTP bytes |
| --- | ---: | ---: | ---: |
| Germany / residual_stream / absolute | 50.97 | 5.97 / 6.51 | 62,970 |
| Germany / residual_stream / positive | 20.57 | 6.02 / 6.83 | 62,722 |
| Germany / residual_stream / negative | 21.70 | 5.80 / 6.16 | 63,488 |
| Germany / mlp_internals / absolute | 16.65 | 4.77 / 5.26 | 44,927 |
| Germany / mlp_internals / positive | 19.27 | 4.61 / 4.72 | 44,614 |
| Germany / mlp_internals / negative | 18.89 | 4.60 / 4.68 | 45,486 |
| France / residual_stream / absolute | 20.14 | 5.70 / 5.83 | 62,830 |
| France / residual_stream / positive | 20.00 | 5.65 / 5.86 | 62,632 |
| France / residual_stream / negative | 20.13 | 5.72 / 5.85 | 63,491 |
| France / mlp_internals / absolute | 19.17 | 4.73 / 4.79 | 44,932 |
| France / mlp_internals / positive | 18.85 | 4.66 / 4.82 | 44,652 |
| France / mlp_internals / negative | 18.91 | 4.85 / 5.51 | 45,492 |

Times are small local wall-time samples including HTTP/polling; they are not throughput or frame-rate guarantees. The benchmark's forward counter stayed **46 → 46**, retained source bytes **101,535,504 → 101,535,504**, and active leases ended at zero. The store total includes previously captured browser/regression sources.

Browser Germany counts **36 → 41**, France **41 → 46**: five explicit captures per prompt, **zero additional exploration forwards**. The nonzero starting counter reflects earlier explicit validation captures. Operation/view/coordinate/ranking changes, Model Journey round trips, token/layer changes and source-error handling do not run the model. The real expiry case remains **0 → 2** after two explicit captures and saved-data recomputation. Backend fail-fast forward sentinels additionally protect analysis execution.

## 11. Frontend functionality and scientific provenance

The existing bright horizontal workspace is preserved. Model Journey layer selection and the exact token strip coordinate follow both inspectors. Clicking microscope MLP/residual operations opens the appropriate view. Stage 6 pinned candidates and sampling settings persist; new unrelated roots clear source associations. Detailed captures are keyed by root source, layer, profile and absolute token, with late-acknowledgement owner isolation. Explicit recapture can upgrade older traces whose vectors exist but reference checks are absent.

Residual Stream renders the eight-step horizontal computation path, active operation highlight, before/update/after coordinate comparison, signed changes, real diagnostic cards, bounded coordinate table and full reconciliation errors. MLP Explorer renders a horizontal gate/up branch from the same normalized input, SiLU/product/down operations, exact shared-coordinate values, hidden/intermediate widths and selected residual addition. SVG arithmetic only determines chart geometry; Python supplies all scientific values and rankings.

Keyboard arrow/Home/End controls change operations, accessible forms validate coordinate bounds, and horizontal scrolling is contained within diagrams/tables. Checks at 1440, 800 and 390 pixels find no document-wide horizontal overflow. Narrow diagrams retain horizontal semantics through contained scrolling rather than distorting the computation. Provenance, source UUID, dtype/device, formulas and method details are visible or available in disclosures.

## 12. Production-browser screenshots and interactions

No final scientific browser test intercepts or mocks HTTP. Both Germany and France use actual model captures and real analyses. Browser-selected gate/up/SiLU/product values match exact source tensor slices; all returned reconciliation records pass. Both residual and MLP views have **zero axe violations** (45 passing rules each), and both runs record no page errors.

| View / state | Germany | France |
| --- | --- | --- |
| Residual Stream | [Screenshot](stage7-germany-residual.png) | [Screenshot](stage7-france-residual.png) |
| Gated MLP, coordinate 17 | [Screenshot](stage7-germany-mlp.png) | [Screenshot](stage7-france-mlp.png) |
| 800-pixel layout | [Screenshot](stage7-germany-narrow.png) | [Screenshot](stage7-france-narrow.png) |
| 390-pixel layout | [Screenshot](stage7-germany-mobile.png) | [Screenshot](stage7-france-mobile.png) |
| Layer 27, token 0 | [Screenshot](stage7-germany-layer27-token0.png) | [Screenshot](stage7-france-layer27-token0.png) |
| Actual deleted source | [Screenshot](stage7-germany-deleted.png) | [Screenshot](stage7-france-deleted.png) |

Additional real states: [analysis expiry](stage7-analysis-expired.png), [source expiry](stage7-source-expired.png), [unavailable backend](stage7-regression/stage4-api-unavailable.png), [original trace-expiry regression](stage7-regression/stage4-expired.png). Desktop residual/MLP and narrow/mobile screenshots were visually inspected for readable labels, formulas, branching, signed axes, selected-stage highlighting and contained scrolling.

[Germany interactions](stage7-germany-browser.json), [France interactions](stage7-france-browser.json), recorded HTTP fixtures and [browser results](stage7-browser-results.json) preserve source UUIDs and actual values. The two normal cases pass in 29.0 seconds. They select before/after attention and MLP, inspect every gated operation, apply negative coordinate rankings, verify invalid-coordinate rejection, return to Model Journey, switch tokens/layers, reuse source associations, delete a detailed source, suppress stale science and explicitly recover with a new capture.

The [analysis/source expiry run](stage7-browser-analysis-expiry-results.json) passes with actual three-second analysis and twelve-second trace TTLs: expired science disappears, saved-data recomputation adds zero forwards, and root expiry clears the workspace. The [offline run](stage7-regression-browser-offline-results.json) and [existing expiry run](stage7-regression-browser-expiry-results.json) pass separately. The [five predecessor browser cases](stage7-regression-browser-results.json) pass for journey, attention/GQA/long windows, both-prompt lens/candidate evolution and sampling. New predecessor screenshots/fixtures are in `stage7-regression/`; originals remain intact.

## 13. Reproduction

Use the existing setup commands in README; no new installation is required. Run one model process at a time for numerical audits. From `apps/api`:

```sh
LLM_LAB_DEVICE=cpu .venv/bin/pytest -q --real-model
LLM_LAB_DEVICE=mps .venv/bin/pytest -q --real-model
.venv/bin/python tests/stage7_validation.py --device cpu --output ../../docs/validation/stage7-scientific-cpu.json
.venv/bin/python tests/stage7_validation.py --device mps --output ../../docs/validation/stage7-scientific-mps.json
.venv/bin/python -m llm_lab.export_openapi ../../packages/shared/openapi.json --check
```

From root, build/start the production frontend and run the actual backend, then:

```sh
npm run check:types
npm run typecheck
npm run lint
npm test
npm run build
LLM_LAB_BROWSER_STAGE=stage7 npm run test:browser --workspace @llm-lab/web -- residual.spec.ts
LLM_LAB_BROWSER_STAGE=stage7-regression LLM_LAB_BROWSER_OUTPUT_DIR=../../docs/validation/stage7-regression npm run test:browser --workspace @llm-lab/web -- journey.spec.ts attention.spec.ts evolution.spec.ts
```

`tests/stage7_http_validation.py` uses the same live backend's recovery UUIDs from the normal browser run. Short-TTL checks use `LLM_LAB_ANALYSIS_TTL_SECONDS=3 LLM_LAB_TRACE_TTL_SECONDS=12` temporarily and `LLM_LAB_BROWSER_FAULT_MODE=analysis-expiry`; the earlier expiry test uses mode `expiry`. The unavailable test uses mode `offline` with the backend actually stopped. Normal backend settings were restored after validation.

## 14. Known limitations and untested scope

- Only pinned Qwen3 eager float32 CPU/MPS is certified. CUDA, other revisions/models, low precision, fused execution, cached generation and other transformer topologies are untested/unsupported for these analyses.
- Per-operation independent module references are retained as scalar original-capture checks with their declared scope. Old captures lacking them need explicit recapture; no reconstruction is substituted for a raw update.
- Captures and analyses are in memory and expire; UI associations are session-local. Reload/restart loses associations/data. External deletion is discovered on subsequent uncached reads, rather than continuously polling ready sources.
- Narrow diagrams/tables use contained horizontal scrolling. Accessibility evidence is axe plus keyboard and responsive checks; it is not exhaustive assistive-technology or touch-device testing.
- Actual retained bytes and encoded response/cache sizes are measured. Total-process RSS and allocator/accelerator peak memory were not measured here; conservative scratch/capture estimates exclude resident model weights and backend allocator workspaces. Arbitrary long-context Stage 7 throughput is untested.
- Coordinate ranks, norms, cosine and gate values are descriptive; semantic neuron labeling and causal attribution require separate evidence.

No remaining requested gate is failed or skipped without a completed separate run. The preservation defect described above was fully recovered and independently hash-verified.

## 15. Recommendations for Stage 8

Begin only after explicit Stage 8 authorization. Reuse the validated saved-logit processor, serialized model executor, trace identities, reservations, cancellation and absolute-coordinate semantics for bounded full-prefix, cache-disabled generation first. Keep each decision's trace/source distinct, append selected tokens once, validate greedy and controlled sampled parity, and stop before another forward when budgets are exhausted. Preserve Stage 7's explicit-capture/no-hidden-forward boundary for historical exploration. Cached decoding requires a separately validated query/key coordinate and provenance contract.

## 16. Exact file inventory

The following source/document/contract inventory is measured against the initial working tree. Pre-existing Stage 5–6 files that were already untracked are listed as **Modified** when Stage 7 changed their content. All created evidence files are listed separately below.

| Change | File |
| --- | --- |
| Modified | [README.md](../../README.md) |
| Modified | [apps/api/src/llm_lab/analysis/jobs.py](../../apps/api/src/llm_lab/analysis/jobs.py) |
| Modified | [apps/api/src/llm_lab/models/adapters/base.py](../../apps/api/src/llm_lab/models/adapters/base.py) |
| Modified | [apps/api/src/llm_lab/models/adapters/qwen3.py](../../apps/api/src/llm_lab/models/adapters/qwen3.py) |
| Modified | [apps/api/src/llm_lab/schemas/analysis.py](../../apps/api/src/llm_lab/schemas/analysis.py) |
| Modified | [apps/api/src/llm_lab/tracing/validation.py](../../apps/api/src/llm_lab/tracing/validation.py) |
| Modified | [apps/web/src/app/globals.css](../../apps/web/src/app/globals.css) |
| Modified | [apps/web/src/components/attention/useAttentionCapture.ts](../../apps/web/src/components/attention/useAttentionCapture.ts) |
| Modified | [apps/web/src/components/evolution/EvolutionWorkspace.tsx](../../apps/web/src/components/evolution/EvolutionWorkspace.tsx) |
| Modified | [apps/web/src/components/inspector/Inspector.tsx](../../apps/web/src/components/inspector/Inspector.tsx) |
| Modified | [apps/web/src/components/inspector/LayerFlow.tsx](../../apps/web/src/components/inspector/LayerFlow.tsx) |
| Modified | [apps/web/src/components/model-journey/ModelJourney.tsx](../../apps/web/src/components/model-journey/ModelJourney.tsx) |
| Modified | [apps/web/src/components/workspace/LLMWorkspace.tsx](../../apps/web/src/components/workspace/LLMWorkspace.tsx) |
| Modified | [apps/web/src/components/workspace/WorkspaceProviders.tsx](../../apps/web/src/components/workspace/WorkspaceProviders.tsx) |
| Modified | [apps/web/src/lib/api/client.ts](../../apps/web/src/lib/api/client.ts) |
| Modified | [apps/web/src/state/selection.tsx](../../apps/web/src/state/selection.tsx) |
| Modified | [apps/web/tests/browser/evolution.spec.ts](../../apps/web/tests/browser/evolution.spec.ts) |
| Modified | [docs/API_CONTRACT.md](../../docs/API_CONTRACT.md) |
| Modified | [docs/IMPLEMENTATION_STAGES.md](../../docs/IMPLEMENTATION_STAGES.md) |
| Modified | [docs/SCIENTIFIC_VALIDATION.md](../../docs/SCIENTIFIC_VALIDATION.md) |
| Modified | [docs/SYSTEM_ARCHITECTURE.md](../../docs/SYSTEM_ARCHITECTURE.md) |
| Modified | [docs/TRACE_DESIGN.md](../../docs/TRACE_DESIGN.md) |
| Modified | [packages/shared/openapi.json](../../packages/shared/openapi.json) |
| Modified | [packages/shared/src/generated/api.ts](../../packages/shared/src/generated/api.ts) |
| Created | [apps/api/src/llm_lab/analysis/block_internals.py](../../apps/api/src/llm_lab/analysis/block_internals.py) |
| Created | [apps/api/tests/stage7_http_validation.py](../../apps/api/tests/stage7_http_validation.py) |
| Created | [apps/api/tests/stage7_validation.py](../../apps/api/tests/stage7_validation.py) |
| Created | [apps/api/tests/test_stage7_analysis.py](../../apps/api/tests/test_stage7_analysis.py) |
| Created | [apps/web/src/components/residual/BlockDiagnostics.tsx](../../apps/web/src/components/residual/BlockDiagnostics.tsx) |
| Created | [apps/web/src/components/residual/BlockFlow.tsx](../../apps/web/src/components/residual/BlockFlow.tsx) |
| Created | [apps/web/src/components/residual/BlockWorkspace.tsx](../../apps/web/src/components/residual/BlockWorkspace.tsx) |
| Created | [apps/web/tests/browser/residual.spec.ts](../../apps/web/tests/browser/residual.spec.ts) |
| Created | [apps/web/tests/residual.test.tsx](../../apps/web/tests/residual.test.tsx) |

## 17. Created validation evidence

- [STAGE_7_ARCHITECTURAL_ASSESSMENT.md](STAGE_7_ARCHITECTURAL_ASSESSMENT.md)
- [STAGE_7_REPORT.md](STAGE_7_REPORT.md)
- [stage7-analysis-expired.png](stage7-analysis-expired.png)
- [stage7-analysis-expiry-browser-tests.txt](stage7-analysis-expiry-browser-tests.txt)
- [stage7-baseline-hashes.json](stage7-baseline-hashes.json)
- [stage7-browser-analysis-expiry-results.json](stage7-browser-analysis-expiry-results.json)
- [stage7-browser-backend.txt](stage7-browser-backend.txt)
- [stage7-browser-frontend.txt](stage7-browser-frontend.txt)
- [stage7-browser-results.json](stage7-browser-results.json)
- [stage7-browser-tests.txt](stage7-browser-tests.txt)
- [stage7-build.txt](stage7-build.txt)
- [stage7-contract-checks.txt](stage7-contract-checks.txt)
- [stage7-cpu-tests.txt](stage7-cpu-tests.txt)
- [stage7-expiry-backend.txt](stage7-expiry-backend.txt)
- [stage7-expiry-browser-tests.txt](stage7-expiry-browser-tests.txt)
- [stage7-expiry-evidence.json](stage7-expiry-evidence.json)
- [stage7-final-backend.txt](stage7-final-backend.txt)
- [stage7-france-browser.json](stage7-france-browser.json)
- [stage7-france-deleted.png](stage7-france-deleted.png)
- [stage7-france-http-fixtures.json](stage7-france-http-fixtures.json)
- [stage7-france-layer27-token0.png](stage7-france-layer27-token0.png)
- [stage7-france-mlp.png](stage7-france-mlp.png)
- [stage7-france-mobile.png](stage7-france-mobile.png)
- [stage7-france-narrow.png](stage7-france-narrow.png)
- [stage7-france-residual.png](stage7-france-residual.png)
- [stage7-frontend-tests.txt](stage7-frontend-tests.txt)
- [stage7-germany-browser.json](stage7-germany-browser.json)
- [stage7-germany-deleted.png](stage7-germany-deleted.png)
- [stage7-germany-http-fixtures.json](stage7-germany-http-fixtures.json)
- [stage7-germany-layer27-token0.png](stage7-germany-layer27-token0.png)
- [stage7-germany-mlp.png](stage7-germany-mlp.png)
- [stage7-germany-mobile.png](stage7-germany-mobile.png)
- [stage7-germany-narrow.png](stage7-germany-narrow.png)
- [stage7-germany-residual.png](stage7-germany-residual.png)
- [stage7-http-performance.json](stage7-http-performance.json)
- [stage7-http-performance.txt](stage7-http-performance.txt)
- [stage7-initial-state.json](stage7-initial-state.json)
- [stage7-installed-qwen3-source.txt](stage7-installed-qwen3-source.txt)
- [stage7-lint.txt](stage7-lint.txt)
- [stage7-mps-tests.txt](stage7-mps-tests.txt)
- [stage7-offline-browser-tests.txt](stage7-offline-browser-tests.txt)
- [stage7-openapi-check.txt](stage7-openapi-check.txt)
- [stage7-recovery-map.json](stage7-recovery-map.json)
- [stage7-regression-browser-expiry-results.json](stage7-regression-browser-expiry-results.json)
- [stage7-regression-browser-offline-results.json](stage7-regression-browser-offline-results.json)
- [stage7-regression-browser-results.json](stage7-regression-browser-results.json)
- [stage7-regression-browser-tests.txt](stage7-regression-browser-tests.txt)
- [stage7-regression/stage4-api-unavailable.png](stage7-regression/stage4-api-unavailable.png)
- [stage7-regression/stage4-browser-evidence.json](stage7-regression/stage4-browser-evidence.json)
- [stage7-regression/stage4-expired.png](stage7-regression/stage4-expired.png)
- [stage7-regression/stage4-initial.png](stage7-regression/stage4-initial.png)
- [stage7-regression/stage4-layer12.png](stage7-regression/stage4-layer12.png)
- [stage7-regression/stage4-loading.png](stage7-regression/stage4-loading.png)
- [stage7-regression/stage4-logits.png](stage7-regression/stage4-logits.png)
- [stage7-regression/stage4-mobile.png](stage7-regression/stage4-mobile.png)
- [stage7-regression/stage4-model-journey.png](stage7-regression/stage4-model-journey.png)
- [stage7-regression/stage4-second-prompt.json](stage7-regression/stage4-second-prompt.json)
- [stage7-regression/stage4-token-germany.png](stage7-regression/stage4-token-germany.png)
- [stage7-regression/stage5-accessibility.json](stage7-regression/stage5-accessibility.json)
- [stage7-regression/stage5-attention-head.png](stage7-regression/stage5-attention-head.png)
- [stage7-regression/stage5-attention-heatmap.png](stage7-regression/stage5-attention-heatmap.png)
- [stage7-regression/stage5-france-attention.png](stage7-regression/stage5-france-attention.png)
- [stage7-regression/stage5-france-browser.json](stage7-regression/stage5-france-browser.json)
- [stage7-regression/stage5-germany-browser.json](stage7-regression/stage5-germany-browser.json)
- [stage7-regression/stage5-http-fixtures.json](stage7-regression/stage5-http-fixtures.json)
- [stage7-regression/stage5-layer-microscope.png](stage7-regression/stage5-layer-microscope.png)
- [stage7-regression/stage5-long-sequence.json](stage7-regression/stage5-long-sequence.json)
- [stage7-regression/stage5-long-sequence.png](stage7-regression/stage5-long-sequence.png)
- [stage7-regression/stage5-qkv-inspector.png](stage7-regression/stage5-qkv-inspector.png)
- [stage7-regression/stage5-token-relation.png](stage7-regression/stage5-token-relation.png)
- [stage7-regression/stage6-france-analysis.json](stage7-regression/stage6-france-analysis.json)
- [stage7-regression/stage6-france-candidates.png](stage7-regression/stage6-france-candidates.png)
- [stage7-regression/stage6-france-http-fixtures.json](stage7-regression/stage6-france-http-fixtures.json)
- [stage7-regression/stage6-france-journey.png](stage7-regression/stage6-france-journey.png)
- [stage7-regression/stage6-france-lens.png](stage7-regression/stage6-france-lens.png)
- [stage7-regression/stage6-france-narrow.png](stage7-regression/stage6-france-narrow.png)
- [stage7-regression/stage6-france-representation.png](stage7-regression/stage6-france-representation.png)
- [stage7-regression/stage6-france-sampling.png](stage7-regression/stage6-france-sampling.png)
- [stage7-regression/stage6-germany-analysis.json](stage7-regression/stage6-germany-analysis.json)
- [stage7-regression/stage6-germany-candidates.png](stage7-regression/stage6-germany-candidates.png)
- [stage7-regression/stage6-germany-http-fixtures.json](stage7-regression/stage6-germany-http-fixtures.json)
- [stage7-regression/stage6-germany-journey.png](stage7-regression/stage6-germany-journey.png)
- [stage7-regression/stage6-germany-lens.png](stage7-regression/stage6-germany-lens.png)
- [stage7-regression/stage6-germany-narrow.png](stage7-regression/stage6-germany-narrow.png)
- [stage7-regression/stage6-germany-representation.png](stage7-regression/stage6-germany-representation.png)
- [stage7-regression/stage6-germany-sampling.png](stage7-regression/stage6-germany-sampling.png)
- [stage7-ruff.txt](stage7-ruff.txt)
- [stage7-runtime.json](stage7-runtime.json)
- [stage7-scientific-cpu.json](stage7-scientific-cpu.json)
- [stage7-scientific-cpu.txt](stage7-scientific-cpu.txt)
- [stage7-scientific-mps.json](stage7-scientific-mps.json)
- [stage7-scientific-mps.txt](stage7-scientific-mps.txt)
- [stage7-source-expired.png](stage7-source-expired.png)
- [stage7-typecheck.txt](stage7-typecheck.txt)
- [stage7-scope-audit.json](stage7-scope-audit.json) — final baseline comparison and preservation checks.
- [stage7-checks.json](stage7-checks.json) — compact actual gate summary.
