# Stage 6 — Token Evolution, Representation Analysis, Logit Lens and Sampling Preview

## 1. Overall status

**PASS · completed 2026-10-11 (Europe/Berlin). Stage 7 has not begun.**

Stage 6 extends the existing Next.js/FastAPI application with five connected views: Token Journey, Representation Evolution, Logit Lens, Candidate Evolution and Sampling Lab. Acceptance is supported by real pinned-Qwen CPU/MPS numerical tests, bounded-cache/lifecycle tests, production Chrome interaction and accessibility checks, explicit forward counters, and a source/evidence audit. No dependencies were added.

Before implementation, the project architecture, trace/API contracts, scientific validation documents, Stage 5 report, existing tests, selection/capture/store mechanisms and actual Qwen normalization/head were inspected. The implementation followed the retained-state → metrics → readout → sampling → shared UI → validation plan. The installed Next.js client-component guidance was read. Validation resumed after interrupted sessions; the final results below supersede exploratory attempts.

## 2. Exact files created and modified

The full measured inventory is in the appendix and [scope audit](stage6-scope-audit.json). Its baseline is the validated Stage 5 **working tree**, captured before Stage 6, not only Git HEAD `631475b` (Stage 4). Stage 5 changes remain present and uncommitted. Every pre-existing Stage 1–5 validation artifact matches its baseline SHA-256. No baseline files were deleted; the implementation-stage plan and Stage 7 definitions are unchanged. Generated Next.js environment-path churn was restored. Regression screenshot filenames are preserved inside a new `stage6-regression/` directory.

## 3. Architectural changes

The existing semantic trace contract remains the source of truth. No duplicate capture engine, model instance or vector store was introduced. `token_journey.py` resolves selected positions through descriptor coordinates; `logit_lens.py` uses the resident adapter; `sampling.py` processes retained final logits. `distributions.py` centralizes full-vocabulary probability and rank policy. `jobs.py` adds a bounded summary cache and one CPU orchestration worker. Weight-dependent jobs enter the existing serialized model executor, sharing its execution boundary with load/trace work. The adapter's existing `native_lm_head` now calls the actual resident head's forward implementation; the existing native final-normalization helper is reused.

The frontend extends the contextual inspector, TanStack Query and Zustand. Token, depth, candidate pins and applied sampler settings survive valid view changes. Chart/timeline selection synchronizes the Model Journey; a selected layer opens the existing Microscope with its actual source trace ID. Returning preserves token/depth. New root traces reset incompatible state. Loading, errors, absolute expiry, cancellation, unavailable vectors and a two-minute polling deadline are explicit; manual status checking remains available after that deadline.

## 4. Backend endpoints and generated contracts

All paths below use `/api/v1/traces`; `/api/traces` is the shared-handler alias.

| Method / relative path | Behavior |
| --- | --- |
| `POST /{trace_id}/analyses` | 202 typed job; deduplicates identical live requests |
| `GET /{trace_id}/analyses/{analysis_id}` | 200 status/progress and bounded ready result |
| `DELETE /{trace_id}/analyses/{analysis_id}` | 200 cancelled job, result cleared |
| `GET /{trace_id}/tokens/{token_index}/journey` | 202 creates/reuses a token-evolution saved-data job |

Pydantic is authoritative. OpenAPI and TypeScript were regenerated and both drift checks pass. Requests use discriminated kinds `token_evolution`, `logit_lens`, `sampling_preview`; responses use `AnalysisJob` with typed results, source ID, method `stage6.v1`, canonical parameters, timestamps, status, checkpoint progress, retained bytes, cache-hit flag and error. Source metadata includes exact token ID/text/native piece/index, model/tokenizer revision, architecture dimensions, configuration/implementation digests, device and precision. Unknown fields, invalid token/candidate IDs and unsupported parameter values fail explicitly.

Lens requests allow 1–20 displayed candidates and at most eight unique, canonically sorted comparison IDs. The recipe is fixed to `final_norm_then_lm_head`. Three default candidates come from the selected position's final raw readout when room remains; users can pin intermediate table entries or exact vocabulary IDs. Strings are never silently treated as single vocabulary tokens. The UI uses generated types, not a competing handwritten contract.

## 5. Trace and capture lifecycle

The ordered checkpoints are actual `embedding.output`, all 28 `layer[i].residual.after_mlp` tensors, then `output.normalized`: **30 checkpoints**, with the final normalization distinctly labeled. Layer count and shapes come from metadata. Absolute token coordinates are resolved even for restricted captures. Missing vectors produce `COMPONENT_NOT_CAPTURED`; summary statistics never substitute for vectors.

Existing overview traces retain the needed states. A missing/summary-only source offers **explicit overview capture** through the established Stage 5 capture hook. This creates trace B for the original prompt while retaining trace A as the root identity. B supplies its own science and Microscope context; data from different executions are not merged. Late capture acknowledgements carry the root owner captured at submission and cannot attach to a new root. Opening a view, changing a token/layer, pinning a candidate or adjusting sampling never invokes trace creation.

Analysis jobs acquire source-store leases, check liveness at progress/publication, and release leases in all outcomes. Expired/deleted sources cannot publish orphan results. Cancellation prevents publication and stops at a checkpoint boundary; an in-flight native operation completes safely. Identical queued/running/ready jobs reuse IDs; failed/cancelled/expired jobs can be explicitly recomputed from retained data. A wrong source/job pair is rejected.

## 6. Cache, resources and measured performance

Defaults: **16 MiB / 128 analysis entries**, at most three active/queued jobs, **600-second absolute analysis TTL** capped by source expiry, **1 MiB response budget**, and **32 MiB estimated tensor scratch budget**. Active jobs reserve one response slot. Terminal summaries are stored as encoded JSON and evicted by LRU; removed-ID metadata is bounded to twice the entry limit. Reads do not renew expiry. Metrics and sampling work without loaded weights; new lens work requires matching resident model/revision/configuration/device/precision. A ready cache hit does not need a new projection.

The implementation streams one checkpoint's vocabulary result at a time and keeps bounded candidate summaries. It never stores a depth-by-vocabulary matrix or duplicates model weights. For this architecture, the conservative preflight estimate `(layers+2)*hidden*16 + vocab*96` is **15,077,376 bytes**. This is an allocation estimate, not a measured total-process RSS or accelerator peak. Source CPU storage remains charged to the existing trace store. Model weights, allocator caches, Python objects and transport copies are outside that tensor estimate; encoded cache/result and entry limits bound their principal retained payloads.

[Performance evidence](stage6-performance.json) measures uncached analyses on already retained traces, then 20 identical POST cache hits per case (120 verified hits total). Times are local HTTP wall time, including polling/serialization; these small samples are not throughput guarantees.

| Prompt | Analysis | Uncached ms | Cached p50 / p95 ms | Largest HTTP response bytes |
| --- | --- | ---: | ---: | ---: |
| Germany | token_evolution | 68.49 | 5.89 / 6.28 | 101,691 |
| Germany | logit_lens | 915.87 | 3.35 / 3.55 | 45,770 |
| Germany | sampling_preview | 57.30 | 1.27 / 1.61 | 4,361 |
| France | token_evolution | 34.18 | 5.90 / 6.44 | 101,691 |
| France | logit_lens | 701.91 | 3.53 / 3.95 | 45,770 |
| France | sampling_preview | 50.66 | 1.43 / 1.61 | 4,357 |

All repeated IDs/results matched. The forward counter stayed **4 → 4**, source retained bytes stayed **14,237,856 → 14,237,856**, and active leases were **0 → 0**. The largest recorded browser result was 101,393 encoded bytes; the measured benchmark's largest complete HTTP response was 101,691 bytes, both below 1 MiB. Tests also exercise LRU eviction, denied admission, scratch/response limits, absolute TTL, cancelled recomputation, source expiry while leased, deferred reclamation, deduplication and cross-source rejection.

## 7. Token evolution numerical examples

The pinned model is `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`: 28 layers, hidden size 1,024, intermediate size 3,072, 16 query / 8 KV heads, head width 128, vocabulary 151,936. Execution is float32 eager, batch one, raw input without a chat template, KV cache or generation. [Runtime versions](stage6-runtime.json) record Python 3.11.15, PyTorch 2.14.1, Transformers 5.19.0, Next.js 16.4.0 and React 19.3.0.

Final browser sources:

- Germany: `c8567e21-559d-4de8-9ffb-514f96c3cd20`, prompt `The capital of Germany is`.
- France: `cf175a51-0d93-4f76-a1bb-0b23fd2bcc06`, prompt `The capital of France is`.

Both have five input tokens. The selected position is **4**, vocabulary ID **374**, exact text ` is`, native piece `Ġis`. Values below come from these saved MPS observations; [full numerical evidence](stage6-scientific-validation.json) preserves precision and provenance.

For vectors `h_i`, backend metrics are `||h_i||₂`, `||h_i||₂−||h_(i−1)||₂`, `||h_i−h_(i−1)||₂`, and dot-product cosine with the previous state, embedding, and last **raw** residual. Reductions use CPU float64. Zero norms yield null cosine plus a reason; the embedding has no preceding checkpoint. Optional collector statistics retain their existing float32, population-standard-deviation semantics. The comparison inspector uses the actual upper-triangle pair summary, explicitly labeled from the earlier to the later checkpoint.

| Prompt | Checkpoint | L2 norm | L2 from previous | Cosine with embedding | Cosine with final raw residual |
| --- | --- | ---: | ---: | ---: | ---: |
| Germany | Embedding | 0.7636887628 | — | 1 | 0.00470203751 |
| Germany | Layer 0 | 7.732839578 | 7.76693628 | 0.00463377255 | 0.316568674 |
| Germany | Layer 12 | 48.60543163 | 13.6008874 | 0.0250868944 | 0.333764535 |
| Germany | Layer 27 | 448.7176692 | 220.867406 | 0.00470203751 | 1 |
| Germany | Final norm | 114.4962216 | 379.844465 | 0.10647691 | 0.682949837 |
| France | Embedding | 0.7636887628 | — | 1 | 0.00348969991 |
| France | Layer 0 | 7.693492443 | 7.72663083 | 0.006146142 | 0.307799848 |
| France | Layer 12 | 48.42867771 | 13.3716115 | 0.0236260298 | 0.334879509 |
| France | Layer 27 | 455.3291834 | 225.805844 | 0.00348969991 | 1 |
| France | Final norm | 113.2930499 | 388.762872 | 0.106956423 | 0.66901778 |

Independent square/sum/sqrt and cosine references validate each selected checkpoint at positions 0 and 4 on CPU and MPS. Reference reductions allow float64 accumulation-order rounding (L2 relative 1e-13 / absolute 1e-12); model-output gates were not relaxed. The raw final residual and final-normalized representation differ and are separately named. Geometry does not establish semantic meaning or causal importance.

## 8. Logit Lens numerical examples

Each raw embedding/post-block vector is independently passed through **the resident Qwen final RMSNorm and actual LM head**. RMSNorm uses the installed model's own per-vector float32 variance/epsilon/learned-weight semantics. The captured final-normalized state uses the head only, avoiding double normalization. Every projection computes all 151,936 logits. The established CPU float64 softmax followed by float32 probabilities is retained to avoid the previously observed CPU float32 normalization discrepancy.

Ranking sorts the complete vocabulary by descending logit, breaking exact ties by ascending vocabulary ID. Returned summaries contain top-N entries, comparison candidates, exact tokenizer mappings, full-vocabulary ranks/probabilities, entropy, probability sum and omitted display mass. Probabilities are never renormalized over displayed tokens. Final defaults and intermediate pinned IDs were derived from actual results, not expected hardcoded words.

| Prompt | Checkpoint | Actual final top candidate / ID | Raw projected logit | Full-vocabulary probability | Rank |
| --- | --- | --- | ---: | ---: | ---: |
| Germany | Embedding | ` Berlin` · 19846 | -9.218358994 | 7.006492322e-45 | 32426 |
| Germany | Layer 0 | ` Berlin` · 19846 | -2.326191425 | 1.774379044e-08 | 10859 |
| Germany | Layer 12 | ` Berlin` · 19846 | -6.864896297 | 4.665747966e-08 | 25083 |
| Germany | Layer 27 | ` Berlin` · 19846 | 16.521492 | 0.4770430624 | 1 |
| Germany | Final norm | ` Berlin` · 19846 | 16.521492 | 0.4770433903 | 1 |
| France | Embedding | ` Paris` · 12095 | -7.106008053 | 5.88545355e-44 | 18941 |
| France | Layer 0 | ` Paris` · 12095 | -2.458047867 | 1.404873906e-08 | 11876 |
| France | Layer 12 | ` Paris` · 12095 | -4.953541756 | 3.782776616e-07 | 10542 |
| France | Layer 27 | ` Paris` · 12095 | 17.49885559 | 0.6568935513 | 1 |
| France | Final norm | ` Paris` · 12095 | 17.49885559 | 0.6568935513 | 1 |

The largest absolute full-vocabulary probability-sum discrepancy across recorded projections was **3.1764449515137017e-08**. Stable float32 output probabilities may underflow to zero for sufficiently small entries; ranking still uses logits. At position 0, the UI correctly projects toward position 1 and says the original-logit comparison is unavailable because only the final input position's original logits were retained.

The UI labels these as **Logit Lens projections**, warns that early-layer readouts may be misleading, and disclaims internal-belief and causal interpretations. Logits, probabilities and ranks have separately labeled charts and quantities. The layer prediction table covers all 30 checkpoints.

## 9. Final projection consistency

Original model values are compared using float64 audit reductions of the original float32 values. Logits and normalization retain the established componentwise `abs(error) <= 1e-5 + 1e-4*abs(reference)` gate; probabilities use `1e-6 + 1e-4*abs(reference)`. No tolerance was widened for Stage 6.

| Prompt | Maximum norm error | Maximum logit error | Maximum probability error | Logit relative L2 error | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Germany | 7.62939453125e-06 | 3.81469726562e-06 | 3.27825546265e-07 | 1.7042153734e-07 | PASS |
| France | 0 | 0 | 0 | 0 | PASS |

France is bitwise equal in all three comparisons. Germany's independently normalized selected row has small float32 rounding differences versus normalization during the original multi-position forward, within all unchanged gates. The report does not claim bitwise equality for Germany. Tests additionally compare embedding, Layer 12, final raw and final-normalized projections against direct calls to the actual model modules, verify full-vocabulary candidate ranks and independent probability normalization, and retain all prior output-reconciliation gates.

## 10. Sampling validation

Saved final-position logits are cloned to CPU float64, then processed by the installed Transformers temperature → top-k → top-p warpers, followed by normalization across **all eligible vocabulary entries**. Top-k retains cutoff ties. Top-p follows the native ascending-sort boundary/minimum-one policy. Raw logits, model probabilities, processed logits and sampling-adjusted probabilities remain distinct. Display count changes presentation only; reset restores temperature 1, top-k 0 and top-p 1.

| Prompt | Temperature / k / p | Eligible entries | Actual top token | Model probability | Adjusted probability | Full adjusted sum |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Germany | 1 / 0 / 1 | 151936 | ` Berlin` | 0.4770433903 | 0.4770433903 | 0.999999991995 |
| Germany | 0.7 / 20 / 0.8 | 1 | ` Berlin` | 0.4770433903 | 1 | 1 |
| France | 1 / 0 / 1 | 151936 | ` Paris` | 0.6568935513 | 0.6568935513 | 0.999999974581 |
| France | 0.7 / 20 / 0.8 | 1 | ` Paris` | 0.6568935513 | 1 | 1 |

The one-entry filtered examples are actual processor results; no token was drawn. Automated parity covers `(1,0,1)`, `(0.7,20,0.8)`, `(2,1,1)`, `(1,4,0.00001)` and `(100,0,0.95)` for both real prompts/devices, plus ties, greedy argmax, out-of-vocabulary k, extreme-temperature overflow, booleans, nonfinite/zero/negative temperatures, invalid p and display bounds. Original tensors and RNG state remain unchanged. CPU analyses work with an unloaded model, and lens tests replace transformer forward/generate entry points with fail-fast sentinels.

## 11. Browser screenshots and interaction evidence

The final production Chrome run has **2 passing Stage 6 tests** in 31.0 seconds, with no HTTP mocks. [Germany observations](stage6-germany-analysis.json), [France observations](stage6-france-analysis.json), [browser results](stage6-browser-results.json), and recorded HTTP fixtures contain the exact source jobs.

| View | Germany | France |
| --- | --- | --- |
| Token Journey | [Screenshot](stage6-germany-journey.png) | [Screenshot](stage6-france-journey.png) |
| Representation Evolution | [Screenshot](stage6-germany-representation.png) | [Screenshot](stage6-france-representation.png) |
| Logit Lens | [Screenshot](stage6-germany-lens.png) | [Screenshot](stage6-france-lens.png) |
| Candidate Evolution | [Screenshot](stage6-germany-candidates.png) | [Screenshot](stage6-france-candidates.png) |
| Sampling Lab | [Screenshot](stage6-germany-sampling.png) | [Screenshot](stage6-france-sampling.png) |
| 800-pixel layout | [Screenshot](stage6-germany-narrow.png) | [Screenshot](stage6-france-narrow.png) |

The journey, metric, candidate, sampling and narrow screenshots were visually reviewed for readable axes, scientific formatting, focus, labels, boundaries and contained horizontal scrolling. Browser checks cover all 30 timeline stops, arrow/End navigation, chart-point selection, layer comparison, Microscope round trips, intermediate candidate pinning, rank selection, all-layer table, token changes, missing original-logit comparison, sampler apply/reset, retained settings and source identity. Germany's forward counter is **2 → 3**, France's **3 → 4**: one explicit capture each, **zero additional exploration forwards**. Prior failed-test captures explain the initial nonzero counter; they are not hidden analysis forwards.

## 12. CPU and MPS regressions

| Gate | Result | Evidence |
| --- | --- | --- |
| Full real-model CPU backend | **269 passed**, 97.98 s | [Log](stage6-cpu-tests.txt) |
| Full real-model MPS backend | **269 passed**, 39.98 s | [Log](stage6-mps-tests.txt) |
| Existing Model Journey and attention Chrome tests | **3 passed**, 33.6 s | [Log](stage6-regression-browser-tests.txt), [results](stage6-regression-browser-results.json) |
| Actual unavailable-backend Chrome test | **1 passed** | [Log](stage6-offline-browser-tests.txt), [results](stage6-regression-browser-offline-results.json) |
| Actual five-second-TTL expiry Chrome test | **1 passed** | [Log](stage6-expiry-browser-tests.txt), [results](stage6-regression-browser-expiry-results.json) |

The backend total includes all 230 prior tests and 39 Stage 6 cases. Both device suites execute the real Germany/France model fixtures. Prior browser coverage retains all 16 GQA head mappings, exact attention/relation/QKV values, explicit detail capture/reuse, the 28-token bounded-window case, horizontal/keyboard journey controls and no exploration forwards. Expiry uses a real configured backend and verifies one explicit trace POST, disappearance of stale layers and an explicit rerun button. The unavailable case uses the actual stopped backend. Earlier-stage artifacts remain intact.

## 13. Frontend, contracts and accessibility

| Gate | Result / evidence |
| --- | --- |
| Full frontend suite | **68 passed** (52 prior + 16 Stage 6), 19.03 s; [log](stage6-frontend-tests.txt) |
| TypeScript | PASS; [log](stage6-typecheck.txt) |
| ESLint | PASS; [log](stage6-lint.txt) |
| Production build | PASS; [log](stage6-build.txt) |
| OpenAPI export + generated TypeScript drift | Both PASS |
| Ruff lint + formatting | PASS; 65 Python files formatted |
| Diff whitespace | PASS |
| Stage 6 axe | **0 violations** in candidate and sampling views for both prompts (45 and 42 passing rules respectively) |
| Existing browser axe checks | PASS; new evidence in `stage6-regression/` |
| Browser JavaScript errors | None in the successful Stage 6 runs |

The 16 new frontend tests cover shared selection/cache reuse, actual recorded numerical results, no original comparison at another position, sampling persistence/reset, five explicit backend error codes, TTL hiding, cancel/recompute/deadline controls, explicit overview capture with B-source Microscope context, late-ACK isolation, root reset and invalid pin rejection. Scientific fixtures are recorded real responses; only transport/lifecycle metadata is deliberately fault-injected. Production code imports none of those fixtures.

Keyboard-accessible native controls and SVG points expose real numeric labels; color legends also include text. The horizontal rail/timeline remains intact. The 800-pixel check shows no document-level horizontal overflow. Automated axe and these keyboard checks do not constitute exhaustive assistive-technology certification.

## 14. Remaining scientific limitations and no-fake-data audit

Validation is scoped to the pinned Qwen3 eager CPU/MPS float32 configuration. New lens execution explicitly rejects unsupported precision and mismatched resident metadata. CUDA, reduced precision, arbitrary models and long-context analysis performance are not newly validated. Positions preceding the final prompt token lack original full-output comparison data. The final-normalized checkpoint is a readout transformation, not another residual block. Norms/cosines are geometric descriptions; diagnostic projections, attention coefficients and candidate probabilities do not prove internal beliefs, semantics or causation. No tuned lens, causal intervention, generation or representation fitting was added.

Production-source review found no random/synthetic scientific values, recorded-response imports, fallback probabilities, browser tensor reductions or hidden forward/generation calls in the Stage 6 analysis path. Browser arithmetic maps server values to chart coordinates and formats them. Scientific work stays on the backend. Missing coverage/metadata fails explicitly. Zero-vector and tie fixtures are isolated algebra tests, not model evidence. Actual HTTP forward counters and fail-fast model sentinels substantiate the no-extra-forward claim.

## 15. Remaining UI limitations

The app remains the existing bright horizontal workspace; no premium redesign was attempted. Wide timelines/tables scroll within their own regions. Dense candidate series and linear probability axes can visually compress small values; exact values remain available in labels, hover text and JSON disclosure. Pair-comparison dropdown state is local to its view, while selected token/layer, candidate pins and applied sampler settings persist. Sampling preview always concerns the original final prompt position, independently of the currently inspected earlier token.

Associations and traces remain in-memory/session scoped. A reload loses UI associations; backend restart loses stored traces/jobs. Ready source deletion is discovered on a subsequent server read rather than continuous ready-state polling. Source/analysis absolute TTL timers hide expired data. Long native operations cannot be interrupted mid-kernel; users see progress and can cancel publication. Measured cache latency is environment-specific, and occasional uncached readouts took longer during concurrent local load.

## 16. Known failures and unsupported cases

There are **no unresolved failures in the mandatory acceptance gates**. Earlier exploratory metric tests used exact equality between different float64 reduction algorithms; their roundoff-only failures were corrected with the tight reference tolerance described above. An existing frontend test exceeded its five-second timeout when CPU tests and the build ran concurrently; the isolated complete suite passed without changing that test. One Chrome assertion originally allowed 15 seconds for a new full-depth readout and expired while the backend continued successfully; it now uses the same explicit 60-second allowance as the first readout. The final two-prompt run passes. The existing Starlette/httpx deprecation warning and Playwright color-environment warning are non-failing.

Unsupported cases fail explicitly: absent raw checkpoints, absent selected-token coverage, unready/expired/deleted source, wrong job/source association, missing or incompatible resident readout, unsupported lens dtype, out-of-range candidate/token IDs, nonfinite observations, budget/queue exhaustion, invalid sampler parameters and overflowing temperatures. Sampling requires retained final-position logits and tokenizer identities. API captures retain up to 100 candidate identities; a rare cutoff tie requesting an identity outside that retained summary raises `CANDIDATE_METADATA_NOT_CAPTURED` instead of inventing token text. Greedy is API-supported and tested; the current UI exposes sampled-distribution controls only. No multi-token phrase encoding is implied by numeric candidate-ID pinning.

## 17. Recommendations for Stage 7 — not implemented

Extend the same trace-qualified selection and explicit capture path into Residual Stream and MLP inspectors. Show retained before/update/after tensors and validate additive reconciliation; distinguish actually captured branch updates from subtraction-derived changes. Reuse generated schemas, bounded slices, leases, absolute TTL and immutable analysis jobs. Add guarded ratios/cosines with zero-denominator reasons, and expose the actual Qwen gate/up/SiLU/product/down path with bounded coordinate rankings. Validate all values on CPU/MPS and preserve Stage 6 final-readout/sampling tests. Do not label neuron coordinates semantically or norm ratios causally without separate evidence. Generation, geometry fitting, tuned lenses and interventions remain outside Stage 7.

## Appendix — exact Stage 6 file inventory

<!-- STAGE6_INVENTORY -->

Paths are relative to the repository root. **24 modified; 80 created; 0 deleted.** [Machine-readable inventory and preserved baseline hashes](stage6-scope-audit.json).

| Change | Exact file |
| --- | --- |
| Modified | [README.md](../../README.md) |
| Modified | [apps/api/src/llm_lab/api/traces.py](../../apps/api/src/llm_lab/api/traces.py) |
| Modified | [apps/api/src/llm_lab/config.py](../../apps/api/src/llm_lab/config.py) |
| Modified | [apps/api/src/llm_lab/main.py](../../apps/api/src/llm_lab/main.py) |
| Modified | [apps/api/src/llm_lab/models/adapters/qwen3.py](../../apps/api/src/llm_lab/models/adapters/qwen3.py) |
| Modified | [apps/api/src/llm_lab/models/manager.py](../../apps/api/src/llm_lab/models/manager.py) |
| Modified | [apps/api/tests/test_trace_api.py](../../apps/api/tests/test_trace_api.py) |
| Modified | [apps/web/src/app/globals.css](../../apps/web/src/app/globals.css) |
| Modified | [apps/web/src/components/attention/useAttentionCapture.ts](../../apps/web/src/components/attention/useAttentionCapture.ts) |
| Modified | [apps/web/src/components/inspector/Inspector.tsx](../../apps/web/src/components/inspector/Inspector.tsx) |
| Modified | [apps/web/src/components/model-journey/ModelJourney.tsx](../../apps/web/src/components/model-journey/ModelJourney.tsx) |
| Modified | [apps/web/src/components/ui/ScientificData.tsx](../../apps/web/src/components/ui/ScientificData.tsx) |
| Modified | [apps/web/src/components/workspace/LLMWorkspace.tsx](../../apps/web/src/components/workspace/LLMWorkspace.tsx) |
| Modified | [apps/web/src/components/workspace/WorkspaceProviders.tsx](../../apps/web/src/components/workspace/WorkspaceProviders.tsx) |
| Modified | [apps/web/src/lib/api/client.ts](../../apps/web/src/lib/api/client.ts) |
| Modified | [apps/web/src/state/selection.tsx](../../apps/web/src/state/selection.tsx) |
| Modified | [apps/web/tests/browser/attention.spec.ts](../../apps/web/tests/browser/attention.spec.ts) |
| Modified | [apps/web/tests/browser/journey.spec.ts](../../apps/web/tests/browser/journey.spec.ts) |
| Modified | [apps/web/tests/browser/lifecycle.spec.ts](../../apps/web/tests/browser/lifecycle.spec.ts) |
| Modified | [docs/API_CONTRACT.md](../../docs/API_CONTRACT.md) |
| Modified | [docs/SYSTEM_ARCHITECTURE.md](../../docs/SYSTEM_ARCHITECTURE.md) |
| Modified | [docs/TRACE_DESIGN.md](../../docs/TRACE_DESIGN.md) |
| Modified | [packages/shared/openapi.json](../../packages/shared/openapi.json) |
| Modified | [packages/shared/src/generated/api.ts](../../packages/shared/src/generated/api.ts) |
| Created | [apps/api/src/llm_lab/analysis/distributions.py](../../apps/api/src/llm_lab/analysis/distributions.py) |
| Created | [apps/api/src/llm_lab/analysis/jobs.py](../../apps/api/src/llm_lab/analysis/jobs.py) |
| Created | [apps/api/src/llm_lab/analysis/logit_lens.py](../../apps/api/src/llm_lab/analysis/logit_lens.py) |
| Created | [apps/api/src/llm_lab/analysis/sampling.py](../../apps/api/src/llm_lab/analysis/sampling.py) |
| Created | [apps/api/src/llm_lab/analysis/token_journey.py](../../apps/api/src/llm_lab/analysis/token_journey.py) |
| Created | [apps/api/src/llm_lab/api/analyses.py](../../apps/api/src/llm_lab/api/analyses.py) |
| Created | [apps/api/src/llm_lab/schemas/analysis.py](../../apps/api/src/llm_lab/schemas/analysis.py) |
| Created | [apps/api/tests/test_stage6_analysis.py](../../apps/api/tests/test_stage6_analysis.py) |
| Created | [apps/web/src/components/evolution/AnalysisStatus.tsx](../../apps/web/src/components/evolution/AnalysisStatus.tsx) |
| Created | [apps/web/src/components/evolution/DepthChart.tsx](../../apps/web/src/components/evolution/DepthChart.tsx) |
| Created | [apps/web/src/components/evolution/EvolutionWorkspace.tsx](../../apps/web/src/components/evolution/EvolutionWorkspace.tsx) |
| Created | [apps/web/src/components/evolution/LensView.tsx](../../apps/web/src/components/evolution/LensView.tsx) |
| Created | [apps/web/src/components/evolution/RepresentationView.tsx](../../apps/web/src/components/evolution/RepresentationView.tsx) |
| Created | [apps/web/src/components/evolution/SamplingLab.tsx](../../apps/web/src/components/evolution/SamplingLab.tsx) |
| Created | [apps/web/src/components/evolution/useAnalysis.ts](../../apps/web/src/components/evolution/useAnalysis.ts) |
| Created | [apps/web/tests/browser/evolution.spec.ts](../../apps/web/tests/browser/evolution.spec.ts) |
| Created | [apps/web/tests/evolution.test.tsx](../../apps/web/tests/evolution.test.tsx) |
| Created | [docs/validation/STAGE_6_REPORT.md](../../docs/validation/STAGE_6_REPORT.md) |
| Created | [docs/validation/stage6-browser-backend.txt](../../docs/validation/stage6-browser-backend.txt) |
| Created | [docs/validation/stage6-browser-results.json](../../docs/validation/stage6-browser-results.json) |
| Created | [docs/validation/stage6-browser-tests.txt](../../docs/validation/stage6-browser-tests.txt) |
| Created | [docs/validation/stage6-build.txt](../../docs/validation/stage6-build.txt) |
| Created | [docs/validation/stage6-checks.json](../../docs/validation/stage6-checks.json) |
| Created | [docs/validation/stage6-contract-checks.txt](../../docs/validation/stage6-contract-checks.txt) |
| Created | [docs/validation/stage6-cpu-tests.txt](../../docs/validation/stage6-cpu-tests.txt) |
| Created | [docs/validation/stage6-expiry-backend.txt](../../docs/validation/stage6-expiry-backend.txt) |
| Created | [docs/validation/stage6-expiry-browser-tests.txt](../../docs/validation/stage6-expiry-browser-tests.txt) |
| Created | [docs/validation/stage6-france-analysis.json](../../docs/validation/stage6-france-analysis.json) |
| Created | [docs/validation/stage6-france-candidates.png](../../docs/validation/stage6-france-candidates.png) |
| Created | [docs/validation/stage6-france-http-fixtures.json](../../docs/validation/stage6-france-http-fixtures.json) |
| Created | [docs/validation/stage6-france-journey.png](../../docs/validation/stage6-france-journey.png) |
| Created | [docs/validation/stage6-france-lens.png](../../docs/validation/stage6-france-lens.png) |
| Created | [docs/validation/stage6-france-narrow.png](../../docs/validation/stage6-france-narrow.png) |
| Created | [docs/validation/stage6-france-representation.png](../../docs/validation/stage6-france-representation.png) |
| Created | [docs/validation/stage6-france-sampling.png](../../docs/validation/stage6-france-sampling.png) |
| Created | [docs/validation/stage6-frontend-tests.txt](../../docs/validation/stage6-frontend-tests.txt) |
| Created | [docs/validation/stage6-germany-analysis.json](../../docs/validation/stage6-germany-analysis.json) |
| Created | [docs/validation/stage6-germany-candidates.png](../../docs/validation/stage6-germany-candidates.png) |
| Created | [docs/validation/stage6-germany-http-fixtures.json](../../docs/validation/stage6-germany-http-fixtures.json) |
| Created | [docs/validation/stage6-germany-journey.png](../../docs/validation/stage6-germany-journey.png) |
| Created | [docs/validation/stage6-germany-lens.png](../../docs/validation/stage6-germany-lens.png) |
| Created | [docs/validation/stage6-germany-narrow.png](../../docs/validation/stage6-germany-narrow.png) |
| Created | [docs/validation/stage6-germany-representation.png](../../docs/validation/stage6-germany-representation.png) |
| Created | [docs/validation/stage6-germany-sampling.png](../../docs/validation/stage6-germany-sampling.png) |
| Created | [docs/validation/stage6-initial-backend.txt](../../docs/validation/stage6-initial-backend.txt) |
| Created | [docs/validation/stage6-lint.txt](../../docs/validation/stage6-lint.txt) |
| Created | [docs/validation/stage6-mps-tests.txt](../../docs/validation/stage6-mps-tests.txt) |
| Created | [docs/validation/stage6-offline-browser-tests.txt](../../docs/validation/stage6-offline-browser-tests.txt) |
| Created | [docs/validation/stage6-performance.json](../../docs/validation/stage6-performance.json) |
| Created | [docs/validation/stage6-regression-browser-expiry-results.json](../../docs/validation/stage6-regression-browser-expiry-results.json) |
| Created | [docs/validation/stage6-regression-browser-offline-results.json](../../docs/validation/stage6-regression-browser-offline-results.json) |
| Created | [docs/validation/stage6-regression-browser-results.json](../../docs/validation/stage6-regression-browser-results.json) |
| Created | [docs/validation/stage6-regression-browser-tests.txt](../../docs/validation/stage6-regression-browser-tests.txt) |
| Created | [docs/validation/stage6-regression/stage4-api-unavailable.png](../../docs/validation/stage6-regression/stage4-api-unavailable.png) |
| Created | [docs/validation/stage6-regression/stage4-browser-evidence.json](../../docs/validation/stage6-regression/stage4-browser-evidence.json) |
| Created | [docs/validation/stage6-regression/stage4-expired.png](../../docs/validation/stage6-regression/stage4-expired.png) |
| Created | [docs/validation/stage6-regression/stage4-initial.png](../../docs/validation/stage6-regression/stage4-initial.png) |
| Created | [docs/validation/stage6-regression/stage4-layer12.png](../../docs/validation/stage6-regression/stage4-layer12.png) |
| Created | [docs/validation/stage6-regression/stage4-loading.png](../../docs/validation/stage6-regression/stage4-loading.png) |
| Created | [docs/validation/stage6-regression/stage4-logits.png](../../docs/validation/stage6-regression/stage4-logits.png) |
| Created | [docs/validation/stage6-regression/stage4-mobile.png](../../docs/validation/stage6-regression/stage4-mobile.png) |
| Created | [docs/validation/stage6-regression/stage4-model-journey.png](../../docs/validation/stage6-regression/stage4-model-journey.png) |
| Created | [docs/validation/stage6-regression/stage4-second-prompt.json](../../docs/validation/stage6-regression/stage4-second-prompt.json) |
| Created | [docs/validation/stage6-regression/stage4-token-germany.png](../../docs/validation/stage6-regression/stage4-token-germany.png) |
| Created | [docs/validation/stage6-regression/stage5-accessibility.json](../../docs/validation/stage6-regression/stage5-accessibility.json) |
| Created | [docs/validation/stage6-regression/stage5-attention-head.png](../../docs/validation/stage6-regression/stage5-attention-head.png) |
| Created | [docs/validation/stage6-regression/stage5-attention-heatmap.png](../../docs/validation/stage6-regression/stage5-attention-heatmap.png) |
| Created | [docs/validation/stage6-regression/stage5-france-attention.png](../../docs/validation/stage6-regression/stage5-france-attention.png) |
| Created | [docs/validation/stage6-regression/stage5-france-browser.json](../../docs/validation/stage6-regression/stage5-france-browser.json) |
| Created | [docs/validation/stage6-regression/stage5-germany-browser.json](../../docs/validation/stage6-regression/stage5-germany-browser.json) |
| Created | [docs/validation/stage6-regression/stage5-http-fixtures.json](../../docs/validation/stage6-regression/stage5-http-fixtures.json) |
| Created | [docs/validation/stage6-regression/stage5-layer-microscope.png](../../docs/validation/stage6-regression/stage5-layer-microscope.png) |
| Created | [docs/validation/stage6-regression/stage5-long-sequence.json](../../docs/validation/stage6-regression/stage5-long-sequence.json) |
| Created | [docs/validation/stage6-regression/stage5-long-sequence.png](../../docs/validation/stage6-regression/stage5-long-sequence.png) |
| Created | [docs/validation/stage6-regression/stage5-qkv-inspector.png](../../docs/validation/stage6-regression/stage5-qkv-inspector.png) |
| Created | [docs/validation/stage6-regression/stage5-token-relation.png](../../docs/validation/stage6-regression/stage5-token-relation.png) |
| Created | [docs/validation/stage6-runtime.json](../../docs/validation/stage6-runtime.json) |
| Created | [docs/validation/stage6-scientific-validation.json](../../docs/validation/stage6-scientific-validation.json) |
| Created | [docs/validation/stage6-scope-audit.json](../../docs/validation/stage6-scope-audit.json) |
| Created | [docs/validation/stage6-typecheck.txt](../../docs/validation/stage6-typecheck.txt) |
