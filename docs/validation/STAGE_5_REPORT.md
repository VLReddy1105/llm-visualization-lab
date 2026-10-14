# Stage 5 — Layer Microscope + Attention Explorer

**PASS · completed 2026-10-09. Stage 6 has not begun.**

This change extends the validated Stage 4 workspace at `631475b` with explicit layer attention capture, shared selections and bounded retained-data views. The model adapter, scientific trace engine, capture profiles and trace store are unchanged. Two additive API capabilities supply selected-slice statistics and output consistency checks. No dependencies were added and no Stage 1–4 validation artifacts were overwritten.

The requested vision, architecture, trace design, visualization workflow, API contract, implementation stages, scientific validation and Stage 1–4 reports were read before implementation. The existing UI, screenshot, selection store, query lifecycle, generated contracts, adapter provenance and GQA mapping were inspected. Relevant installed Next.js guidance was read. Work paused at an automatic approval-review usage limit and resumed on October 9; that interruption did not execute the blocked browser command. Final browser validation subsequently completed successfully.

## Layer architecture and navigation

The existing contextual inspector contains the Layer Microscope. Its simple horizontal view is Residual → Attention → Add → MLP → Add. The detailed view exposes residual input, RMSNorm, q/k/v projections, q/k normalization, RoPE, attention, o_proj, first residual addition, RMSNorm, gated MLP, second residual addition and layer output. The caption states that V bypasses Q/K normalization and RoPE and identifies both residual bypasses. The long detailed flow scrolls horizontally instead of shrinking text.

Major operation buttons select their actual descriptor, a bounded architectural explanation, or an explicit unavailable state. MLP has a summary and Stage 7 placeholder; no MLP/neuron explorer was implemented. Attention opens within the same inspector. The Model Journey map remains above it; its detailed rail collapses during this zoom and returns when zooming out. Model/layer context, query/KV mapping, query/key controls and relation labels identify the current computation. Back to Layer and Back to Model Journey use shared state rather than browser history.

## Explicit capture, identity and lifecycle

Opening attention on an overview shows “Detailed attention data was not captured in this trace.” It sends **no creation request**. The explicit Capture Layer 12 Attention action posts the original raw prompt with `capture: {profile: "attention", layers: [12]}`. All query heads and input tokens are retained for that selected layer by omitting head/token restrictions. Residual boundaries and Q/K/V stages are included by the existing profile.

The original overview remains trace A. The attention panel uses only trace B for its tokens, layer descriptors, head mapping, matrix, mask, relation, vectors and head output. It visibly labels this a separate execution. Trace Details records both UUIDs, profile, exact prompt, expiry and execution metadata. Comparing A and B is explicit in the data model; their tensors are never presented as one execution.

Session associations are keyed by `source overview ID : layer : attention profile` and store only the detailed trace UUID. TanStack Query owns every API payload, keyed by its true trace ID and semantic selectors. Returning to Layer 12 or changing heads/tokens reuses B. Layer 13 requires its own explicit capture. A new overview clears associations and attention selections. Late creation acknowledgements from a replaced source cannot attach to the new session.

Queued/running traces poll every 500 ms, stop after terminal/error states, and expose a two-minute deadline with read-only status checking and cancellation. Creation mutations do not retry automatically. Absolute `expires_at` timers and focus checks remove displayed detail; expired, evicted and missing traces expose explicit recapture. Missing tensors and API errors have no numerical fallback. Backend cancellation follows the existing store contract. Ready traces do not continuously poll for external deletion; a subsequent uncached read discovers it. The overview still governs the lifetime of the exploration session.

## Minimal backend/API additions

1. `SliceRequest.include_statistics` is optional, default false. JSON `TensorSlice.statistics` is nullable; binary responses carry the same optional statistics in their bounded metadata header. Selection and element-budget checks occur before tensor access/reduction. The serializer calls the existing `tensor_statistics` collector helper on the selected CPU slice, with population standard deviation and float32 accumulation. Native masked positions are excluded for mask/masked-score statistics; nonfinite and undefined values preserve the existing schema. The response declares **requested slice coordinates**, preventing whole-tensor norms from masquerading as a selected-vector norm.
2. `GET /api/v1/traces/{trace_id}/compare-output/{reference_trace_id}` acquires both retained read leases and invokes `analysis/trace_comparison.py`. It checks compatibility and compares full retained final-position vocabulary logits/probabilities with the existing `compare_tensors` gates. It also compares the top token. Float32 logits use atol 1e-5 / rtol 1e-4; probabilities use atol 1e-6 / rtol 1e-4. Incompatible revisions, tokenization, prompt, device, dtype, architecture or prediction position return an explicit failed compatibility result. Expired/unready sources retain normal typed errors.

The comparison returns bounded scalar metrics and trace-qualified sources with DERIVED provenance, never full vocabulary arrays. Existing response-byte limits apply. Neither route runs inference or accesses model weights, and route handlers contain no transformer arithmetic. Existing mask, relation, head, matrix and Q/K/V endpoints were reused. OpenAPI, generated TypeScript and the API documentation were updated. Canonical trace path count increases additively from 16 to 17.

## Rendering and selection

The heatmap is a native semantic table/grid containing at most **16 × 16 cells**, with one query head at a time. At this bound, DOM text/buttons provide readable numeric cells, focus and tooltips without Canvas accessibility duplication. Query rows and key columns use exact API token positions. Sticky labels and native horizontal/vertical scrolling preserve cell size; window controls access the remaining sequence. Selected cells center using their bounds relative to the scroll container, including after the native mask arrives and after resize. Arrow keys and Home/End move focus; Enter selects allowed relations. The keyboard pattern follows the [WAI-ARIA grid guidance](https://www.w3.org/WAI/ARIA/apg/patterns/grid/).

Matrix slices use `attention.weights_reconstructed`, visibly RECONSTRUCTED. A separate bounded slice of `position.attention_mask` supplies the mask; TypeScript never constructs `j > i` values. Masked cells have hatching, an accessible unavailable label and cannot select a valid relation. A masked key chosen from the token control displays a masked message and suppresses relation mathematics and vectors. Tokens retain decoded whitespace, native piece, ID and position through accessible labels, tooltips and exact-identity disclosure.

The connection view uses SVG paths plus keyboard-accessible key buttons for the **full selected-query row**. Only native-mask-allowed keys are included. Line width represents the retained weight; opacity is constant. Percentages are presentation formatting; exact weights remain in hover/focus labels. Long rows scroll horizontally. The visible caveat is “Attention weights describe internal routing, not causal attribution.” No displayed subset is renormalized.

The relation inspector reads retained similarity, scaling, scaled score, native mask state and reconstructed weight. Q/K operands are explicitly post-normalization and post-RoPE; the denominator is the complete allowed key row. The frontend performs no dot product, norm, softmax, entropy, averaging or scientific reduction.

Vector plots use small SVG signed dimension strips around zero, with separate Q/K presentation scales based on backend min/max. Selected Q and K appear side by side; stage controls distinguish projected, normalized and reconstructed rotary observations. V uses the mapped KV head at the selected key and bypasses Q/K norm/RoPE. A captured pre-o_proj head-output summary is available. Length, L2 norm, min, max, mean and population std come from the selected-slice response. Raw arrays and provenance are collapsed by default.

Zustand adds `selectedQueryToken`, `selectedKeyToken`, `selectedOperation`, attention trace associations and detail issues, and extends visualization mode/component to attention. Changing the query through the input strip synchronizes `selectedToken` and query, selecting that token’s self-key initially. Clicking a valid matrix/connection relation updates query, key and input token together. Head changes preserve token positions. Returning to the same layer preserves head/relation context; changing layers resets head-specific context. API tensors and descriptors remain outside Zustand.

## Actual numerical validation

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`, MPS float32, eager attention, one raw unpadded sequence, no cache, no sampling or chat template. Head dimension is **128** from model metadata, not hidden-size/query-head division. All query/KV mappings come from adapter/API metadata.

| Observation | Germany | France |
| --- | --- | --- |
| Raw prompt | `The capital of Germany is` | `The capital of France is` |
| Overview UUID | `6bd23005-a85e-4dd1-8d4b-c9d7e039675e` | `b068159e-fc8f-41c3-8f24-6c0dd8bc4969` |
| Detailed UUID | `700f8496-bc57-48dd-badb-913c4f262fdb` | `5602ef47-a328-4223-8099-1ec357eb98dc` |
| Layer / query head / KV head | 12 / 4 / 2 | 12 / 4 / 2 |
| Query | index 4, `" is"`, native `Ġis`, ID 374 | index 4, `" is"`, native `Ġis`, ID 374 |
| Key | index 3, `" Germany"`, native `ĠGermany`, ID 9856 | index 3, `" France"` |
| Post-norm/RoPE Q · K | 78.73936462402344 | 74.0020980834961 |
| Scale | 0.08838834764831845 | 0.08838834764831845 |
| Scaled score | 6.95964241027832 | 6.540923118591309 |
| Mask | Allowed | Allowed |
| Reconstructed weight | 0.4407791793346405 | 0.3483966588973999 |
| Q L2 norm after RoPE | 13.268375396728516 | 13.460029602050781 |
| K L2 norm after RoPE | 38.815914154052734 | 38.10700607299805 |
| V L2 norm | 9.517403602600098 | 10.00419807434082 |

The Germany detail retains **1,703,884 bytes** and passes **96 Stage 2 checks**, with no failed checks. Layer 12’s reconstructed weights and head output are bitwise equal to captured eager observations. Full key-row normalization’s maximum error is 1.1082738637924194e-07; the existing tolerance is 1e-5. All 160 masked head/query/key positions across 16 heads have exactly zero weight; each five-token heatmap displays ten distinct masked cells. The native finite mask sentinel is preserved, not treated as an ordinary score.

The two Germany executions have bitwise-equal complete logits and probabilities, the same top token, and zero maximum absolute difference. The UI warns on a failed/mismatched comparison and marks comparison errors unavailable. No equality is inferred when the original trace is missing.

GQA was checked for every query head: 0/1 → 0, 2/3 → 1, 4/5 → 2, 6/7 → 3, 8/9 → 4, 10/11 → 5, 12/13 → 6, 14/15 → 7. This describes measured adapter data; the UI does not hardcode that division. Backend tests compare all serialized reconstructed matrices to retained tensors and each selected relation to the exact corresponding matrix cell. Chrome verifies identical serialized weight strings between the selected cell, relation API and mathematical output, plus exact Q/K/V statistics.

**Zero-forward proof:** the final browser run’s backend counter was **6 before Germany, 8 after its overview/detail captures, and 8 after all head/token/cell/vector/navigation exploration**. That is two explicit executions and zero retrieval executions. France adds exactly two explicit executions, reaching 10. No browser HTTP mocking was used for these tests. Isolated component tests replay recorded real responses and deliberately inject transport/lifecycle errors only.

The extended prompt contains **28 tokens**. Its final matrix window renders **144 cells**, remains bounded by 256, preserves full-row weights, supports both window axes and shows the selected final cell after an 800-pixel viewport resize. No document-level horizontal overflow was observed in that checked layout. Connections scroll within their own region.

## Verification results

| Gate | Result / evidence |
| --- | --- |
| Full backend, real Qwen MPS | **230 passed**, 20.09 s; [log](stage5-mps-tests.txt), 2026-10-08 |
| Full backend, real Qwen CPU | **230 passed**, 20.20 s; [log](stage5-cpu-tests.txt), 2026-10-08 |
| Frontend | **52 passed** (31 existing + 21 attention/lifecycle), 18.21 s; [log](stage5-frontend-tests.txt) |
| Real Chrome, production build | **2 passed**, 19.6 s; [results](stage5-browser-results.json) |
| TypeScript / ESLint | PASS / PASS |
| Production webpack build | PASS; [log](stage5-build.txt) |
| OpenAPI export / generated types | Both drift checks PASS |
| Python lint / format | Ruff PASS / 57 files formatted |
| Accessibility | **0 axe violations**, 48 passing rules with Q/K/V open; long-sequence state also 0 violations |
| Diff whitespace | `git diff --check` PASS |

Backend coverage adds real selected Q/K/V/head-output statistics and binary parity; full retained-output compatibility and mismatch checks; mask-sentinel exclusion; all-head GQA, matrix and relation equality; nonfinite slice semantics; and element-budget checks before tensor access. All prior Stage 1–3 tests remain in both 230-test runs. The one warning is the existing Starlette/httpx TestClient deprecation. An initial sandbox-only attempt could not see MPS; the recorded passing MPS suite ran with approved GPU access.

Frontend coverage includes explicit creation/plan/prompt, existing-detail reuse, 16 selectors and GQA switching, exact cells and relation scalars, masked clicks, query/key synchronization, allowed connections, Q/K/V stages and raw disclosure, return navigation, independent Layer 13 capture, absolute expiry, expired/evicted/missing errors, missing tensors, output mismatch/unavailability, failed capture, cancellation, deadline/read-only status checking, budget failure, reset and late acknowledgement isolation. Numerical fixtures are development-only and are not imported by production source.

## Screenshot review

Required real-backend screenshots:

- [stage5-layer-microscope.png](stage5-layer-microscope.png)
- [stage5-attention-heatmap.png](stage5-attention-heatmap.png)
- [stage5-attention-head.png](stage5-attention-head.png)
- [stage5-token-relation.png](stage5-token-relation.png)
- [stage5-qkv-inspector.png](stage5-qkv-inspector.png)

Additional evidence: [France](stage5-france-attention.png) and [long sequence](stage5-long-sequence.png). The microscope, matrix/relation, vectors, full explorer and long layout were visually inspected. Refinements fixed selected-token label contrast, heatmap numeric contrast, long connection-number wrapping and selected-cell centering after asynchronous mask loading/resizing. The final images retain the white Stage 4 language, restrained color, readable horizontal operations, explicit mask hatching and side-by-side Q/K plots. Wide detailed architecture and long connections deliberately use contained scrolling.

## No-fake-data audit and limitations

Production source contains no recorded-response imports, random attention, fabricated matrices, synthetic vectors, fallback scientific constants or inferred GQA formula. CSS/SVG arithmetic only maps real values into position, line width, color or signed pixel height; numeric formatting changes presentation only. Scientific statistics, scaling, scores, mask observations and weights originate in the retained backend. Capture creation has one explicit mutation path. Retrieval and selection do not call inference.

Validation covers the pinned Qwen3 eager CPU/MPS float32 configuration, short Germany/France examples and a 28-token browser prompt, together with the broader existing backend suite. CUDA, fp16/bfloat16 visualization, arbitrary long-context performance and other models are not newly validated. Attention weights are routing coefficients, not causal attribution or proof of a named head function. Head aggregation, entropy, cosine comparison and logit-lens analysis are omitted. Vector axes are coordinates without assigned semantic meanings; Q/K charts have independent scales.

UI limits: the matrix uses paged 16 × 16 windows rather than simultaneous full-sequence rendering; long connections require horizontal scrolling. Desktop/800-pixel layouts and keyboard interactions were checked, but this is not an exhaustive screen-reader or touch-device audit. Associations are in-session only; reload loses them and restart loses retained traces. Ready traces discover external removals on subsequent uncached reads rather than continuous polling. Reaching the original overview’s expiry ends its exploration session. No request claims an unmeasured frame rate.

## Required completion checklist (38 items)

| # | Requested item | Result |
| --- | --- | --- |
| 1 | Stage 5 status | PASS |
| 2 | Exact files | Full inventory below: 18 modified, 31 created |
| 3 | Backend/API changes | Optional selected-slice statistics; retained-output comparison; generated contracts |
| 4 | Layer architecture | Two horizontal levels with actual Qwen operations and V/residual bypass explanations |
| 5 | Deep workflow | Explicit attention-profile execution for the same raw prompt |
| 6 | Reuse | Source UUID + layer + profile association; query cache by actual detail UUID |
| 7 | Attention rendering | Bounded semantic DOM grid and SVG connections |
| 8 | Vector rendering | Signed SVG dimension strips; backend statistics; raw disclosure |
| 9 | Selection | Shared query/key/head/token/mode, IDs only for associations |
| 10 | Layer 12 validation | 96 trace checks passed; retained/reconstructed weights and head output agree bitwise |
| 11 | Deep ID | `700f8496-bc57-48dd-badb-913c4f262fdb` |
| 12 | Query head | 4 |
| 13 | KV head | 2 from API |
| 14 | Query token | index 4, `" is"` |
| 15 | Key token | index 3, `" Germany"` |
| 16 | Attention weight | 0.4407791793346405 |
| 17 | QK / scaled score | 78.73936462402344 / 6.95964241027832 |
| 18 | Q norm | 13.268375396728516 |
| 19 | K norm | 38.815914154052734 |
| 20 | V norm | 9.517403602600098 |
| 21 | Heatmap correctness | Exact retained matrix/cell/relation equality; reconstructed provenance |
| 22 | Causal mask | Native mask; 10 distinct masked cells per five-token head; invalid selection rejected |
| 23 | GQA | All 16 API mappings checked against adapter data |
| 24 | Zero extra forwards | 8 → 8 during Germany exploration |
| 25 | Germany browser | PASS, real MPS backend and production build |
| 26 | France browser | PASS, distinct overview/detail executions, real Q/K/V and relation |
| 27 | Frontend tests | 52 passed |
| 28 | MPS regression | 230 passed |
| 29 | CPU regression | 230 passed |
| 30 | TypeScript | PASS |
| 31 | ESLint | PASS |
| 32 | Production build | PASS |
| 33 | Accessibility | 0 axe violations; keyboard, focus, mask and resize checks |
| 34 | Screenshots | Five required paths above, plus France and long-sequence evidence |
| 35 | No-fake-data audit | PASS; production uses retained API values only |
| 36 | Scientific limitations | Pinned model/device/dtype scope; attention is not causal attribution |
| 37 | UI limitations | Bounded windows, contained scrolling, session-only associations, non-exhaustive accessibility audit |
| 38 | Stage 6 | Specified below; not implemented |

## Stage 6 boundary — future work only

Stage 6 will follow one selected input position through model depth using retained residual checkpoints: norms, cosine/change/update summaries and token-journey APIs. It will add immutable analysis jobs/caching and a clearly labeled **final-norm-then-LM-head Logit Lens** recipe, with candidate logit/probability/rank evolution, checkpoint/source-position controls and full-vocabulary normalization. Sampling preview will apply a shared validated processor recipe to saved final-position logits without drawing tokens. Candidate vocabulary IDs remain distinct from input positions; multi-token candidate encoding and missing resident-weight/revision/expiry conditions require explicit handling. Gates include raw-checkpoint agreement, undefined zero-norm cosine, final-layer agreement with actual logits and parity with pinned HF sampling processors. Full residual/MLP screens remain Stage 7; actual autoregressive generation remains Stage 8. **No Stage 6 work was started.**

## Exact file inventory

Paths are relative to the repository root. This inventory is measured against the validated Stage 4 HEAD (`631475b`); generated Next.js environment-path churn was restored to its original content. There are no changes to package dependencies, model/adapter code, trace engine, capture-plan semantics or trace store.

| Change | File |
| --- | --- |
| Modified | [README.md](../../README.md) |
| Modified | [apps/api/src/llm_lab/api/traces.py](../../apps/api/src/llm_lab/api/traces.py) |
| Modified | [apps/api/src/llm_lab/schemas/trace_api.py](../../apps/api/src/llm_lab/schemas/trace_api.py) |
| Modified | [apps/api/src/llm_lab/tracing/serialization.py](../../apps/api/src/llm_lab/tracing/serialization.py) |
| Modified | [apps/api/tests/test_trace_api.py](../../apps/api/tests/test_trace_api.py) |
| Modified | [apps/api/tests/test_trace_serialization.py](../../apps/api/tests/test_trace_serialization.py) |
| Modified | [apps/web/playwright.config.ts](../../apps/web/playwright.config.ts) |
| Modified | [apps/web/src/app/globals.css](../../apps/web/src/app/globals.css) |
| Modified | [apps/web/src/components/inspector/Inspector.tsx](../../apps/web/src/components/inspector/Inspector.tsx) |
| Modified | [apps/web/src/components/inspector/LayerInspector.tsx](../../apps/web/src/components/inspector/LayerInspector.tsx) |
| Modified | [apps/web/src/components/model-journey/ModelJourney.tsx](../../apps/web/src/components/model-journey/ModelJourney.tsx) |
| Modified | [apps/web/src/components/workspace/LLMWorkspace.tsx](../../apps/web/src/components/workspace/LLMWorkspace.tsx) |
| Modified | [apps/web/src/components/workspace/WorkspaceProviders.tsx](../../apps/web/src/components/workspace/WorkspaceProviders.tsx) |
| Modified | [apps/web/src/lib/api/client.ts](../../apps/web/src/lib/api/client.ts) |
| Modified | [apps/web/src/state/selection.tsx](../../apps/web/src/state/selection.tsx) |
| Modified | [docs/API_CONTRACT.md](../API_CONTRACT.md) |
| Modified | [packages/shared/openapi.json](../../packages/shared/openapi.json) |
| Modified | [packages/shared/src/generated/api.ts](../../packages/shared/src/generated/api.ts) |
| Created | [apps/api/src/llm_lab/analysis/trace_comparison.py](../../apps/api/src/llm_lab/analysis/trace_comparison.py) |
| Created | [apps/web/src/components/attention/AttentionExplorer.tsx](../../apps/web/src/components/attention/AttentionExplorer.tsx) |
| Created | [apps/web/src/components/attention/AttentionHeatmap.tsx](../../apps/web/src/components/attention/AttentionHeatmap.tsx) |
| Created | [apps/web/src/components/attention/RelationInspector.tsx](../../apps/web/src/components/attention/RelationInspector.tsx) |
| Created | [apps/web/src/components/attention/TokenConnections.tsx](../../apps/web/src/components/attention/TokenConnections.tsx) |
| Created | [apps/web/src/components/attention/VectorInspector.tsx](../../apps/web/src/components/attention/VectorInspector.tsx) |
| Created | [apps/web/src/components/attention/data.ts](../../apps/web/src/components/attention/data.ts) |
| Created | [apps/web/src/components/attention/useAttentionCapture.ts](../../apps/web/src/components/attention/useAttentionCapture.ts) |
| Created | [apps/web/src/components/inspector/LayerFlow.tsx](../../apps/web/src/components/inspector/LayerFlow.tsx) |
| Created | [apps/web/tests/attention.test.tsx](../../apps/web/tests/attention.test.tsx) |
| Created | [apps/web/tests/browser/attention.spec.ts](../../apps/web/tests/browser/attention.spec.ts) |
| Created | [docs/validation/STAGE_5_REPORT.md](STAGE_5_REPORT.md) |
| Created | [docs/validation/stage5-accessibility.json](stage5-accessibility.json) |
| Created | [docs/validation/stage5-attention-head.png](stage5-attention-head.png) |
| Created | [docs/validation/stage5-attention-heatmap.png](stage5-attention-heatmap.png) |
| Created | [docs/validation/stage5-browser-results.json](stage5-browser-results.json) |
| Created | [docs/validation/stage5-build.txt](stage5-build.txt) |
| Created | [docs/validation/stage5-checks.json](stage5-checks.json) |
| Created | [docs/validation/stage5-cpu-tests.txt](stage5-cpu-tests.txt) |
| Created | [docs/validation/stage5-france-attention.png](stage5-france-attention.png) |
| Created | [docs/validation/stage5-france-browser.json](stage5-france-browser.json) |
| Created | [docs/validation/stage5-frontend-tests.txt](stage5-frontend-tests.txt) |
| Created | [docs/validation/stage5-germany-browser.json](stage5-germany-browser.json) |
| Created | [docs/validation/stage5-http-fixtures.json](stage5-http-fixtures.json) |
| Created | [docs/validation/stage5-layer-microscope.png](stage5-layer-microscope.png) |
| Created | [docs/validation/stage5-long-sequence.json](stage5-long-sequence.json) |
| Created | [docs/validation/stage5-long-sequence.png](stage5-long-sequence.png) |
| Created | [docs/validation/stage5-mps-tests.txt](stage5-mps-tests.txt) |
| Created | [docs/validation/stage5-qkv-inspector.png](stage5-qkv-inspector.png) |
| Created | [docs/validation/stage5-scientific-validation.json](stage5-scientific-validation.json) |
| Created | [docs/validation/stage5-token-relation.png](stage5-token-relation.png) |
