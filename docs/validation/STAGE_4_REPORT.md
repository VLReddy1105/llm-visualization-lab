# Stage 4 — Model Journey UI

**Result: PASS.** Validated on 2026-10-08. Stage 4 is complete; Stage 5 has not begun. Stages 1–3, their model execution, trace storage, capture presets and generated API contracts are unchanged.

The workspace presents one continuous computation through a compact control header, a dominant horizontal Model Journey and a subordinate context inspector. It uses a white research aesthetic, readable node labels, restrained blue/violet accents, aligned connectors and explicit selection. No per-layer execution timing is simulated.

## Rendering decision

The alternatives were evaluated before implementation against a fixed left-to-right computation, 35 overview nodes, keyboard semantics, navigation, maintenance and later semantic drill-down.

| Approach | Assessment |
| --- | --- |
| **React + CSS — chosen** | Native buttons, text, focus and horizontal overflow provide readable labels and browser scrolling. CSS supplies consistent connectors, transformer grouping and the small layer topology. Thirty-five nodes do not warrant a rendering engine. |
| SVG | Useful for arbitrary curved relationships, but this rail requires no geometry engine. SVG would add manual text layout, scrolling and focus work without improving the Stage 4 task. |
| Canvas | Useful for future dense numerical matrices. Its hit testing, keyboard alternatives and text scaling are unnecessary for this small semantic rail. |
| React Flow | Its free graph layout, pan/zoom engine and graph-editing affordances offer little value for a fixed computation. No dependency was added. |
| Hybrid React/SVG | A plausible later choice for complex selected relationships. CSS handles all current connectors and residual bypass paths, so a second rendering surface is unnecessary now. |

Fit Journey switches to tighter node spacing and returns to the start; Reset restores comfortable spacing. A persistent semantic map shows the whole computation and its actual prediction on desktop. The detailed rail remains horizontally scrollable, preserving readable labels instead of scaling 28 layers into tiny text. Numerical canvas zoom is intentionally absent.

## Frontend and state architecture

`page.tsx` stays small and mounts `LLMWorkspace`. Components are grouped by meaning:

- `workspace/`: providers, prompt/trace orchestration, lifecycle and cache isolation.
- `model-journey/`: semantic overview, connected rail, transformer region, navigation and node buttons.
- `tokens/`: real paginated tokenizer output and persistent token selection.
- `inspector/`: context routing, layer structure, component metadata and logits/prediction.
- `trace/`: controls, structured errors and technical metadata disclosure.
- `ui/`: formatting, typed descriptors and collapsible raw evidence.

TanStack Query **5.104.1** owns server state: model readiness, trace lifecycle, tokens, layers, selected-layer descriptors, logits and bounded embedding slices. Stable retained data has infinite stale time, five-minute unused cache collection, no automatic retry and no focus/reconnect refresh. Model readiness has its own loading/focus behavior. Polling belongs only to pending trace overview queries. Generated OpenAPI types remain authoritative; dictionary fields in the existing layer contract are narrowed through runtime record guards rather than duplicate scientific interfaces.

Zustand **5.0.15** owns UI state in a store created per workspace: `traceId`, `selectedToken`, `selectedLayer`, `selectedComponent`, `selectedHead`, `selectedCandidateToken`, `selectedGenerationStep`, `activeVisualizationMode`, viewport mode and an active-trace issue. API payloads are not copied into Zustand. Head and generation selections remain null and the active mode remains Model Journey. A new explicit trace atomically resets incompatible identities and clears/cancels the previous trace cache. Selection APIs are independent of routing, allowing future URL synchronization without requiring it now.

The typed client gained optional external AbortSignals for the Stage 4 calls, composed with its existing timeout. Aborted obsolete reads preserve cancellation instead of becoming API-unavailable errors. Every trace query key includes its trace ID; late deletion and manual status-check results also verify active identity before updating lifecycle state. A discovered duplicate sibling React key was fixed by distinct token/journey/inspector keys and regression checks that a replacement trace still renders exactly five tokens.

Query configuration follows the official [TanStack cancellation documentation](https://tanstack.com/query/latest/docs/framework/react/guides/query-cancellation) and [cache defaults documentation](https://tanstack.com/query/latest/docs/framework/react/guides/important-defaults). Relevant installed Next.js 16.4 guides were read locally, including server/client boundaries, layouts, CSS and `devIndicators`.

## API integration and trace workflow

1. Model status is read from the existing typed client. Loading readiness polls every 1.5 seconds; failures expose Reconnect or the backend error. Trace submission stays disabled while the model is unavailable.
2. TRACE MODEL sends raw text, `capture: { profile: "overview" }`, `display_top_k: 10` to `POST /api/v1/traces?wait=false`. Input is not templated, normalized, truncated or silently resubmitted. Whitespace-only input is rejected clearly.
3. The UI immediately shows submitting, then the actual queued/running state. Stage 3 currently acknowledges a running trace with HTTP 202 after preflight; queued envelopes are supported and tested.
4. `GET /api/v1/traces/{id}?display_top_k=10` polls every 500 ms only while queued/running, with no error. A ready/failed/cancelled response or an error stops polling. The ready ACK/overview is cached without page reload.
5. Ready data enables tokens and layer summaries. `GET /tokens?limit=200` follows all cursors; `GET /layers?limit=200` covers the current 28-layer model. Only the selected layer uses `GET /layers/{index}`. Component metadata and `GET /logits?display_top_k=10` load on selection.
6. With a selected token, Embedding requests a bounded JSON slice of hidden coordinates 0–63 using semantic batch/token/hidden selectors. All 1,024 dimensions are never rendered as a wall of numbers.
7. Ready traces stop polling. `expires_at` drives an absolute deadline timer and a focus-time check. Expiry removes the displayed scientific data and offers Run trace again without creating another trace.
8. Structured errors include API unavailable, model not ready, invalid input, capture/store budget, failed capture, missing component, expiry/eviction and missing trace. A two-minute pending deadline shows Check trace status and Cancel trace; it cannot spin indefinitely. A read-only check does not create inference. Explicit deletion uses the existing DELETE route and releases the retained trace.

A ready trace deleted/evicted externally is recognized on the next uncached read or explicit Verify retained trace; stable ready results deliberately do not continuously poll. Stage 3 returns 404 for an externally deleted UUID without distinguishing it from an unknown UUID. Own deletion is displayed specifically as Trace deleted.

### Capture plan

The unchanged **overview** preset retains embeddings, all-layer base checkpoints/updates, summaries and final outputs, but no deep Q/K/V, attention matrices or gate/activation detail. It supports meaningful inspection of every layer and the bounded token embedding slice. The capital trace retains **3,559,464 bytes (about 3.39 MiB)**. Summary alone would omit the raw embedding slice, while deep capture would retain unnecessary tensors. No Stage 3 preset configuration was modified. Existing token, capture and store budgets continue to apply; longer prompts may be rejected by those existing limits.

## Interaction and scientific presentation

The full order is Input → Tokens → Embedding → L0 … L27 → Final Norm → LM Head → Logits → Next token. All layer indices and counts come from `available.layers` and the layer endpoint, not a hardcoded array. The transformer is one visibly grouped region. A stronger prediction node and map endpoint show the first actual API candidate and its probability.

Mouse wheel/trackpad, background drag, native scrolling, a layer jump selector, selected-node centering, Fit Journey and Reset navigate the rail. Buttons accept Tab/Enter/Space; arrow keys move between nodes and Home/End reach the input/output. Auto-centering honors reduced motion. The status pulse also honors reduced motion; it signals processing without attributing time to individual layers.

The token strip marks spaces with `·`, newlines with `↵`, tabs with `⇥` and carriage returns with `␍`. Accessible labels/tooltips and the token inspector retain the exact quoted decoded token, native tokenizer piece, vocabulary ID, position and character span. Clicking Germany selects input index 3 globally; it remains highlighted when inspecting a layer, embedding or logits. Candidate vocabulary selection is separate from input-token position.

The context inspector explains the current selection:

| Selection | Retained evidence displayed |
| --- | --- |
| Input | Exact prompt, Unicode character count, token count, execution device/dtype and duration. |
| Tokens / individual token | Actual tokenizer output and exact native/decoded representations with IDs and positions. |
| Embedding | Shape, semantic axes, dtype/device, provenance, reduction scope and summary statistics; selected-token bounded slice. |
| Layer | Real single-layer endpoint, residual → RMSNorm → attention → add → RMSNorm → MLP → add → output structural flow, shapes, scoped L2 norms, reconciliation counts and capture coverage. |
| Final Norm | Semantic role, real descriptor, shape, provenance and statistics. |
| LM Head | Retained input descriptor, vocabulary size, output position and a short projection explanation. |
| Logits / Prediction | Actual final-position top ten, raw logits, full-vocabulary probabilities, omitted mass and separate candidate selection. |

Overview attention and MLP updates are captured, while their **deep detail** is explicitly “Not captured in this trace.” Layer summaries cover all captured positions and are not relabeled as the selected token's norm. Logits remain at the final input position even when Germany is selected; the inspector states the distinction. Probabilities are never renormalized over the displayed top ten. The next token is an argmax preview: no token has been appended or generated.

## Actual Qwen and browser validation

The saved HTTP trace is `ffb6c0b3-2406-4d36-bddb-5d79c5e74775`; the final production browser trace is `3ada2086-0185-4e47-a191-c84dcfb62f83`. Both use Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, MPS, float32, the validated eager implementation, raw input and overview capture.

For **The capital of Germany is**, both expose:

- **5 tokens**: exact decoded pieces `The`, ` capital`, ` of`, ` Germany`, ` is`.
- **28 layers**, L0–L27, all represented by real availability metadata.
- Top candidate **` Berlin`**, native `ĠBerlin`, vocabulary ID 19846.
- Probability **0.47704339027404785**, displayed **47.70%**; raw logit **16.52149200439453**.
- Correct metadata, an actual Layer 12 descriptor, final-position logits and retained embedding slice.

The changed prompt **The capital of France is** created a different trace (`363cc7aa-4016-481f-ba27-cbd981249f87`) and returned **` Paris`**, probability **0.6568935513496399**. Editing alone created no trace; explicit execution reset selection and replaced the token strip cleanly.

Browser checks ran in installed headless Google Chrome against the actual FastAPI model and the optimized production Next.js build. They verified initial/loading/ready views, actual values, Layer 12 and token selection, logits/candidate/embedding inspection, wheel scrolling, exact selected-node centering, keyboard navigation, Fit/Reset, ready polling cessation, replacement, deletion and 390-pixel mobile layout. No console/page errors were reported. There was no document-level horizontal overflow on mobile; the journey itself remains horizontal.

The normal run passed one comprehensive test and explicitly skipped the two incompatible fault scenarios. Each fault scenario then passed separately against real backend state: stopped API and five-second absolute trace TTL. Expiry created exactly one trace, removed all layer nodes and offered explicit rerun. No browser HTTP mocking was used.

Accessibility: meaningful button/region/heading semantics, associated prompt/jump labels, visible focus and pressed states, keyboard node movement, error alerts, processing status and readable contrast. The axe scan reported **zero violations** in the tested ready state. This automated scan and keyboard checks do not replace exhaustive assistive-technology testing.

### Visual review

The real screenshots were inspected at desktop and mobile sizes. The title, prompt, token strip, connected computation and selected inspector have a clear hierarchy. Layers stay compact and grouped, connectors align, selection has a strong blue focus treatment and the persistent prediction remains legible while the rail pans. The center rail and semantic overview dominate the desktop view; the inspector sits beneath them. Deep inspector content may require page scrolling. No chart collection, decorative execution animation or dark dashboard was introduced.

## Verification results

| Gate | Result | Evidence |
| --- | --- | --- |
| Frontend isolated HTTP/component tests | **31 passed**, 0 skipped | [JUnit](stage4-frontend-tests.xml) |
| Existing real-model backend MPS suite | **224 passed**, 0 skipped, 21.422 s | [JUnit](stage4-backend-mps.xml) |
| Existing real-model backend CPU suite | **224 passed**, 0 skipped, 18.690 s | [JUnit](stage4-backend-cpu.xml) |
| Production Chrome journey | **1 passed**, 2 separately run fault cases skipped | [Results](stage4-browser-results.json), [actual browser evidence](stage4-browser-evidence.json) |
| Real API unavailable | **1 passed** | [Results](stage4-browser-offline-results.json) |
| Real five-second expiry | **1 passed** | [Results](stage4-browser-expiry-results.json) |
| TypeScript strict checks | **PASS** | `npm run typecheck` |
| ESLint | **PASS** | `npm run lint` |
| Generated TypeScript/OpenAPI drift | **PASS** | `npm run check:types` and backend snapshot `--check` |
| Production build | **PASS**, Next.js 16.4.0 webpack; `/` static | `npm run build` |
| Backend Ruff / formatting | **PASS** | 56 Python files already formatted |
| Scientific source audit | **PASS** | [Audit](stage4-no-fake-data-audit.json) |
| Backend/contract preservation | **PASS**, no modified paths under `apps/api` or `packages/shared` | Git diff inspection |

The backend suites were run serially with the serving process stopped to avoid duplicate resident models. Each reports the existing Starlette/httpx deprecation warning; there were no test failures. Frontend checks cover raw submission, status transitions, all 28 layers, whitespace tokens, linked selection, all inspectors, real-response formatting, errors, expiration, deletion, missing detail, token pagination, no automatic retrace, cache stability, cancellation and late response isolation. Only isolated tests mock HTTP, using recorded actual responses. The browser tests execute real inference.

Environment: Node **22.23.1**, npm **10.9.8**, Next **16.4.0**, React **19.3.0**, TypeScript **5.9.3**, Playwright **1.64.0**, Vitest **5.0.3**. Dependencies are pinned in npm workspace manifests and `package-lock.json`. Install audited zero vulnerabilities. The generated Next.js `AGENTS.md` is retained as instructed by Next's agent-file generator. Detailed results and executed quality commands are consolidated in [stage4-checks.json](stage4-checks.json).

## Performance observations

The browser rendered **35 journey nodes** (28 layers plus seven stages). A 12-change cached Layer 12/27 observation measured **33.30 ms median**, **33.50 ms maximum**, through two `requestAnimationFrame` callbacks per change with reduced motion. This includes the two-frame observation interval and is not a React render-duration measurement, frame-rate guarantee or load benchmark. All changes committed the selected state successfully.

The network log shows one tokens request, one layer-summary request and one read per visited layer; repeated cached layer selection produced no additional reads. Ready overview polling stopped. Both direct HTTP and browser store counters prove **one creation forward and zero additional retrieval forwards**. Native scrolling updates the viewport without storing pointer positions in React state. Only compact summaries/descriptors are displayed; raw disclosures stay closed by default, and the sole vector value request is bounded to 64 coordinates. No full hidden-state tensor, attention matrix or large visualization library is downloaded for this page. The final screenshot trace reports its actual capture timing in the inspector; it is not claimed as a stable performance benchmark.

## No-fake-data audit

All 18 production TypeScript/TSX files were searched for Berlin, the validated probability/logit literals, recorded token/candidate IDs, fixed 28/1,024/151,936 model sizes and `any`. The only `any` matches are the native `AbortSignal.any` API and the English sentence “Select any stage to inspect.” There are no TypeScript `any` declarations or scientific result matches. CSS/SVG numbers define presentation geometry/colors, not model results. Manual source review confirms scientific numbers originate from generated API payloads; production imports never reference the recorded evidence. The allowed default prompt remains. A real changed-prompt browser test confirms output changes through Qwen.

## Screenshots and evidence

All screenshots show the actual application. Ready selections came from actual Qwen traces; no mock scientific fixture was injected.

| View | Path |
| --- | --- |
| Initial application | [stage4-initial.png](stage4-initial.png) |
| Trace processing | [stage4-loading.png](stage4-loading.png) |
| Completed Model Journey | [stage4-model-journey.png](stage4-model-journey.png) |
| Layer 12 with inspector | [stage4-layer12.png](stage4-layer12.png) |
| Germany token selected | [stage4-token-germany.png](stage4-token-germany.png) |
| Logits / prediction inspector | [stage4-logits.png](stage4-logits.png) |
| Mobile | [stage4-mobile.png](stage4-mobile.png) |
| API actually stopped | [stage4-api-unavailable.png](stage4-api-unavailable.png) |
| Actual retention deadline | [stage4-expired.png](stage4-expired.png) |

HTTP artifacts preserve the creation ACK, ready overview, tokens, layer page, Layer 12, embedding/final-norm/LM-head descriptors and logits. These are historical evidence: their UUIDs expire normally and are not a persistence feature. [stage4-http-validation.json](stage4-http-validation.json) records the store forward counts, metadata and memory. [stage4-second-prompt.json](stage4-second-prompt.json) records the changed-prompt output.

## Known limitations

**UI:** Desktop/laptop is the primary target. Detailed layers require horizontal navigation and long inspectors require page scrolling. Fit Journey is semantic overview plus compact spacing, not a numeric zoom level or an all-28-label fit. There is no URL synchronization, inspector resizing, session persistence or multi-model selection. Native semantic map/tokens scroll separately on narrow screens. Externally deleted/evicted traces are detected on an explicit verification or uncached read; ready data is intentionally stable. The current complete layer-summary page fits the fixed 28-layer model; generic future larger models would need cursor traversal there too. Screen-reader and cross-browser testing beyond Chrome/axe remain future validation work.

**Scientific:** Values belong to the pinned revision/configuration and recorded execution path; approximate probabilities may vary on other device/precision paths. Overview contains base checkpoints and summaries, not deep attention/MLP operands. Structural glyphs and connectors communicate architecture, not measured timing, causal influence or importance. Norms/statistics state their captured-position reduction scope. Selected-token embedding previews are 64-coordinate slices without semantic claims. Candidate probabilities refer to the full vocabulary at the final input position, not the selected earlier input token. Argmax preview performs no generation. No logit lens, causal inference, interventions, dimensionality reduction or nearest-neighbor semantics are implemented.

## Stage 5 boundary

Stage 5 will build the **Layer and Attention Microscope** described in `docs/IMPLEMENTATION_STAGES.md`: an interactive residual/norm/attention/MLP operation graph; Attention Explorer heatmap, head selector, query/key selection and token connections; Q/K/V Inspector and a relation walkthrough using the exact retained mathematical operands. It must show Q/K normalization, RoPE and correct grouped-query KV-head mapping, request bounded rows/tiles, keep full-row probabilities, distinguish masked cells from numerical zero and preserve token/layer context. Missing attention detail must offer an **explicit new attention/deep capture with lineage**. Validation must reconcile a selected cell/relation with stored layer/head/query/key values and supply keyboard/table alternatives. No Stage 5 feature was started in this change.

## Requested completion points

| # | Item | Result |
| --- | --- | --- |
| 1 | Stage 4 | **PASS**. |
| 2 | Exact created/modified files | Complete inventory below; no deleted files. |
| 3 | Rendering technology | React/CSS; semantic native controls and horizontal scrolling, no graph/canvas dependency. |
| 4 | Components | Workspace, model journey, tokens, inspector, trace and UI directories. |
| 5 | State | Query owns server data; Zustand owns selections/viewport and trace identity. |
| 6 | Creation/polling | Explicit overview POST; 500 ms pending GETs, stopped on ready/terminal/error. |
| 7 | Journey | Input → Tokens → Embedding → transformer → Final Norm → LM Head → Logits → Next token. |
| 8 | Layers | All 28 actual available indices represented as compact L0–L27 buttons. |
| 9 | Navigation | Native wheel/trackpad, background drag, arrows/Home/End, jump, centering, Fit/Reset. |
| 10 | Global selection | All seven requested fields present; incompatible identities reset per trace. |
| 11 | Tokens | Exact real whitespace/native/ID/index access; persistent Germany selection. |
| 12 | Layers | Select any layer, retain token, read only its existing layer endpoint. |
| 13 | Inspector | Contextual input/token/component/layer/logits views and raw disclosures. |
| 14 | Logits/prediction | Actual final-position raw logits/full-vocabulary probabilities; distinct candidate selection. |
| 15 | Real prompt | Germany capital prompt validated via actual HTTP and production Chrome; France changes output. |
| 16 | Token count | **5**. |
| 17 | Layer count | **28**. |
| 18 | Final prediction | **` Berlin` / 47.70%** from Qwen. |
| 19 | Browser | Main production flow and two actual fault scenarios pass; zero console errors. |
| 20 | Accessibility | Keyboard checks and axe **0 violations** in tested state. |
| 21 | Frontend tests | **31 passed**. |
| 22 | Backend regression | **224 passed MPS; 224 passed CPU**. |
| 23 | TypeScript | **PASS**. |
| 24 | ESLint | **PASS**. |
| 25 | Production build | **PASS**. |
| 26 | Screenshots | Nine real UI screenshots, linked above. |
| 27 | Performance | 35 nodes, stable caches, bounded slice, zero retrieval forwards; measured two-frame selection observation. |
| 28 | No-fake-data audit | **PASS**, recorded production source scan and changed-prompt execution. |
| 29 | UI limitations | Horizontal navigation, semantic Fit, no URL/persistence/resizer, external deletion on read. |
| 30 | Scientific limitations | Pinned computation, summary scope, missing deep detail, final-position argmax preview. |
| 31 | Stage 5 | Interactive Layer/Attention Microscope with real bounded operands and fidelity gates; **not started**. |

## Exact file inventory

Paths below are repository-relative. The inventory is the complete Stage 4 diff against the clean validated Stage 3 HEAD; local dependencies, build products and runtime test outputs excluded by `.gitignore` are not source changes.

### Modified (10)

- `.gitignore`
- `README.md`
- `apps/web/next.config.ts`
- `apps/web/package.json`
- `apps/web/src/app/globals.css`
- `apps/web/src/app/layout.tsx`
- `apps/web/src/app/page.tsx`
- `apps/web/src/lib/api/client.ts`
- `package-lock.json`
- `package.json`

### Created (53)

- `apps/web/AGENTS.md`
- `apps/web/playwright.config.ts`
- `apps/web/src/app/icon.svg`
- `apps/web/src/components/inspector/ComponentInspector.tsx`
- `apps/web/src/components/inspector/Inspector.tsx`
- `apps/web/src/components/inspector/LayerInspector.tsx`
- `apps/web/src/components/inspector/LogitsInspector.tsx`
- `apps/web/src/components/model-journey/JourneyNode.tsx`
- `apps/web/src/components/model-journey/ModelJourney.tsx`
- `apps/web/src/components/tokens/TokenStrip.tsx`
- `apps/web/src/components/trace/ApiErrorNotice.tsx`
- `apps/web/src/components/trace/TraceControls.tsx`
- `apps/web/src/components/trace/TraceDetails.tsx`
- `apps/web/src/components/ui/ScientificData.tsx`
- `apps/web/src/components/workspace/LLMWorkspace.tsx`
- `apps/web/src/components/workspace/WorkspaceProviders.tsx`
- `apps/web/src/lib/api/queries.ts`
- `apps/web/src/state/selection.tsx`
- `apps/web/tests/browser/journey.spec.ts`
- `apps/web/tests/browser/lifecycle.spec.ts`
- `apps/web/tests/setup.ts`
- `apps/web/tests/workspace.test.tsx`
- `apps/web/vitest.config.mts`
- `docs/validation/STAGE_4_REPORT.md`
- `docs/validation/stage4-api-unavailable.png`
- `docs/validation/stage4-backend-cpu.xml`
- `docs/validation/stage4-backend-mps.xml`
- `docs/validation/stage4-browser-evidence.json`
- `docs/validation/stage4-browser-expiry-results.json`
- `docs/validation/stage4-browser-offline-results.json`
- `docs/validation/stage4-browser-results.json`
- `docs/validation/stage4-checks.json`
- `docs/validation/stage4-create-trace.json`
- `docs/validation/stage4-embedding.json`
- `docs/validation/stage4-expired.png`
- `docs/validation/stage4-final-norm.json`
- `docs/validation/stage4-frontend-tests.xml`
- `docs/validation/stage4-http-validation.json`
- `docs/validation/stage4-initial.png`
- `docs/validation/stage4-layer12.json`
- `docs/validation/stage4-layer12.png`
- `docs/validation/stage4-layers.json`
- `docs/validation/stage4-lm-head.json`
- `docs/validation/stage4-loading.png`
- `docs/validation/stage4-logits.json`
- `docs/validation/stage4-logits.png`
- `docs/validation/stage4-mobile.png`
- `docs/validation/stage4-model-journey.png`
- `docs/validation/stage4-no-fake-data-audit.json`
- `docs/validation/stage4-second-prompt.json`
- `docs/validation/stage4-token-germany.png`
- `docs/validation/stage4-tokens.json`
- `docs/validation/stage4-trace-overview.json`

**No files were deleted. No files under `apps/api` or `packages/shared` changed.**
