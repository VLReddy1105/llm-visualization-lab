# Implementation stages

Status: **Stage 1 implemented and exercised with real Qwen inference.** The [Stage 1 report](validation/STAGE_1_REPORT.md) records actual completion checks. Stage 2 and subsequent modules, APIs and tests below remain planned; documenting them does not authorize implementation or imply validation.

## Stage execution rules

An instruction such as “Implement Stage 3 only” authorizes that stage's tasks and required verification, then a stop. Check predecessor gates first. If a prerequisite is missing, identify it instead of silently implementing multiple stages. Routine repairs needed to preserve already-implemented predecessor behavior belong to the current stage; report them. Do not add later-stage UI or endpoints because their contracts are documented.

For each stage, update README/status with actual implementation and tests, preserve the semantic contract, and report validation evidence and untested device paths. Scientific features cannot substitute fixtures or attractive mock charts for a real trace. Generated contract files are updated whenever implemented Pydantic schemas change; speculative future schemas remain documentation.

The dependency chain is Stage 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11. A later stage also depends on the specific validation gates named below. Representation-space and causal extensions do not excuse weaknesses in the core trace system.

## Stage 0 — Architecture and workflow

**Objective:** Produce the complete blueprint before writing application code.

**Deliverables:** `README.md` and the seven requested documents in `docs/`.

**Completion gate:** Vision, Qwen mapping, trace vocabulary/provenance, three architecture diagrams, all visualization workflows, API contract, stages, scientific validation, and material risks are documented consistently. Check local links and examples. Structural research is cited and runtime claims remain unverified.

**Stop boundary:** No scaffold, dependency installation, model download/loading, inference, scientific fixtures, or polished UI.

## Stage 1 — Project foundation and model smoke test

**Objective:** Establish separate frontend/backend toolchains and prove real next-token inference on the initial model.

**Dependencies:** Stage 0 blueprint and explicit instruction to begin Stage 1.

**Exact tasks:**

1. Create `apps/web`, `apps/api`, and `packages/shared` workspaces. Configure Next.js, React, TypeScript, Tailwind CSS, Python environment, FastAPI, Pydantic, and Uvicorn; pin compatible versions and lock dependencies.
2. Install only needed ML dependencies (PyTorch, Transformers and their required tokenizer/weight dependencies). Add pytest for meaningful backend checks. Use explicit `.to(device)` placement; add Accelerate only if a specific approved loading strategy needs it.
3. Add local configuration, explicit model registry entry, default limits, local CORS, model manager, asynchronous loading status, and CUDA/MPS/CPU selection with recorded dtype policy.
4. Resolve and pin actual model/tokenizer revisions; download/load Qwen/Qwen3-0.6B only in this authorized stage. Set eval/inference modes and validated eager attention for the baseline. Record software/runtime provenance.
5. Implement `/api/v1/health`, `/status`, `/models`, `/models/{model_id}`, `/models/{model_id}/load`, and `/inference` for a single raw input. Reject chat and unsupported options rather than silently templating text.
6. Run `The capital of Germany is` as a deterministic raw completion, cache disabled. Return effective token IDs/pieces/positions, model/runtime metadata, last-position top raw logits, base probabilities, omitted mass, and argmax preview. Do not assume a particular tokenization or a Berlin result.
7. Generate only the implemented OpenAPI/TypeScript contract and a minimal typed API client. Show backend loading/status and real tokens/output as simple readable text in the frontend, proving communication.
8. Document exact local setup/run commands and tested hardware. Keep blocking inference off the event loop; serialize model execution.

**Expected files/modules:** Root workspace manifest/lockfile and `.gitignore`; `apps/web/package.json`, Next/TS/Tailwind configuration, `src/app/page.tsx`, `src/lib/api/client.ts`; `apps/api/pyproject.toml` and environment lockfile, `src/llm_lab/main.py`, `config.py`, `api/health.py`, `api/models.py`, `api/inference.py`, `schemas/model.py`, `schemas/inference.py`, `models/manager.py`, `models/devices.py`, `models/baseline.py`; `packages/shared/openapi.json`, `src/generated/api.ts`; backend foundation/inference tests. The baseline uses HF Auto classes and config metadata; deep Qwen module mapping starts Stage 2.

**Test criteria:** Process liveness works without weights; loading/error states are truthful; detected device matches effective execution; identical same-device raw-input inference matches a direct HF baseline within the planned dtype tolerance; tokens match the tokenizer; final logits cover V candidates internally; full-vocabulary probabilities normalize and displayed omitted mass is correct; an HTTP result renders in the browser. Invalid/too-long input and unavailable device errors are bounded. Run frontend typecheck/build and appropriate API tests. Device paths not present are marked untested/skipped, not passed.

**Definition of done:** A fresh workspace can follow the documented commands, load the pinned model on at least one available device, and show real deterministic token/output results through the typed frontend/backend connection. Report actual results, memory constraints, and runtime limitations.

**Must not implement yet:** Internal hooks, Trace Engine/Trace Store, trace APIs, model graphs, attention heatmaps, Logit Evolution, sampled generation, geometry, interventions, other models, or decorative fake data.

## Stage 2 — Trace Engine

**Objective:** Produce a trustworthy semantic trace directly from actual execution, without HTTP detail APIs or advanced UI.

**Dependencies:** Stage 1 baseline, pinned revisions/library versions, and usable local execution.

**Exact tasks:**

1. Introduce `BaseModelAdapter`, registry, Qwen3Adapter, metadata/capabilities/component topology, and canonical named-axis schema. Migrate baseline execution through the adapter rather than keeping duplicate inference paths.
2. Inspect the installed Qwen3 implementation and validate module/functional boundaries. Configure a run-scoped instrumentation context with pre/forward hooks and scoped functional observations where safe.
3. Capture embedding, residual checkpoints, actual attention/MLP updates, normalization boundaries, and output. Add requested Q/K/V variants, head outputs, actual eager weights, gate/up/gated values, and mask/scale/position metadata.
4. Distinguish captured and derived SiLU/rotary/score values. Add pure server-side metrics and attention relation derivation from actual operands; do not falsely mark reconstructed tensors as captured.
5. Implement profiles, coverage, summaries, native-precision snapshots, aliases, detach/CPU copying, budget estimation, and finally cleanup. Produce an in-process immutable TraceRecord; storage service comes next stage.
6. Run numerical tests and record capture overhead and peak host/accelerator memory for bounded prompts. Establish a version/device/dtype compatibility report.

**Expected files/modules:** `models/adapters/base.py`, `qwen3.py`, `registry.py`; `schemas/trace.py`, `schemas/components.py`; `tracing/engine.py`, `collector.py`, `capture_plan.py`, `instrumentation.py`, `provenance.py`; `analysis/metrics.py`, `analysis/attention.py`; `tests/test_trace_equivalence.py`, `test_qwen3_shapes.py`, `test_residual_reconciliation.py`, `test_instrumentation_cleanup.py`, `test_capture_budget.py`.

**Test criteria:** Trace output matches matched untraced HF inference; every declared Q/K/V axis agrees with config; GQA head mapping is correct; Q/K normalization precedes RoPE; residual and gated MLP equations reconcile; attention rows/masks/head output reconcile; final pre-norm and normalized states are distinct; handles/wrappers are restored on failure and repeat execution; captured arrays do not retain device storage/graphs or mutable aliases. No inference observer consumes RNG. Unrequested/unsupported data reports the correct availability.

**Definition of done:** All required scientific gates in [Scientific validation](SCIENTIFIC_VALIDATION.md) pass on the supported execution path, and a real in-process trace can supply overview plus requested microscope operands. Release-specific gaps and untested devices are explicit.

**Must not implement yet:** Trace HTTP endpoints/store, graph UI, generation, Logit Lens, PCA/UMAP, interventions, multi-model adapters, or a rewritten transformer forward pass.

## Stage 3 — Trace API

**Objective:** Store actual traces and serve bounded overview/component/slice data through the typed contract.

**Dependencies:** Stage 2 fidelity gates and valid immutable TraceRecords.

**Exact tasks:**

1. Add Trace Store interface and byte-accounted in-memory CPU implementation with reservations, aliases, read leases, TTL, capacity eviction, tombstones, and deletion.
2. Add a bounded asynchronous trace job service over the existing serialized executor; expose queued/running/ready/failed/cancelled states.
3. Implement all Stage 3 endpoints in [API contract](API_CONTRACT.md), including tokens/layers/components, attention/head/relationship details, MLP/residual manifests, logits, and bounded JSON/binary slices.
4. Enforce named selectors, coverage, element/response budgets, pagination, typed errors, stale-source handling, safe cancellation, local origins, and prompt-free diagnostic logs.
5. Generate the implemented schema/types, add TanStack Query and query-key helpers, and extend the simple frontend connection to submit a trace and inspect textual overview/status. This is a contract probe, not Model Journey UI.
6. Demonstrate lazy requests reuse one stored trace and do not invoke additional inference or model loading.

**Expected files/modules:** `storage/base.py`, `memory.py`, `leases.py`, `expiration.py`; `api/traces.py`, `api/components.py`, `api/tensors.py`, `api/errors.py`; `schemas/requests.py`, `schemas/responses.py`, `tracing/jobs.py`; `apps/web/src/lib/api/queries.ts`, query provider; updated generated contracts; `tests/test_trace_api.py`, `test_slice_selectors.py`, `test_store_lifecycle.py`, `test_job_cleanup.py`.

**Test criteria:** Real trace creates/polls/serves correctly; counters prove detail requests do not rerun inference; returned slice values match stored originals; JSON/binary shape/coordinate/dtype metadata agrees; bounds/coverage errors are typed; expiration yields 410 while tombstones exist, unknown IDs 404; simultaneous read/expiry is lease-safe; queued and running cancellation clean up; store bytes include alias deduplication and stay within budget; malformed requests cannot trigger oversized serialization.

**Definition of done:** A typed client can create a real trace, inspect ready summaries, request retained component detail, and handle unavailable/expired states. API/store lifecycle and payload limits pass tests.

**Must not implement yet:** Horizontal Model Journey, visual microscope screens, token-journey analysis route, Logit Lens/sampling-analysis jobs, multi-step generation, geometry, interventions, disk/database persistence, or trace export/import.

## Stage 4 — Model Journey UI

**Objective:** Build the horizontal navigation backbone around real stored traces.

**Dependencies:** Stage 3 API/client/store gates.

**Exact tasks:** Add global selection with Zustand; render input/tokens/embedding/layers/final norm/head/logits/prediction using semantic topology; add zoom/pan/horizontal navigation, fit controls, token/layer/component selection, breadcrumbs, and context panel shell. Show genuine summary values and provenance/capture/expiry states. Add keyboard selection and accessible DOM labels. Virtualize offscreen detail and cancel stale requests.

**Expected files/modules:** `features/model-journey/ModelJourney.tsx`, `JourneyNode.tsx`, `JourneyViewport.tsx`; `features/tokens/TokenStrip.tsx`; `state/selection.ts`, `state/selection-rules.ts`; `features/inspector/ContextPanel.tsx`; shared provenance/availability components; `tests/model-journey.spec.ts`.

**Test criteria:** A fresh prompt produces a new real trace and changes its own summary values; layer count comes from metadata; horizontal/keyboard navigation works; token/layer selection persists appropriately; new unrelated traces reset incompatible positions; late detail responses cannot populate the wrong trace; expired/failed states are truthful. Browser interaction plus build/typecheck pass.

**Definition of done:** Users can follow a real input-to-output journey, select a token/layer, and open context while remaining in one coherent workspace.

**Must not implement yet:** Detailed attention or layer operation screens beyond context placeholders/descriptors, Token Journey curves, Logit Evolution, detailed residual/MLP explorers, generation, geometry, or interventions.

## Stage 5 — Layer and attention microscope

**Objective:** Connect layer topology, real attention relationships, and mathematical operands.

**Dependencies:** Stage 4 linked selection and Stage 2/3 validated attention data/capture APIs.

**Exact tasks:** Render Layer Microscope residual/norm/attention/MLP operation graph; add Attention Explorer heatmap, head selector, query/key selection, connection view, Q/K/V Inspector, and relation walkthrough. Show Q/K normalization and RoPE stages with correct GQA mapping. Use selected rows/tiles and full-row normalization. Offer a new deep/attention capture if missing; show lineage. Include descriptive head aggregation only with declared axes.

**Expected files/modules:** `features/layer-microscope/LayerMicroscope.tsx`; `features/attention/AttentionExplorer.tsx`, `AttentionHeatmap.tsx`, `HeadSelector.tsx`, `TokenConnections.tsx`, `QKVInspector.tsx`, `AttentionRelation.tsx`; adapter-neutral operation renderer; bounded formatting helpers; attention/layer interaction checks.

**Test criteria:** Selected cell maps to the exact stored layer/head/query/key values; head 3 maps to the correct KV head; causal mask differs visually from zero; tile/connection subsets do not renormalize rows; QKV stages and named dimensions are correct; layer/token context survives drill-down; Canvas has keyboard/table alternatives; unsupported/uncaptured details never show numeric substitutes.

**Definition of done:** A user can explain one selected attention relation mathematically using actual operands and weights while navigating the layer's true structure.

**Must not implement yet:** Signature all-depth Token Journey, Logit Evolution, detailed residual/MLP analysis screens, autoregressive timeline, representation fitting, interventions, or other models.

## Stage 6 — Token Journey and Logit Evolution

**Objective:** Deliver persistent token-following across depth and explicitly labeled vocabulary projection analysis.

**Dependencies:** Stage 5 interaction/fidelity gates; retained residual checkpoints and exact matching output weights.

**Exact tasks:** Add token-journey reductions/API; render depth norms, cosine/change/update summaries; implement immutable analysis jobs and result caching; add final-norm-then-LM-head Logit Lens recipe, candidate logit/probability/rank comparisons, source-position/checkpoint controls, and full-vocabulary normalization. Add sampling-preview analysis of saved last-position logits with shared processor recipe; it does not draw tokens. Distinguish input token selection from output candidate IDs and from prediction position. Validate candidate encoding and unavailable resident-weight handling.

**Expected files/modules:** `analysis/token_journey.py`, `logit_lens.py`, `sampling.py`, `jobs.py`; `api/analyses.py`, `api/token_journey.py`, analysis schemas; `features/token-journey/TokenJourney.tsx`, `features/logit-evolution/LogitEvolution.tsx`, `CandidateSelector.tsx`, `SamplingPreview.tsx`; relevant analysis/interaction tests and generated contracts.

**Test criteria:** Journey reductions agree with raw checkpoints; zero-norm cosine is marked undefined; final-layer lens endpoint matches actual final logits within tolerance; candidate probabilities use full vocabulary, not only displayed IDs; switching prediction position changes interpretation/labels; multi-token words are not silently treated as one ID; sampling preview and pinned HF processors agree under documented settings; source expiry/resident-revision errors are truthful.

**Definition of done:** A user follows one position through depth and compares real intermediate projections for compatible candidates, with the method and interpretation limits visible. Previewed sampling settings operate on real saved logits.

**Must not implement yet:** Detailed Residual Stream/MLP Explorer screens, actual sampled autoregressive run/timeline, tuned lens training, PCA/UMAP, interventions, or extra model adapters.

## Stage 7 — Residual Stream and MLP Explorer

**Objective:** Expose the additive residual path and real gated MLP in detail.

**Dependencies:** Stage 6 selected-position analysis; Stage 2 residual/MLP reconciliation and deep capture.

**Exact tasks:** Render before/update/after residual comparisons, update norms/ratios/cosines and reconciliation status; add gated MLP branches and actual activation/product/output summaries; show bounded exact dimensions and signed/absolute unit rankings. Provide a labeled common linear projection only if needed to illustrate vector addition. Reuse existing traces and component endpoints.

**Expected files/modules:** `features/residual-stream/ResidualStream.tsx`, `ResidualComparison.tsx`; `features/mlp-explorer/MLPExplorer.tsx`, `ActivationDistribution.tsx`, `MLPUnitInspector.tsx`; necessary `analysis/residual.py`, `analysis/mlp.py` helpers; residual/MLP interaction and metric tests.

**Test criteria:** Every before/after/update metric matches selected source tensors; actual updates are distinct from subtraction-derived updates; zero denominators are handled; gate/up/SiLU/product/down path follows semantic graph; unit rankings identify actual coordinates; no norm ratio is labeled causal responsibility; unavailable capture offers explicit recapture rather than fabricated plots.

**Definition of done:** Users inspect and numerically reconcile one token's layer updates and gated MLP path with consistent linked selection.

**Must not implement yet:** SAE integration, semantic neuron labels, autoregressive timeline, nonlinear residual-vector diagrams presented as exact addition, representation fitting, causal interventions, or other models.

## Stage 8 — Generation Timeline

**Objective:** Preserve and inspect real autoregressive next-token decisions.

**Dependencies:** Stage 7 core trace UI; Stage 6 validated sampling processor; store reservations and cancellation.

**Exact tasks:** Add generation service/routes and frozen sampler settings; implement bounded full-prefix cache-disabled generation through the same Trace Engine; link each decision step to its own trace; append actual chosen token IDs; expose EOS/length/cancel/budget/failure outcomes; render timeline, step selection, and branch-with-new-settings. Preserve absolute token identities and distinguish chosen tokens from consumed input. Optimize KV caching only after its explicit parity/coverage tests pass; retain full-prefix mode as scientific reference.

**Expected files/modules:** `generation/service.py`, `steps.py`, `sampler.py`, `records.py`; `api/generations.py`, generation schemas; `features/generation-timeline/GenerationTimeline.tsx`, `GenerationControls.tsx`; `tests/test_generation_parity.py`, `test_generation_limits.py`, generation browser checks. Sampler calls the existing `analysis/sampling.py` processor, not a duplicate implementation.

**Test criteria:** Greedy traced/untraced generation matches for equivalent settings; sampled decisions match with restored sampler RNG state on the same supported runtime; each step uses the exact previous IDs; EOS and budget limits stop correctly; cancellation removes observers and live cache references; earlier-position selection persists only where present. If cache mode is enabled, rectangular attention and query/key maps agree with cached computation and matching full-prefix rows under tolerance.

**Definition of done:** A bounded run yields actual chosen tokens and inspectable per-decision traces with reproducible settings, truthful coverage, and bounded retention. Cached mode may remain disabled if its gate is not passed.

**Must not implement yet:** Unbounded streaming retention, editing historical decisions in place, automatic recapture of expired steps, PCA/UMAP/3D, causal intervention with live caches, or additional models.

## Stage 9 — Representation Space

**Objective:** Add reproducible lower-dimensional analysis over real compatible hidden states.

**Dependencies:** Stage 8 trace navigation, Stage 6 analysis jobs, and retained vectors with provenance.

**Exact tasks:** Add scikit-learn PCA and later umap-learn only for this scope; implement fit population/normalization/metric/seed metadata, bounded source extraction, common-basis cross-layer PCA, explained variance, insufficient-data states, and immutable fit caching. Render a 2D linked point plot and layer slider. UMAP is seeded and labeled with its geometry limitations. A separate explicit 3D scope may add Three.js/React Three Fiber after the 2D analysis is validated.

**Expected files/modules:** `analysis/projections.py`, `projection_schema.py`; analysis-route kind extensions; `features/representation-space/RepresentationSpace.tsx`, `ProjectionControls.tsx`; `tests/test_projection_provenance.py`, projection interaction checks; optional 3D module only if later requested.

**Test criteria:** Plot points resolve to correct source trace/token/layer positions; PCA results and explained variance match the implemented fit recipe; a common fit is reused across layer transitions; per-layer fits cannot masquerade as comparable trajectories; seed/version provenance is recorded; insufficient populations and missing captures are explicit; linked selections and browser payload limits remain correct.

**Definition of done:** A defensible 2D analysis of genuine residual geometry works alongside the core journey and declares fit/method limitations. UMAP can be included only with its own verified population/parameter/reproducibility behavior.

**Must not implement yet:** Mandatory 3D, invented semantic city clusters, automatic semantic labels, t-SNE without separately scoped justification, Causal Lab, cross-model coordinate equivalence, or extra adapters.

## Stage 10 — Causal Lab

**Objective:** Add controlled interventions with matched baseline evidence.

**Dependencies:** Stage 9 completed core, validated Stage 2 observer cleanup, Stage 8 execution/sampling controls.

**Exact tasks:** Define semantic intervention targets, query-head output ablation, bounded activation replacement and patching; validate source/target revision/shape/position; reserve paired budgets; execute matched cache-disabled baseline/intervened traces under identical conditions; restore observers/modifications in finally; compute signed raw-logit/probability differences before sampling; render specification, controls, paired data, and experiment provenance.

**Expected files/modules:** `interventions/specs.py`, `executor.py`, `patching.py`; `analysis/intervention_comparison.py`; `api/interventions.py`, intervention schemas; `features/causal-lab/CausalLab.tsx`, `InterventionEditor.tsx`, `ComparisonView.tsx`; `tests/test_intervention_controls.py`, `test_patch_compatibility.py`, `test_intervention_cleanup.py`.

**Test criteria:** A no-op intervention matches baseline; ablated query-head output is zero at the specified pre-projection location; other target slices remain unchanged at that boundary; patch values equal the selected actual source; incompatible/expired sources are rejected; post-intervention ordinary execution again matches baseline; displayed differences reconcile with paired distributions. Do not require every intervention to change the prediction.

**Definition of done:** A saved experiment demonstrates the effect, or absence of effect, of its exact intervention with matched conditions and correct cleanup. Claim language stays limited to the tested context.

**Must not implement yet:** Automatic discovery of the model's reasoning, broad causal feature labels, causal assertions from attention alone, unvalidated cached interventions, additional model adapters, or unsafe serialized executable experiments.

## Stage 11 — Multi-model support

**Objective:** Validate the adapter boundary with a second genuinely different causal LM.

**Dependencies:** Stage 10 stable semantic contract and initial adapter's validation suite. Choose a specific second checkpoint based on available hardware and research needs; do not silently download a model collection.

**Exact tasks:** Inspect the second model's official config/source; implement the appropriate Llama/Gemma/Pythia/TinyLlama-family adapter, revised topology/capabilities and tokenizer policies; run the model-independent conformance suite; preserve Qwen behavior; validate model switching, vocabulary/position reset rules, and unsupported nodes. Add additional adapters incrementally, each with revisions/device/fidelity evidence.

**Expected files/modules:** `models/adapters/<family>.py`, registry/config entry, conformance fixtures sourced from real executions, `tests/test_adapter_conformance.py`, `test_model_switching.py`; capability-driven renderer extensions only where a new operation type actually requires them.

**Test criteria:** Second model's logits/observations match its HF baseline; operation topology represents actual normalization/residual/attention/MLP structure; unsupported Qwen-style concepts remain unavailable; frontend has no module-name logic; incompatible candidates/positions reset correctly; both adapters pass their scientific gates and store/API bounds.

**Definition of done:** At least one additional pinned model works through the same trace/API/selection infrastructure with little frontend restructuring and no scientific substitutions. Document any new semantic operation types and compatibility changes.

**Must not implement yet:** Claims that every HF model works, a universal heuristic hook mapper, silent cross-model feature/coordinate equivalence, or unrelated architecture/platform expansion.

## Release and review discipline

Each stage delivers a legitimate subset of the final system. Preserve the existing baseline and scientific tests while adding the new scope. A stage is complete only when its definition of done is met, measured evidence is recorded, and unsupported paths are identified. Documentation of a future feature does not authorize its implementation.
