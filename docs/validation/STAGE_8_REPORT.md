# Stage 8 — Autoregressive Generation and Token-by-Token Visualization

**Result: PASS on CPU and Apple MPS, float32, 2026-10-14.** Bounded real-Qwen generation, independent per-step comparisons, saved-source inspector reuse, expiry, cancellation, and production-browser regressions passed. This report records the completed implementation and the limits of the measurements.

## 1. Overall result

The pinned `Qwen/Qwen3-0.6B` now produces short greedy or seeded sampled continuations through an explicit full-prefix loop. Every completed token decision maps to one real, cache-disabled model forward. The API retains exact prefix IDs, raw argmax, chosen ID, model and sampler probabilities, source identity, and bounded candidate summaries. Optional internal traces are captured from that same forward. Selecting a historical step and opening its original inspectors performs no inference.

The configured default limits are 16 new tokens, 64 prefix tokens, 16 sessions, 2 in-flight generation loops, 4 detailed traces per session, 1 MiB serialized result per session, 16 MiB total reserved session state, 32 MiB detailed trace bytes per session, 32 MiB estimated scratch, 120 seconds of execution time, and a 600-second absolute session TTL. The UI reflects the default 16-token limit. These bounds are configurable within the ceilings in `apps/api/src/llm_lab/config.py` and are listed in `apps/api/.env.example`.

## 2. Implementation scope

Implemented the generation request/session/step contracts; serialized generation service; create, status, session, paginated-step, single-step, cancel, and delete endpoints; exact-token-ID forwarding; same-forward trace capture; sampling/greedy decision summaries; dynamic source-availability reporting; polling and cancellation UI; keyboard-selectable horizontal timeline; probability candidates; and source-qualified reuse of existing Journey, Attention, Residual, and MLP views.

OpenAPI and generated TypeScript were regenerated from Pydantic. No dependencies, model weights, lockfiles, or existing validation reports were changed. No commit, push, merge, or deployment was performed.

## 3. Existing modules reused

Generation runs as one bounded job submitted to `ModelService.submit_trace_job`, the existing serialized model executor. It reuses the resident Qwen3 adapter and `ModelService._compute`, `TraceEngine`, capture collector/plans, `TraceStore` reservations/leases/TTL, Stage 6 `process_logits` and full-vocabulary `distribution`, Stage 7 base residual capture, existing API error/serializer conventions, and the typed TanStack Query client. No second model or parallel model queue is introduced.

## 4. Changed-file inventory

- Backend: `apps/api/.env.example`; `src/llm_lab/config.py`, `main.py`, `models/manager.py`, `schemas/inference.py`, `schemas/trace_api.py`, `tracing/capture_plan.py`, `tracing/collector.py`, `tracing/engine.py`; new `api/generations.py`, `generation/service.py`, `schemas/generation.py`, `tests/test_stage8_generation.py`, and `tests/stage8_validation.py`.
- Frontend/contracts: `apps/web/src/app/globals.css`, `components/workspace/LLMWorkspace.tsx`, new `components/generation/GenerationWorkspace.tsx`, `src/lib/api/client.ts`, `tests/generation.test.tsx`, and `tests/browser/generation.spec.ts`; regenerated `packages/shared/openapi.json` and `packages/shared/src/generated/api.ts`.
- Documentation/evidence: `README.md`, `docs/IMPLEMENTATION_STAGES.md`, `docs/TRACE_DESIGN.md`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/API_CONTRACT.md`, this report, `STAGE_8_ARCHITECTURAL_ASSESSMENT.md`, initial-state/hash records, device audit JSON, browser JSON, and screenshots under `stage8/`.

The clean baseline was branch `stage7-validated`, HEAD `045c6dee422ba2cd8edaebaef51804239b0aac90`, with 379 tracked files hashed before edits. The full hash comparison is in [`stage8-scope-audit.json`](stage8-scope-audit.json); the initial metadata is in [`stage8-initial-state.json`](stage8-initial-state.json) and the baseline SHA-256 map is in [`stage8-baseline-hashes.json`](stage8-baseline-hashes.json). Prior Stage 1–7 evidence is unchanged.

## 5. Generation algorithm

The service tokenizes the original raw prompt once, with special-token insertion disabled and without truncation. For step `i`, it passes `prompt_ids + selected_ids[:i]` exactly to one cache-disabled resident-model forward, obtains last-position raw logits, computes the full-vocabulary base distribution, applies greedy or the configured Stage 6 temperature → top-k → top-p processors, selects one token, publishes a bounded immutable decision summary, then appends that ID once. It checks cancellation, expiry, duration, context, scratch and result-size limits before another forward. `model.generate()` and KV caching are not used.

The original forward can be captured through `TraceEngine`; the trace UUID is separate from the forward execution UUID. Captured traces identify the generation UUID, step index, forward UUID, exact input IDs and decision-position coverage. Deep capture is limited to Layer 12 plus base residual states across layers. A missing or expired trace never triggers recapture.

## 6. Session and step contract

The session records the original prompt/IDs/decode, model and tokenizer revision/identity, device/dtype, execution settings, immutable sampler settings, start/completion and absolute expiry times, lifecycle status, stop reason, progress, resource reservations, trace retention, selected IDs and full decoded sequence. Sessions distinguish `queued`, `running`, `completed`, `stopped`, `cancelled`, and `failed`.

Each step has a generation-qualified step ID, exact prefix IDs and SHA-256, decoded prefix, absolute decision and append positions, vocabulary size, selected ID and native token piece, contextual continuation/full-sequence decode, raw selected logit, raw model probability, adjusted probability when sampled, raw argmax ID, bounded candidate data, distribution summaries, trace coverage/source ID, forward execution ID, timings and provenance. Greedy adjusted probabilities are null. Sampled tokens can differ from argmax. Text is decoded from full token sequences; isolated pieces are never concatenated to invent display text.

## 7. Model-forward provenance

The Stage 8 audit compared every captured greedy step’s retained logits with a separate direct HF forward over the exact stored prefix; all were bitwise equal on both devices. Trace metadata reported exactly one forward and matched the session, step and forward UUID. Request-forward counters increased by exactly one per completed generation decision. Browser inspection and source-expiry checks showed no additional forwards. Independent reference forwards in the audit are separate validation operations and are not generation steps.

## 8. Greedy validation

On both CPU and MPS, the three-token greedy sequences were identical:

| Prompt | IDs | Full decoded continuation |
| --- | --- | --- |
| `The capital of Germany is` | `19846, 13, 576` | ` Berlin. The` |
| `The capital of France is` | `12095, 13, 576` | ` Paris. The` |

Each saved prefix was verified before the independent forward, each selected ID equaled the direct-reference raw-logit argmax, and the final full-sequence decode reconciled with the selected IDs. Exact logits, selected logit/probability, position, and source provenance are recorded in [`stage8-scientific-cpu.json`](stage8-scientific-cpu.json) and [`stage8-scientific-mps.json`](stage8-scientific-mps.json).

## 9. Sampling validation

The audit compared the implementation’s processed logits exactly against the installed Transformers temperature, top-k and top-p warpers on independently obtained real model logits. It checked the unfiltered boundary (`temperature=1`, `top_k=0`, `top_p=1`, 151,936 eligible IDs), a sampled boundary case (`0.7`, `40`, `0.9`), and the one-eligible-token boundary (`0.7`, `1`, `0.01`). All processor outputs were bitwise equal. The selected token was independently drawn from the full adjusted distribution with a separate CPU generator seeded identically; raw selected-token probability and raw argmax were also checked.

Each configuration was run twice per device. Same-device IDs repeated exactly. Seed 53 with temperature 0.7, top-k 40 and top-p 0.9 produced a sampled ID different from raw argmax in the real France browser run (selected ID 12095, raw argmax ID 13 at the first decision). CPU/MPS equality is not claimed as a general property; the generator is explicitly CPU-based on both devices.

## 10. Tokenization and decoding checks

The audit and browser acceptance verified the exact tokenizer IDs for the raw prompts, no implicit chat template or special-token insertion, exact generated-prefix progression, absolute final-prefix decision position, one-time append, contextual full-sequence decoding, and final selected-ID reconciliation. The API now supports exact IDs in `ModelService` without tokenizing decoded text between steps. Whitespace-only prompts, token/prefix limits and strict sampling configurations are rejected before generation starts.

## 11. Memory and lifecycle behavior

The store retains bounded candidate summaries rather than full vocabulary histories. In the measured deep-capture runs, six saved step traces per device used 1,873,532–1,909,012 retained bytes each; sampled sessions without detailed traces remained about 9.3–16.9 KiB. Session state reservations, trace reservations, scratch estimates, request deduplication, TTL, LRU capacity, cancellation and failure cleanup are explicit. Synthetic isolated tests cover EOS, stable greedy ties, exact prefix appends, cancellation during an in-progress forward, partial failure and session expiry.

Source expiry was tested in production Chrome on the real MPS model using a three-second trace TTL and longer session TTL. The decision summary remained visible, inspector actions disappeared, and forward counters did not change. Session expiry and an actually stopped backend were also separately verified in the rebuilt production UI. The service records estimated scratch limits separately from actual retained bytes.

## 12. Backend CPU results

`LLM_LAB_DEVICE=cpu .venv/bin/pytest -q --real-model`: **359 passed**, one existing Starlette deprecation warning. Ruff passed. Isolated tests confirm whitespace-only requests and over-limit prompts are rejected before a model forward, request-key retries deduplicate, and a full-length valid prefix can produce one final decision without attempting an invalid next forward. The independent Stage 8 CPU audit passed for Germany/France greedy runs and three sampling configurations (six repeated sampled sessions). Exact logits, processors, selection, sequence IDs, termination, trace identity and counters were checked.

## 13. Backend MPS results

`LLM_LAB_DEVICE=mps .venv/bin/pytest -q --real-model`: **359 passed**, one existing Starlette deprecation warning. The independent Stage 8 MPS audit passed under float32 eager execution on the pinned Qwen revision. The audit ran in a process with MPS available; the sandboxed default process did not expose MPS, so the device audit and tests were run with the approved local-device execution context.

## 14. Frontend results

The generated contract check, TypeScript, ESLint, production Next build, and Vitest all passed. Frontend tests: **77 passed across 5 files**. The UI includes keyboard timeline navigation, contained horizontal scrolling, backend candidate values, sampled-only controls, source availability, cancellation and session deletion/expiry.

## 15. Production-browser results

The final combined production Chrome run against the real MPS API completed **11 passed, 6 skipped out of 17**. The six skips are separately scoped offline/expiry cases that require a stopped backend or altered TTL. The passing cases include Stage 5 attention, Stage 6 lens/sampling on both prompts, Stage 4 Model Journey, Stage 7 residual/MLP on both prompts, and Stage 8 greedy, sampled, cancellation and invalid-parameter behavior. Stage 8 alone passed four normal cases; session expiry, offline handling and source-trace expiry each passed in separate real-browser runs.

The Germany greedy browser case had no axe violations and passed keyboard navigation and 390-pixel layout width checks. The France sample case showed the distinct selected and argmax IDs and reused Attention Microscope. Screenshots and identities are under [`stage8/`](stage8/) and [`stage8/regression/`](stage8/regression/); the combined Playwright JSON is [`stage8-regression-browser-results.json`](stage8-regression-browser-results.json).

## 16. Measured performance

Across the twelve traced greedy forwards, measured per-forward durations were 143.1–167.4 ms on CPU (median 145.6 ms) and 139.0–193.6 ms on MPS (median 162.6 ms). Three-token session durations were 718–877 ms CPU and 734–1,163 ms MPS. Seeded three-token sampled sessions measured 476–1,131 ms CPU and 229–239 ms MPS. These are short local measurements, not throughput claims; the independent comparisons add validation time outside the recorded generation session.

Captured source traces used about 1.87–1.91 MB each. Generated session JSON measured about 9.3–17.0 KiB. Peak process RSS and accelerator allocator peaks were not measured; the 32 MiB scratch setting is a conservative preflight estimate, not an observed peak-memory result. The cache-disabled full-prefix path recomputes earlier positions for every token, so total work grows with prefix length.

## 17. No-hidden-forward audit

Both device audits checked generation forward deltas against completed decisions. The real browser suite checked counters before and after Model Journey/Attention/Residual/MLP exploration; all exploration deltas were zero. The source-expiry browser test also checked that checking availability caused no forward. No React component or saved-data GET invokes inference.

## 18. Stage 1–7 regression results

All prior numerical backend cases remain passing: the expanded full real-model suite passed 359/359 on CPU and 359/359 on MPS, including the 340 Stage 7 cases and predecessor sampling, attention, trace, residual, and MLP tests. The combined production-browser run passed 11 cases spanning Stages 4–8. Earlier reports and evidence were preserved. Stage 8 browser screenshots and results are written to new paths rather than replacing Stage 1–7 artifacts.

## 19. Known limitations

Generation is intentionally short, local, single-process, cache-disabled and in-memory. The default UI ceiling is 16 new tokens; server configuration may raise it within the 32-token schema ceiling. Detailed attention/QKV and MLP coverage is captured at Layer 12, while base residual states span layers. Source traces may expire or be evicted before their session; the summary survives until its own TTL. CPU and MPS use the same explicit RNG policy, but device-to-device sequence equality is not guaranteed. Peak RSS and MPS allocator memory remain unmeasured.

## 20. Stage 9 recommendations

Stage 9 can add reproducible low-dimensional representation analysis using the retained per-layer base states. Fit population, normalization, method, seed, source IDs and common-basis policy should be explicit. Keep generation step/position identity attached to every point, require compatible retained sources, bound points and response sizes, and test that selecting or changing a projection never starts inference.
