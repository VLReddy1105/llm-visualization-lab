# Stage 7 architectural assessment — 2026-10-12

This assessment precedes Stage 7 implementation. The starting branch is `main`; the full initial dirty inventory and baseline SHA-256 values are preserved in `stage7-initial-state.json` and `stage7-baseline-hashes.json`. Existing Stage 5–6 changes remain uncommitted and must be preserved.

## Existing execution and capture

The installed stack is pinned: Transformers 5.19.0, PyTorch 2.14.1, FastAPI 0.142.2, Pydantic 2.13.5, Next.js 16.4.0 and React 19.3.0. No dependency upgrade is needed. The actual Qwen3 adapter and installed source, architecture/trace/API/scientific policy, Stage 5–6 reports, trace collector/store, analysis manager and linked inspector were inspected.

The real Qwen3 block runs input RMSNorm → attention output projection → residual addition → post-attention RMSNorm → gate projection → SiLU and up projection → elementwise product → down projection → residual addition. RMSNorm uses the model's learned weights, epsilon and native cast order. Hidden width is 1024; gated intermediate width is 3072. The checkpoint revision is `c1899de289a04d12100db370d81485cdf75e47ca`.

Existing run-scoped module hooks directly observe `residual.before` (r0), `attention.output` (a, after o_proj), `residual.after_attention` (r1), `mlp.output` (m), and `residual.after_mlp` (r2). Overview retains those vectors. Selected MLP/layer/deep captures already observe `mlp.input_normalized` (n_mlp), `mlp.gate` (g), `mlp.up` (u), `mlp.activated_gate` (s, the actual act_fn module output) and `mlp.gated` (p, actual down_proj input). The attention-normalized input (n_attn) exists in attention/layer/deep captures, but is absent from the narrower MLP profile. Extend that profile with this one scoped hook, avoiding QKV/attention matrix capture.

## Integration plan

1. Extend the existing `mlp` capture profile for all eleven block operands at an explicit selected layer and absolute token. Preserve the original model forward and run-scoped hook cleanup; no second inference architecture or model is needed.
2. Add `residual_stream` and `mlp_internals` to authoritative Pydantic analysis requests/results. Reuse `/api/v1/traces/{trace_id}/analyses`, the existing summary cache and execution queue. Regenerate OpenAPI and TypeScript.
3. Compute selected-token norms, signed differences, cosines, ratios, reconciliation, coordinate selections and deterministic rankings in Python. Return bounded scalar summaries and coordinate rows with exact trace/layer/token/provenance. Undefined ratios/cosines remain null with reasons; fail inconsistent/nonfinite science explicitly. Preserve existing float32 reconstruction tolerances.
4. Extend the existing Inspector and horizontal Model Journey with Residual Stream and MLP Explorer. Zustand retains only selection and trace associations; TanStack Query owns results. All views follow the existing trace-qualified layer/token state and retain Stage 6 candidate settings. Missing detail offers an explicit capture through the existing workflow, creating a separate source UUID and never merging executions.
5. Validate independent actual-module references on CPU/MPS for Germany/France, positions 0/4, layers 0/12/27; preserve all predecessor tests. Production Chrome tests will use the actual API, collect screenshots, check keyboard/narrow layouts and prove that only explicit captures increment the forward counter.

## Reused lifecycle and bounds

`TraceStore` owns immutable CPU snapshots, alias accounting, reservations, read leases, absolute TTL, LRU eviction and tombstones. The existing analysis service deduplicates canonical trace/version/parameter requests, reserves one max-response slot while active, retains JSON summaries only, checks source liveness before publication and releases source leases on every outcome. Cancellation prevents result publication; an in-flight native operation completes safely. Source expiry/deletion invalidates results; mismatched trace/analysis associations are rejected.

Default analysis bounds are 16 MiB / 128 entries, 32 MiB estimated scratch, one CPU analysis worker, queue capacity 3 and 600-second absolute TTL capped by source expiry. Existing response limit is 1 MiB. These mechanisms will be extended, not duplicated. Stage 7 arithmetic needs no resident weights after capture. Switching layers, positions, tabs, operations or rankings may request an analysis of retained data; it must never create a trace automatically.
