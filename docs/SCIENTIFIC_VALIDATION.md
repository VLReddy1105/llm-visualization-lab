# Scientific validation

Status: Stage 1 output/tokenization/baseline gates and Stage 2 internal capture gates have been exercised on actual pinned Qwen weights using CPU and Apple MPS float32. See the preserved [Stage 1 report](validation/STAGE_1_REPORT.md) and the [Stage 2 report](validation/STAGE_2_REPORT.md) for measured outcomes. Store/API, generation, visualization and intervention gates remain later-stage requirements.

## Reference experiment and compatibility record

For every supported model revision, adapter version, Transformers/PyTorch version, device, dtype, and attention backend, record a compatibility report. A pass on CPU float32 is not a pass on CUDA bfloat16, MPS, cached generation, fused attention, or quantized weights. Missing hardware is explicitly untested.

The primary reference is an ordinary uninstrumented Hugging Face forward pass using the same loaded weights, exact input IDs, masks, positions, eval mode, dtype/device, backend, and cache settings. Compare traced and untraced executions with all observers removed/restored between them. Also retain direct HF tokenization/output evidence so the adapter is not validating itself against its own copied equations.

Initial conditions: batch 1, unpadded raw input, no chat template, no KV cache, eager attention, deterministic next-token argmax preview, and no quantization/compilation/kernel substitution. Inspect the pinned implementation's output semantics before making index assumptions.

Use prompts covering the requested capital example, a one-token sequence, repeated tokens at different positions, whitespace/newlines, non-ASCII/byte-fragment tokenization, and inputs near configured limits. Chat/template inputs are added when implemented. Do not demand “Berlin” or any particular probability; correctness is parity with the actual model under the recorded conditions.

Reports include exact model/tokenizer/config hashes, software versions, hardware, execution/capture settings, seeds/RNG controls, observed errors, tolerances, measured timings/memory, and excluded capabilities. Keep real fixture data clearly identified and bounded. Small algebraic fixtures may test selectors or reductions, but they must never be presented as real model internals or used to replace real-model integration gates.

## Validation matrix

| Test | Method | Required result | Stage |
| --- | --- | --- | --- |
| Tokenizer fidelity | Compare IDs/pieces/effective text with direct tokenizer call | Exact IDs and recorded template/special-token policy | 1 |
| Baseline output | Compare `/inference` last-position logits and top candidates with HF | Equivalent values/ranks within justified profile | 1 |
| Observer neutrality | Same model/input with and without capture | No excess divergence beyond baseline repeatability | 2 |
| Shape/axis fidelity | Assert config-derived projections and named-axis maps | Exact dimensions and absolute coordinate coverage | 2 |
| Normalization and RoPE | Compare observed boundaries and independent reconstruction | Correct stage order and values within dtype tolerance | 2 |
| Residual reconstruction | Compare actual additions and checkpoints | Both residual equations reconcile at every retained position | 2 |
| Attention fidelity | Inspect mask, full row, score/softmax/value mixing | Correct mapped operands, weights, output and mask behavior | 2 |
| Gated MLP fidelity | Compare gate/up/activation/product/down boundaries | Correct activation and element-wise product, output matches | 2 |
| Snapshot and cleanup | Failure/repetition/mutation/weak-reference tests | Owned CPU data; no hooks, wrappers, or device graphs retained | 2 |
| Store/API fidelity | Compare raw stored slices with JSON/binary retrieval | Correct coordinates, values, coverage, availability, expiry | 3 |
| UI selection fidelity | Compare selected plot values with API references | Correct trace/layer/head/position/dimension on every view | 4–7 |
| Logit Lens endpoint | Project final pre-norm layer output with actual norm/head | Matches actual final logits under recipe tolerance | 6 |
| Sampling fidelity | Compare processor chain and RNG-controlled decisions | Same order/filter policy/distribution on same runtime | 6, 8 |
| Generation parity | Compare traced/untraced equivalent generation | Same controlled token sequence or explain numerical tie limits | 8 |
| Geometry provenance | Recompute fit from declared sources/parameters | Consistent population, fit ID, method and coordinates | 9 |
| Intervention controls | No-op, specific target, patch, restoration checks | Matched baseline and scope-correct controlled effects | 10 |
| Adapter conformance | Run equivalent suite on each added family | Honest topology/capabilities and baseline parity | 11 |

## 1. Prove tracing does not modify inference

1. Warm up the reference runtime and repeat untraced inference to establish its own repeatability.
2. Run with the overview capture profile; compare the complete retained last-position vocabulary logits, not only the top candidate.
3. Repeat with attention and deep profiles for selected layers/heads and with all supported observers enabled within limits.
4. Compare actual layer/normalization/attention/MLP boundaries against untraced HF outputs or separately enabled reference outputs where available.
5. Remove all observers, rerun ordinary inference, and require restoration of the baseline.
6. Inject a collector exception and cancellation path; verify finally cleanup and successful subsequent baseline inference.

Observers never replace output values, modify masks/cache/positions, change model.training, or consume sampler randomness. Record hook counts before/during/after and wrapper identity restoration. With deterministic kernels and identical operations, same-backend logits should be bitwise equal; a difference is investigated even if it is below a permissive low-precision tolerance.

Switching from a fused backend to eager attention is a separate execution-mode comparison, not an invisible tracing side effect. Compare untraced eager with traced eager for neutrality, then compare eager with fused inference under a separately justified numeric profile if fused support is later offered. Do not claim captured eager weights belong to the original fused run.

PyTorch notes that floating-point results can differ across devices, releases, and equivalent calculation arrangements. Treat cross-runtime comparisons separately from same-runtime observer neutrality. [Numerical accuracy reference](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

## 2. Tensor shapes, axes, and architecture

Use configuration-derived `D`, `Hq`, `Hkv`, explicit `d`, `I`, and `V`. Assert that query projection width is `Hq*d`, KV widths are `Hkv*d`, concatenation width is `Hq*d`, output-projection width is `D`, and MLP width is `I`. The initial model's explicit head width cannot be inferred as `D/Hq`.

Check every canonical reshape/transposition against original module data, including sample coordinates, rather than comparing shape alone. Verify GQA query-to-KV mapping and compact K/V storage. Distinct copies/views must preserve the head order used by output projection. Shape equality is not evidence that axes are correctly aligned.

Confirm layer count/order, embedding/output tying, normalization epsilon/weights, activation function, positional mechanism, dropout state, and absence/presence of biases against actual loaded modules. Compare observed topology with the installed source, not only a generic diagram. Layer output and final normalized state must be captured distinctly; HF hidden-state tuple conventions cannot silently substitute the normalized last entry for pre-norm final residual.

Tokens carry absolute positions independent of displayed text. Full-prefix traces have query/key positions 0…N−1. Cached traces require separate absolute query/key maps and cache positions. Future padding/packing/sliding windows need new mask/axis tests before support is advertised.

## 3. Residual reconciliation

At each retained layer/position compare:

```text
actual_after_attention ≈ actual_before + actual_attention_output
actual_after_mlp       ≈ actual_after_attention + actual_mlp_output
next_layer_before     = previous_layer_after_mlp
output_residual_final = last_layer_after_mlp
```

Reconstruct using execution dtype and addition order first. A float32 reconstruction of a bfloat16 addition may differ from the rounded model checkpoint; report the distinction. Check native observations before optional lossy storage conversion. Alias deduplication is allowed only after the boundary equivalence is established.

Compare vector values, max absolute error, relative L2 error, and per-position norms. Verify relative update ratio and cosine definitions including zero vectors. A large update norm cannot be reported as a percentage of influence. If a diagram claims to show vector addition, all its points/vectors must share a linear transformation and centering policy that preserves the asserted relationship. Nonlinear projections do not validate this equation.

## 4. Q/K/V normalization, RoPE, and attention

Check projected → per-head normalized → rotary query/key stages against observed boundaries. Independently reconstruct RMSNorm and rotation with actual parameters, positions, and native cast order, within declared tolerance. V must match its actual projection without an invented Q/K normalization or rotary transform.

For selected query head and row:

1. Resolve its actual normalized/rotated Q and mapped context K/V, plus scale and mask.
2. Reconstruct full allowed-key scores; apply the **actual** mask representation and scale/order.
3. Compute softmax over the full key row using the implementation's accumulation/cast policy, then its dropout policy.
4. Compare captured weights with reconstruction as a validation check; preserve their distinct RAW/DERIVED classification in output.
5. Reconstruct weighted values, head ordering/concatenation, and output projection; compare with actual projection input/output.

For initial eval-mode eager attention, valid unpadded query rows should sum approximately to one and weights should be nonnegative. Future/padded/masked positions should receive zero mass under the actual implementation; fully masked or padded query rows require explicit policy instead of a universal sum-to-one assertion. Low-precision casting affects row sums. If training/dropout were ever supported, post-dropout weights need different normalization expectations.

A clicked relation needs a full-row softmax denominator. Test that row/connection/tile display does not renormalize a subset. Test that head means are labeled aggregation and received-attention reductions specify observed query scope. An unavailable/reconstructed fused attention matrix must never be mislabeled captured. The release's [Qwen3 implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py) and [attention interface](https://huggingface.co/docs/transformers/en/attention_interface) are references; actual installed behavior is the test authority.

## 5. MLP reconciliation

Capture the normalized MLP input, actual gate/up outputs, down-projection input, and final MLP update. Verify activated gate matches the configured activation, gated product matches activation(gate) × up, and down projection matches the actual update. If the activation is derived rather than observed, label it and compare its product against the captured down-projection input.

Validate strongest-unit rankings with exact source coordinates and explicit signed/absolute ranking. Histograms declare bin edges, counts, finite handling, source token/layer coverage, and population size. No numerical test can validate an invented natural-language unit label; semantic feature claims need separate evidence/artifacts.

## 6. Output, Logit Evolution, and probability definitions

Check actual final normalization and LM head against direct HF last-position logits. Where weight tying applies, verify the relationship in loaded storage without retaining weights in each trace. Predictions at position `i` refer to the next position, not the selected input token itself.

Compute base probabilities from the actual full-vocabulary logits. Stage 1 found a concrete normalization problem with this runtime's CPU float32 softmax: sum `1.0001225397008613` (float64 audit reduction), exceeding the float32 gate; its maximum probability difference from float64 softmax was `5.845935186632456e-05`. Accordingly, promote only final-vocabulary normalization to CPU float64 and cast probabilities to float32. Record both dtypes; preserve model dtype and raw logits. This replaces the original float32-only output softmax plan based on measured evidence, without relaxing the normalization tolerance. A separate max-shifted exponential/reduction check validates the distribution.

Check finiteness and normalization, top candidates, vocabulary IDs, decoded pieces, rank/tie policy, and omitted mass. Display-top-k must not change those base probabilities. A candidate list is not a full softmax vocabulary.

Logit Lens applies the actual final norm/head to a saved intermediate checkpoint. Verify that applying the same recipe at final layer after-MLP reproduces final inference within justified numeric tolerance; do not apply final norm twice to `output.normalized`. For other checkpoints, validate the recipe and its provenance, not whether its projections match nonexistent intermediate model predictions. Vocabulary chunking may change matmul rounding; compare chunked/candidate results and log-normalizers with a full projection on bounded cases before using it.

Validate that input-token selection, prediction position, and candidate vocabulary IDs remain distinct in both API and UI. Candidate text mapping to multiple IDs is rejected or explicitly converted into a separately scoped sequence-scoring analysis.

## 7. Sampling and generation

Use one backend processor implementation for preview and generation. Disable inherited/default repetition/presence/min-p or other processors unless explicitly requested and recorded. Compare temperature → top-k → top-p → renormalization against equivalent pinned Transformers processors, including ties, boundary-crossing tokens, top-k=0, top-p=1, and numerical edge cases. Verify preview draws no token and consumes no generation RNG.

Greedy mode uses explicit argmax and canonical disabled filters. Temperature zero is not passed into sampled-mode division. For sampled parity, capture/restore the same sampler RNG state, processor settings, device, dtype, vocabulary order, and source logits. A seed alone is not a guarantee of cross-hardware identity. Record deterministic greedy parity separately from stochastic repeatability.

Compare traced and untraced bounded sequences at every decision's full last-position logits, processed distribution, and chosen token. A near tie may make a small numerical backend difference change argmax; report logit margins and the discrepancy rather than widening tolerance until tokens appear equivalent.

Stage 8 first validates full-prefix cache-disabled traces. Cached execution, if added, compares its selected query rows and current-position logits with full-prefix reference under a dedicated profile. Check that cached K/V use correct absolute positions, historical rows are labeled originating data, the newly selected token is appended exactly once, and template processing is not reapplied each step. Test EOS, max length, budget stop, cancellation, and cache cleanup.

[Transformers generation utilities](https://huggingface.co/docs/transformers/en/internal/generation_utils) are the processor/output reference. Deterministic smoke tests are correctness tests rather than claims of recommended conversational decoding quality.

## 8. Numerical tolerance policy

Record errors before evaluating pass/fail. Use an elementwise criterion `abs(actual-reference) <= atol + rtol*abs(reference)`, plus max absolute error, relative L2 error with a stated epsilon, and relevant ranks/logit margins. Do not use relative error alone near zero. Masked/non-finite entries are checked separately. Summaries specify their accumulation dtype and reduction axes.

The following are **provisional reconstruction ceilings**, not measured accepted errors and not permission to hide observer effects:

| Execution dtype | Initial absolute ceiling | Initial relative ceiling | Valid-row sum ceiling |
| --- | --- | --- | --- |
| float32 | 1e-5 | 1e-4 | 1e-5 |
| float16 | 2e-3 | 1e-2 | 2e-3 |
| bfloat16 | 1e-2 | 5e-2 | 1e-2 |

Same-path deterministic observer-neutrality checks expect exact equality first. If reference repeats are themselves variable, traced divergence must fit the measured untraced-repeat envelope and the reviewed profile's ceiling. Explain the kernel/runtime source of variability. Same-runtime captured API slices require exact numeric equality after lossless wire widening; broad dtype tolerances are inappropriate for a wrong slice.

Stage 2 calibrates tolerances separately by operation/device/dtype with bounded real traces, distinguishes native arithmetic from promoted audit reconstruction, and stores the accepted profile. Some values may need scale-aware criteria or a stricter ceiling. A failing test triggers investigation of axes, masks, cast order, missing operations, or backend changes before any tolerance change. Review and document every relaxation. Non-finite logits/hidden states fail execution unless a precisely justified masked-score representation explains them.

Cross-device and cross-backend comparisons have separate reports and cannot certify same-path neutrality. Lossy storage accuracy is a distinct measured property with its own provenance, not proof that the observer is faithful.

## 9. Memory, lifetime, and performance validation

Measure host peak memory, accelerator allocated/reserved peak when available, retained unique CPU bytes, inference time, capture/copy time, analysis time, and response bytes. Warm up separately, synchronize timing when required, and state the hardware/software/runtime. Document CPU/MPS measurement limitations rather than inventing accelerator counters.

For several increasing sequence lengths and capture profiles, compare estimates with actual retained/peak memory and attention's expected N² trend. Confirm selected retention does not falsely claim to avoid eager score allocation. Verify last-position logits avoid full-position vocabulary storage by default. Repeated runs and expiration should return live references/retained bytes to the expected baseline; allocator caching can keep reserved memory high and must be distinguished from a leak.

Test owned CPU snapshots where an observer's source could mutate, aliases counted once, graph/device references released, reservations released after errors, request leases against eviction, source expiry during analysis, and budget exhaustion during generation. No per-trace GPU weights or live cache history remain retained after completion.

Provisional frontend targets to measure on recorded hardware: overview JSON ≤1 MiB, slices within configured element/byte bounds, cached selection response around 100 ms, and main-thread frame work within about 16 ms for ordinary pan/zoom. These are engineering targets, not achieved benchmarks. Record percentile interaction latency and payload sizes. Optimize only when measured results justify extra libraries/workers or changed limits.

## 10. Visualization and interpretation review

Numerical correctness does not establish scientifically defensible wording. At every relevant stage, manually review a fresh real trace and inspect whether:

- every plotted value has its correct trace/component/axis scope and provenance;
- captured, derived, summary, projection, interpretability, and controlled-intervention evidence are distinguishable;
- rounding, omitted mass, mask status, missing data, precision, and stale/expired states are visible;
- attention language describes routing/association, not proof of cause;
- Logit Evolution declares its intermediate projection recipe and position;
- representation coordinates disclose fit population/basis, method, variance when applicable, and geometry limits;
- residual-vector graphics use a defensible linear transformation for any asserted vector addition;
- generation decision and consumed token positions are not confused;
- MLP coordinates do not receive unsupported semantic labels;
- intervention conclusions name the tested modification and matched conditions.

Keyboard/table inspection must resolve the same numerical cells as graphical selection. Use linked-view interaction tests to detect stale trace responses, off-by-one positions, repeated-token confusion, query/KV head confusion, and incompatible selection after model changes. Accessibility and clarity are tested alongside numerical mapping.

## 11. Causal controls and adapter expansion

Stage 10 requires a no-op intervention control, exact boundary-target verification, genuine patch-source checks, and ordinary baseline execution after modification cleanup. Compare distributions before sampling. Initial query-head ablation zeroes its head output before output projection; zeroing a KV head is a different intervention and cannot share the same label. A valid intervention can have no visible effect; the test must not require a dramatic result.

Match model/revision/input/mask/positions/backend/dtype and seeds between baseline/intervention. Activation patches require compatible semantic stage, coordinates, and shape. Record off-distribution and interaction limits when interpreting results. Do not reuse a contaminated cache. Every new model adapter runs the appropriate conformance and baseline suite, with architecture-specific equations and explicitly unsupported components.

## Gate and evidence ownership

Stage 1 establishes a pinned usable baseline. Stage 2 owns scientific capture gates; Stage 3 owns storage/serialization correctness; Stages 4–7 own linked-view and analysis correctness; Stage 8 adds generation parity; Stages 9–11 add projection, intervention, and additional-adapter evidence. Failures block the dependent capability. A schema-valid trace can remain scientifically unvalidated, and the UI must state that distinction.

Evidence reports belong under `docs/validation/` with actual execution dates and compatibility keys. Stage 1 contains measured output evidence and a completion report; later capture reports must be produced by their own real validation runs.
