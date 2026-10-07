# Trace design

Status: Stage 0 specification. Schema version planned: `1.0.0`. This document defines semantic meaning; it does not contain measured model data.

## Trace identity and invariants

A **trace** records one actual forward execution over an explicit effective input. After successful commit it is immutable. All views use its identity, runtime provenance, capture specification, and availability manifest. Analyses reference that trace; they do not replace its values.

| Identifier | Meaning |
| --- | --- |
| `trace_id` | Opaque UUID identifying one execution record |
| `parent_trace_id` | Optional earlier trace used to request expanded capture or a related experiment; never implies identical execution |
| `generation_id` | Opaque UUID identifying an autoregressive run |
| `generation_step` | Zero-based index of a prediction decision within that run |
| `token_index` | Zero-based absolute position in the effective sequence, including template/special tokens |
| `position_id` | Position supplied to the model; distinct from token identity and cache position |
| `layer_index` | Zero-based decoder layer index |
| `head_index` | Zero-based query-head index |
| `kv_head_index` | Zero-based key/value-head index; distinct namespace |
| `candidate_token_id` | Vocabulary ID of one possible next token; not an input position |
| `component_id` | Stable semantic path within the trace, e.g. `layer[4].attention.query.rotary` |
| `tensor_id` | Trace-local opaque storage reference; may be shared by semantic aliases |
| `analysis_id` | Immutable derived result, parameters, method version, and source references |
| `intervention_id` | Later controlled modification specification linked to baseline/intervened traces |

The initial contract supports batch size 1 and unpadded input. Reject requests for multiple sequences instead of flattening their axes. Future batching adds explicit sequence identity. Vocabulary ID domains and token indices never cross model/tokenizer revisions implicitly.

## Value classification and availability

Classification dimensions are independent. A tensor may be RAW, OPTIONAL, and EXPENSIVE simultaneously.

| Term | Meaning | Required label |
| --- | --- | --- |
| RAW | Observed from the actual forward computation, possibly exposed through a reshape/view | `value_kind: raw`, capture point and execution provenance |
| DERIVED | Computed from retained operands using a specified method | `value_kind: derived`, method/version/parameters and source components |
| SUMMARY | Reduced statistics, histogram, or ranked selection over a specified source scope | `value_kind: summary`, axes and reduction definition |
| OPTIONAL | Adapter/backend/profile may omit it | `optional: true`, support and capture coverage |
| EXPENSIVE | Materialization, storage, or analysis is significant | `cost_class: expensive`, estimate and limits |

Interpretation adds `evidence_type`: `observation`, `descriptive_metric`, `projection`, `interpretability_method`, or `controlled_intervention`. A projected point is DERIVED with projection evidence, not a RAW hidden state. A norm is SUMMARY/descriptive_metric. Captured attention weights are RAW/observation, not causal evidence.

Support capability is one of `supported`, `derivable`, or `unsupported`. Per-trace availability is `captured`, `derivable`, `not_captured`, `unsupported`, or `failed`. Expiration is a trace-level lifecycle state, not a tensor filled with nulls. A summary-only component exposes its summary and `not_captured` raw availability. Missing data is never represented by an all-zero array.

An advertised derivation includes required operands and whether matching resident model weights are needed. If its prerequisites are missing, it is not available. No GET or slice request reruns inference. Expanded capture creates a new trace with explicit lineage.

## Component descriptor

Every component describes:

- `component_id`, display `name`, `semantic_type`, and its role in the operation graph;
- logical `shape`, named `axes`, and retained axis coverage;
- `dtype` of the observed computation, `storage_dtype`, original device, and CPU storage device;
- `tensor_id` or alias reference, support/availability, and unavailable reason;
- `value_kind`, `evidence_type`, source references, capture/derivation method, and precision policy;
- bounded statistics, reduction scope, finite/masked counts, and `available_visualizations`;
- optional/expensive flags, estimated retained bytes, and available slice/analysis operations.

Each axis has its size and semantics. Retained subsets carry absolute axis coordinates. Example: logical `[1,16,128,128]` attention weights can retain only head 3 and query positions 120–127; storage shape is then `[1,1,8,128]`. Logical shape is not permission to retrieve missing coordinates. A `coverage` map records this explicitly.

Structural descriptor example, **not an inference result**:

```json
{
  "component_id": "layer[4].attention.query.rotary",
  "name": "Query after rotary encoding",
  "semantic_type": "attention_query_rotary",
  "shape": [1, 16, 128, 128],
  "axes": ["batch", "query_head", "query_token", "head_dimension"],
  "coverage": {"batch": "all", "query_head": "all", "query_token": "all", "head_dimension": "all"},
  "dtype": "float32",
  "storage_dtype": "float32",
  "original_device": "cpu",
  "storage_device": "cpu",
  "tensor_id": null,
  "support": "supported",
  "availability": "not_captured",
  "unavailable_reason": "not_requested",
  "value_kind": "raw",
  "evidence_type": "observation",
  "optional": true,
  "cost_class": "linear",
  "statistics": null,
  "available_visualizations": []
}
```

Summary statistics use float32 accumulation by default, with optional float64 audit reductions. Define `std` with population correction 0, L2 over stated dimensions, and histograms with explicit bins, counts, and scope. Ignore no non-finite values silently: record counts and whether a metric was computed over finite entries only. Exclude causal-mask sentinels from score distribution statistics and report the excluded mask count. Undefined cosine for a zero vector returns null plus a reason.

## Global trace and token data

The trace envelope contains schema/adapter versions, status/timestamps/expiration, request fingerprint, lineage, model metadata, tokenizer metadata, effective input, runtime metadata, capture plan, capability/topology manifest, token list, layer summaries, output summary, and storage accounting.

Record model/tokenizer IDs and resolved immutable revisions, config digest, layer/head/hidden/MLP/vocabulary sizes, normalization and positional-encoding descriptions, tied weights, backend, dtype/device, software versions, random seed/settings, cache mode, and determinism flags. Module paths may be kept in diagnostic provenance but must not drive frontend rendering.

Keep the user prompt and exact effective text separately. `input_mode` is `raw` or `chat`; record template digest, added special tokens, generation-prompt insertion, thinking control when applicable, and special-token policy. The microscope's deterministic starting example uses raw completion to make its actual input transparent. Chat support must show inserted tokens rather than concealing them.

For each token retain `token_index`, vocabulary ID, tokenizer piece, display text/escaped whitespace, special-token status, `position_id`, role/origin (`user`, `template`, `generated`), and offsets into effective text where valid. Original-prompt offsets are nullable when inserted text has no original span. Tokenizer pieces can represent partial byte sequences; individual decoded tokens need not concatenate into exact text. Preserve full-sequence decoding and verified offsets, not guessed word boundaries.

Immediately send token metadata and model dimensions once ready. Weight matrices, hidden-state arrays, masks, RNG state blobs, and raw input metadata too large for the response stay server-side. Summaries paginate when the response budget would be exceeded.

## Canonical tensor axes

Use `B=1`, `Nq` current query positions, `Nk` visible key positions, `D` residual width, `Hq` query heads, `Hkv` KV heads, `d` head width, `I` MLP width, and `V` vocabulary size. Prefill without cache has `Nq=Nk=N`. Every tensor names its axes; shape alone is insufficient.

Query/KV tensors use head-major canonical storage. Module outputs that use token-major axes are reshaped/transposed with a recorded layout mapping. Head output may share storage with concatenated output if the verified reshape permits it. GQA never requires storing physically repeated K/V tensors.

## Complete semantic vocabulary

Unless qualified, RAW tensors remain server-side, summaries/descriptors are immediately eligible for the overview, and full values are retrieved by bounded slices. OPTIONAL status depends on profile and adapter. Sizes follow the installed model configuration; [System architecture](SYSTEM_ARCHITECTURE.md#verified-qwen3-structure) records the initial checkpoint's dimensions.

### Embedding trace

| Component | Axes / shape | Classification | Meaning |
| --- | --- | --- | --- |
| `embedding.output` | `[B,Nq,D]` | RAW | Actual embedding output after any adapter-declared embedding transform |
| `embedding.statistics` | Token-indexed scalars/histograms | SUMMARY | Norm and distribution over hidden dimensions |
| `embedding.position_metadata` | Position IDs and encoding description | Metadata | RoPE or other model-specific position mechanism |

Do not label token embeddings as position-added vectors for a model that applies RoPE later. If an adapter scales embeddings, expose input lookup and scaled output as distinct nodes.

### Per-layer and attention trace

Prefix every layer component below with `layer[i].`.

| Component suffix | Axes / shape | Classification / cost | Meaning |
| --- | --- | --- | --- |
| `residual.before` | `[B,Nq,D]` | RAW | Input residual to this layer |
| `attention.input_normalized` | `[B,Nq,D]` | RAW, OPTIONAL | Actual attention input after normalization |
| `attention.query.projected` | `[B,Hq,Nq,d]` | RAW, OPTIONAL | Linear projection, before Q normalization/position transform |
| `attention.key.projected` | `[B,Hkv,Nq,d]` | RAW, OPTIONAL | Newly projected keys before normalization/rotation |
| `attention.value` | `[B,Hkv,Nq,d]` | RAW, OPTIONAL | Newly projected values |
| `attention.query.normalized` | `[B,Hq,Nq,d]` | RAW, OPTIONAL | Q per-head normalization where present |
| `attention.key.normalized` | `[B,Hkv,Nq,d]` | RAW, OPTIONAL | K per-head normalization where present |
| `attention.query.rotary` | `[B,Hq,Nq,d]` | RAW or explicitly DERIVED, OPTIONAL | Queries after actual positional transform |
| `attention.key.rotary` | `[B,Hkv,Nq,d]` | RAW or explicitly DERIVED, OPTIONAL | Newly rotated keys |
| `attention.key.context` | `[B,Hkv,Nk,d]` | RAW, OPTIONAL | Keys supplied to attention, including cache positions |
| `attention.value.context` | `[B,Hkv,Nk,d]` | RAW, OPTIONAL | Values supplied to attention |
| `attention.similarity` | `[B,Hq,Nq,Nk]` | Usually DERIVED, OPTIONAL, EXPENSIVE | Unscaled Q·K per query/KV mapping |
| `attention.scaled_scores` | `[B,Hq,Nq,Nk]` | RAW if observed; otherwise DERIVED; EXPENSIVE | Scaled dot products before mask |
| `attention.masked_scores` | `[B,Hq,Nq,Nk]` | RAW if observed; otherwise DERIVED; EXPENSIVE | Actual additive mask applied to scores |
| `attention.weights` | `[B,Hq,Nq,Nk]` | RAW when observed; OPTIONAL, EXPENSIVE | Actual weights consumed by value mixing, with dropout policy |
| `attention.weights_reconstructed` | `[B,Hq,Nq,Nk]` | DERIVED, OPTIONAL, EXPENSIVE | Recomputed weights; never relabel as captured |
| `attention.head_output` | `[B,Hq,Nq,d]` | RAW when captured through projection input | Weighted values before output projection |
| `attention.concatenated` | `[B,Nq,Hq*d]` | RAW view/alias | Head concatenation in the model's actual order |
| `attention.output` | `[B,Nq,D]` | RAW | Projected attention update added to residual |
| `attention.statistics` | Stated selected heads/positions | SUMMARY | Entropy, distributions, row/column summaries |

Also retain actual scale, dropout/training state, mask kind/representation, RoPE parameters and positions, query-to-KV mapping, query/key absolute index maps, and any positional bias/logit transformation. Initial Qwen attention uses scale `1/sqrt(d)`, but future adapters must declare their actual operation instead of assuming this equation universally.

For selected query head `h`, position `i`, and key `j`, the mathematical inspector resolves the actual Q vector and mapped K vector, then dot product, scale, mask, and row-wise softmax. The denominator covers **all allowed keys in the full row**, not only displayed connections or fetched tiles. A causal relation with `j>i` in an uncached raw prefix is masked; show mask status separately rather than printing a huge sentinel as a meaningful similarity.

### Residual trace

| Component suffix | Axes / shape | Classification | Meaning |
| --- | --- | --- | --- |
| `residual.before` | `[B,Nq,D]` | RAW | Layer input |
| `attention.output` | `[B,Nq,D]` | RAW | Actual attention update |
| `residual.after_attention` | `[B,Nq,D]` | RAW | Actual sum after attention |
| `mlp.output` | `[B,Nq,D]` | RAW | Actual MLP update |
| `residual.after_mlp` | `[B,Nq,D]` | RAW | Actual layer output before any later layer/final norm |
| `residual.statistics` | Token-indexed metrics | SUMMARY | Norms, update norms, cosine and relative magnitudes |

Reconcile `after_attention ≈ before + attention.output` and `after_mlp ≈ after_attention + mlp.output` under the execution's numeric rules. `after_mlp` aliases the next layer's `before` when verified. The final layer output aliases `output.residual_final`, **not** `output.normalized`.

Define relative update magnitude as `||update||₂ / ||before||₂` with an explicit zero-denominator policy. It is not a percentage of causal responsibility. Norms do not add: `||a+b||` is not `||a||+||b||`. An update obtained by subtracting stored residuals may differ from the actual captured update through rounding; label that subtraction DERIVED.

### MLP trace

| Component suffix | Axes / shape | Classification | Meaning |
| --- | --- | --- | --- |
| `mlp.input_normalized` | `[B,Nq,D]` | RAW, OPTIONAL | Actual pre-MLP normalized state |
| `mlp.gate` | `[B,Nq,I]` | RAW, OPTIONAL | Gate projection before activation |
| `mlp.up` | `[B,Nq,I]` | RAW, OPTIONAL | Parallel value projection |
| `mlp.activated_gate` | `[B,Nq,I]` | RAW if observed; otherwise DERIVED; OPTIONAL | SiLU(gate) for initial model |
| `mlp.gated` | `[B,Nq,I]` | RAW, OPTIONAL | Element-wise product entering down projection |
| `mlp.output` | `[B,Nq,D]` | RAW | Down-projected residual update |
| `mlp.statistics` | Bounded token/unit reductions | SUMMARY | Distribution, top absolute units, update magnitude |
| `mlp.features` | Feature-model-specific | DERIVED, OPTIONAL, future | Requires an identified SAE/feature model; unsupported initially |

The strongest signed/absolute units are descriptive coordinates, not named semantic concepts. Non-gated MLP adapters mark gate/product unsupported and supply a different operation graph. A future MoE adapter must expose routing and expert structure rather than compressing it into a fictitious dense MLP.

### Output trace

| Component | Axes / shape | Classification / retention |
| --- | --- | --- |
| `output.residual_final` | `[B,Nq,D]` | RAW, aliases final layer output |
| `output.normalized` | `[B,Nq,D]` | RAW, actual final normalization output |
| `output.logits` | `[B,P,V]` with explicit prediction-position map | RAW, retain last position by default; all positions EXPENSIVE |
| `output.probabilities` | Same position/vocabulary scope | DERIVED via full-vocabulary float32 softmax, lazy |
| `output.top_candidates` | Ranked bounded vocabulary items | SUMMARY, include raw logits/probabilities, total omitted mass |
| `output.sampling_distribution` | Position/vocabulary scope and recipe | DERIVED, separate from base probabilities |
| `output.selected_token` | Decision metadata | Actual sampled/greedy vocabulary ID and RNG provenance when generation occurs |

Logits at input position `i` predict the next position `i+1`. The final visible prompt position predicts the first generated token. A standalone forward trace can have `selected_token: null`; argmax summary is not evidence that a token was generated.

Top-k display is independent of top-k sampling. Displayed probabilities retain their full-vocabulary denominator, and omitted mass is explicit. A truncated display must not renormalize itself to 100%. Candidate text may encode to multiple IDs, especially with leading spaces. Candidate comparison accepts vocabulary IDs with visible pieces; multi-token word scoring is a separate future analysis.

## Derived analyses

**Token Journey:** reductions of a selected position across embedding/layer boundaries and updates. Cosine compares vectors in the same residual coordinate system. Attention sent is a selected query row; received is a column sum/mean with head/query scope stated and is not a probability distribution. Missing raw tensors can still support saved scalar summaries, but not arbitrary new metrics.

**Logit Evolution:** default Logit Lens recipe is `LM_head(final_norm(r_checkpoint))`, followed by float32 full-vocabulary softmax if probabilities are requested. Use the checkpoint's final norm and vocabulary head without modifying weights. Record checkpoint stage (`embedding`, `after_attention`, or `after_mlp`), selected prediction position, head/norm revision, and method version. It is an interpretability projection, not an actual intermediate prediction made by the original forward pass. A tuned lens would require distinct learned artifacts and labeling. Layer-depth ranking queries require the full vocabulary denominator/ranking or a correctly labeled restricted candidate set.

**Representation Space:** record source tensor stage/layers/positions, centering, normalization, distance metric, fitting population, seed, method version, and fit ID. Default PCA fits one common basis over selected layers' comparable residual vectors; layer changes reuse that basis. Per-layer independently fitted coordinates cannot be treated as trajectories without an explicit alignment and its limitations. UMAP is a later seeded analysis; tiny token sets or invalid neighbor counts yield insufficient-data states.

**Sampling:** from saved last-position logits compute base softmax, then for sampled mode apply temperature, top-k, then top-p using a documented order and boundary/tie policy, followed by renormalization. Top-p keeps at least one token and the boundary-crossing token. `temperature>0`, `top_k=0` disables the k filter, and `top_p=1` disables the nucleus filter. Greedy is an explicit mode, not division by temperature zero. Actual generation and preview must share the same server-side processor implementation and parity checks against the pinned Transformers behavior. Other model-default penalties/processors are disabled unless specified and recorded.

**Causal Lab:** a comparison references a matched baseline and intervention trace, exact token positions/heads/components, patch source when relevant, and effect metrics. Intervention results are scope-limited evidence under those conditions, not a complete explanation of model reasoning.

## Capture profiles and immutable coverage

| Profile | Retained data | Intended use |
| --- | --- | --- |
| `overview` | Tokens; embedding and residual boundary vectors; actual attention/MLP updates; output normalized states; last-position logits; layer/token scalar summaries | Model Journey, Token Journey, Residual Stream, Logit Evolution prerequisites |
| `attention` | Overview plus normalized inputs, Q/K/V stages, actual weights and head outputs for requested layers/heads/query positions | Attention Explorer and Q/K/V Inspector |
| `deep` | Overview plus requested attention and MLP intermediates, operation operands and mask metadata | Detailed Layer Microscope and MLP Explorer |

Profiles are deterministic capture expansions, not promises of unsupported capabilities. Each request specifies layers, heads, query positions, and tensor precision. Selected-head attention capture still needs all keys for a valid row. Overview captures scalar layer summaries while raw retention can be constrained only if the effective coverage manifest and affected views reflect the constraint. Default overview retains all residual token positions within the initial sequence limit so token switching does not rerun inference.

Derive all-layer received-attention summaries only when all relevant query rows were actually observed, and record the reduction scope. Selected-row capture cannot pretend to provide a full received-attention statistic. Scalar summary-only Q/K/V/MLP metrics may be calculated during execution without keeping arrays; label their saved scope.

Full pre-softmax score tensors are omitted by default. Compute bounded score/row analyses from retained Q/K/context and exact mask semantics. Do not store physically expanded GQA K/V, every permutation of one array, or model weights per trace.

Default storage preserves execution dtype and widens reductions to float32. A requested lossy storage conversion adds its source/storage dtype and policy to provenance and disables exact raw-value claims. JSON widening of a stored low-precision number to float32 does not recreate information lost earlier. Validation uses native captured values before any optional lossy conversion.

## Collection lifecycle

1. Resolve the exact model/tokenizer/runtime and validate input/capture scope.
2. Estimate unique retained bytes and peak execution/copy memory; reserve store capacity.
3. Acquire the model execution lock and register run-scoped observers.
4. Execute the real model with evaluation and inference modes. Observers must return no replacement outputs, mutate no inputs, and never consume the sampler RNG.
5. Detach and snapshot only requested tensors. Moving an already-CPU tensor to CPU can alias its storage; ensure an owned immutable snapshot where later writes are possible. Release device references immediately after safe copy/summary creation.
6. Validate descriptors and required lightweight invariants; remove observers and restore any scoped wrappers in `finally`, including errors/cancellation.
7. Atomically commit a ready trace. Release reservation on failure and publish a failed status without exposing partial arrays as a complete trace.

Module hooks observe boundaries; functional instrumentation observes selected internal operations. Do not permanently monkey-patch global attention functions. A release-specific scoped wrapper must invoke the original operation exactly once with unchanged operands, returning its original result. If safe observation is unavailable, use labeled derivations or unsupported capabilities. [PyTorch hook documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) is the implementation reference.

## Generation trace semantics

Generation decision step 0 runs the prompt and predicts output token 0. Step 1 runs the context with output token 0 appended and predicts output token 1. The sampled token belongs to the decision record; it has no hidden representation in that step's input trace until a later forward pass consumes it.

Start Stage 8 with full-prefix, cache-disabled reruns per decision for complete inspectable histories. Each step is one new trace, preserving absolute positions and generation lineage. Prefix inputs are `prompt_ids + generated_ids[:step]`, avoiding repeated chat-template application. Generation parameters are frozen for a run; changing them creates a new branch/generation ID. Previewing settings does not resample a historical step.

Future cache-enabled decoding has `Nq=1` for an ordinary decode step and `Nk=prefix_length`. Record query/key absolute-position maps, actual cache policy, and new versus cached K/V. Historical query rows and earlier-token hidden states are not recomputed in that step; link earlier trace data with its origin rather than filling a square attention matrix. Compare cache execution against matched full-prefix execution before enabling it.

Do not copy every previous trace or cache tensor into every step. Link immutable records and count unique retained bytes. When capacity is exhausted, stop before another forward execution with a meaningful reason; never evict the active context invisibly. Earlier steps may expire under the documented retention policy and must show an explicit unavailable state.

## Immediate versus lazy data

The ready overview includes IDs, revision metadata, token metadata, topology, per-layer scalar summaries, bounded descriptors, last-position top candidates, execution/capture settings, validation status, expiry, and storage usage. Paginate expanded manifests and summaries as necessary.

Raw embedding/residual/QKV/MLP tensors, full attention matrices, full logits, and large mask arrays remain server-side. Serve slices with named selectors and absolute coordinates. Analyses are parameter-keyed and immutable. Pure retained-data analysis survives model unload; projection through the vocabulary head or weights requires an exact matching resident revision. An unavailable dependency yields a structured error, not automatic loading or a new model run.

## Schema evolution

Semantic component meanings and axes cannot change within a compatible version. Add optional fields/capabilities without breaking old consumers; major changes require a new schema/API major version. Unknown optional nodes can display generic tensor inspection. Preserve operation graphs for architecture-specific topology and optional branches. An imported trace without a supported major version is rejected rather than interpreted heuristically.
