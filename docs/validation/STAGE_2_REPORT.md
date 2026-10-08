# Stage 2 validation report

**Status: PASS.** Recorded 2026-10-07T23:35:04.261314+00:00.

Stage 2 real run-scoped model instrumentation; no trace persistence or HTTP

## Execution identity

| Property | Measured value |
| --- | --- |
| model_id | qwen3-0.6b |
| hf_model_id | Qwen/Qwen3-0.6B |
| revision | c1899de289a04d12100db370d81485cdf75e47ca |
| tokenizer_revision | c1899de289a04d12100db370d81485cdf75e47ca |
| device | mps |
| dtype | float32 |
| load_duration_ms | 7126.471208001021 |

| Software | Version / environment |
| --- | --- |
| python | 3.11.15 |
| pytorch | 2.14.1 |
| transformers | 5.19.0 |
| platform | macOS-26.6.1-arm64-arm-64bit |

## Installed architecture and adapter

| Property | Actual config / implementation value |
| --- | --- |
| layers | 28 |
| hidden_size | 1024 |
| intermediate_size | 3072 |
| query_heads | 16 |
| kv_heads | 8 |
| head_dim | 128 |
| vocabulary_size | 151936 |
| rms_norm_epsilon | 1e-06 |
| rope_theta | 1000000.0 |
| tied_embeddings | True |
| attention_dropout | 0.0 |
| attention_backend | eager |
| config_digest | 7870eed078fc404f7c0621149c22cb265a52664129662bb5519eaaf4ef1809d3 |
| implementation_digest | ca9535e36ea4e689c76d7d753b91aec9ea00da837fbf223c8ea4c01a3b50cac0 |
| adapter_id | qwen3 |
| adapter_version | 1.0.0 |

`ModelService` serializes loading, inference and tracing on its existing executor. `TraceEngine` validates the plan and budget, then a run-scoped `TraceSession` centrally registers module observers. `TraceCollector` receives semantic observations through `BaseModelAdapter`; Qwen paths remain inside `Qwen3Adapter`. Observers return no replacement outputs. Hooks are removed in `finally`, including failed inference. Final tensors are detached owned CPU snapshots in an immutable internal `TraceResult`.

`BaseModelAdapter` declares architecture, hook targets, semantic components, unchanged tokenization/forward, trace eligibility and native reconstruction helpers. `Qwen3Adapter` checks actual module widths, per-head RMSNorm, compact GQA, RoPE, SiLU gated MLP and embedding/head tying against the installed model/config. It preserves the existing eager backend, evaluation mode, device and dtype.

## Capture policy and provenance

`CapturePlan` supports summary, overview, selected layer, attention, MLP and deep profiles, explicit layer/query-head/token bounds, optional raw retention, statistics and a 256 MiB default byte budget. Summary retains global final outputs and scalar internal statistics; selected queries preserve complete mapped key contexts. Oversized requests fail before inference rather than truncating scientific tensors.

RAW values are actual module-boundary observations, including projections, normalizations, residual checkpoints, eager attention weights, MLP activation/product and final outputs when supported by this installed release. DERIVED output probabilities use the trusted Stage 1 full-vocabulary normalization. Unscaled, scaled and masked scores are DERIVED. RECONSTRUCTED RoPE Q/K, softmax weights and head mixing preserve named sources and are checked against observed eager weights/head output. SUMMARY statistics record their finite population, masked/nonfinite counts, accumulation dtype and population standard-deviation definition.

Validation metadata only: null means full logical coverage; selected components record each axis as selection=all with size, or selection=indices with absolute indices.

## Numerical acceptance criterion

```json
{
  "logit_absolute": 1e-05,
  "logit_relative": 0.0001,
  "probability_absolute": 1e-06,
  "probability_relative": 0.0001,
  "criterion": "abs(actual-reference) <= absolute + relative*abs(reference)",
  "comparison_scope": "same resident weights, same device/dtype/backend and exact raw input"
}
```

## Prompt 1: measured traces

Exact raw prompt: `The capital of Germany is`. Sequence length: **5**.

Untraced Stage 1 forward: `66.41545798629522` ms; total inference: `105.41262500919402` ms. Capture overhead differences are recorded per profile. These are single runs, with allocator/cache/scheduling effects, not a controlled latency benchmark.

| Index | ID | Native piece | Decoded token |
| --- | --- | --- | --- |
| 0 | 785 | The | "The" |
| 1 | 6722 | Ġcapital | " capital" |
| 2 | 315 | Ġof | " of" |
| 3 | 9856 | ĠGermany | " Germany" |
| 4 | 374 | Ġis | " is" |

| Capture | Gate | Tensor components | Retained bytes | Estimated retained | Transient estimate | Duration ms | Max logit error | Mean logit error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| summary | PASS | 2 | 1215488 | 1215488 | 22545024 | 143.3 | 0.0 | 0.0 |
| deep_selected_layers | PASS | 91 | 3296844 | 3522124 | 58320064 | 177.194 | 0.0 | 0.0 |
| layer_12 | PASS | 37 | 1970124 | 2031564 | 34471104 | 136.854 | 0.0 | 0.0 |
| attention_selected_token_heads | PASS | 32 | 1295620 | 1299716 | 21188672 | 216.763 | 0.0 | 0.0 |
| mlp_statistics_only | PASS | 2 | 1215488 | 1215488 | 28770944 | 123.508 | 0.0 | 0.0 |
| overview | PASS | 177 | 3559464 | 4726824 | 22545024 | 121.381 | 0.0 | 0.0 |

All-layer deep metadata-only estimate:

```json
{
  "retained_bytes": 22154124,
  "transient_bytes": 354465984,
  "peak_bytes": 376620108
}
```

Untraced before/after neutrality:

```json
{
  "shape_match": true,
  "max_absolute_error": 0.0,
  "mean_absolute_error": 0.0,
  "relative_l2_error": 0.0,
  "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
  "bitwise_equal": true,
  "forward_duration_ms": 44.60966700571589,
  "inference_duration_ms": 47.169750003376976,
  "token_ids_equal": true,
  "top_10_ids_equal": true,
  "within_tolerance": true
}
```

Stage 1 baseline versus independent direct Hugging Face forward:

```json
{
  "logits": {
    "shape_match": true,
    "max_absolute_error": 0.0,
    "mean_absolute_error": 0.0,
    "relative_l2_error": 0.0,
    "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
    "bitwise_equal": true
  },
  "probabilities": {
    "shape_match": true,
    "max_absolute_error": 0.0,
    "mean_absolute_error": 0.0,
    "relative_l2_error": 0.0,
    "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
    "bitwise_equal": true
  },
  "exact_token_ids": true,
  "logits_within_tolerance": true,
  "probabilities_within_tolerance": true,
  "top_10_ids_equal": true,
  "reference_method": "independent direct HF all-position forward; final position selected afterward"
}
```

### Layer 12 semantic shapes and provenance

| Component | Logical shape | Retained shape | Semantic axes | Kind | Availability |
| --- | --- | --- | --- | --- | --- |
| embedding.output | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| layer[12].residual.before | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| layer[12].attention.input_normalized | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| layer[12].attention.query.projected | [1, 16, 5, 128] | [1, 16, 5, 128] | batch, query_head, query_token, head_dimension | raw | captured |
| layer[12].attention.query.normalized | [1, 16, 5, 128] | [1, 16, 5, 128] | batch, query_head, query_token, head_dimension | raw | captured |
| layer[12].attention.key.projected | [1, 8, 5, 128] | [1, 8, 5, 128] | batch, kv_head, key_token, head_dimension | raw | captured |
| layer[12].attention.key.normalized | [1, 8, 5, 128] | [1, 8, 5, 128] | batch, kv_head, key_token, head_dimension | raw | captured |
| layer[12].attention.value | [1, 8, 5, 128] | [1, 8, 5, 128] | batch, kv_head, key_token, head_dimension | raw | captured |
| layer[12].attention.concatenated | [1, 5, 2048] | [1, 5, 2048] | batch, token, head_feature | raw | captured |
| layer[12].attention.head_output | [1, 16, 5, 128] | [1, 16, 5, 128] | batch, query_head, query_token, head_dimension | raw | captured |
| layer[12].attention.output | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| layer[12].attention.weights | [1, 16, 5, 5] | [1, 16, 5, 5] | batch, query_head, query_token, key_token | raw | captured |
| layer[12].residual.after_attention | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| layer[12].mlp.input_normalized | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| layer[12].mlp.gate | [1, 5, 3072] | [1, 5, 3072] | batch, token, intermediate | raw | captured |
| layer[12].mlp.activated_gate | [1, 5, 3072] | [1, 5, 3072] | batch, token, intermediate | raw | captured |
| layer[12].mlp.up | [1, 5, 3072] | [1, 5, 3072] | batch, token, intermediate | raw | captured |
| layer[12].mlp.gated | [1, 5, 3072] | [1, 5, 3072] | batch, token, intermediate | raw | captured |
| layer[12].mlp.output | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| layer[12].residual.after_mlp | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| layer[12].output | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| output.residual_final | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| output.normalized | [1, 5, 1024] | [1, 5, 1024] | batch, token, hidden | raw | captured |
| output.lm_head_input | [1, 1, 1024] | [1, 1, 1024] | batch, prediction_position, hidden | raw | captured |
| output.logits | [1, 1, 151936] | [1, 1, 151936] | batch, prediction_position, vocabulary | raw | captured |
| layer[12].attention.query.rotary | [1, 16, 5, 128] | [1, 16, 5, 128] | batch, query_head, query_token, head_dimension | reconstructed | captured |
| layer[12].attention.key.rotary | [1, 8, 5, 128] | [1, 8, 5, 128] | batch, kv_head, key_token, head_dimension | reconstructed | captured |
| layer[12].attention.similarity | [1, 16, 5, 5] | [1, 16, 5, 5] | batch, query_head, query_token, key_token | derived | captured |
| layer[12].attention.scaled_scores | [1, 16, 5, 5] | [1, 16, 5, 5] | batch, query_head, query_token, key_token | derived | captured |
| layer[12].attention.masked_scores | [1, 16, 5, 5] | [1, 16, 5, 5] | batch, query_head, query_token, key_token | derived | captured |
| layer[12].attention.weights_reconstructed | [1, 16, 5, 5] | [1, 16, 5, 5] | batch, query_head, query_token, key_token | reconstructed | captured |
| layer[12].attention.head_output_reconstructed | [1, 16, 5, 128] | [1, 16, 5, 128] | batch, query_head, query_token, head_dimension | reconstructed | captured |
| output.probabilities | [1, 1, 151936] | [1, 1, 151936] | batch, prediction_position, vocabulary | derived | captured |

Residual, layer-output, gated-MLP and attention reconstruction validation:

| Check | Pass / status | Max absolute | Mean absolute | Relative L2 | Bitwise |
| --- | --- | --- | --- | --- | --- |
| layer[0].residual.attention | True | 0.0 | 0.0 | 0.0 | True |
| layer[0].residual.mlp | True | 0.0 | 0.0 | 0.0 | True |
| layer[0].layer_output | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].residual.attention | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].residual.mlp | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].layer_output | True | 0.0 | 0.0 | 0.0 | True |
| layer[27].residual.attention | True | 0.0 | 0.0 | 0.0 | True |
| layer[27].residual.mlp | True | 0.0 | 0.0 | 0.0 | True |
| layer[27].layer_output | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].attention.rmsnorm | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].query_norm | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].key_norm | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].attention.weights | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].attention.head_output | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].attention.row_normalization | True | 1.1082738637924194e-07 | — | — | — |
| layer[12].attention.causal_mask | True | — | — | — | — |
| layer[12].attention.actual_mask | True | — | — | — | — |
| layer[12].mlp.rmsnorm | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].mlp.activation | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].mlp.gated | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].mlp.down_projection | True | 0.0 | 0.0 | 0.0 | True |
| output.final_rmsnorm | True | 0.0 | 0.0 | 0.0 | True |
| output.lm_head | True | 0.0 | 0.0 | 0.0 | True |
| output.lm_head_input | True | 0.0 | 0.0 | 0.0 | True |
| hook_cleanup | True | — | — | — | — |

Actual Layer 12 statistics and retained coordinate coverage are in the machine-readable artifact; no complete tensor arrays are stored here.

| Rank | Decoded candidate | ID | Raw logit | Full-vocabulary probability |
| --- | --- | --- | --- | --- |
| 1 | " Berlin" | 19846 | 16.52149200439453 | 0.47704339027404785 |
| 2 | " located" | 7407 | 14.624664306640625 | 0.07157742977142334 |
| 3 | " ______" | 32671 | 14.192410469055176 | 0.046456944197416306 |
| 4 | " ____" | 30743 | 13.501934051513672 | 0.023290593177080154 |
| 5 | " in" | 304 | 13.070024490356445 | 0.015121839009225368 |
| 6 | " " | 220 | 13.063224792480469 | 0.015019363723695278 |
| 7 | " (" | 320 | 13.024328231811523 | 0.014446377754211426 |
| 8 | " a" | 264 | 12.992985725402832 | 0.014000614173710346 |
| 9 | " one" | 825 | 12.930341720581055 | 0.013150465674698353 |
| 10 | " the" | 279 | 12.914087295532227 | 0.012938440777361393 |

## Prompt 2: measured traces

Exact raw prompt: `After the rain, Berlin feels quiet.`. Sequence length: **8**.

Untraced Stage 1 forward: `56.71845798497088` ms; total inference: `69.54483399749734` ms. Capture overhead differences are recorded per profile. These are single runs, with allocator/cache/scheduling effects, not a controlled latency benchmark.

| Index | ID | Native piece | Decoded token |
| --- | --- | --- | --- |
| 0 | 6025 | After | "After" |
| 1 | 279 | Ġthe | " the" |
| 2 | 11174 | Ġrain | " rain" |
| 3 | 11 | , | "," |
| 4 | 19846 | ĠBerlin | " Berlin" |
| 5 | 11074 | Ġfeels | " feels" |
| 6 | 11340 | Ġquiet | " quiet" |
| 7 | 13 | . | "." |

| Capture | Gate | Tensor components | Retained bytes | Estimated retained | Transient estimate | Duration ms | Max logit error | Mean logit error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| summary | PASS | 2 | 1215488 | 1215488 | 24364032 | 149.721 | 0.0 | 0.0 |
| deep_selected_layers | PASS | 91 | 4566336 | 4926784 | 81974272 | 183.146 | 0.0 | 0.0 |
| layer_12 | PASS | 37 | 2428224 | 2526528 | 43570176 | 151.738 | 0.0 | 0.0 |
| attention_selected_token_heads | PASS | 32 | 1311136 | 1315232 | 21436928 | 212.167 | 0.0 | 0.0 |
| mlp_statistics_only | PASS | 2 | 1215488 | 1215488 | 34325504 | 143.358 | 0.0 | 0.0 |
| overview | PASS | 177 | 4963392 | 6831168 | 24364032 | 150.381 | 0.0 | 0.0 |

All-layer deep metadata-only estimate:

```json
{
  "retained_bytes": 34929984,
  "transient_bytes": 558879744,
  "peak_bytes": 593809728
}
```

Untraced before/after neutrality:

```json
{
  "shape_match": true,
  "max_absolute_error": 0.0,
  "mean_absolute_error": 0.0,
  "relative_l2_error": 0.0,
  "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
  "bitwise_equal": true,
  "forward_duration_ms": 48.913209000602365,
  "inference_duration_ms": 51.19291698792949,
  "token_ids_equal": true,
  "top_10_ids_equal": true,
  "within_tolerance": true
}
```

Stage 1 baseline versus independent direct Hugging Face forward:

```json
{
  "logits": {
    "shape_match": true,
    "max_absolute_error": 0.0,
    "mean_absolute_error": 0.0,
    "relative_l2_error": 0.0,
    "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
    "bitwise_equal": true
  },
  "probabilities": {
    "shape_match": true,
    "max_absolute_error": 0.0,
    "mean_absolute_error": 0.0,
    "relative_l2_error": 0.0,
    "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
    "bitwise_equal": true
  },
  "exact_token_ids": true,
  "logits_within_tolerance": true,
  "probabilities_within_tolerance": true,
  "top_10_ids_equal": true,
  "reference_method": "independent direct HF all-position forward; final position selected afterward"
}
```

### Layer 12 semantic shapes and provenance

| Component | Logical shape | Retained shape | Semantic axes | Kind | Availability |
| --- | --- | --- | --- | --- | --- |
| embedding.output | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| layer[12].residual.before | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| layer[12].attention.input_normalized | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| layer[12].attention.query.projected | [1, 16, 8, 128] | [1, 16, 8, 128] | batch, query_head, query_token, head_dimension | raw | captured |
| layer[12].attention.query.normalized | [1, 16, 8, 128] | [1, 16, 8, 128] | batch, query_head, query_token, head_dimension | raw | captured |
| layer[12].attention.key.projected | [1, 8, 8, 128] | [1, 8, 8, 128] | batch, kv_head, key_token, head_dimension | raw | captured |
| layer[12].attention.key.normalized | [1, 8, 8, 128] | [1, 8, 8, 128] | batch, kv_head, key_token, head_dimension | raw | captured |
| layer[12].attention.value | [1, 8, 8, 128] | [1, 8, 8, 128] | batch, kv_head, key_token, head_dimension | raw | captured |
| layer[12].attention.concatenated | [1, 8, 2048] | [1, 8, 2048] | batch, token, head_feature | raw | captured |
| layer[12].attention.head_output | [1, 16, 8, 128] | [1, 16, 8, 128] | batch, query_head, query_token, head_dimension | raw | captured |
| layer[12].attention.output | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| layer[12].attention.weights | [1, 16, 8, 8] | [1, 16, 8, 8] | batch, query_head, query_token, key_token | raw | captured |
| layer[12].residual.after_attention | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| layer[12].mlp.input_normalized | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| layer[12].mlp.gate | [1, 8, 3072] | [1, 8, 3072] | batch, token, intermediate | raw | captured |
| layer[12].mlp.activated_gate | [1, 8, 3072] | [1, 8, 3072] | batch, token, intermediate | raw | captured |
| layer[12].mlp.up | [1, 8, 3072] | [1, 8, 3072] | batch, token, intermediate | raw | captured |
| layer[12].mlp.gated | [1, 8, 3072] | [1, 8, 3072] | batch, token, intermediate | raw | captured |
| layer[12].mlp.output | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| layer[12].residual.after_mlp | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| layer[12].output | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| output.residual_final | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| output.normalized | [1, 8, 1024] | [1, 8, 1024] | batch, token, hidden | raw | captured |
| output.lm_head_input | [1, 1, 1024] | [1, 1, 1024] | batch, prediction_position, hidden | raw | captured |
| output.logits | [1, 1, 151936] | [1, 1, 151936] | batch, prediction_position, vocabulary | raw | captured |
| layer[12].attention.query.rotary | [1, 16, 8, 128] | [1, 16, 8, 128] | batch, query_head, query_token, head_dimension | reconstructed | captured |
| layer[12].attention.key.rotary | [1, 8, 8, 128] | [1, 8, 8, 128] | batch, kv_head, key_token, head_dimension | reconstructed | captured |
| layer[12].attention.similarity | [1, 16, 8, 8] | [1, 16, 8, 8] | batch, query_head, query_token, key_token | derived | captured |
| layer[12].attention.scaled_scores | [1, 16, 8, 8] | [1, 16, 8, 8] | batch, query_head, query_token, key_token | derived | captured |
| layer[12].attention.masked_scores | [1, 16, 8, 8] | [1, 16, 8, 8] | batch, query_head, query_token, key_token | derived | captured |
| layer[12].attention.weights_reconstructed | [1, 16, 8, 8] | [1, 16, 8, 8] | batch, query_head, query_token, key_token | reconstructed | captured |
| layer[12].attention.head_output_reconstructed | [1, 16, 8, 128] | [1, 16, 8, 128] | batch, query_head, query_token, head_dimension | reconstructed | captured |
| output.probabilities | [1, 1, 151936] | [1, 1, 151936] | batch, prediction_position, vocabulary | derived | captured |

Residual, layer-output, gated-MLP and attention reconstruction validation:

| Check | Pass / status | Max absolute | Mean absolute | Relative L2 | Bitwise |
| --- | --- | --- | --- | --- | --- |
| layer[0].residual.attention | True | 0.0 | 0.0 | 0.0 | True |
| layer[0].residual.mlp | True | 0.0 | 0.0 | 0.0 | True |
| layer[0].layer_output | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].residual.attention | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].residual.mlp | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].layer_output | True | 0.0 | 0.0 | 0.0 | True |
| layer[27].residual.attention | True | 0.0 | 0.0 | 0.0 | True |
| layer[27].residual.mlp | True | 0.0 | 0.0 | 0.0 | True |
| layer[27].layer_output | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].attention.rmsnorm | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].query_norm | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].key_norm | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].attention.weights | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].attention.head_output | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].attention.row_normalization | True | 1.5506520867347717e-07 | — | — | — |
| layer[12].attention.causal_mask | True | — | — | — | — |
| layer[12].attention.actual_mask | True | — | — | — | — |
| layer[12].mlp.rmsnorm | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].mlp.activation | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].mlp.gated | True | 0.0 | 0.0 | 0.0 | True |
| layer[12].mlp.down_projection | True | 0.0 | 0.0 | 0.0 | True |
| output.final_rmsnorm | True | 0.0 | 0.0 | 0.0 | True |
| output.lm_head | True | 0.0 | 0.0 | 0.0 | True |
| output.lm_head_input | True | 0.0 | 0.0 | 0.0 | True |
| hook_cleanup | True | — | — | — | — |

Actual Layer 12 statistics and retained coordinate coverage are in the machine-readable artifact; no complete tensor arrays are stored here.

| Rank | Decoded candidate | ID | Raw logit | Full-vocabulary probability |
| --- | --- | --- | --- | --- |
| 1 | " The" | 576 | 15.821012496948242 | 0.3680817186832428 |
| 2 | " It" | 1084 | 14.208876609802246 | 0.07341799885034561 |
| 3 | " But" | 1988 | 13.465045928955078 | 0.03489482030272484 |
| 4 | " What" | 3555 | 13.246477127075195 | 0.028043856844305992 |
| 5 | " In" | 758 | 13.085619926452637 | 0.023876922205090523 |
| 6 | " A" | 362 | 13.06991958618164 | 0.02350497432053089 |
| 7 | " He" | 1260 | 13.009546279907227 | 0.022127889096736908 |
| 8 | " She" | 2932 | 12.924297332763672 | 0.020319679751992226 |
| 9 | " This" | 1096 | 12.893558502197266 | 0.019704578444361687 |
| 10 | " " | 220 | 12.702042579650879 | 0.016270197927951813 |

## Hook lifecycle, memory and automated evidence

Initial hooks: `0`; final hooks: `0`; equality: `True`. Each capture also records its own before/after count and detached CPU storage checks.

Process peak RSS includes model loading and all earlier operations. MPS driver allocation includes allocator cache; it is not a measure of retained trace tensors. Trace-retained bytes count unique owned storage, separate from the conservative transient/workspace estimate.

```json
{
  "process_lifetime_peak_rss_bytes": 1743716352,
  "rss_scope": "process lifetime, includes model load and all preceding captures",
  "mps_current_allocated_bytes": 2399404544,
  "mps_driver_allocated_bytes": 2827616256
}
```

Automated test outcomes:

```json
{
  "unique_automated_tests": 137,
  "stage1_preserved_tests": 30,
  "stage2_added_tests": 107,
  "completed_cpu_mps_test_executions": 274,
  "devices": {
    "cpu": {
      "status": "PASS",
      "passed": 137,
      "failed": 0,
      "errors": 0,
      "skipped": 0,
      "duration_seconds": 14.241,
      "timestamp": "2026-10-08T01:37:32.541438+02:00",
      "artifact": "/Users/likhithreddy/Documents/llm interpreter/docs/validation/stage2-tests-cpu.xml",
      "command": "LLM_LAB_DEVICE=cpu .venv/bin/pytest -q --real-model",
      "warning_count": 1,
      "warning": "Existing Starlette TestClient httpx deprecation; no numerical warning."
    },
    "mps": {
      "status": "PASS",
      "passed": 137,
      "failed": 0,
      "errors": 0,
      "skipped": 0,
      "duration_seconds": 15.567,
      "timestamp": "2026-10-08T01:39:16.507863+02:00",
      "artifact": "/Users/likhithreddy/Documents/llm interpreter/docs/validation/stage2-tests-mps.xml",
      "command": "LLM_LAB_DEVICE=mps .venv/bin/pytest -q --real-model",
      "warning_count": 1,
      "warning": "Existing Starlette TestClient httpx deprecation; no numerical warning."
    }
  },
  "without_real_model": {
    "passed": 86,
    "skipped": 51,
    "failed": 0
  },
  "static_contract_checks": {
    "ruff_check": "PASS",
    "ruff_format_check": "PASS",
    "openapi_snapshot_unchanged": "PASS",
    "generated_types_check": "PASS",
    "frontend_typecheck": "PASS",
    "frontend_lint": "PASS"
  }
}
```

Additional device validation:

```json
{
  "artifact": "/Users/likhithreddy/Documents/llm interpreter/docs/validation/stage2-validation-cpu.json",
  "status": "PASS",
  "device": "cpu",
  "dtype": "float32",
  "prompt_count": 2,
  "profile_captures": 12,
  "same_path_max_absolute_logit_error": 0.0,
  "same_path_mean_absolute_logit_error": 0.0,
  "same_path_probabilities_bitwise_equal": true,
  "top_10_prediction_agreement": true,
  "max_residual_attention_error": 0.0,
  "max_residual_mlp_error": 0.0,
  "max_mlp_gating_error": 0.0,
  "max_mlp_down_projection_error": 0.0,
  "max_attention_weight_reconstruction_error": 0.0,
  "max_attention_head_reconstruction_error": 0.0,
  "hooks_initial": 0,
  "hooks_final": 0,
  "independent_direct_hf": [
    {
      "prompt": "The capital of Germany is",
      "logits": {
        "shape_match": true,
        "max_absolute_error": 2.47955322265625e-05,
        "mean_absolute_error": 2.130601541172186e-06,
        "relative_l2_error": 6.891273579146543e-07,
        "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
        "bitwise_equal": false
      },
      "probabilities": {
        "shape_match": true,
        "max_absolute_error": 2.1606683731079102e-07,
        "mean_absolute_error": 1.4115446212560053e-11,
        "relative_l2_error": 7.693673088519924e-07,
        "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
        "bitwise_equal": false
      },
      "exact_token_ids": true,
      "logits_within_tolerance": true,
      "probabilities_within_tolerance": true,
      "top_10_ids_equal": true,
      "reference_method": "independent direct HF all-position forward; final position selected afterward"
    },
    {
      "prompt": "After the rain, Berlin feels quiet.",
      "logits": {
        "shape_match": true,
        "max_absolute_error": 1.7642974853515625e-05,
        "mean_absolute_error": 1.5553461732605986e-06,
        "relative_l2_error": 8.022709900706027e-07,
        "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
        "bitwise_equal": false
      },
      "probabilities": {
        "shape_match": true,
        "max_absolute_error": 1.3113021850585938e-06,
        "mean_absolute_error": 2.9744362181461826e-11,
        "relative_l2_error": 3.764281847203577e-06,
        "relative_error_definition": "L2(actual-reference)/L2(reference); null when denominator is zero",
        "bitwise_equal": false
      },
      "exact_token_ids": true,
      "logits_within_tolerance": true,
      "probabilities_within_tolerance": true,
      "top_10_ids_equal": true,
      "reference_method": "independent direct HF all-position forward; final position selected afterward"
    }
  ],
  "capture_performance": [
    {
      "prompt": "The capital of Germany is",
      "untraced_forward_ms": 135.03945799311623,
      "profiles": {
        "summary": {
          "duration_ms": 348.05716600385495,
          "forward_ms": 325.0657079915982,
          "forward_overhead_ms": 190.02624999848194,
          "retained_bytes": 1215488
        },
        "deep_selected_layers": {
          "duration_ms": 189.54150000354275,
          "forward_ms": 158.82495799451135,
          "forward_overhead_ms": 23.78550000139512,
          "retained_bytes": 3296844
        },
        "layer_12": {
          "duration_ms": 178.4129170118831,
          "forward_ms": 156.57150000333786,
          "forward_overhead_ms": 21.53204201022163,
          "retained_bytes": 1970124
        },
        "attention_selected_token_heads": {
          "duration_ms": 205.75133297825232,
          "forward_ms": 145.12379100779071,
          "forward_overhead_ms": 10.084333014674485,
          "retained_bytes": 1295620
        },
        "mlp_statistics_only": {
          "duration_ms": 175.05624998011626,
          "forward_ms": 154.166541993618,
          "forward_overhead_ms": 19.12708400050178,
          "retained_bytes": 1215488
        },
        "overview": {
          "duration_ms": 171.2920830177609,
          "forward_ms": 152.8209169919137,
          "forward_overhead_ms": 17.78145899879746,
          "retained_bytes": 3559464
        }
      }
    },
    {
      "prompt": "After the rain, Berlin feels quiet.",
      "untraced_forward_ms": 126.84291700134054,
      "profiles": {
        "summary": {
          "duration_ms": 172.17779098427854,
          "forward_ms": 153.50033401045948,
          "forward_overhead_ms": 26.657417009118944,
          "retained_bytes": 1215488
        },
        "deep_selected_layers": {
          "duration_ms": 195.45812500291504,
          "forward_ms": 165.55295800208114,
          "forward_overhead_ms": 38.7100410007406,
          "retained_bytes": 4566336
        },
        "layer_12": {
          "duration_ms": 179.28087498876266,
          "forward_ms": 156.86054198886268,
          "forward_overhead_ms": 30.01762498752214,
          "retained_bytes": 2428224
        },
        "attention_selected_token_heads": {
          "duration_ms": 206.1730409914162,
          "forward_ms": 147.93041700613685,
          "forward_overhead_ms": 21.08750000479631,
          "retained_bytes": 1311136
        },
        "mlp_statistics_only": {
          "duration_ms": 179.71908301115036,
          "forward_ms": 157.97312499489635,
          "forward_overhead_ms": 31.130207993555814,
          "retained_bytes": 1215488
        },
        "overview": {
          "duration_ms": 175.8958749996964,
          "forward_ms": 156.30687499651685,
          "forward_overhead_ms": 29.463957995176315,
          "retained_bytes": 4963392
        }
      }
    }
  ],
  "memory_after_all_captures": {
    "process_lifetime_peak_rss_bytes": 3172679680,
    "rss_scope": "process lifetime, includes model load and all preceding captures"
  }
}
```

## Exact file inventory

```json
{
  "created_count": 30,
  "modified_count": 9,
  "created": [
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/analysis/__init__.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/analysis/attention.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/models/adapters/__init__.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/models/adapters/base.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/models/adapters/qwen3.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/models/adapters/registry.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/schemas/components.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/schemas/trace.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/trace_validation.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/tracing/__init__.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/tracing/capture_plan.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/tracing/collector.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/tracing/engine.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/tracing/instrumentation.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/tracing/validation.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_adapter_structure.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_capture_budget.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_capture_plan.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_collector.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_future_cleanup.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_instrumentation_cleanup.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_qwen3_shapes.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_residual_reconciliation.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_trace_equivalence.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_trace_validation.py",
    "/Users/likhithreddy/Documents/llm interpreter/docs/validation/STAGE_2_REPORT.md",
    "/Users/likhithreddy/Documents/llm interpreter/docs/validation/stage2-tests-cpu.xml",
    "/Users/likhithreddy/Documents/llm interpreter/docs/validation/stage2-tests-mps.xml",
    "/Users/likhithreddy/Documents/llm interpreter/docs/validation/stage2-validation-cpu.json",
    "/Users/likhithreddy/Documents/llm interpreter/docs/validation/stage2-validation.json"
  ],
  "modified": [
    "/Users/likhithreddy/Documents/llm interpreter/README.md",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/.env.example",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/config.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/src/llm_lab/models/manager.py",
    "/Users/likhithreddy/Documents/llm interpreter/apps/api/tests/test_foundation.py",
    "/Users/likhithreddy/Documents/llm interpreter/docs/IMPLEMENTATION_STAGES.md",
    "/Users/likhithreddy/Documents/llm interpreter/docs/SCIENTIFIC_VALIDATION.md",
    "/Users/likhithreddy/Documents/llm interpreter/docs/SYSTEM_ARCHITECTURE.md",
    "/Users/likhithreddy/Documents/llm interpreter/docs/TRACE_DESIGN.md"
  ]
}
```

## Scientific and implementation limits

- Real numerical validation covers the pinned Qwen3-0.6B checkpoint with installed Transformers 5.19.0, eager attention, batch one, unpadded raw input, cache disabled, and float32 on CPU/Apple MPS. CUDA and reduced precision are unverified.
- Post-RoPE Q/K are functional reconstructions using actual observed cos/sin and installed math. Reconstructed attention is checked against RAW eager weights/head output and separate direct-HF observations; reconstruction is never labeled RAW.
- Attention weights describe value routing, not causal responsibility or reasoning. MLP coordinates are not named concepts, and no intervention or intermediate prediction is claimed.
- CPU same-path tracing is bitwise identical. The independent all-position HF LM-head reference has maximum logit differences 2.47955322265625e-05 and 1.7642974853515625e-05 for the two prompts; both pass the preserved abs-error <= 1e-5 + 1e-4*abs(reference) criterion. MPS independent references are bitwise identical.
- Only two short prompts (5 and 8 tokens) were exercised numerically. Long-context safety is enforced by selection/byte estimates, not claimed as a measured numerical or latency benchmark.
- The 256 MiB default cap applies to a conservative extra-tensor peak estimate, including retained snapshots and float64/native working scratch. It excludes resident weights/backend allocation and may reject a plan whose eventual retained bytes alone would fit.
- Full all-layer detail was estimated without allocating it. For the 5-token prompt, retained estimate is 22154124 bytes and extra tensor peak estimate is 376620108 bytes, exceeding the default cap. Reduce scope or explicitly configure an adequate bounded cap.
- Timings are single-run measurements affected by cache, allocator and scheduling. Process peak RSS includes loading and previous captures; MPS driver bytes include allocator cache and must not be added to RSS as independent physical memory.
- Trace metadata/mappings are frozen and public tensor access copies by default. Internal copy=False borrowers can mutate PyTorch tensors and must obey the read-only contract.
- Statistics are finite population reductions over explicitly retained coordinates; masked scores exclude actual masked positions. Omitted raw data raises KeyError. Later arbitrary analyses require suitable raw capture.
- Unsupported capture modes (other backends, sliding window, quantization or permanent HF output-capture hooks) are rejected. No multi-model adapter, cache/generation, Logit Lens, representation projection or intervention is implemented.
- Failed queue tasks preserve typed error messages, causal chains and bounded text origin notes while dropping worker frame references, preventing retained Futures from owning model weights after service close.
- No Stage3 trace store, TTL, trace HTTP endpoints, slice serialization or frontend trace consumption is implemented. Existing Stage1 HTTP and generated client contracts remain unchanged.

## Stage 3 boundary

Stage 3 will add bounded trace persistence, expiration and leases, trace-creation and retrieval HTTP contracts, component/layer/attention slice endpoints, explicit serialization and generated client types. It will serve this validated semantic data without rerunning inference on reads. No Stage 3 API, store, expiration behavior, serialization or frontend trace consumption has been implemented in Stage 2. Visualization screens remain later work.
