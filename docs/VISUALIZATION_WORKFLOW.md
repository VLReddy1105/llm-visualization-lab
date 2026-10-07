# Visualization workflow

Status: Stage 0 interaction specification. All numerical displays described here must consume real traces; no screens or scientific demo data are created in this stage.

## One continuous workspace

The top bar contains the project name, model selector and loading status, prompt editor, raw/chat input choice, Run / Trace, generation controls, capture profile, and settings. Keep detailed scientific settings in an inspector, while effective input length, device, and trace status stay discoverable.

The center is a large horizontal Model Journey: Input → Tokens → Embeddings → decoder layers → Final Norm → LM Head → Logits → Prediction. Layer count and topology come from the trace. Horizontal scrolling, explicit zoom controls, fit-to-selection, and keyboard navigation provide access to depth without shrinking labels into unreadable text.

A resizable bottom context panel changes with selection: token → Token Journey; layer → Layer Microscope; attention → Attention Explorer; head → head details within Attention Explorer; MLP → MLP Explorer; residual → Residual Stream; logits → output details and Logit Evolution. Breadcrumbs preserve model/layer/component/operation depth. Returning to the model restores its viewport and selections.

Use a white-dominant layout, large labels, clear spacing, and limited panels. Suggested concept identities are neutral blue for tokens, teal for attention, violet for residuals, amber for MLP, and blue for logits/prediction. Verify actual palette contrast during implementation, and reinforce each concept with labels/icons rather than color alone. Masks, missing values, and zero values need distinct appearances.

## Global selection model

| State | Canonical identity | Rule |
| --- | --- | --- |
| `selectedToken` | Effective-sequence `token_index` plus trace/generation context | A position, never matched by displayed text |
| `selectedLayer` | `layer_index` | Validate against current topology |
| `selectedHead` | Query `head_index` | Derive and display `kv_head_index` separately |
| `selectedComponent` | Semantic `component_id` | Includes residual stage and inspector depth |
| `selectedGenerationStep` | `generation_id`, `generation_step` → `trace_id` | Resolve trace before dependent data requests |
| `selectedCandidateToken` | Tokenizer revision and vocabulary ID | Distinct from selected input token |
| `activeVisualizationMode` | One canonical mode name | Preserve applicable linked selection |

Selection changes are atomic. Selecting token position 3 updates token highlights across the layer, attention query row, residual/MLP summaries, and projection point. A separate `selectedKeyToken` supports a clicked attention relation; it does not overwrite the query token. Selecting a heatmap cell sets layer/head/query/key together. Hover is transient and cannot overwrite persistent selection. Selecting an MLP unit or vector dimension is local inspector state.

On a new unrelated prompt/model trace, reset token/candidate selection and validate or reset layer/head/component selection; never carry a same-spelled token as if it had the same identity. Within a generation, preserve absolute token positions present in both steps; if a selected generated position is absent in an earlier step, show that state explicitly. Preserve vocabulary candidates only when tokenizer/model revision is compatible. Changing model requires fresh capability validation.

Server data is cached by `trace_id`, component, selectors, analysis method/version, and parameters. Selection/viewport state lives in Zustand; API tensors live in the query cache. Cancel obsolete detail requests and ignore late responses from a previously selected trace. Deep links can store IDs/selections without embedding the prompt. An expired linked trace opens an expiry state rather than an empty scientific plot.

## Shared scientific and missing-data UI

Every scientific view provides a compact provenance label: **Captured**, **Derived metric**, **Projection**, **Interpretability method**, or **Controlled intervention**, with source stage, axes, dtype, coverage, and method available on demand. Color scales and units are labeled. Display values use a consistent formatting policy; exact bounded inspection is separate from rounded labels.

Statuses include no trace, queued/running, ready, failed, expired/evicted, unsupported, not captured, and insufficient analysis data. Unavailable detail offers an explicit new capture only when supported. The previous trace remains immutable. Animation explains order and focus; tensor values do not interpolate into fabricated intermediate states. Prefill token computations inside a layer should not be presented as sequential token execution.

## 1. Model Journey

**Purpose:** Navigate the complete text-to-prediction computation and choose a detail level.

**Input data:** Effective text/tokens, model metadata, semantic operation graph, capture manifest, layer summaries, output summary, generation association.

**Visual representation:** Horizontal chain with the residual stream visible through every layer. At model scale, layers show a small norm/update summary and capture status. Final Norm, LM Head, logits, and next-token decision are separate nodes.

**User interaction:** Pan/scroll horizontally, zoom, select a token or layer, open a component, return through breadcrumbs, and move through generation steps. Clicking a token preserves selection while entering another component.

**Scientific interpretation:** A navigation map of the actual architecture and its recorded execution. Summary values originate from that trace.

**Potential misinterpretation:** Spatial length and playback speed are not measured compute cost; changing a summary color is not evidence of importance. An argmax preview is not an actual generated token.

**Performance considerations:** Render only visible layer labels; load one overview; avoid per-layer tensor queries at overview scale. Preserve viewport separately from data.

## 2. Token Journey

**Purpose:** Follow one input position's representation through embedding, residual checkpoints, and final normalization.

**Input data:** Selected position's norms, cosine changes, update norms, residual checkpoint vectors when retained, scoped attention summaries, and optional vocabulary projections.

**Visual representation:** Linked depth strip with norm/change curves and selected-layer details. Distinguish embedding, pre-attention, after-attention, after-MLP, and final-normalized stages. Highlight the same position in other views.

**User interaction:** Select a token chip, scrub depth, compare adjacent checkpoints, open the relevant residual/attention/MLP component, and pin a compatible vocabulary candidate.

**Scientific interpretation:** Changes to that position's state across depth. A projection at an earlier position predicts its next position under the projection recipe; it does not quantify contribution to the last-position prediction.

**Potential misinterpretation:** Large norm/change does not imply semantic importance or causal influence. Identical display text at two positions identifies different tokens. Attention received is a scoped column reduction, not causal responsibility.

**Performance considerations:** Use saved all-position scalar summaries for switching tokens. Retrieve one selected-position vector per requested checkpoint, not all hidden states in browser memory.

## 3. Layer Microscope

**Purpose:** Explain one decoder layer as a connected sequence of actual operations.

**Input data:** Adapter-supplied layer topology, shapes, normalization types, residual checkpoints, attention/MLP updates and summaries.

**Visual representation:** For initial Qwen3: Residual Input → RMSNorm → Attention → Residual Addition → RMSNorm → gated MLP → Residual Addition → Layer Output. Residual bypass paths remain visible. Expanding Attention opens its Q/K/V, normalization, rotation, score, softmax, and projection sequence.

**User interaction:** Select layer/token, open an operation, inspect operands/shapes, switch layers without losing the selected position, and choose deep capture when requested details were not retained.

**Scientific interpretation:** Real operation topology and observed checkpoints; formulas correspond to selected operands.

**Potential misinterpretation:** A generic sequential diagram cannot substitute for another model's parallel residuals or different normalization. The post-attention norm normalizes the state entering the MLP; it is not normalization of attention weights.

**Performance considerations:** Fetch one layer manifest and its bounded summaries. Expand detail lazily; reuse shared residual tensor aliases.

## 4. Attention Explorer

**Purpose:** Inspect token-to-token value routing across layers and query heads.

**Input data:** Actual attention weights or separately labeled reconstructed weights, head-to-KV mapping, absolute query/key position maps, masks, layer/head summaries, and head output references.

**Visual representation:** Token × Token heatmap with **queries as rows and keys as columns**, selected-query connection view, and a mathematical relation inspector. Full prefill is square; cached decoding can be rectangular. Masked cells have a distinct treatment. A selected relation opens dot product → scale → mask → full-row softmax → weight → weighted values.

**User interaction:** Change layer/query head, select query/key, inspect a full row, switch heatmap/connection views, zoom to tiles, and choose an explicit arithmetic mean over selected heads when useful. Connection limits show displayed versus omitted mass.

**Scientific interpretation:** Weights route values at this operation. A mean across query heads is a descriptive aggregation, not a physical model head. Entropy and sent/received summaries disclose their axes and normalization.

**Potential misinterpretation:** High attention is not proof that a token caused the prediction. A softmax denominator cannot be restricted to the shown heatmap tile. Q/K cosine alone does not reproduce a scaled dot-product weight, and value magnitude/output projection also affect the update.

**Performance considerations:** A row costs O(Nk); a full matrix costs O(Nq·Nk). Fetch selected heads/rows/tiles, render dense cells with Canvas, and keep labels/keyboard alternatives in the DOM. Only compute received-attention statistics over actually observed query scope.

## 5. Q/K/V Inspector

**Purpose:** Connect one attention relation to its real query, mapped key, and value vectors.

**Input data:** Projected, normalized, and rotary Q/K variants, V, selected query/KV mapping, norm/shape summaries, and bounded dimensions.

**Visual representation:** Three compact vector panels with shared head-dimension axes, magnitude/distribution summaries, and an optional exact-value table. Stage labels make normalization and rotation visible. Projected and rotary values are not mixed in the same unlabeled chart.

**User interaction:** Select query/key/head, compare stages, inspect dimensions, and open projection or positional operation operands.

**Scientific interpretation:** The selected head's actual operands. GQA shares K/V across query heads; normalized/rotated Q and K determine the score used by the attention operation.

**Potential misinterpretation:** Q/K/V are not fixed vocabulary token embeddings. Coordinate magnitudes do not establish named concepts. Head width is model metadata and must not be inferred from residual width divided by head count.

**Performance considerations:** Retrieve 128-dimensional head slices for the initial model, not all projected vectors. Missing normalized/rotary captures expose the derivation source or unavailable state.

## 6. Residual Stream

**Purpose:** Make representation updates visible and reconcile the layer's additions.

**Input data:** `residual.before`, `attention.output`, `residual.after_attention`, `mlp.output`, `residual.after_mlp`, and their saved metrics.

**Visual representation:** Before + attention update = middle; middle + MLP update = after. Use aligned norm/change plots and bounded coordinate comparisons. An optional 2D vector illustration uses one shared linear projection over all operands, with its projection label visible.

**User interaction:** Select token/layer, compare checkpoint pairs, inspect a dimension or reconstruction error, and compare actual updates with derived residual differences.

**Scientific interpretation:** Additive updates to one residual coordinate system. Relative norm is a descriptive ratio with a defined denominator.

**Potential misinterpretation:** Norms do not sum and ratios are not shares of causal importance. UMAP does not preserve vector addition; its arrows cannot demonstrate the residual equation. Stored subtraction may differ from the actual update due to rounding.

**Performance considerations:** Saved scalars power depth navigation. Fetch selected vectors only. Compute reconciliation server-side with the execution's dtype tolerance.

## 7. MLP Explorer

**Purpose:** Inspect the actual gated feed-forward path and its residual update.

**Input data:** Normalized input, gate/up projections, activated gate, gated product, down-projection output, activation metadata, and bounded unit statistics.

**Visual representation:** Two branches: gate → SiLU and up → element-wise product → down projection → MLP update → residual addition. Show distributions and selected strongest signed/absolute coordinates beside the operation graph.

**User interaction:** Select token/layer/unit, compare pre/post activation, inspect product operands, and view update magnitude. Choose signed versus absolute ranking explicitly.

**Scientific interpretation:** Recorded coordinates and transformation of the selected token state. A derived SiLU result is labeled as derived; an observed down-projection input is a captured gated product.

**Potential misinterpretation:** A strongly active coordinate is not automatically a human-interpretable feature. Hidden coordinates and MLP units have different dimensionalities. SAE labels require a separate validated feature model.

**Performance considerations:** Use histograms/top-unit summaries and optional slices, not thousands of bars. Fetch only selected token/unit windows. Deep captures are planned before inference and must fit memory limits.

## 8. Logit Evolution

**Purpose:** Examine when vocabulary candidates become strong under a defined intermediate projection.

**Input data:** Residual checkpoints, exact matching final norm/LM head, selected prediction position, candidate vocabulary IDs, and server-side Logit Lens analysis.

**Visual representation:** Candidate logit, full-vocabulary probability, or rank versus depth; a bounded candidate × checkpoint table; actual final inference as a separately labeled endpoint. Name the projection recipe in the view.

**User interaction:** Default to the final prompt position, choose checkpoint stage, add vocabulary candidates, select logit/probability/rank, inspect the normalization/head recipe, and jump to a layer.

**Scientific interpretation:** `LM_head(final_norm(intermediate residual))` is an interpretability technique. Its final-layer after-MLP endpoint can be compared with actual output under tolerance.

**Potential misinterpretation:** Intermediate projected probabilities are not distributions the original model explicitly produced at each layer. A curve for a token elsewhere in the input does not measure its causal contribution to final prediction. Multi-token words cannot be treated as one candidate ID. Probabilities require a full-vocabulary denominator.

**Performance considerations:** Cache by trace/checkpoint/position/method/candidates. Retain candidate results and log-normalizers rather than all layer × token × vocabulary arrays. Compute vocabulary chunks with validated numerical behavior; require matching resident weights for new projection analysis.

## 9. Generation Timeline

**Purpose:** Explore successive next-token decisions and their separate traces.

**Input data:** Generation record, immutable step → trace mapping, context position maps, actual chosen token, raw/base/processed distributions, sampler settings/seed, and termination reason.

**Visual representation:** Prompt followed by ordered decision steps, chosen token chips, and selected-step status. Show the context used for each decision and the token chosen after that forward pass. Distinguish sampled distribution from base model probabilities.

**User interaction:** Generate within a budget, cancel at a step boundary, move backward/forward, select an earlier trace, and branch a run with changed settings. Preview controls use real saved logits and do not rewrite history.

**Scientific interpretation:** Each step predicts the next token from its recorded input. Prefill and cached decoding have different coverage; a newly chosen token is represented only when a subsequent forward pass consumes it.

**Potential misinterpretation:** The sampled token's hidden state is not present in the decision trace that selected it. Cache traces do not contain freshly recomputed earlier-token states. Same seed is insufficient for cross-device bitwise reproducibility.

**Performance considerations:** Default to bounded full-prefix traces initially; cumulative capture grows with steps. Fetch one selected step, release completed live cache state, enforce per-run/store budgets, and mark expired historical records explicitly.

## 10. Representation Space

**Purpose:** Inspect lower-dimensional geometry of selected residual representations as an additional analysis mode.

**Input data:** Compatible residual vectors, fitted PCA/UMAP analysis, fit population, coordinate transformations, metric, seed, and method metadata.

**Visual representation:** Start with a 2D point plot and a layer slider, linked token highlights, and an optional trail using a common fit. PCA includes explained variance. A future 3D City of Meaning view consumes the same projection result.

**User interaction:** Choose layer/checkpoint, fit population and method, select a point to select the token, inspect original norms/distances, and compare supported aligned views.

**Scientific interpretation:** A dimensionality-reduced projection with method-specific distance limitations. PCA's common basis supports a consistent display across layers; it does not prove semantic clusters.

**Potential misinterpretation:** Independently fitted per-layer coordinates can rotate/translate and cannot imply movement without alignment. UMAP axes have no inherent semantic meaning and visual distance is not exact high-dimensional distance. Too few points cannot support a persuasive manifold claim.

**Performance considerations:** Fit server-side with bounded points and reproducible parameters. Cache fit IDs; return points rather than source matrices. Report insufficient-data states. Add browser workers or 3D libraries only after measured need. [PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) and [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) guide later implementation.

## 11. Causal Lab

**Purpose:** Compare a specified intervention with a matched unmodified execution.

**Input data:** Baseline/intervention traces, exact intervention location/scope, optional patch source, aligned input/revisions/runtime, and paired effect metrics.

**Visual representation:** Matched side-by-side logits/probabilities, signed differences, selected-component comparison, and a visible intervention specification. Do not prepopulate effect numbers.

**User interaction:** Choose query-head ablation, activation replacement, or patching; validate source/target shapes; execute a paired comparison; inspect results and controls. Historical traces remain unchanged.

**Scientific interpretation:** Evidence of an effect under the specified intervention and matched execution conditions. Initial head ablation means zeroing the selected query head's weighted-value output before concatenation/output projection; this differs from zeroing a shared KV head.

**Potential misinterpretation:** Off-distribution interventions, nonlinear interactions, and context dependence limit generalization. Patching is not automatic proof of a named concept. A sampled sentence difference alone is a weak control; compare distributions before sampling.

**Performance considerations:** Reserve paired execution/storage budgets, serialize scoped modifications, and restore state on every exit. Reject incompatible patches. Cache-disabled full-prefix comparisons precede any intervention with live KV caches.

## Interaction acceptance story

A later implementation must let a user run a fresh raw prompt, inspect real tokens, select the position containing a piece of `Germany`, open a layer, select a query head, click one relation, and inspect its actual operand/mask/softmax context. The same selected position must remain highlighted while opening Residual Stream and MLP Explorer. Logit Evolution defaults to the final prediction position and explains a deliberate switch to another position. Changing sampling controls produces a new analysis of real saved logits. New prompt execution, expanded capture, and changed generation settings each have explicit identities.

This story is tested incrementally through [Implementation stages](IMPLEMENTATION_STAGES.md); it does not authorize building later stages early.
