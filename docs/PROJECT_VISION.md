# Project vision

Status: Stage 0 design, 2026-10-07. All capabilities below are planned unless explicitly described as documentation.

## Purpose and research question

LLM Visualization Lab makes a real causal language model's computation inspectable at several scales. A learner starts with text such as `The capital of Germany is`, follows its tokens into the network, and explores the transformations that produce the next-token distribution. A researcher can open the same trace to inspect shapes, operations, retained tensor slices, and validation evidence.

The research question is: **How can transformer computation become visible and understandable while preserving its mathematical structure, numerical provenance, and uncertainty?**

An ordinary attention visualizer exposes a narrow observation. This project connects tokenization, embeddings, normalization, attention, residual additions, gated MLPs, final normalization, vocabulary projection, and sampling in one navigable record. Token selection survives changes of view. The residual stream provides continuity between components. The user can move from a model overview to an individual operation and its actual operands.

## Evidence and scope

The supplied `LLM_Visualization_Conceptual_Architecture.pdf` is a conceptual reference, not a record of implemented capabilities. The pasted request controls scope and terminology. Its Stage 0 stopping instruction overrides any implementation language in the reference. Reference labels such as Architecture View and Layer Explorer map to **Model Journey** and **Layer Microscope** here. The reference's development phases are superseded by the requested Stages 1–11.

Initial execution targets Qwen/Qwen3-0.6B. Structural facts have been checked against official sources in [System architecture](SYSTEM_ARCHITECTURE.md#verified-qwen3-structure). No weights have been downloaded, no forward pass has been run, and no numerical fidelity claim has yet been experimentally established.

## Mandatory principles

1. **Real model data.** Every numerical view has a trace ID, component reference, capture scope, and provenance. No invented attention patterns, probabilities, activations, embeddings, or demo results.
2. **Trace once, visualize many ways.** One forward execution produces an immutable trace consumed by every view. Additional analysis can use its retained data; an uncaptured component requires an explicitly requested new trace with a parent link.
3. **Model-independent frontend.** The UI consumes semantic operations, named axes, topology, and capability metadata. Python module paths belong to adapters and diagnostic provenance.
4. **Scientific honesty.** Distinguish captured tensors, derived measurements, summaries, projections, interpretability methods, and intervention evidence. A visualization's appearance cannot turn an association into a causal claim.
5. **Semantic zoom.** Begin with a horizontal Model Journey. Open a layer, then a component, then a tensor or operation. Keep prompt and token context visible at every depth.
6. **Bounded local execution.** Sequence length, capture coverage, trace retention, and payload size are explicit limits. An honest unavailable state is preferable to silently approximating or exhausting memory.
7. **Reproducibility.** Preserve effective token IDs, model/tokenizer revision, execution settings, capture specification, software versions, device/dtype, and analysis parameters.

## Three exploration levels

| Level | User question | Default information | Next action |
| --- | --- | --- | --- |
| Model Journey | What happened between input and prediction? | Tokens, ordered components, layer summaries, output summary | Select token or layer |
| Layer Microscope | What changed inside this layer? | Architecture-specific operation graph, residual checkpoints, attention/MLP updates | Open component or head |
| Tensor / mathematical microscope | How was this value produced? | Operand references, named axes, bounded values, formulas, provenance | Select relation or dimension |

Semantic zoom changes detail and navigation, not the underlying trace. The representation-space view remains an optional analysis of this same record; a future 3D scene cannot replace the computational journey.

## Questions the system should eventually answer

- How does the tokenizer split this exact input, including whitespace, special tokens, and chat-template text?
- How do a selected token's norm, direction, and residual updates evolve with depth?
- Which earlier token positions receive attention from a selected query head, and how are the scores normalized?
- What do normalization, Q/K/V projections, RoPE, and value mixing contribute to the actual attention update?
- How do the gated MLP's units and output modify a representation?
- How do vocabulary candidates rank under an intermediate projection, and how do those projections differ from final inference?
- How do sampling controls transform the saved final logits, and which token was actually chosen?
- Under a specified intervention, what changes relative to a matched baseline?

For an earlier input token, a vocabulary projection describes the distribution associated with that position; it is not automatically its contribution to the last-position prediction. The Token Journey must preserve this distinction.

## Scientific fidelity

A scientifically faithful trace preserves the computation's identity: exact input, architecture, execution backend, and tensor axes. Captured values reconcile with the instrumented execution. Tracing leaves that execution's output within justified tolerance of its matched untraced baseline. Derived values name their transformation and source tensors. Stored precision changes and incomplete coverage remain visible.

The UI describes attention as a normalized value-routing weight. It describes Logit Evolution as an intermediate vocabulary projection. It describes PCA/UMAP as a lower-dimensional representation of specified source data. It reserves causal language for a controlled intervention comparison, with the intervention and comparison conditions displayed.

Fidelity does not mean identifying the model's complete reasoning or assigning natural-language meanings to individual coordinates. Nor does it guarantee identical outputs across hardware or software versions. The validation plan defines the practical limits.

## Product character

The workspace should feel like a scientific instrument: white-dominant surfaces, large readable labels, generous spacing, restrained concept colors, precise hover inspection, and a single continuous computation. Attention, residuals, MLPs, and logits have stable identities without relying on color alone. Loading, missing data, expired traces, and reduced precision are legible states.

Animation may explain order or highlight an operation. Playback timing is illustrative unless tied to recorded timing metadata; it must not imply that positions are computed serially within a prefill layer. All plotted values come from the trace.

## Success criteria

| Quality dimension | Required evidence |
| --- | --- |
| Scientific fidelity | Matched baseline, reconciliation tests, provenance on every scientific view |
| End-to-end coverage | Inspect text through final logits and next-token choice |
| Multi-scale exploration | Navigate model → layer → component → tensor → operation |
| Understandability | Explain one layer using its real data without requiring a raw tensor dump |
| Interactivity | Preserve linked token, layer, head, candidate, and generation selection |
| Performance | Responsive navigation with bounded retrieval and measured memory use |
| Reproducibility | Save execution and analysis recipes with pinned revisions and settings |
| Causal analysis | Eventually compare controlled interventions against matched baselines |

A new prompt creates a new execution. Sampling adjustments use real saved logits. Additional visualizations reuse the same trace. Unsupported internals remain unavailable. These are completion gates, not optional polish.

## Current boundary

Stage 0 delivers the eight Markdown documents only. No application scaffold, dependencies, UI screens, model inference, numerical fixtures, or scientific test results are part of this task. [Implementation stages](IMPLEMENTATION_STAGES.md) defines the later work and its gates.
