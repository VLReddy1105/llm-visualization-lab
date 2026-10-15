# Stage 8 architectural integration assessment

Baseline: 2026-10-14, branch `stage7-validated`, HEAD `045c6dee422ba2cd8edaebaef51804239b0aac90`, clean working tree; 379 file SHA-256 hashes saved before edits. Required stage/trace/system/API/scientific documents and Stage 6/7 reports were inspected. Prior evidence remains immutable.

The resident ModelService owns the serialized executor. Generation submits one bounded session loop to it and uses the existing adapter forward, TraceEngine, scoped collector and TraceStore reservations. Each decision receives exact prefix IDs; decoded text is never retokenized. TraceEngine accepts these IDs while standalone tracing remains intact.

Generation sessions have a separate bounded metadata lifecycle, absolute start-based expiry, idempotent request keys, frozen sampling settings and observable cancellation. They store compact candidate/decision summaries, never full-vocabulary history matrices. Original step traces share TraceStore accounting; retention is bounded and independent source expiry remains visible. Model/logit scratch estimates are distinct from measured retained bytes.

Sampling calls the existing Stage 6 processor chain and full-vocabulary normalizer. A dedicated seeded CPU generator draws from its processed distribution on both model devices; this policy does not promise CPU/MPS sequence equality. Greedy uses stable smallest-ID argmax ties. Generation remains full-prefix, cache disabled and raw prompt input without chat templating.

The bright existing workspace receives a Generation Journey mode. Step selection resolves its original source UUID and exact decision coordinate before enabling the existing inspectors. Missing historical coverage is explicit and never triggers recapture. Generation and trace mutation controls remain explicit. Independent real-model CPU/MPS comparisons and production Chrome acceptance have passed; synthetic faults remain isolated lifecycle tests. See [Stage 8 report](STAGE_8_REPORT.md) for measured results and evidence.
