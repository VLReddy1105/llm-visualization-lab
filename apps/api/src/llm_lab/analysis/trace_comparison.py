"""Bounded output audit between two retained executions; never invokes a model."""

from llm_lab.schemas.trace import TraceResult
from llm_lab.tracing.validation import compare_tensors


def compare_outputs(actual: TraceResult, reference: TraceResult) -> dict:
    a, b = actual.metadata, reference.metadata
    fields = (
        "hf_model_id",
        "revision",
        "tokenizer_revision",
        "device",
        "dtype",
        "prompt",
        "token_ids",
        "architecture",
    )
    mismatches = [name for name in fields if getattr(a, name) != getattr(b, name)]
    if actual.output.prediction_position != reference.output.prediction_position:
        mismatches.append("prediction_position")
    result = {
        "trace_id": a.trace_id,
        "reference_trace_id": b.trace_id,
        "compatible": not mismatches,
        "mismatches": mismatches,
        "passed": False,
        "same_top_token": None,
        "logits": None,
        "probabilities": None,
        "provenance": "derived",
        "sources": [
            f"{m.trace_id}:{name}"
            for m in (a, b)
            for name in ("output.logits", "output.probabilities")
        ],
        "method": (
            "Compare full retained final-position vocabulary outputs with Stage 2 "
            "elementwise tolerances; float64 audit of native values. No model forward."
        ),
    }
    if mismatches:
        return result
    atol, rtol, probability_atol = {
        "float32": (1e-5, 1e-4, 1e-6),
        "float16": (2e-3, 1e-2, 2e-3),
        "bfloat16": (1e-2, 5e-2, 1e-2),
    }[a.dtype]
    for name, tolerance in (("logits", atol), ("probabilities", probability_atol)):
        result[name] = compare_tensors(
            actual.tensor(f"output.{name}", copy=False),
            reference.tensor(f"output.{name}", copy=False),
            atol=tolerance,
            rtol=rtol,
        )
    result["same_top_token"] = bool(
        actual.tensor("output.logits", copy=False).argmax()
        == reference.tensor("output.logits", copy=False).argmax()
    )
    result["passed"] = (
        result["same_top_token"]
        and result["logits"]["passed"]
        and result["probabilities"]["passed"]
    )
    return result
