"""Run real inference and verify the independent tracing-off HF baseline."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from llm_lab.config import Settings
from llm_lab.models.manager import ModelService

PROMPT = "The capital of Germany is"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    service = ModelService(Settings(auto_load=False))
    try:
        service.start_loading()
        service.wait_until_ready()
        first = service.infer_for_validation(PROMPT, 10).result(timeout=180)
        repeated = service.infer_for_validation(PROMPT, 10).result(timeout=180)
        baseline = service.reference_baseline(PROMPT).result(timeout=180)
        # Float32 same-device reconstruction profile from the blueprint.
        atol, rtol = (1e-5, 1e-4) if first.response.model.dtype == "float32" else (1e-2, 5e-2)
        torch.testing.assert_close(first.logits, repeated.logits, atol=atol, rtol=rtol)
        torch.testing.assert_close(first.logits, baseline.logits, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            first.probabilities, baseline.probabilities, atol=1e-6, rtol=1e-4
        )
        assert baseline.token_ids == [token.token_id for token in first.response.input.tokens]
        assert abs(float(first.probabilities.sum()) - 1) < 1e-6
        assert all(0 <= c.probability <= 1 for c in first.response.prediction.top_k)
        assert len(first.response.prediction.top_k) == 10
        expected_ids = baseline.logits.topk(10).indices.tolist()
        assert expected_ids == [c.token_id for c in first.response.prediction.top_k]
        assert first.response.input.text == PROMPT
        report = {
            "status": "PASS",
            "recorded_at": datetime.now(UTC).isoformat(),
            "scope": "Stage 1 tracing-off real inference; no instrumentation",
            "result": first.response.model_dump(mode="json"),
            "verification": {
                "absolute_tolerance": atol,
                "relative_tolerance": rtol,
                "baseline_max_absolute_logit_error": float(
                    (first.logits - baseline.logits).abs().max()
                ),
                "repeat_max_absolute_logit_error": float(
                    (first.logits - repeated.logits).abs().max()
                ),
                "repeat_logits_bitwise_equal": bool(torch.equal(first.logits, repeated.logits)),
                "full_probability_sum": float(first.probabilities.sum()),
                "baseline_token_ids_match": True,
                "baseline_top_10_ids_match": True,
                "full_logit_count": first.logits.numel(),
                "tracing_enabled": False,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        service.close()


if __name__ == "__main__":
    main()
