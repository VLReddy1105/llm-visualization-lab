import pytest
import torch
from fastapi.testclient import TestClient

from llm_lab.config import Settings
from llm_lab.errors import ServiceError
from llm_lab.main import create_app
from llm_lab.schemas.inference import NextTokenResponse

pytestmark = pytest.mark.real_model
PROMPT = "The capital of Germany is"


def test_real_tokenizer_and_raw_text(real_service, real_result):
    baseline = real_service.reference_baseline(PROMPT).result(timeout=180)
    result = real_result.response
    assert result.input.text == result.input.effective_text == PROMPT
    assert result.input.decoded_text == PROMPT
    assert result.input.mode == "raw" and result.input.add_special_tokens is False
    assert [t.token_id for t in result.input.tokens] == baseline.token_ids
    assert [t.position_id for t in result.input.tokens] == list(range(len(baseline.token_ids)))
    assert all(t.tokenizer_piece and t.decoded_token for t in result.input.tokens)


def test_real_logits_probabilities_and_count(real_result):
    response = real_result.response
    assert len(real_result.logits) == response.model.vocabulary_size
    assert response.prediction.logits_shape == [1, 1, response.model.vocabulary_size]
    assert not real_result.logits.requires_grad
    assert real_result.logits.device.type == "cpu"
    assert bool(torch.isfinite(real_result.logits).all())
    assert bool(torch.isfinite(real_result.probabilities).all())
    assert abs(float(real_result.probabilities.sum()) - 1) < 1e-6
    # Independent max-shifted exp/reduction verifies the full distribution.
    shifted = real_result.logits.double() - real_result.logits.double().max()
    exponentials = torch.exp(shifted)
    expected_probabilities = (exponentials / exponentials.sum()).float()
    torch.testing.assert_close(
        real_result.probabilities, expected_probabilities, atol=1e-8, rtol=1e-6
    )
    candidates = response.prediction.top_k
    assert len(candidates) == 10
    assert [c.rank for c in candidates] == list(range(1, 11))
    assert all(0 <= c.probability <= 1 for c in candidates)
    assert [c.probability for c in candidates] == sorted(
        [c.probability for c in candidates], reverse=True
    )
    for candidate in candidates:
        assert candidate.raw_logit == float(real_result.logits[candidate.token_id])
        assert candidate.probability == float(real_result.probabilities[candidate.token_id])
    assert (
        abs(
            sum(c.probability for c in candidates)
            + response.prediction.omitted_probability_mass
            - 1
        )
        < 1e-6
    )


def test_direct_hf_full_vocabulary_baseline(real_service, real_result):
    baseline = real_service.reference_baseline(PROMPT).result(timeout=180)
    atol, rtol = (1e-5, 1e-4) if real_result.response.model.dtype == "float32" else (1e-2, 5e-2)
    torch.testing.assert_close(real_result.logits, baseline.logits, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        real_result.probabilities, baseline.probabilities, atol=1e-6, rtol=1e-4
    )
    assert baseline.logits.topk(10).indices.tolist() == [
        c.token_id for c in real_result.response.prediction.top_k
    ]


def test_repeated_deterministic_inference(real_service, real_result):
    repeated = real_service.infer_for_validation(PROMPT, 10).result(timeout=180)
    torch.testing.assert_close(real_result.logits, repeated.logits, atol=1e-5, rtol=1e-4)
    assert [c.token_id for c in repeated.response.prediction.top_k] == [
        c.token_id for c in real_result.response.prediction.top_k
    ]


def test_real_http_schema_and_top_k(real_service):
    with TestClient(create_app(Settings(auto_load=False), real_service)) as client:
        response = client.post(
            "/api/inference/next-token", json={"input": {"text": PROMPT}, "top_k": 3}
        )
        assert response.status_code == 200
        result = NextTokenResponse.model_validate(response.json())
        assert len(result.prediction.top_k) == 3
        assert result.input.text == PROMPT
        assert result.runtime.tracing_enabled is False
        assert result.runtime.cache_enabled is False
        assert not any(
            name in response.json() for name in ["hidden_states", "attentions", "trace_id"]
        )


def test_whitespace_and_unicode_are_preserved(real_service):
    text = "  Hello\n café 🧠 "
    result = real_service.infer_for_validation(text, 1).result(timeout=180)
    baseline = real_service.reference_baseline(text).result(timeout=180)
    assert result.response.input.text == text
    assert result.response.input.decoded_text == text
    assert [t.token_id for t in result.response.input.tokens] == baseline.token_ids


def test_input_limit_no_silent_truncation(real_service):
    text = "hello " * (real_service.settings.max_input_tokens + 10)
    with pytest.raises(ServiceError) as error:
        real_service.infer_for_validation(text).result(timeout=180)
    assert error.value.code == "INPUT_LIMIT_EXCEEDED"


def test_inference_failure_is_safe_and_next_request_recovers(real_service, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Injected inference failure")

    with TestClient(create_app(Settings(auto_load=False), real_service)) as client:
        with monkeypatch.context() as patch:
            patch.setattr(real_service._model, "forward", fail)
            response = client.post("/api/v1/inference", json={"input": {"text": PROMPT}})
            assert response.status_code == 500
            assert response.json()["error"]["code"] == "INFERENCE_FAILED"
            assert "Injected" not in response.text
        recovered = client.post("/api/v1/inference", json={"input": {"text": PROMPT}, "top_k": 1})
        assert recovered.status_code == 200
        assert len(recovered.json()["prediction"]["top_k"]) == 1
