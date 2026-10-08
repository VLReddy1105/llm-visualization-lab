"""Real Stage 1/Stage 2 observer-neutrality gates; no numerical stand-ins."""

import random

import pytest
import torch

from llm_lab.tracing.capture_plan import CapturePlan

pytestmark = pytest.mark.real_model
CAPITAL_PROMPT = "The capital of Germany is"
COMPLEX_PROMPT = "After the rain, Berlin feels quiet."
DETAIL_LAYERS = (0, 12, 27)


@pytest.fixture(scope="session")
def capital_trace(real_service):
    return real_service.trace(
        CAPITAL_PROMPT, CapturePlan(profile="deep", layers=DETAIL_LAYERS), top_k=10
    ).result(timeout=180)


@pytest.fixture(scope="session")
def complex_trace(real_service):
    return real_service.trace(
        COMPLEX_PROMPT, CapturePlan(profile="deep", layers=DETAIL_LAYERS), top_k=10
    ).result(timeout=180)


@pytest.fixture(scope="session", params=("capital", "complex"))
def matched_case(request, real_service, real_result, capital_trace, complex_trace):
    prompt = CAPITAL_PROMPT if request.param == "capital" else COMPLEX_PROMPT
    trace = capital_trace if request.param == "capital" else complex_trace
    baseline = (
        real_result
        if request.param == "capital"
        else real_service.infer_for_validation(prompt, 10).result(timeout=180)
    )
    reference = real_service.reference_baseline(prompt).result(timeout=180)
    return prompt, baseline, reference, trace


def test_tracing_preserves_exact_input_and_model_identity(matched_case):
    prompt, baseline, reference, trace = matched_case
    response = baseline.response
    assert trace.metadata.prompt == prompt == response.input.effective_text
    assert trace.metadata.token_ids == tuple(reference.token_ids)
    assert tuple(token.token_id for token in trace.tokens) == trace.metadata.token_ids
    assert trace.metadata.sequence_length == response.input.token_count
    assert trace.metadata.revision == response.model.revision
    assert trace.metadata.device == response.model.device
    assert trace.metadata.dtype == response.model.dtype
    assert tuple(token.position_id for token in trace.tokens) == tuple(
        range(response.input.token_count)
    )


def test_same_path_trace_is_bitwise_neutral(matched_case):
    _, baseline, _, trace = matched_case
    logits = trace.tensor("output.logits")
    probabilities = trace.tensor("output.probabilities")
    assert logits.shape == (1, 1, baseline.logits.numel())
    assert probabilities.shape == logits.shape
    # Observers do not replace any model operation: same-path equality is stricter
    # than the independent full-position HF reference's floating-point tolerance.
    assert torch.equal(logits.reshape(-1), baseline.logits)
    assert torch.equal(probabilities.reshape(-1), baseline.probabilities)
    assert [item.token_id for item in trace.output.candidates] == [
        item.token_id for item in baseline.response.prediction.top_k
    ]
    for traced, original in zip(
        trace.output.candidates, baseline.response.prediction.top_k, strict=True
    ):
        assert traced.rank == original.rank
        assert traced.raw_logit == original.raw_logit
        assert traced.probability == original.probability


def test_trace_matches_independent_full_position_hf_reference(matched_case):
    _, baseline, reference, trace = matched_case
    atol, rtol = (1e-5, 1e-4) if baseline.response.model.dtype == "float32" else (1e-2, 5e-2)
    torch.testing.assert_close(
        trace.tensor("output.logits").reshape(-1), reference.logits, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        trace.tensor("output.probabilities").reshape(-1),
        reference.probabilities,
        atol=1e-6,
        rtol=1e-4,
    )


def test_output_probabilities_use_full_vocabulary(matched_case):
    _, baseline, _, trace = matched_case
    logits = trace.tensor("output.logits").reshape(-1)
    probabilities = trace.tensor("output.probabilities").reshape(-1)
    exponentials = torch.exp(logits.double() - logits.double().max())
    independent = (exponentials / exponentials.sum()).float()
    torch.testing.assert_close(probabilities, independent, atol=1e-8, rtol=1e-6)
    assert abs(float(probabilities.double().sum()) - 1.0) < 1e-6
    assert torch.isfinite(logits).all() and torch.isfinite(probabilities).all()
    assert probabilities.numel() == baseline.response.model.vocabulary_size
    assert sum(candidate.probability for candidate in trace.output.candidates) < 1.0
    assert trace.output.prediction_position == trace.metadata.sequence_length - 1


@pytest.mark.parametrize("profile", ("summary", "overview", "layer", "attention", "mlp"))
def test_each_supported_capture_profile_preserves_final_logits(real_service, real_result, profile):
    options = {} if profile in ("summary", "overview") else {"layers": (12,)}
    if profile == "layer":
        options["token_indices"] = (3,)
    trace = real_service.trace(CAPITAL_PROMPT, CapturePlan(profile=profile, **options)).result(
        timeout=180
    )
    assert torch.equal(trace.tensor("output.logits").reshape(-1), real_result.logits)


def test_tracing_preserves_sampling_random_generator_states(real_service):
    mps_state = (
        torch.mps.get_rng_state()
        if torch.device(real_service.status().device).type == "mps"
        and hasattr(torch.mps, "get_rng_state")
        else None
    )
    cpu_state = torch.get_rng_state()
    python_state = random.getstate()
    trace = real_service.trace(CAPITAL_PROMPT, CapturePlan(profile="deep", layers=(12,))).result(
        timeout=180
    )
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert random.getstate() == python_state
    if mps_state is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps_state)
    assert trace.metadata.runtime["sampling"] is False
