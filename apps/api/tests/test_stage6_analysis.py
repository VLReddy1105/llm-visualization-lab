"""Stage 6 science uses real saved Qwen tensors; algebra fixtures test edge cases only."""

import time
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from llm_lab.analysis.distributions import distribution
from llm_lab.analysis.jobs import AnalysisService
from llm_lab.analysis.logit_lens import logit_lens
from llm_lab.analysis.sampling import process_logits, sampling_preview
from llm_lab.analysis.token_journey import cosine, evolution, vector
from llm_lab.config import Settings
from llm_lab.errors import ServiceError
from llm_lab.main import create_app
from llm_lab.models.manager import ModelService
from llm_lab.schemas.analysis import (
    AnalysisJob,
    EvolutionRequest,
    LensParameters,
    PositionParameters,
    SamplingParameters,
    SamplingRequest,
)
from llm_lab.store.trace_store import TraceStore
from llm_lab.tracing.capture_plan import CapturePlan


@pytest.fixture(scope="module", params=["Germany", "France"])
def overview(real_service, request):
    return real_service.trace(
        f"The capital of {request.param} is", CapturePlan(profile="overview"), 100
    ).result(180)


def insert(store, trace):
    identity = store.reserve(trace.metadata.actual_retained_bytes)["trace_id"]
    store.commit(identity, replace(trace, metadata=replace(trace.metadata, trace_id=identity)))
    return identity


def done(service, identity, job):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = service.get(identity, job.analysis_id)
        if result.status not in {"queued", "running"}:
            return result
        time.sleep(0.01)
    pytest.fail("Analysis did not complete")


def test_metrics_order_alignment_and_independent_reductions(overview):
    for pos in (0, overview.metadata.sequence_length - 1):
        result = evolution(overview, pos)
        assert result.source.token.token_id == overview.tokens[pos].token_id
        states = result.checkpoints
        assert len(states) == overview.metadata.architecture.layers + 2
        assert [s.layer_index for s in states[1:-1]] == list(
            range(overview.metadata.architecture.layers)
        )
        assert states[0].stage == "embedding" and states[-1].stage == "final_normalized"
        assert (
            states[-2].component_id
            == f"layer[{overview.metadata.architecture.layers - 1}].residual.after_mlp"
        )
        original = vector(overview, states[0].component_id, pos).double()
        final = vector(overview, states[-2].component_id, pos).double()
        for i, state in enumerate(states):
            v = vector(overview, state.component_id, pos).double()
            assert state.l2_norm == pytest.approx(
                float(v.square().sum().sqrt()), rel=1e-13, abs=1e-12
            )
            assert state.cosine_embedding == pytest.approx(
                float(torch.nn.functional.cosine_similarity(v, original, dim=0)), abs=1e-12
            )
            assert state.cosine_final_residual == pytest.approx(
                float(torch.nn.functional.cosine_similarity(v, final, dim=0)), abs=1e-12
            )
            if i:
                before = vector(overview, states[i - 1].component_id, pos).double()
                assert state.consecutive_distance == pytest.approx(
                    float((v - before).square().sum().sqrt()), rel=1e-13, abs=1e-12
                )
                assert state.norm_change == state.l2_norm - states[i - 1].l2_norm
        assert not torch.equal(final, vector(overview, states[-1].component_id, pos).double())
        pair = next(
            p for p in result.comparisons if p.from_checkpoint == 3 and p.to_checkpoint == 14
        )
        assert pair.l2_distance == float(
            torch.linalg.vector_norm(
                vector(overview, states[14].component_id, pos).double()
                - vector(overview, states[3].component_id, pos).double()
            )
        )
        assert len(result.model_dump_json()) < 1024**2


def test_zero_cosine_and_full_distribution_tie_policy():
    assert cosine(torch.zeros(4), torch.ones(4)) is None
    assert cosine(torch.zeros(4), torch.zeros(4)) is None
    assert cosine(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0])) == 1
    probabilities, order, ranks = distribution(torch.tensor([1.0, 2.0, 2.0, -100.0]))
    assert order.tolist() == [1, 2, 0, 3]
    assert ranks.tolist() == [3, 1, 2, 4]
    assert float(probabilities.double().sum()) == pytest.approx(1, abs=1e-7)
    assert float(probabilities[order[:1]].sum()) < 1


def test_zero_metric_reasons_and_no_preceding_state(overview):
    tensors = dict(overview._tensors)
    tensors[overview.embedding.output] = torch.zeros_like(tensors[overview.embedding.output])
    result = evolution(replace(overview, _tensors=tensors), 0)
    assert result.checkpoints[0].l2_norm == 0
    assert result.checkpoints[0].consecutive_distance is None
    assert result.checkpoints[0].undefined["cosine_previous"] == "no preceding checkpoint"
    assert result.checkpoints[1].undefined["cosine_embedding"] == "zero vector norm"


def test_actual_norm_head_full_vocab_and_no_forward(real_service, overview):
    pos = overview.output.prediction_position
    adapter = real_service.adapter
    before = real_service.execution_metrics()["request_forward_passes"]
    rng = torch.get_rng_state().clone()
    with (
        patch.object(adapter, "forward", side_effect=AssertionError("No transformer forward")),
        patch.object(adapter.model, "generate", side_effect=AssertionError("No generation")),
    ):
        lens = real_service.submit_analysis_job(
            logit_lens, overview, LensParameters(token_index=pos), adapter
        ).result(180)
        assert lens.final_logits_comparison.passed
        assert lens.final_probabilities_comparison.passed
        assert lens.final_normalization_comparison.passed
        assert not lens.checkpoints[-1].normalization_applied
        assert all(s.normalization_applied for s in lens.checkpoints[:-1])
        for index in (0, 13, len(lens.checkpoints) - 2, len(lens.checkpoints) - 1):
            checkpoint = lens.checkpoints[index]
            values = (
                vector(overview, checkpoint.component_id, pos).reshape(1, 1, -1).to(adapter.device)
            )
            with torch.inference_mode():
                normalized = (
                    values
                    if checkpoint.stage == "final_normalized"
                    else adapter.model.model.norm(values)
                )
                logits = adapter.model.lm_head(normalized).detach().cpu().reshape(-1)
            exp = torch.exp(logits.double() - logits.double().max())
            probabilities = (exp / exp.sum()).float()
            assert checkpoint.probability_sum == pytest.approx(1, abs=1e-6)
            for candidate in checkpoint.top_candidates + checkpoint.comparison_candidates:
                t = candidate.token_id
                assert candidate.raw_logit == float(logits[t])
                assert candidate.probability == pytest.approx(float(probabilities[t]), abs=1e-8)
                rank = int((logits > logits[t]).sum()) + int((logits[:t] == logits[t]).sum()) + 1
                assert candidate.rank == rank
                assert candidate.tokenizer_piece == adapter.tokenizer.convert_ids_to_tokens(t)
                assert candidate.decoded_token == adapter.tokenizer.decode(
                    [t], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
        assert len(lens.model_dump_json()) < 1024**2
    assert torch.equal(rng, torch.get_rng_state())
    assert real_service.execution_metrics()["request_forward_passes"] == before


def test_other_position_and_pinned_intermediate_candidate(real_service, overview):
    adapter = real_service.adapter
    params = LensParameters(token_index=0)
    lens = real_service.submit_analysis_job(logit_lens, overview, params, adapter).result(180)
    assert lens.source.token.token_index == 0 and lens.next_position == 1
    assert lens.final_logits_comparison is None
    chosen = lens.checkpoints[0].top_candidates[0].token_id
    rerun = real_service.submit_analysis_job(
        logit_lens, overview, LensParameters(token_index=0, candidate_ids=[chosen]), adapter
    ).result(180)
    assert chosen in rerun.candidate_ids
    assert all(chosen in [c.token_id for c in p.comparison_candidates] for p in rerun.checkpoints)
    with pytest.raises(ServiceError, match="match"):
        logit_lens(
            replace(overview, metadata=replace(overview.metadata, revision="wrong")),
            params,
            adapter,
        )


@pytest.mark.parametrize(
    "temperature,k,p",
    [(1.0, 0, 1.0), (0.7, 20, 0.8), (2.0, 1, 1.0), (1.0, 4, 0.00001), (100.0, 0, 0.95)],
)
def test_sampling_independent_pinned_processors(overview, temperature, k, p):
    params = SamplingParameters(
        token_index=overview.output.prediction_position, temperature=temperature, top_k=k, top_p=p
    )
    logits = overview.tensor(overview.output.logits).reshape(-1)
    rng = torch.get_rng_state().clone()
    result = sampling_preview(overview, params)
    reference = TemperatureLogitsWarper(temperature)(torch.empty((1, 0)), logits.double()[None])
    if k:
        reference = TopKLogitsWarper(k)(torch.empty((1, 0)), reference)
    if p < 1:
        reference = TopPLogitsWarper(p)(torch.empty((1, 0)), reference)
    assert torch.equal(process_logits(logits, params), reference.reshape(-1))
    exp = torch.exp(reference - reference.max())
    expected = (exp / exp.sum()).float().reshape(-1)
    for c in result.candidates:
        assert c.probability == pytest.approx(float(expected[c.token_id]), abs=1e-8)
        assert c.model_probability == float(
            overview.tensor(overview.output.probabilities).reshape(-1)[c.token_id]
        )
    assert result.probability_sum == pytest.approx(1, abs=1e-6)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(logits, overview.tensor(overview.output.logits).reshape(-1))


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": 0},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"temperature": True},
        {"top_k": True},
        {"top_k": -1},
        {"top_p": 0},
        {"top_p": 1.1},
        {"display_top_k": 101},
        {"mode": "greedy", "top_k": 10},
    ],
)
def test_sampling_invalid_boundaries(fields):
    with pytest.raises(ValidationError):
        SamplingParameters(token_index=0, **fields)


def test_sampling_ties_boundary_greedy_and_extreme():
    logits = torch.tensor([2.0, 2.0, 1.0, 0.0])
    params = SamplingParameters(token_index=0, top_k=1)
    assert torch.isfinite(process_logits(logits, params)).tolist() == [True, True, False, False]
    params = SamplingParameters(token_index=0, top_p=0.6)
    assert torch.isfinite(process_logits(logits, params)).sum() == 2
    greedy = process_logits(logits, SamplingParameters(token_index=0, mode="greedy"))
    assert torch.isfinite(greedy).tolist() == [True, False, False, False]
    with pytest.raises(ServiceError):
        process_logits(logits, SamplingParameters(token_index=0, top_k=5))
    with pytest.raises(ServiceError):
        process_logits(logits, SamplingParameters(token_index=0, temperature=5e-324))


def test_cache_identity_bounded_results_and_unloaded_cpu_analyses(overview):
    settings = Settings(auto_load=False)
    store = TraceStore()
    model = ModelService(settings)
    jobs = AnalysisService(model, store, settings)
    try:
        identity = insert(store, overview)
        req = EvolutionRequest(kind="token_evolution", parameters=PositionParameters(token_index=0))
        first = jobs.create(identity, req)
        repeated = jobs.create(identity, req)
        assert repeated.analysis_id == first.analysis_id and repeated.cache_hit
        result = done(jobs, identity, first)
        assert result.status == "ready", result.error
        assert jobs.create(identity, req).analysis_id == first.analysis_id
        assert result.retained_bytes < settings.trace_api_max_response_bytes
        other = insert(store, overview)
        with pytest.raises(ServiceError) as exc:
            jobs.get(other, first.analysis_id)
        assert exc.value.code == "ANALYSIS_NOT_FOUND"
        sample = jobs.create(
            identity,
            SamplingRequest(
                kind="sampling_preview",
                parameters=SamplingParameters(token_index=overview.output.prediction_position),
            ),
        )
        assert done(jobs, identity, sample).status == "ready"
        assert model.execution_metrics()["request_forward_passes"] == 0
        assert store.status()["active_leases"] == 0
        store.delete(identity)
        with pytest.raises(ServiceError):
            jobs.get(identity, first.analysis_id)
    finally:
        jobs.close()
        model.close()
        store.close()


def test_http_jobs_missing_capture_and_explicit_recapture(real_service, overview):
    settings = real_service.settings
    store = TraceStore()
    with TestClient(create_app(settings, real_service, store)) as client:
        identity = insert(store, overview)
        path = f"/api/v1/traces/{identity}"
        before = real_service.execution_metrics()["request_forward_passes"]
        body = {
            "kind": "logit_lens",
            "parameters": {"token_index": overview.output.prediction_position},
        }
        response = client.post(path + "/analyses", json=body)
        assert response.status_code == 202, response.text
        job = done(
            client.app.state.analysis_service, identity, AnalysisJob.model_validate(response.json())
        )
        assert job.status == "ready", job.error
        assert client.get(path + "/analyses/" + job.analysis_id).json()["result"][
            "final_logits_comparison"
        ]["passed"]
        assert client.get(path + "/tokens/0/journey").status_code == 202
        assert (
            client.post(
                path + "/analyses", json={"kind": "logit_lens", "parameters": {"token_index": 999}}
            ).status_code
            == 422
        )
        assert real_service.execution_metrics()["request_forward_passes"] == before
        summary = client.post(
            "/api/v1/traces?wait=true", json={"prompt": overview.metadata.prompt}
        ).json()["trace_id"]
        missing = client.post(f"/api/v1/traces/{summary}/analyses", json=body)
        assert (
            missing.status_code == 409
            and missing.json()["error"]["code"] == "COMPONENT_NOT_CAPTURED"
        )
        assert real_service.execution_metrics()["request_forward_passes"] == before + 1
        captured = client.post(
            "/api/v1/traces?wait=true",
            json={"prompt": overview.metadata.prompt, "capture": {"profile": "overview"}},
        ).json()["trace_id"]
        assert client.post(f"/api/v1/traces/{captured}/analyses", json=body).status_code == 202
        assert real_service.execution_metrics()["request_forward_passes"] == before + 2


def test_selected_capture_absolute_positions_and_missing_data(real_service):
    trace = real_service.trace(
        "one two three", CapturePlan(profile="overview", token_indices=(1,)), 100
    ).result(180)
    result = evolution(trace, 1)
    assert result.source.token.token_index == 1
    assert result.checkpoints[0].l2_norm == float(
        trace.tensor("embedding.output")[0, 0].double().norm()
    )
    for position in (0, 99):
        with pytest.raises(ServiceError):
            evolution(trace, position)
    summary = real_service.trace("one two three", CapturePlan(profile="summary"), 100).result(180)
    with pytest.raises(ServiceError) as exc:
        evolution(summary, 1)
    assert exc.value.code == "COMPONENT_NOT_CAPTURED"


def test_job_cancel_leases_expiry_and_budget(real_service, overview, monkeypatch):
    import threading

    from llm_lab.analysis import jobs as jobs_module

    class Clock:
        value = time.time()

        def __call__(self):
            return self.value

    clock = Clock()
    settings = Settings(auto_load=False, analysis_ttl_seconds=10, analysis_cache_max_entries=2)
    store = TraceStore(ttl_seconds=20, clock=clock, wall_clock=clock)
    jobs = AnalysisService(real_service, store, settings, clock=clock, wall_clock=clock)
    req = EvolutionRequest(kind="token_evolution", parameters=PositionParameters(token_index=0))
    entered, release = threading.Event(), threading.Event()
    original = jobs_module.evolution

    def held(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    try:
        identity = insert(store, overview)
        monkeypatch.setattr(jobs_module, "evolution", held)
        first = jobs.create(identity, req)
        assert entered.wait(10)
        assert store.status()["active_leases"] == 1
        assert jobs.create(identity, req).analysis_id == first.analysis_id
        jobs.cancel(identity, first.analysis_id)
        release.set()
        jobs._executor.submit(lambda: None).result(10)
        assert jobs.get(identity, first.analysis_id).status == "cancelled"
        assert store.status()["active_leases"] == 0
        monkeypatch.setattr(jobs_module, "evolution", original)
        second = done(jobs, identity, jobs.create(identity, req))
        assert second.status == "ready" and second.analysis_id != first.analysis_id
        expiry = second.expires_at
        clock.value += 9
        assert jobs.create(identity, req).expires_at == expiry
        clock.value += 1
        with pytest.raises(ServiceError) as error:
            jobs.get(identity, second.analysis_id)
        assert error.value.code == "ANALYSIS_EXPIRED"
        # Expiry while leased must discard work and then release the source bytes.
        entered.clear()
        release.clear()
        monkeypatch.setattr(jobs_module, "evolution", held)
        third = jobs.create(identity, req)
        assert entered.wait(10)
        clock.value += 10
        store.sweep()
        assert store.status()["deferred_removals"] == 1
        release.set()
        jobs._executor.submit(lambda: None).result(10)
        assert store.status()["retained_bytes"] == 0
        with pytest.raises(ServiceError) as error:
            jobs.get(identity, third.analysis_id)
        assert error.value.code == "TRACE_EXPIRED"
    finally:
        release.set()
        jobs.close()
        store.close()


def test_analysis_lru_and_fail_closed_response_scratch_budgets(real_service, overview):
    req = EvolutionRequest(kind="token_evolution", parameters=PositionParameters(token_index=0))
    store = TraceStore()
    identity = insert(store, overview)
    settings = Settings(auto_load=False, analysis_cache_max_entries=1)
    jobs = AnalysisService(real_service, store, settings)
    try:
        first = done(jobs, identity, jobs.create(identity, req))
        second = done(
            jobs,
            identity,
            jobs.create(
                identity,
                EvolutionRequest(
                    kind="token_evolution", parameters=PositionParameters(token_index=1)
                ),
            ),
        )
        assert second.status == "ready"
        with pytest.raises(ServiceError) as error:
            jobs.get(identity, first.analysis_id)
        assert error.value.code == "ANALYSIS_EXPIRED"
        assert len(jobs._entries) == 1 and jobs._accounted() < settings.analysis_cache_max_bytes
    finally:
        jobs.close()
    for overrides in (
        {"analysis_work_max_bytes": 1024},
        {"trace_api_max_response_bytes": 1024},
        {"analysis_cache_max_bytes": 1024},
    ):
        bounded = AnalysisService(real_service, store, Settings(auto_load=False, **overrides))
        try:
            with pytest.raises(ServiceError):
                bounded.create(identity, req)
            assert not bounded._entries
        finally:
            bounded.close()
    store.close()
