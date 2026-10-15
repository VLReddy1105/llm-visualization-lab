"""Independent real-Qwen Stage 8 audit for one device per process.

Only bounded prompt, per-step logits, measured times and scalar comparisons are saved;
vocabulary vectors are released after each audit and never retained in sessions.
"""

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from llm_lab.analysis.distributions import distribution
from llm_lab.analysis.sampling import process_logits
from llm_lab.config import Settings
from llm_lab.generation.service import GenerationService
from llm_lab.models.devices import synchronize
from llm_lab.models.manager import ModelService
from llm_lab.schemas.generation import GenerationRequest, GenerationSampling
from llm_lab.schemas.inference import RawInput
from llm_lab.store.trace_store import TraceStore


def independent_logits(model, prefix):
    def run(ids):
        tensor = torch.tensor([ids], dtype=torch.long, device=model.adapter.device)
        with torch.inference_mode():
            output = model._model(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                logits_to_keep=1,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            values = output.logits[0, -1].float().cpu().clone()
        synchronize(model.adapter.device)
        return values

    return model.submit_trace_job(run, prefix).result(timeout=180)


def await_terminal(service, identity, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = service.get(identity)
        if result.status in {"completed", "cancelled", "failed"}:
            return result
        time.sleep(0.1)
    raise TimeoutError(f"Generation {identity} did not finish within {timeout}s")


def request(
    prompt,
    key,
    *,
    mode="greedy",
    seed=None,
    maximum=3,
    capture="none",
    traces=0,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
):
    return GenerationRequest(
        input=RawInput(text=prompt),
        request_key=key,
        max_new_tokens=maximum,
        capture_profile=capture,
        capture_layer=12,
        retain_traces=traces,
        sampling=GenerationSampling(
            token_index=0,
            mode=mode,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            display_top_k=10,
        ),
    )


def tokenize_ids(adapter, prompt):
    encoded = adapter.tokenize(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    if isinstance(encoded, torch.Tensor):
        return encoded[0].tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        return list(encoded[0])
    return list(encoded)


def assert_processor_reference(logits, parameters):
    ids = torch.empty((1, 0), dtype=torch.long)
    ref = logits.detach().cpu().double().reshape(1, -1).clone()
    ref = TemperatureLogitsWarper(parameters.temperature)(ids, ref)
    if parameters.top_k:
        ref = TopKLogitsWarper(parameters.top_k)(ids, ref)
    if parameters.top_p < 1:
        ref = TopPLogitsWarper(parameters.top_p)(ids, ref)
    actual = process_logits(logits, parameters)
    torch.testing.assert_close(actual, ref.reshape(-1), rtol=0, atol=0, equal_nan=True)
    return ref.reshape(-1)


def audit_device(device, output_path):
    settings = Settings(auto_load=False, device=device)
    model = ModelService(settings)
    store = TraceStore(
        settings.trace_store_max_bytes, settings.trace_ttl_seconds, settings.trace_store_max_entries
    )
    model.start_loading(device=device)
    model.wait_until_ready()
    service = GenerationService(model, store, settings)
    adapter = model.adapter
    prompts = ("The capital of Germany is", "The capital of France is")
    records = []
    processor_cases = []
    sampler_settings = (
        GenerationSampling(
            mode="sample", seed=31, temperature=1.0, top_k=0, top_p=1.0, display_top_k=10
        ),
        GenerationSampling(
            mode="sample", seed=31, temperature=0.7, top_k=40, top_p=0.9, display_top_k=10
        ),
        GenerationSampling(
            mode="sample", seed=31, temperature=0.7, top_k=1, top_p=0.01, display_top_k=10
        ),
    )
    try:
        for index, prompt in enumerate(prompts):
            ids = tokenize_ids(adapter, prompt)
            before = model.execution_metrics()["request_forward_passes"]
            ack = service.create(
                request(prompt, f"stage8_greedy_{index}", maximum=3, capture="deep", traces=3)
            )
            session = await_terminal(service, ack.generation_id)
            after_generation = model.execution_metrics()["request_forward_passes"]
            assert session.status == "completed", session.error
            assert session.stop_reason == "max_new_tokens"
            assert len(session.steps) == 3
            assert session.attempted_forwards == session.completed_steps == 3
            assert after_generation - before == 3
            expected_prefix = list(ids)
            steps = []
            for step in session.steps:
                assert step.prefix_token_ids == expected_prefix
                assert step.decision_position == len(expected_prefix) - 1
                assert step.selected_position == len(expected_prefix)
                assert step.forward_execution_id
                assert step.source_trace_id
                with store.acquire(step.source_trace_id) as (trace_status, trace):
                    assert trace_status["status"] == "ready"
                    assert trace.metadata.runtime["forward_count"] == 1
                    assert trace.metadata.runtime["generation_id"] == session.generation_id
                    assert trace.metadata.runtime["generation_step"] == step.step_index
                    assert (
                        trace.metadata.runtime["forward_execution_id"] == step.forward_execution_id
                    )
                    retained_logits = trace.tensor("output.logits", copy=False).reshape(-1).clone()
                reference = independent_logits(model, expected_prefix)
                torch.testing.assert_close(retained_logits, reference, rtol=0, atol=0)
                raw, order, _ = distribution(reference)
                selected = int(order[0])
                assert step.raw_argmax_token_id == selected
                assert step.selected_token_id == selected
                assert step.selected_raw_logit == float(reference[selected])
                assert abs(step.selected_model_probability - float(raw[selected])) <= 1e-8
                assert step.selected_adjusted_probability is None
                assert step.distribution_statistics["adjusted_probability_sum"] is None
                expected_prefix.append(step.selected_token_id)
                assert step.prefix_sha256
                assert step.selected_position == len(expected_prefix) - 1
                assert step.decoded_continuation == adapter.tokenizer.decode(
                    expected_prefix[len(ids) :],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                steps.append(
                    {
                        "step_index": step.step_index,
                        "prefix_ids": step.prefix_token_ids,
                        "selected_token_id": step.selected_token_id,
                        "raw_argmax_token_id": step.raw_argmax_token_id,
                        "selected_raw_logit": step.selected_raw_logit,
                        "selected_model_probability": step.selected_model_probability,
                        "independent_logits_bitwise_equal": True,
                        "forward_execution_id": step.forward_execution_id,
                        "source_trace_id": step.source_trace_id,
                        "forward_ms": step.forward_duration_ms,
                        "elapsed_ms": step.elapsed_ms,
                        "source_retained_bytes": trace_status["retained_bytes"],
                    }
                )
            assert session.selected_token_ids == expected_prefix[len(ids) :]
            assert session.decoded_continuation == adapter.tokenizer.decode(
                session.selected_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            records.append(
                {
                    "mode": "greedy",
                    "prompt": prompt,
                    "input_ids": ids,
                    "generation_id": session.generation_id,
                    "selected_ids": session.selected_token_ids,
                    "decoded_continuation": session.decoded_continuation,
                    "forward_attempt_delta": after_generation - before,
                    "published_steps": len(session.steps),
                    "steps": steps,
                    "session_json_bytes": session.result_bytes,
                    "session_duration_ms": session.session_duration_ms,
                }
            )

        prompt = prompts[0]
        ids = tokenize_ids(adapter, prompt)
        initial_logits = independent_logits(model, ids)
        divergence_parameters = GenerationSampling(
            mode="sample",
            seed=0,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            display_top_k=10,
        )
        divergence_logits = process_logits(initial_logits, divergence_parameters)
        divergence_probabilities, _, _ = distribution(divergence_logits)
        chosen_seed = None
        for seed in range(256):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            sample = int(
                torch.multinomial(divergence_probabilities.double(), 1, generator=generator)
            )
            if sample != int(initial_logits.argmax()):
                chosen_seed = seed
                break
        assert chosen_seed is not None, "seed sweep found no sample differing from raw argmax"

        for case_index, params in enumerate(sampler_settings):
            chosen = chosen_seed if case_index == 1 else int(params.seed or 0)
            configured = params.model_copy(update={"seed": chosen})
            reference_logits = independent_logits(model, ids)
            independently_processed = assert_processor_reference(reference_logits, configured)
            eligible = int(torch.isfinite(independently_processed).sum())
            raw, _, _ = distribution(reference_logits)
            processor_cases.append(
                {
                    "temperature": configured.temperature,
                    "top_k": configured.top_k,
                    "top_p": configured.top_p,
                    "eligible_count": eligible,
                    "raw_probability_sum": float(raw.double().sum()),
                    "transformers_processors_bitwise_equal": True,
                }
            )
            summaries = []
            for repeat in range(2):
                before = model.execution_metrics()["request_forward_passes"]
                body = request(
                    prompt,
                    f"stage8_sample_{case_index}_{repeat}",
                    mode="sample",
                    seed=chosen,
                    maximum=3,
                    capture="none",
                    traces=0,
                    temperature=configured.temperature,
                    top_k=configured.top_k,
                    top_p=configured.top_p,
                )
                ack = service.create(body)
                session = await_terminal(service, ack.generation_id)
                after = model.execution_metrics()["request_forward_passes"]
                assert session.status == "completed", session.error
                assert session.stop_reason == "max_new_tokens"
                assert after - before == session.attempted_forwards == 3
                prefix = list(ids)
                rng = torch.Generator(device="cpu").manual_seed(chosen)
                for step in session.steps:
                    assert step.prefix_token_ids == prefix
                    reference = independent_logits(model, prefix)
                    processed = assert_processor_reference(reference, configured)
                    probabilities, _, _ = distribution(processed)
                    raw_probabilities, _, _ = distribution(reference)
                    expected = int(torch.multinomial(probabilities.double(), 1, generator=rng))
                    assert step.selected_token_id == expected
                    assert step.raw_argmax_token_id == int(reference.argmax())
                    assert step.selected_adjusted_probability == float(probabilities[expected])
                    assert step.selected_model_probability == float(raw_probabilities[expected])
                    prefix.append(expected)
                summaries.append(
                    {
                        "generation_id": session.generation_id,
                        "selected_ids": session.selected_token_ids,
                        "attempted_forwards": session.attempted_forwards,
                        "session_json_bytes": session.result_bytes,
                        "duration_ms": session.session_duration_ms,
                        "no_trace_reason": session.steps[0].capture_unavailable_reason,
                    }
                )
            assert summaries[0]["selected_ids"] == summaries[1]["selected_ids"]
            records.append(
                {
                    "mode": "sample",
                    "seed": chosen,
                    "differs_from_argmax_seed_case": case_index == 1,
                    "temperature": configured.temperature,
                    "top_k": configured.top_k,
                    "top_p": configured.top_p,
                    "repeatable_same_device": True,
                    "runs": summaries,
                }
            )

        status = store.status()
        result = {
            "result": "PASS",
            "completed_at": datetime.now(UTC).isoformat(),
            "device": adapter.device,
            "dtype": str(adapter.dtype).removeprefix("torch."),
            "model_id": adapter.model.config._name_or_path,
            "model_revision": adapter.revision,
            "tokenizer_revision": adapter.revision,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "configuration": {
                "use_cache": False,
                "attention_backend": "eager",
                "max_new_tokens": 3,
                "prefix_ids_are_not_retokenized": True,
            },
            "greedy_and_sampling_runs": records,
            "processor_cases": processor_cases,
            "store_status": status,
            "model_forward_counter_scope": model.execution_metrics(),
            "limitations": [
                "Runs are short local research audits, not throughput claims.",
                (
                    "This report records Python retained bytes and timings; allocator peak "
                    "RSS/MPS memory is not asserted."
                ),
                (
                    "CPU and MPS use the same explicitly seeded CPU-generator policy; "
                    "cross-device token equality is not claimed."
                ),
            ],
        }
        Path(output_path).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "result": result["result"],
                    "device": adapter.device,
                    "sessions": len(records),
                    "output": str(output_path),
                },
                indent=2,
            )
        )
    finally:
        service.close()
        store.close()
        model.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit_device(args.device, args.output)


if __name__ == "__main__":
    main()
