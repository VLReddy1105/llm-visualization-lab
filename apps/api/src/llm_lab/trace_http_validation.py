"""Record bounded evidence from an actual running uvicorn server over loopback HTTP."""

import argparse
import json
import struct
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROMPT = "The capital of Germany is"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("../../docs/validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    measurements = []

    def call(method, path, body=None, artifact=None, expected=200):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(
            args.url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        try:
            response = urlopen(request, timeout=180)
        except HTTPError as failure:
            response = failure
        with response:
            content = response.read()
            status = response.status
            location = response.headers.get("Location")
            binary_metadata = response.headers.get("X-Tensor-Metadata")
        elapsed = (time.perf_counter() - started) * 1000
        if binary_metadata:
            parsed = {
                "metadata": json.loads(binary_metadata),
                "values": list(struct.unpack(f"<{len(content) // 4}f", content)),
            }
        else:
            parsed = json.loads(content) if content else None
        assert status == expected, (path, status, parsed)
        measurements.append(
            {
                "method": method,
                "path": path,
                "status": status,
                "latency_ms": elapsed,
                "response_bytes": len(content),
            }
        )
        if artifact:
            (args.output / artifact).write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )
        return parsed, location

    deadline = time.monotonic() + 180
    while True:
        status, _ = call("GET", "/api/model/status")
        if status["state"] == "ready":
            break
        assert status["state"] != "failed", status
        if time.monotonic() > deadline:
            raise TimeoutError("Resident model did not become ready")
        time.sleep(0.5)
    baseline, _ = call(
        "POST", "/api/inference/next-token", {"input": {"text": PROMPT}, "top_k": 10}
    )
    before, _ = call("GET", "/api/traces/status")
    created, location = call(
        "POST",
        "/api/traces?wait=true",
        {
            "prompt": PROMPT,
            "capture": {"profile": "deep", "layers": [12]},
            "display_top_k": 10,
        },
        "stage3-create-trace.json",
        201,
    )
    identity = created["trace_id"]
    base = f"/api/traces/{identity}"
    assert location == f"/api/v1/traces/{identity}"
    assert created["overview"]["prediction"]["top_k"] == baseline["prediction"]["top_k"]
    after_create, _ = call("GET", "/api/traces/status")
    overview, _ = call("GET", base, artifact="stage3-trace-overview.json")
    tokens, _ = call("GET", base + "/tokens")
    layers, _ = call("GET", base + "/layers")
    call("GET", base + "/layers/12", artifact="stage3-layer12.json")
    call("GET", base + "/layers/12/attention")
    head, _ = call(
        "GET",
        base + "/layers/12/attention/heads/4?query_token_index=3"
        "&include_weights=true&include_query=true",
        artifact="stage3-attention-head.json",
    )
    call("GET", base + "/layers/12/mlp", artifact="stage3-mlp.json")
    call("GET", base + "/layers/12/residual", artifact="stage3-residual.json")
    call("GET", base + "/logits", artifact="stage3-logits.json")
    component, _ = call("GET", base + "/components/layer%5B12%5D.residual.before")
    sliced, _ = call(
        "POST",
        base + f"/tensors/{component['tensor_id']}/slice",
        {
            "component_id": component["component_id"],
            "selectors": {
                "batch": {"index": 0},
                "token": {"index": 3},
                "hidden": {"start": 0, "stop": 128},
            },
        },
        "stage3-tensor-slice.json",
    )
    call(
        "GET", base + "/layers/12/attention/heads/4/relation?query_token_index=3&key_token_index=4"
    )
    binary, _ = call(
        "POST",
        base + f"/tensors/{component['tensor_id']}/slice",
        {
            "component_id": component["component_id"],
            "format": "binary",
            "selectors": {
                "batch": {"index": 0},
                "token": {"index": 3},
                "hidden": {"start": 0, "stop": 128},
            },
        },
    )
    assert binary["values"] == sliced["values"]
    assert binary["metadata"]["shape"] == sliced["shape"]
    call("GET", base + "/layers/15/attention", expected=409)
    call("POST", base + "/components/output.logits/slice", {}, expected=413)
    after_reads, _ = call("GET", "/api/traces/status")
    assert after_create["request_forward_passes"] == before["request_forward_passes"] + 1
    assert after_reads["request_forward_passes"] == after_create["request_forward_passes"]
    assert tokens["total"] == 5 and layers["total"] == 28
    assert head["kv_head_index"] == 2
    assert sliced["shape"] == [1, 1, 128]
    assert not overview["overview"]["validation"]["failed"]
    call("DELETE", base, expected=204)
    call("DELETE", base, expected=204)
    deleted, _ = call("GET", base, expected=404)
    final_status, _ = call("GET", "/api/traces/status")
    assert final_status["retained_bytes"] == before["retained_bytes"]
    evidence = {
        "status": "PASS",
        "recorded_at": datetime.now(UTC).isoformat(),
        "transport": "urllib over loopback to actual uvicorn process",
        "trace_id": identity,
        "prompt": PROMPT,
        "model": created["overview"]["model"],
        "baseline_prediction": baseline["prediction"],
        "baseline_matches_trace": True,
        "before_create": before,
        "after_create": after_create,
        "after_reads": after_reads,
        "after_delete": final_status,
        "additional_creation_forwards": 1,
        "additional_retrieval_forwards": 0,
        "binary_matches_json": True,
        "delete_access_error": deleted,
        "measurements": measurements,
    }
    (args.output / "stage3-http-validation.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "trace_id": identity,
                "retained_bytes": created["retained_bytes"],
                "retrieval_forwards": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
