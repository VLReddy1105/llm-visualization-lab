"""Measure bounded Stage 7 HTTP summaries using the real browser's saved sources.

Run after residual.spec.ts against the same live backend. No model execution is
requested; source UUIDs and scientific activations come from the real browser run.
"""

import json
import time
from pathlib import Path

import httpx


def main():
    output = Path(__file__).resolve().parents[3] / "docs" / "validation"
    records = []
    with httpx.Client(base_url="http://127.0.0.1:8000/api/v1", timeout=30) as client:
        before = client.get("/traces/status").json()
        for country in ("germany", "france"):
            source = json.loads((output / f"stage7-{country}-browser.json").read_text())["recovery"]
            for kind in ("residual_stream", "mlp_internals"):
                for ranking in ("absolute", "positive", "negative"):
                    body = {
                        "kind": kind,
                        "parameters": {
                            "token_index": 4,
                            "layer_index": 12,
                            "hidden_coordinate_index": 23,
                            "intermediate_coordinate_index": 17,
                            "top_k": 64,
                            "ranking": ranking,
                        },
                    }
                    started = time.perf_counter()
                    response = client.post(f"/traces/{source}/analyses", json=body)
                    response.raise_for_status()
                    first = response.json()
                    deadline = time.monotonic() + 30
                    while first["status"] in {"queued", "running"}:
                        assert time.monotonic() < deadline
                        time.sleep(0.01)
                        response = client.get(f"/traces/{source}/analyses/{first['analysis_id']}")
                        response.raise_for_status()
                        first = response.json()
                    assert first["status"] == "ready", first.get("error")
                    uncached_ms = (time.perf_counter() - started) * 1000
                    assert first["result"]["source"]["trace_id"] == source
                    assert all(len(o["coordinates"]) <= 64 for o in first["result"]["operations"])
                    assert all(c["metrics"]["passed"] for c in first["result"]["reconciliation"])
                    sizes, timings = [len(response.content)], []
                    for _ in range(20):
                        started = time.perf_counter()
                        response = client.post(f"/traces/{source}/analyses", json=body)
                        response.raise_for_status()
                        hit = response.json()
                        timings.append((time.perf_counter() - started) * 1000)
                        sizes.append(len(response.content))
                        assert hit["cache_hit"] and hit["analysis_id"] == first["analysis_id"]
                        assert hit["result"] == first["result"]
                    ordered = sorted(timings)
                    records.append(
                        {
                            "country": country,
                            "source_trace_id": source,
                            "kind": kind,
                            "ranking": ranking,
                            "top_k": 64,
                            "uncached_ms": uncached_ms,
                            "cache_p50_ms": (ordered[9] + ordered[10]) / 2,
                            "cache_p95_ms": ordered[18],
                            "verified_cache_hits": len(timings),
                            "max_http_response_bytes": max(sizes),
                            "encoded_result_bytes": first["retained_bytes"],
                        }
                    )
        after = client.get("/traces/status").json()
    assert before["request_forward_passes"] == after["request_forward_passes"]
    assert before["retained_bytes"] == after["retained_bytes"]
    assert after["active_leases"] == 0
    assert max(r["max_http_response_bytes"] for r in records) < 1024 * 1024
    evidence = {
        "status": "PASS",
        "before": before,
        "after": after,
        "cases": records,
        "additional_model_forwards": 0,
        "verified_cache_hits": sum(r["verified_cache_hits"] for r in records),
    }
    (output / "stage7-http-performance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: {len(records)} bounded HTTP cases, {evidence['verified_cache_hits']} cache hits")


if __name__ == "__main__":
    main()
