"""Storage-only fixtures test ownership and budgets, not model computation."""

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import UUID

import pytest
import torch

from llm_lab.errors import ServiceError
from llm_lab.store.trace_store import TraceStore


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def store():
    clock = Clock()
    result = TraceStore(max_bytes=100, ttl_seconds=10, clock=clock, wall_clock=clock)
    return result, clock


def metadata(size):
    return SimpleNamespace(
        actual_retained_bytes=size,
        model_id="storage-test",
        revision="test",
        sequence_length=1,
        capture_plan=SimpleNamespace(profile="summary", layers=()),
    )


def insert(store, size=40):
    identity = store.reserve(size)["trace_id"]
    trace = SimpleNamespace(metadata=metadata(size), tensor=torch.zeros(size // 4))
    store.commit(identity, trace)
    return identity


def error_code(code, function, *args):
    with pytest.raises(ServiceError) as failure:
        function(*args)
    assert failure.value.code == code


def test_reserve_commit_unique_ids_and_counters(store):
    s, _ = store
    entry = s.reserve(80)
    assert UUID(entry["trace_id"]).version == 4
    assert s.status()["reserved_bytes"] == 80
    trace = SimpleNamespace(metadata=metadata(40))
    s.commit(entry["trace_id"], trace)
    assert s.status()["accounted_bytes"] == 40
    assert s.snapshot(entry["trace_id"])["status"] == "ready"
    assert s.reserve(10)["trace_id"] != entry["trace_id"]


def test_ttl_is_absolute_and_sweeps(store):
    s, clock = store
    identity = insert(s)
    expiry = s.snapshot(identity)["expires_at"]
    clock.now = 9
    assert s.snapshot(identity)["expires_at"] == expiry
    clock.now = 10
    error_code("TRACE_EXPIRED", s.snapshot, identity)
    assert s.status()["retained_bytes"] == 0
    assert s.status()["expiration_count"] == 1
    clock.now += 86400
    error_code("TRACE_NOT_FOUND", s.snapshot, identity)


def test_lru_reservation_evicts_eligible_oldest(store):
    s, _ = store
    first, second = insert(s), insert(s)
    s.snapshot(first)
    third = insert(s)
    error_code("TRACE_EVICTED", s.snapshot, second)
    assert s.snapshot(first)["status"] == s.snapshot(third)["status"] == "ready"
    assert s.status()["retained_bytes"] == 80
    assert s.status()["eviction_count"] == 1


def test_budget_rejection_is_atomic_and_does_not_evict(store):
    s, _ = store
    identity = insert(s)
    error_code("TRACE_STORE_BUDGET_EXCEEDED", s.reserve, 101)
    assert s.snapshot(identity)["status"] == "ready"
    assert s.status()["accounted_bytes"] == 40
    assert s.status()["eviction_count"] == 0


def test_reservations_cannot_be_evicted(store):
    s, _ = store
    identity = s.reserve(90)["trace_id"]
    error_code("TRACE_STORE_BUDGET_EXCEEDED", s.reserve, 11)
    s.fail(identity, {"code": "INFERENCE_FAILED"})
    assert s.snapshot(identity)["status"] == "failed"
    assert s.status()["accounted_bytes"] == 0


def test_unexpected_capture_overrun_fails_closed(store):
    s, _ = store
    identity = s.reserve(10)["trace_id"]
    trace = SimpleNamespace(metadata=metadata(20))
    error_code("TRACE_STORE_BUDGET_EXCEEDED", s.commit, identity, trace)
    s.fail(identity, {})
    assert s.status()["accounted_bytes"] == 0


@pytest.mark.parametrize("removal", ["delete", "expire"])
def test_active_lease_defers_physical_removal(store, removal):
    s, clock = store
    identity = insert(s, 80)
    with s.acquire(identity) as (_, trace):
        ref = weakref.ref(trace.tensor)
        if removal == "delete":
            assert s.delete(identity) is None
            error_code("TRACE_NOT_FOUND", s.snapshot, identity)
        else:
            clock.now = 10
            s.sweep()
            error_code("TRACE_EXPIRED", s.snapshot, identity)
        assert s.status()["retained_bytes"] == 80
        assert s.status()["deferred_removals"] == 1
        error_code("TRACE_STORE_BUDGET_EXCEEDED", s.reserve, 21)
        assert ref() is not None
    del trace
    gc.collect()
    assert ref() is None
    assert s.status()["retained_bytes"] == 0


def test_leases_release_after_serializer_exception(store):
    s, _ = store
    identity = insert(s)
    with pytest.raises(RuntimeError), s.acquire(identity):
        raise RuntimeError("serializer failure")
    assert s.status()["active_leases"] == 0
    s.delete(identity)
    assert s.delete(identity) is None
    assert s.status()["accounted_bytes"] == 0


def test_running_cancellation_discards_result(store):
    s, _ = store
    identity = s.reserve(40)["trace_id"]
    assert s.delete(identity)["cancellation_requested"]
    assert s.snapshot(identity)["status"] == "cancelled"
    s.commit(identity, SimpleNamespace(metadata=metadata(40)))
    assert s.snapshot(identity)["status"] == "cancelled"
    assert s.status()["accounted_bytes"] == 0
    s.delete(identity)
    error_code("TRACE_NOT_FOUND", s.snapshot, identity)


def test_metadata_and_tombstones_are_bounded():
    s = TraceStore(max_bytes=100, max_entries=2, tombstone_limit=2)
    identities = []
    for _ in range(8):
        identity = s.reserve(0)["trace_id"]
        s.fail(identity, {})
        identities.append(identity)
    assert s.status()["trace_count"] == 2
    assert s.status()["tombstone_count"] == 2
    error_code("TRACE_NOT_FOUND", s.snapshot, identities[0])


def test_concurrent_reads_insertion_and_deletion(store):
    s, _ = store
    identity = insert(s, 80)
    barrier = threading.Barrier(9)
    release = threading.Event()

    def read():
        with s.acquire(identity) as (_, trace):
            barrier.wait(timeout=5)
            assert release.wait(timeout=5)
            return trace.tensor.numel()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(read) for _ in range(8)]
        barrier.wait(timeout=5)
        assert s.status()["active_leases"] == 8
        extra = s.reserve(20)["trace_id"]
        s.fail(extra, {})
        s.delete(identity)
        assert s.status()["retained_bytes"] == 80
        release.set()
        assert [f.result(timeout=5) for f in futures] == [20] * 8
    assert s.status()["retained_bytes"] == 0
    assert s.status()["active_leases"] == 0


def test_store_close_keeps_lease_safe_and_disallows_reservation(store):
    s, _ = store
    identity = insert(s)
    with s.acquire(identity):
        s.close()
        assert s.status()["retained_bytes"] == 40
        error_code("EXECUTION_UNAVAILABLE", s.reserve, 1)
    assert s.status()["retained_bytes"] == 0
