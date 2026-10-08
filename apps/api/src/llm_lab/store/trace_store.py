"""Atomic byte reservations and absolute TTL; leases defer physical reclamation."""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from llm_lab.errors import ServiceError
from llm_lab.schemas.trace import TraceResult


def utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


@dataclass
class Entry:
    trace_id: str
    created_at: str
    last_access_at: str
    status: str = "running"
    completed_at: str | None = None
    expires_at: str | None = None
    deadline: float | None = None
    reserved_bytes: int = 0
    retained_bytes: int = 0
    trace: TraceResult | None = None
    error: dict | None = None
    leases: int = 0
    removed: str | None = None
    working: bool = True

    def snapshot(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "status": self.status,
            "created_at": self.created_at,
            "last_access_at": self.last_access_at,
            "completed_at": self.completed_at,
            "expires_at": self.expires_at,
            "retained_bytes": self.retained_bytes,
            "reserved_bytes": self.reserved_bytes,
            "error": self.error,
            "metadata": {
                "model_id": self.trace.metadata.model_id,
                "revision": self.trace.metadata.revision,
                "sequence_length": self.trace.metadata.sequence_length,
                "capture_profile": self.trace.metadata.capture_plan.profile,
                "detail_layers": list(self.trace.metadata.capture_plan.layers),
            }
            if self.trace is not None
            else None,
        }


class TraceStore:
    def __init__(
        self,
        max_bytes: int = 512 * 1024 * 1024,
        ttl_seconds: float = 1800,
        max_entries: int = 1024,
        tombstone_limit: int = 2048,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        self.max_bytes, self.ttl_seconds = max_bytes, ttl_seconds
        self.max_entries, self.tombstone_limit = max_entries, tombstone_limit
        self._clock, self._wall = clock, wall_clock
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, Entry] = OrderedDict()
        self._tombstones: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._retained = self._reserved = self._evictions = self._expirations = 0
        self._closed = False

    def _reclaim(self, entry: Entry) -> None:
        if entry.removed and not entry.leases and not entry.working:
            self._retained -= entry.retained_bytes
            self._reserved -= entry.reserved_bytes
            entry.trace = None
            entry.retained_bytes = entry.reserved_bytes = 0
            self._entries.pop(entry.trace_id, None)

    def _remove(self, entry: Entry, reason: str) -> None:
        if entry.removed:
            return
        entry.removed = reason
        self._tombstones[entry.trace_id] = (reason, self._clock() + 86400)
        while len(self._tombstones) > self.tombstone_limit:
            self._tombstones.popitem(last=False)
        self._reclaim(entry)

    def _sweep(self) -> None:
        now = self._clock()
        for entry in list(self._entries.values()):
            if not entry.removed and entry.deadline is not None and now >= entry.deadline:
                self._expirations += 1
                self._remove(entry, "expired")
        for identity, (_, deadline) in list(self._tombstones.items()):
            if now >= deadline:
                del self._tombstones[identity]

    def sweep(self) -> None:
        with self._lock:
            self._sweep()

    def _make_room(self, requested: int, *, inserting: bool = False) -> None:
        self._sweep()
        # Decide eligibility before evicting anything: a rejected reservation
        # must not unnecessarily discard existing scientific results.
        eligible = [
            e for e in self._entries.values() if not e.removed and not e.leases and not e.working
        ]
        need_bytes = max(0, self._retained + self._reserved + requested - self.max_bytes)
        need_entries = max(0, len(self._entries) + int(inserting) - self.max_entries)
        victims = []
        reclaimed = 0
        for entry in eligible:
            if reclaimed >= need_bytes and len(victims) >= need_entries:
                break
            victims.append(entry)
            reclaimed += entry.retained_bytes
        if reclaimed < need_bytes or len(victims) < need_entries:
            raise ServiceError(
                "TRACE_STORE_BUDGET_EXCEEDED",
                "No unleased trace capacity is available.",
                503,
                details={
                    "max_bytes": self.max_bytes,
                    "requested_bytes": requested,
                    "hint": "Delete a trace, release active reads, or request less capture.",
                },
            )
        for entry in victims:
            self._evictions += 1
            self._remove(entry, "evicted")

    def reserve(self, estimated_bytes: int) -> dict:
        if estimated_bytes < 0:
            raise ValueError("Reservation cannot be negative")
        with self._lock:
            if self._closed:
                raise ServiceError("EXECUTION_UNAVAILABLE", "Trace store is closed.", 503)
            self._make_room(estimated_bytes, inserting=True)
            now = utc_timestamp(self._wall())
            entry = Entry(str(uuid4()), now, now, reserved_bytes=estimated_bytes)
            self._entries[entry.trace_id] = entry
            self._reserved += estimated_bytes
            return entry.snapshot()

    def _complete(self, entry: Entry) -> None:
        entry.working = False
        entry.completed_at = utc_timestamp(self._wall())
        entry.expires_at = utc_timestamp(self._wall() + self.ttl_seconds)
        entry.deadline = self._clock() + self.ttl_seconds
        self._reserved -= entry.reserved_bytes
        entry.reserved_bytes = 0

    def commit(self, trace_id: str, trace: TraceResult) -> None:
        with self._lock:
            entry = self._entries[trace_id]
            actual = trace.metadata.actual_retained_bytes
            if entry.removed or entry.status == "cancelled":
                self._complete(entry)
                self._reclaim(entry)
                return
            # The estimator is conservative. An unexpected overrun fails closed.
            if actual > entry.reserved_bytes:
                raise ServiceError(
                    "TRACE_STORE_BUDGET_EXCEEDED", "Capture exceeded its reserved bytes.", 503
                )
            self._complete(entry)
            entry.trace, entry.retained_bytes, entry.status = trace, actual, "ready"
            self._retained += actual
            self._entries.move_to_end(trace_id)

    def fail(self, trace_id: str, error: dict) -> None:
        with self._lock:
            entry = self._entries[trace_id]
            self._complete(entry)
            if entry.status != "cancelled":
                entry.status, entry.error = "failed", error
            self._reclaim(entry)

    def _lookup(self, trace_id: str) -> Entry:
        self._sweep()
        entry = self._entries.get(trace_id)
        if entry is None or entry.removed:
            reason = entry.removed if entry else self._tombstones.get(trace_id, (None, 0))[0]
            if reason in {"expired", "evicted"}:
                raise ServiceError(
                    f"TRACE_{reason.upper()}",
                    f"This trace was {reason}; create a new trace.",
                    410,
                    details={"trace_id": trace_id},
                )
            raise ServiceError(
                "TRACE_NOT_FOUND", "Trace does not exist.", 404, details={"trace_id": trace_id}
            )
        entry.last_access_at = utc_timestamp(self._wall())
        self._entries.move_to_end(trace_id)
        return entry

    def snapshot(self, trace_id: str) -> dict:
        with self._lock:
            return self._lookup(trace_id).snapshot()

    @contextmanager
    def acquire(self, trace_id: str) -> Iterator[tuple[dict, TraceResult | None]]:
        with self._lock:
            entry = self._lookup(trace_id)
            entry.leases += 1
            snapshot, trace = entry.snapshot(), entry.trace
        try:
            yield snapshot, trace
        finally:
            with self._lock:
                entry.leases -= 1
                self._reclaim(entry)

    def delete(self, trace_id: str) -> dict | None:
        with self._lock:
            # Repeat DELETE is idempotent while its bounded tombstone exists.
            self._sweep()
            if self._tombstones.get(trace_id, (None, 0))[0] == "deleted":
                return None
            entry = self._lookup(trace_id)
            if entry.working:
                entry.status = "cancelled"
                return {"trace_id": trace_id, "status": "cancelled", "cancellation_requested": True}
            self._remove(entry, "deleted")
            return None

    def listing(self) -> list[dict]:
        with self._lock:
            self._sweep()
            return [e.snapshot() for e in self._entries.values() if not e.removed]

    def status(self) -> dict:
        with self._lock:
            self._sweep()
            return {
                "trace_count": sum(not e.removed for e in self._entries.values()),
                "retained_bytes": self._retained,
                "reserved_bytes": self._reserved,
                "accounted_bytes": self._retained + self._reserved,
                "max_bytes": self.max_bytes,
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
                "eviction_count": self._evictions,
                "expiration_count": self._expirations,
                "active_leases": sum(e.leases for e in self._entries.values()),
                "deferred_removals": sum(bool(e.removed) for e in self._entries.values()),
                "tombstone_count": len(self._tombstones),
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for entry in list(self._entries.values()):
                self._remove(entry, "deleted")
