"""Failed queue tasks release resident references without erasing diagnostic errors."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from llm_lab.config import Settings
from llm_lab.models.manager import ModelService


def _fail_with_resident_reference(service):
    resident = service._adapter.model

    def fail_inside_operation():
        # This local reference represents the resident weights, not numerical
        # model inference. It must disappear from retained error tracebacks.
        weights = resident.weight
        assert weights.numel() > 0
        raise ValueError("original operation failure")

    try:
        fail_inside_operation()
    except ValueError as cause:
        raise RuntimeError("queued operation failed") from cause


def test_failed_future_keeps_diagnostics_and_releases_resident_after_close():
    service = ModelService(Settings(auto_load=False))
    resident = torch.nn.Linear(3, 3)
    reference = weakref.ref(resident)
    service._model = resident
    service._adapter = SimpleNamespace(model=resident)
    del resident
    future = service._submit(_fail_with_resident_reference, service)
    try:
        with pytest.raises(RuntimeError, match="queued operation failed"):
            future.result(timeout=10)
    finally:
        # Closing joins the executor, so callback cleanup has finished before
        # checking ownership. Keep the failed Future and its exception alive.
        service.close()
    error = future.exception(timeout=10)
    assert str(error) == "queued operation failed"
    assert isinstance(error.__cause__, ValueError)
    assert str(error.__cause__) == "original operation failure"
    gc.collect()
    assert reference() is None
    assert future.done()
