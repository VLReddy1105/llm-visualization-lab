"""Validation metadata remains bounded and never emits scientific tensor arrays."""

import json

import pytest
import torch

from llm_lab.schemas.components import ComponentSpec, TensorDescriptor
from llm_lab.trace_validation import _descriptor_metadata, _plain


def test_validation_artifact_refuses_tensor_arrays():
    with pytest.raises(TypeError, match="must not serialize tensor arrays"):
        _plain({"accidental_tensor": torch.ones(2, 3)})


def test_validation_coverage_compacts_full_axes_and_keeps_selected_absolute_indices():
    spec = ComponentSpec(
        "layer[12].mlp.gate",
        ("batch", "token", "intermediate"),
        (1, 8, 3072),
        coordinates=((0,), (2,), tuple(range(3072))),
    )
    descriptor = TensorDescriptor(
        spec,
        (1, 1, 3072),
        "float32",
        "float32",
        "mps:0",
        3072,
        12288,
    )
    metadata = _descriptor_metadata(descriptor)
    coverage = metadata["spec"]["coordinates"]
    assert coverage == [
        {"axis": "batch", "selection": "all", "size": 1},
        {"axis": "token", "selection": "indices", "indices": [2]},
        {"axis": "intermediate", "selection": "all", "size": 3072},
    ]
    assert metadata["spec"]["logical_shape"] == [1, 8, 3072]
    assert metadata["spec"]["value_kind"] == "raw"
    assert len(json.dumps(metadata)) < 1200
    assert spec.coordinates[1] == (2,)
    assert len(spec.coordinates[2]) == 3072
