"""Synthetic CPU tensors exercise transport only; real science is tested separately."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from llm_lab.config import Settings
from llm_lab.errors import ServiceError
from llm_lab.schemas.components import ComponentSpec, TensorDescriptor, TensorStatistics, ValueKind
from llm_lab.schemas.trace_api import ComponentDescriptor, SliceRequest, TensorSlice
from llm_lab.tracing.serialization import TraceSerializer, page


@pytest.fixture
def transport():
    values = torch.tensor([[[1.0, float("nan")], [float("inf"), -float("inf")]]])
    spec = ComponentSpec(
        "embedding.output",
        ("batch", "token", "hidden"),
        (1, 4, 2),
        coordinates=((0,), (1, 3), (0, 1)),
        value_kind=ValueKind.RECONSTRUCTED,
        sources=("source.observation",),
        method="transport test",
    )
    d = TensorDescriptor(
        spec,
        (1, 2, 2),
        "float32",
        "float32",
        "cpu",
        4,
        16,
        tensor_id="opaque-id",
        availability="captured",
        statistics=TensorStatistics(1, 3, mean=float("nan")),
    )
    trace = SimpleNamespace(
        metadata=SimpleNamespace(trace_id="transport-test"),
        descriptors={spec.component_id: d},
        _tensors={spec.component_id: values},
        tensor=lambda name, copy: values,
    )
    return TraceSerializer(Settings(auto_load=False)), trace, d


def test_descriptor_preserves_axes_provenance_coverage_and_summary(transport):
    serializer, _, d = transport
    body = serializer.descriptor(d)
    ComponentDescriptor.model_validate(body)
    assert body["provenance"] == "reconstructed"
    assert body["coverage"]["token"] == {"count": 2, "start": 1, "stop": 4, "step": 2}
    assert body["statistics"]["value_kind"] == "summary"
    assert body["statistics"]["finite"] is False
    assert body["statistics"]["mean"] is None
    assert "mean" in body["statistics"]["undefined_fields"]


def test_nonfinite_json_is_null_with_flat_index_and_kind(transport):
    serializer, trace, _ = transport
    body = serializer.slice(trace, "embedding.output", SliceRequest())
    TensorSlice.model_validate(body)
    assert body["values"] == [1, None, None, None]
    assert body["shape"] == [1, 2, 2]
    assert [v["kind"] for v in body["nonfinite"]] == [
        "nan",
        "positive_infinity",
        "negative_infinity",
    ]
    encoded = serializer.response(body).body
    assert b"NaN" not in encoded and b"Infinity" not in encoded and b"tensor(" not in encoded


def test_slice_uses_absolute_not_storage_token_indices(transport):
    serializer, trace, _ = transport
    body = serializer.slice(
        trace,
        "embedding.output",
        SliceRequest(selectors={"token": {"index": 3}, "hidden": {"start": 0, "stop": 1}}),
    )
    assert body["shape"] == [1, 1, 1]
    assert body["values"] == [None]
    assert body["coordinates"]["token"]["start"] == 3


@pytest.mark.parametrize(
    "selectors,code",
    [
        ({"token": {"index": 4}}, "INVALID_TOKEN"),
        ({"token": {"index": 2}}, "COMPONENT_NOT_CAPTURED"),
        ({"hidden": {"start": 1, "stop": 1}}, "INVALID_SLICE"),
        ({"hidden": {"start": 0, "stop": 3}}, "INVALID_SLICE"),
        ({"python_attribute": {"index": 0}}, "INVALID_SLICE"),
    ],
)
def test_semantic_slice_errors(transport, selectors, code):
    serializer, trace, _ = transport
    with pytest.raises(ServiceError) as failure:
        serializer.slice(trace, "embedding.output", SliceRequest(selectors=selectors))
    assert failure.value.code == code


@pytest.mark.parametrize(
    "body",
    [
        {"selectors": "[0,:,3,...]"},
        {"selectors": {"token": {"index": True}}},
        {"selectors": {"token": {"index": "3"}}},
        {"selectors": {"token": {"index": -1}}},
        {"selectors": {"token": {"start": 0, "stop": 2, "step": 0}}},
        {"selectors": {"token": {"index": 1, "stop": 2}}},
        {"format": "python"},
    ],
)
def test_no_executable_or_coerced_selectors(body):
    with pytest.raises(ValidationError):
        SliceRequest.model_validate(body)


def test_element_limit_checked_before_tensor_access(transport):
    _, trace, _ = transport
    serializer = TraceSerializer(Settings(auto_load=False, trace_api_max_elements=2))
    trace.tensor = lambda *args, **kwargs: pytest.fail("tensor accessed before limit check")
    with pytest.raises(ServiceError) as failure:
        serializer.slice(trace, "embedding.output", SliceRequest())
    assert failure.value.code == "RESPONSE_TOO_LARGE"


def test_exact_encoded_byte_limit(transport):
    _, trace, _ = transport
    serializer = TraceSerializer(Settings(auto_load=False, trace_api_max_response_bytes=1024))
    with pytest.raises(ServiceError) as failure:
        serializer.response({"unicode": "😊" * 300})
    assert failure.value.code == "RESPONSE_TOO_LARGE"
    assert failure.value.details["bytes"] > 1024
    assert serializer.response({"small": 1}).status_code == 200


def test_summary_only_descriptor_succeeds_but_slice_conflicts(transport):
    serializer, trace, d = transport
    trace.descriptors[d.component_id] = replace(d, tensor_id=None, availability="summary_only")
    assert not serializer.descriptor(serializer.component(trace, d.component_id))["raw_available"]
    with pytest.raises(ServiceError) as failure:
        serializer.slice(trace, d.component_id, SliceRequest())
    assert failure.value.code == "COMPONENT_NOT_CAPTURED"


def test_unknown_semantic_component_is_not_an_attribute_path(transport):
    serializer, trace, _ = transport
    with pytest.raises(ServiceError) as failure:
        serializer.component(trace, "__dict__._tensors")
    assert failure.value.code == "COMPONENT_NOT_FOUND"


def test_page_cursor_order_and_validation():
    items = [{"index": i} for i in range(5)]
    first = page(items, 2, None)
    second = page(items, 2, first["next_cursor"])
    last = page(items, 2, second["next_cursor"])
    assert [item["index"] for part in (first, second, last) for item in part["items"]] == list(
        range(5)
    )
    assert last["next_cursor"] is None
    with pytest.raises(ServiceError):
        page(items, 2, "invalid!")


def test_binary_float32_metadata_and_nonfinite_roundtrip(transport):
    import json

    serializer, trace, _ = transport
    response = serializer.slice_response(trace, "embedding.output", SliceRequest(format="binary"))
    assert response.media_type == "application/octet-stream"
    metadata = json.loads(response.headers["X-Tensor-Metadata"])
    assert metadata["shape"] == [1, 2, 2] and metadata["provenance"] == "reconstructed"
    assert metadata["wire_dtype"] == "little_endian_float32"
    values = torch.frombuffer(bytearray(response.body), dtype=torch.float32)
    assert values[0] == 1 and torch.isnan(values[1])
    assert torch.isposinf(values[2]) and torch.isneginf(values[3])


def test_binary_cannot_bypass_element_limit_or_round_integer_ids(transport):
    _, trace, _ = transport
    serializer = TraceSerializer(Settings(auto_load=False, trace_api_max_elements=2))
    with pytest.raises(ServiceError) as failure:
        serializer.slice_response(trace, "embedding.output", SliceRequest(format="binary"))
    assert failure.value.code == "RESPONSE_TOO_LARGE"
    serializer = TraceSerializer(Settings(auto_load=False))
    trace.tensor = lambda *args, **kwargs: torch.ones((1, 2, 2), dtype=torch.int64)
    with pytest.raises(ServiceError) as failure:
        serializer.slice_response(trace, "embedding.output", SliceRequest(format="binary"))
    assert failure.value.code == "INVALID_SLICE"


def test_opt_in_statistics_preserve_nonfinite_semantics_and_slice_scope(transport):
    serializer, trace, _ = transport
    body = serializer.slice(trace, "embedding.output", SliceRequest(include_statistics=True))
    stats = body["statistics"]
    assert stats["finite_count"] == 1 and stats["nonfinite_count"] == 3
    assert stats["l2_norm"] == 1 and stats["std"] == 0 and not stats["finite"]
    assert "requested slice" in stats["reduction_scope"]
    empty = serializer.slice(
        trace,
        "embedding.output",
        SliceRequest(selectors={"token": {"index": 3}}, include_statistics=True),
    )
    assert empty["statistics"]["finite_count"] == 0
    assert empty["statistics"]["l2_norm"] is None
    assert "l2_norm" in empty["statistics"]["undefined_fields"]
    assert serializer.slice(trace, "embedding.output", SliceRequest())["statistics"] is None


def test_statistics_request_cannot_bypass_element_budget(transport):
    _, trace, _ = transport
    serializer = TraceSerializer(Settings(auto_load=False, trace_api_max_elements=2))
    trace.tensor = lambda *args, **kwargs: pytest.fail("tensor accessed before limit check")
    with pytest.raises(ServiceError) as failure:
        serializer.slice(trace, "embedding.output", SliceRequest(include_statistics=True))
    assert failure.value.code == "RESPONSE_TOO_LARGE"
