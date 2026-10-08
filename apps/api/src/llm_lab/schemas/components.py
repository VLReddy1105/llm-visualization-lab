"""Internal semantic tensor metadata; deliberately independent of HTTP schemas."""

from dataclasses import dataclass
from enum import StrEnum


class ValueKind(StrEnum):
    RAW = "raw"
    DERIVED = "derived"
    RECONSTRUCTED = "reconstructed"
    SUMMARY = "summary"


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Meaning and complete logical extent of a component.

    ``coordinates`` records retained absolute indices for each semantic axis.
    ``None`` means the entire logical extent. Module paths are not identities.
    """

    component_id: str
    axes: tuple[str, ...]
    logical_shape: tuple[int, ...]
    layer_index: int | None = None
    value_kind: ValueKind = ValueKind.RAW
    sources: tuple[str, ...] = ()
    method: str = "module observation"
    optional: bool = False
    expensive: bool = False
    coordinates: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id must not be empty")
        if len(self.axes) != len(self.logical_shape):
            raise ValueError("Every logical dimension must have one semantic axis")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Semantic axis names must be unique")
        if any(type(size) is not int or size < 1 for size in self.logical_shape):
            raise ValueError("Logical dimensions must be positive integers")
        if self.coordinates is not None:
            if len(self.coordinates) != len(self.axes):
                raise ValueError("Coverage must name retained coordinates for every axis")
            for coordinates, size in zip(self.coordinates, self.logical_shape, strict=True):
                if not coordinates or len(set(coordinates)) != len(coordinates):
                    raise ValueError("Retained coordinates must be nonempty and unique")
                if any(type(index) is not int or not 0 <= index < size for index in coordinates):
                    raise ValueError("Retained coordinates exceed the logical axis bounds")


@dataclass(frozen=True, slots=True)
class TensorStatistics:
    """Population reductions over the explicitly recorded finite-value scope."""

    finite_count: int
    nonfinite_count: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    l2_norm: float | None = None
    accumulation_dtype: str = "float32"
    value_kind: ValueKind = ValueKind.SUMMARY
    reduction_scope: str = "all retained coordinates; finite entries only; std correction=0"
    masked_count: int = 0
    total_count: int = 0


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    spec: ComponentSpec
    storage_shape: tuple[int, ...]
    dtype: str
    storage_dtype: str
    original_device: str
    numel: int
    byte_size: int
    storage_device: str = "cpu"
    tensor_id: str | None = None
    statistics: TensorStatistics | None = None
    availability: str = "not_captured"
    unavailable_reason: str | None = None

    @property
    def component_id(self) -> str:
        return self.spec.component_id

    @property
    def axes(self) -> tuple[str, ...]:
        return self.spec.axes

    @property
    def shape(self) -> tuple[int, ...]:
        """The full logical shape, not permission to retrieve omitted positions."""
        return self.spec.logical_shape

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return self.spec.logical_shape

    @property
    def coordinates(self) -> tuple[tuple[int, ...], ...] | None:
        return self.spec.coordinates
