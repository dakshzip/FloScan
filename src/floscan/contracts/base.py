"""Shared primitives for FloScan's internal records (project-owned schema).

Every record model is strict: unknown fields, numeric strings, booleans as
numbers, NaN and Infinity are all rejected. Missing values are ``None`` with a
reason, never zero. Semantic invariants that JSON Schema cannot express live in
model validators. Conventions follow docs/implementation-strategy/02-data-contracts.md.

Records are deeply immutable once validated: attribute assignment is refused
(``frozen``) and every list or dict inside a record becomes a ``FrozenList`` or
``FrozenDict``, which refuse in-place changes. They remain ``list``/``dict``
subclasses, so JSON shapes and schemas are unchanged. To change a record, build
a new one through validation.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, NoReturn

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from floscan.geometry.frames import FrameError, validate_rotation

INTERNAL_SCHEMA_VERSION = "internal-v0"
CAPTURE_SCHEMA_VERSION = "capture-v0"


class FrozenList(list):
    """A list that refuses in-place modification after validation."""

    __slots__ = ()

    def _refuse(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        raise TypeError("validated records are immutable; build a new record instead")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse
    append = extend = insert = pop = remove = clear = sort = reverse = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (list(self),))


class FrozenDict(dict):
    """A dict that refuses in-place modification after validation."""

    __slots__ = ()

    def _refuse(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        raise TypeError("validated records are immutable; build a new record instead")

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (dict(self),))


def _freeze(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value  # Frozen by its own validation.
    if isinstance(value, list) and not isinstance(value, FrozenList):
        return FrozenList(_freeze(item) for item in value)
    if isinstance(value, dict) and not isinstance(value, FrozenDict):
        return FrozenDict((key, _freeze(item)) for key, item in value.items())
    return value


class Contract(BaseModel):
    """Base for every record and sub-record: strict, closed and deeply immutable."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
        # Never turn a non-finite number into null on output; emit NaN so any
        # reader (ours rejects it) fails loudly instead of losing the value.
        ser_json_inf_nan="constants",
    )

    @model_validator(mode="after")
    def _freeze_containers(self) -> Contract:
        for name in type(self).model_fields:
            value = getattr(self, name)
            frozen = _freeze(value)
            if frozen is not value:
                object.__setattr__(self, name, frozen)
        return self

    def to_json(self, indent: int | None = 2) -> str:
        """Re-validate the whole record tree, then serialise it.

        Re-validation guarantees that nothing invalid reaches output even if
        a record was altered by a path that bypasses validation.
        """
        type(self).model_validate(self.model_dump(mode="python"))
        return self.model_dump_json(indent=indent)


Id = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegative = Annotated[float, Field(ge=0.0)]
Positive = Annotated[float, Field(gt=0.0)]

Vec2 = Annotated[list[float], Field(min_length=2, max_length=2)]
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Vec4 = Annotated[list[float], Field(min_length=4, max_length=4)]

Tier = Literal["photo", "video", "lidar"]
LengthUnit = Literal["m", "reconstruction_unit"]
MeasurementUnit = Literal["m", "m2", "rad", "s", "count", "ratio"]
ProcessingMode = Literal["live", "replay"]
RecordStatus = Literal[
    "ok",
    "partial",
    "unavailable",
    "invalid_input",
    "insufficient_evidence",
    "ambiguous",
    "unsupported",
    "failed",
]


def _square(size: int) -> object:
    def check(rows: list[list[float]]) -> list[list[float]]:
        if len(rows) != size or any(len(row) != size for row in rows):
            raise ValueError(f"expected a {size}x{size} matrix")
        return rows

    return AfterValidator(check)


Matrix3 = Annotated[list[list[float]], _square(3)]
Matrix4 = Annotated[list[list[float]], _square(4)]
Matrix6 = Annotated[list[list[float]], _square(6)]


def check_rotation(rows: list[list[float]], where: str) -> None:
    """Raise ``ValueError`` unless ``rows`` is a proper rotation matrix."""
    try:
        validate_rotation(np.asarray(rows, dtype=np.float64))
    except FrameError as error:
        raise ValueError(f"{where}: {error}") from error


def check_unit_vector(vector: list[float], where: str, tolerance: float = 1e-6) -> None:
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > tolerance:
        raise ValueError(f"{where} must be a unit vector (norm {norm:.6f})")


# Relative tolerance for "positive semidefinite": eigenvalues may dip below
# zero by at most this fraction of the largest eigenvalue magnitude (round-off).
PSD_RELATIVE_TOLERANCE = 1e-9


def check_symmetric_covariance(
    matrix: list[list[float]], where: str, size: int | None = None
) -> None:
    """Covariance/information must be symmetric, PSD and not all zero.

    Singular PSD matrices are allowed (partial observability). An indefinite
    matrix describes negative variance in some direction and is rejected, not
    projected. An all-zero matrix claims certainty; mark it unknown instead.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or (size and m.shape[0] != size):
        expected = f"{size}x{size}" if size else "square"
        raise ValueError(f"{where} must be a {expected} matrix")
    scale = max(1.0, float(np.abs(m).max()))
    if np.abs(m - m.T).max() > 1e-9 * scale:
        raise ValueError(f"{where} must be symmetric")
    if not m.any():
        raise ValueError(
            f"{where} is all zeros, which claims certainty; mark it unknown instead"
        )
    eigenvalues = np.linalg.eigvalsh((m + m.T) / 2.0)
    floor = -PSD_RELATIVE_TOLERANCE * float(np.abs(eigenvalues).max())
    if eigenvalues.min() < floor:
        raise ValueError(
            f"{where} is not positive semidefinite (smallest eigenvalue "
            f"{eigenvalues.min():.3g})"
        )


class Provenance(Contract):
    """Where a record came from. Inference provenance never holds ground truth."""

    stage: Id
    stage_version: Text
    mode: ProcessingMode
    config_hash: Sha256 | None = None
    model_hash: Sha256 | None = None
    source_asset_hashes: list[Sha256] = Field(default_factory=list)
    upstream_artifact_hashes: list[Sha256] = Field(default_factory=list)
    observation_ids: list[Id] = Field(default_factory=list)


def _relative_uri(uri: str) -> str:
    path = PurePosixPath(uri)
    if not uri or path.is_absolute() or ".." in path.parts or "\\" in uri:
        raise ValueError(
            f"asset uri {uri!r} must be relative to its root, without '..'"
        )
    return uri


class AssetRef(Contract):
    """A file inside a capture or run root, identified by content hash."""

    uri: Annotated[str, AfterValidator(_relative_uri)]
    sha256: Sha256
    byte_count: Annotated[int, Field(ge=0)]
    mime_type: Text
    encoding: Text | None = None
    shape: list[Annotated[int, Field(ge=0)]] | None = None
    dtype: Annotated[str, Field(pattern=r"^[<|][a-z]\d+$")] | None = None

    @model_validator(mode="after")
    def _array_metadata(self) -> AssetRef:
        if (self.shape is None) != (self.dtype is None):
            raise ValueError(
                "array assets need both shape and an explicit-endian dtype"
            )
        return self


class Record(Contract):
    """Fields shared by every top-level record."""

    id: Id
    schema_version: Literal["internal-v0"] = INTERNAL_SCHEMA_VERSION
    provenance: Provenance
    status: RecordStatus = "ok"
    status_reason: Text | None = None

    @model_validator(mode="after")
    def _status_needs_reason(self) -> Record:
        if self.status != "ok" and self.status_reason is None:
            raise ValueError(f"status {self.status!r} needs a status_reason")
        return self
