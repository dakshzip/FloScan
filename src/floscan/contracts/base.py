"""Shared primitives for FloScan's internal records (project-owned schema).

Every record model is strict: unknown fields, numeric strings, booleans as
numbers, NaN and Infinity are all rejected. Missing values are ``None`` with a
reason, never zero. Semantic invariants that JSON Schema cannot express live in
model validators. Conventions follow docs/implementation-strategy/02-data-contracts.md.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from floscan.geometry.frames import FrameError, validate_rotation

INTERNAL_SCHEMA_VERSION = "internal-v0"
CAPTURE_SCHEMA_VERSION = "capture-v0"


class Contract(BaseModel):
    """Base for every record and sub-record: strict, closed and immutable."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
    )


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


def check_symmetric_covariance(matrix: list[list[float]], where: str) -> None:
    """Covariances are symmetric with non-negative variances and not all zero.

    An all-zero covariance claims perfect certainty; unknown covariance must be
    represented as unknown instead.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if np.abs(m - m.T).max() > 1e-9 * max(1.0, np.abs(m).max()):
        raise ValueError(f"{where} must be symmetric")
    if (np.diag(m) < 0).any():
        raise ValueError(f"{where} has negative variances")
    if not m.any():
        raise ValueError(
            f"{where} is all zeros, which claims certainty; mark it unknown instead"
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
