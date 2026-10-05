"""Reconstruction bundle: point observations, planes and diagnostics.

A bundle is what a reconstruction stage hands to room building. Every point
keeps the frame and pixel it came from, so later stages can return to the
raw observation. Points are stored per submap as ``.npy`` arrays and listed
by content hash in ``bundle.json``, together with typed ``PointCloud`` and
``Plane`` records. Nothing here is a room, wall or measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from floscan.contracts.base import AssetRef, Provenance
from floscan.contracts.geometry import Plane, PointCloud
from floscan.io.assets import asset_ref

BUNDLE_KIND = "floscan-reconstruction-bundle"
BUNDLE_VERSION = "0.1.0"


@dataclass(frozen=True)
class SubmapPoints:
    """Points of one submap in the session world, with their provenance.

    ``observations[i] = (source_index, u, v)``: the frame and depth pixel of
    point i. ``viewpoints[i]`` is that frame's camera centre.
    """

    submap_id: str
    xyz: NDArray[np.float32]
    observations: NDArray[np.int32]
    normals: NDArray[np.float32]
    viewpoints: NDArray[np.float32]

    def __post_init__(self) -> None:
        count = len(self.xyz)
        for name in ("xyz", "observations", "normals", "viewpoints"):
            array = getattr(self, name)
            if array.ndim != 2 or array.shape != (count, 3):
                raise ValueError(f"{name} must have shape ({count}, 3)")


@dataclass
class PlaneResult:
    record: Plane
    support: NDArray[np.int64]
    diagnostics: dict[str, Any]


@dataclass
class SubmapResult:
    submap_id: str
    record: PointCloud
    keyframes: list[int]
    diagnostics: dict[str, Any]
    planes: list[PlaneResult] = field(default_factory=list)


def _save_array(directory: Path, relative: str, array: NDArray) -> AssetRef:
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    dtype = np.dtype(array.dtype).newbyteorder("<").str
    return asset_ref(directory, relative, shape=list(array.shape), dtype=dtype)


def save_submap(
    directory: Path, points: SubmapPoints, provenance: Provenance
) -> PointCloud:
    """Write a submap's arrays (new files only) and return its record."""
    prefix = f"submaps/{points.submap_id}"
    xyz = _save_array(directory, f"{prefix}/xyz.npy", points.xyz.astype("<f4"))
    observations = _save_array(
        directory, f"{prefix}/observations.npy", points.observations.astype("<i4")
    )
    normals = _save_array(
        directory, f"{prefix}/normals.npy", points.normals.astype("<f4")
    )
    _save_array(directory, f"{prefix}/viewpoints.npy", points.viewpoints.astype("<f4"))
    return PointCloud(
        id=f"cloud:{points.submap_id}",
        provenance=provenance,
        frame_id="W",
        xyz=xyz,
        unit="m",
        normals=normals,
        observation_index=observations,
        submap_id=points.submap_id,
        scale_status="metric",
    )


def save_support(
    directory: Path, submap_id: str, plane_id: str, support: NDArray[np.int64]
) -> AssetRef:
    return _save_array(
        directory, f"submaps/{submap_id}/{plane_id}_support.npy", support.astype("<i8")
    )


def write_manifest(directory: Path, manifest: dict[str, Any]) -> Path:
    """Write ``bundle.json`` (strict JSON, never overwriting)."""
    path = directory / "bundle.json"
    text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path


def submap_entry(result: SubmapResult) -> dict[str, Any]:
    return {
        "submap_id": result.submap_id,
        "keyframes": result.keyframes,
        "point_cloud": result.record.model_dump(mode="json"),
        "diagnostics": result.diagnostics,
        "planes": [
            {
                "plane": plane.record.model_dump(mode="json"),
                "diagnostics": plane.diagnostics,
            }
            for plane in result.planes
        ],
    }
