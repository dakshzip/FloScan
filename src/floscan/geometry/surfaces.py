"""Versioned planar surfaces of a room: floor, ceiling and wall faces.

Every surface lies on an observed plane and has a metric (u, v) chart with
``u x v = normal``, the normal facing into the room (toward where it was
observed from): floors face up, ceilings down, walls inward. A floor or
ceiling keeps its fitted plane, tilt included; its boundary is the room
outline lifted vertically onto that plane. Wall faces run from the floor
plane up to the room's ceiling plane when one was observed over the room,
otherwise to the observed top of the wall's support, which is reported as
such and never extended to an assumed height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from floscan.contracts.base import Provenance
from floscan.contracts.geometry import Material, Surface
from floscan.geometry.planes import plane_basis
from floscan.geometry.polygons import to_polygon2d

UNKNOWN_MATERIAL = Material(label="unknown", source="not estimated")


@dataclass(frozen=True)
class FittedPlane:
    """``normal . p + offset = 0`` in W; not vertical (``normal[2] != 0``)."""

    normal: NDArray[np.float64]
    offset: float
    plane_ids: tuple[str, ...]

    def z_at(self, xy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Height of the plane above each (x, y)."""
        xy = np.asarray(xy, dtype=np.float64)
        return -(xy @ self.normal[:2] + self.offset) / self.normal[2]

    def lift(self, xy: NDArray[np.float64]) -> NDArray[np.float64]:
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        return np.column_stack([xy, self.z_at(xy)])

    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(min(1.0, abs(self.normal[2])))))


def _horizontal_surface(
    surface_id: str,
    room_id: str,
    kind: str,
    outline: Polygon,
    plane: FittedPlane,
    provenance: Provenance,
) -> Surface:
    centre = plane.lift(np.asarray(outline.centroid.coords[0]))[0]
    u, v = plane_basis(plane.normal)

    def chart(ring) -> list[tuple[float, float]]:
        points = plane.lift(np.asarray(ring.coords)[:-1]) - centre
        return list(zip((points @ u).tolist(), (points @ v).tolist(), strict=True))

    boundary = Polygon(chart(outline.exterior), [chart(h) for h in outline.interiors])
    return Surface(
        id=surface_id,
        provenance=provenance,
        room_id=room_id,
        kind=kind,
        geometry_version=1,
        frame_id="W",
        origin=centre.tolist(),
        basis_u=u.tolist(),
        basis_v=v.tolist(),
        normal=plane.normal.tolist(),
        boundary_uv=to_polygon2d(boundary),
        material=UNKNOWN_MATERIAL,
        observation_ids=list(plane.plane_ids),
        lineage=list(plane.plane_ids),
    )


def floor_surface(
    surface_id: str,
    room_id: str,
    outline: Polygon,
    plane: FittedPlane,
    provenance: Provenance,
) -> Surface:
    """Floor on its fitted plane; the normal faces up."""
    if plane.normal[2] <= 0:
        raise ValueError("a floor plane's normal must face up")
    return _horizontal_surface(surface_id, room_id, "floor", outline, plane, provenance)


def ceiling_surface(
    surface_id: str,
    room_id: str,
    outline: Polygon,
    plane: FittedPlane,
    provenance: Provenance,
) -> Surface:
    """Ceiling on its fitted plane; the normal faces down."""
    if plane.normal[2] >= 0:
        raise ValueError("a ceiling plane's normal must face down")
    return _horizontal_surface(
        surface_id, room_id, "ceiling", outline, plane, provenance
    )


def wall_surface(
    surface_id: str,
    room_id: str,
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    floor: FittedPlane,
    top_z: tuple[float, float],
    plane_ids: list[str],
    provenance: Provenance,
) -> Surface:
    """Wall face along a CCW outline edge start -> end (room on its left).

    The chart runs horizontally from ``end`` back to ``start`` (u) and
    upward (v), so ``u x v`` is the inward normal. The face spans from the
    floor plane to ``top_z`` (at start, at end) at both ends.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    u = np.append((start - end) / length, 0.0)
    v = np.array([0.0, 0.0, 1.0])
    floor_start, floor_end = (float(z) for z in floor.z_at(np.stack([start, end])))
    base = floor_end
    boundary = Polygon(
        [
            (0.0, 0.0),
            (length, floor_start - base),
            (length, top_z[0] - base),
            (0.0, top_z[1] - base),
        ]
    )
    return Surface(
        id=surface_id,
        provenance=provenance,
        room_id=room_id,
        kind="wall",
        geometry_version=1,
        frame_id="W",
        origin=[float(end[0]), float(end[1]), base],
        basis_u=u.tolist(),
        basis_v=v.tolist(),
        normal=np.cross(u, v).tolist(),
        boundary_uv=to_polygon2d(boundary),
        material=UNKNOWN_MATERIAL,
        observation_ids=plane_ids,
        lineage=plane_ids,
    )
