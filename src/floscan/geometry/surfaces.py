"""Versioned planar surfaces of a room: floor, ceiling and wall faces.

Every surface has a metric (u, v) chart with ``u x v = normal`` and the
normal facing into the room (toward the space it was observed from). Floors
face up, ceilings down, walls inward. Boundaries are polygons in (u, v)
metres. A wall surface's vertical extent is the floor-to-ceiling height only
when a ceiling was observed; otherwise it is the observed support extent and
is reported as such, never extended to an assumed height.
"""

from __future__ import annotations

import numpy as np
from shapely.affinity import scale
from shapely.geometry import Polygon, box

from floscan.contracts.base import Provenance
from floscan.contracts.geometry import Material, Surface
from floscan.geometry.polygons import to_polygon2d

UNKNOWN_MATERIAL = Material(label="unknown", source="not estimated")


def floor_surface(
    surface_id: str,
    room_id: str,
    outline: Polygon,
    floor_z: float,
    plane_ids: list[str],
    provenance: Provenance,
) -> Surface:
    """Floor: chart (x, y) at the floor height, normal +z."""
    return Surface(
        id=surface_id,
        provenance=provenance,
        room_id=room_id,
        kind="floor",
        geometry_version=1,
        frame_id="W",
        origin=[0.0, 0.0, float(floor_z)],
        basis_u=[1.0, 0.0, 0.0],
        basis_v=[0.0, 1.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        boundary_uv=to_polygon2d(outline),
        material=UNKNOWN_MATERIAL,
        observation_ids=plane_ids,
        lineage=plane_ids,
    )


def ceiling_surface(
    surface_id: str,
    room_id: str,
    outline: Polygon,
    ceiling_z: float,
    plane_ids: list[str],
    provenance: Provenance,
) -> Surface:
    """Ceiling: chart (x, -y) at the ceiling height, normal -z (facing down)."""
    mirrored = scale(outline, xfact=1.0, yfact=-1.0, origin=(0, 0))
    return Surface(
        id=surface_id,
        provenance=provenance,
        room_id=room_id,
        kind="ceiling",
        geometry_version=1,
        frame_id="W",
        origin=[0.0, 0.0, float(ceiling_z)],
        basis_u=[1.0, 0.0, 0.0],
        basis_v=[0.0, -1.0, 0.0],
        normal=[0.0, 0.0, -1.0],
        boundary_uv=to_polygon2d(mirrored),
        material=UNKNOWN_MATERIAL,
        observation_ids=plane_ids,
        lineage=plane_ids,
    )


def wall_surface(
    surface_id: str,
    room_id: str,
    start: np.ndarray,
    end: np.ndarray,
    floor_z: float,
    height: float,
    plane_ids: list[str],
    provenance: Provenance,
) -> Surface:
    """Wall face along a CCW outline edge start -> end (room on its left).

    The chart runs from ``end`` back to ``start`` (u) and upward (v), so
    ``u x v`` is the inward normal. ``height`` is the face's vertical extent.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    u = np.append((start - end) / length, 0.0)
    v = np.array([0.0, 0.0, 1.0])
    return Surface(
        id=surface_id,
        provenance=provenance,
        room_id=room_id,
        kind="wall",
        geometry_version=1,
        frame_id="W",
        origin=[float(end[0]), float(end[1]), float(floor_z)],
        basis_u=u.tolist(),
        basis_v=v.tolist(),
        normal=np.cross(u, v).tolist(),
        boundary_uv=to_polygon2d(box(0.0, 0.0, length, height)),
        material=UNKNOWN_MATERIAL,
        observation_ids=plane_ids,
        lineage=plane_ids,
    )
