"""Linear quantities: wall lengths and heights, ceiling heights, room extents.

Pure functions of record geometry, in metres, never rounded.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from shapely.geometry import Polygon


def plan_length(start: ArrayLike, end: ArrayLike) -> float:
    """Horizontal distance between two points (x, y[, z])."""
    a = np.asarray(start, dtype=np.float64)[:2]
    b = np.asarray(end, dtype=np.float64)[:2]
    return float(np.linalg.norm(b - a))


def mean_profile_height(heights: list[float]) -> float:
    if not heights:
        raise ValueError("no height samples")
    return float(np.mean(heights))


def plane_height(normal: ArrayLike, offset: float, xy: ArrayLike) -> float:
    """z of the plane n . p + d = 0 above (x, y)."""
    n = np.asarray(normal, dtype=np.float64)
    if abs(n[2]) < 1e-9:
        raise ValueError("a vertical plane has no height above a point")
    p = np.asarray(xy, dtype=np.float64)
    return float(-(n[:2] @ p + offset) / n[2])


def vertical_gap(
    floor: tuple[ArrayLike, float], ceiling: tuple[ArrayLike, float], xy: ArrayLike
) -> float:
    """Ceiling plane height minus floor plane height above (x, y)."""
    return plane_height(*ceiling, xy) - plane_height(*floor, xy)


def plan_extents(outline: Polygon) -> tuple[float, float]:
    """(longer, shorter) sides of the minimum-area enclosing rectangle."""
    with np.errstate(divide="ignore", invalid="ignore"):  # GEOS internals
        rectangle = outline.minimum_rotated_rectangle
    coords = np.asarray(rectangle.exterior.coords)
    sides = np.linalg.norm(np.diff(coords[:3], axis=0), axis=1)
    return float(sides.max()), float(sides.min())
