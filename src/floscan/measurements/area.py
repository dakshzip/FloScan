"""Area quantities on plan polygons (metres squared, never rounded)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def ring_area(ring: ArrayLike) -> float:
    """Signed shoelace area of a ring (positive when counter-clockwise)."""
    xy = np.asarray(ring, dtype=np.float64)
    if len(xy) and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(outer: ArrayLike, holes: list[ArrayLike] = ()) -> float:
    """Area enclosed by ``outer`` minus its holes, independent of orientation."""
    return abs(ring_area(outer)) - sum(abs(ring_area(hole)) for hole in holes)
