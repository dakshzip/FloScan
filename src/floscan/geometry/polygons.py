"""2D polygons for room boundaries: rasters, line arrangements, edge evidence.

Coordinates are metres in the horizontal (x, y) plane of one frame. Room
outlines come from the arrangement of observed wall lines: the plane is cut
by every wall line, and the cells (faces) mostly covered by observed floor
are merged. Concave rooms, diagonal walls and enclosed pillars (holes) fall
out of the arrangement without any assumed shape. Where no wall was seen,
the observed floor extent closes the outline, and those edges are marked
unknown rather than presented as walls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import polygonize, unary_union

from floscan.contracts.geometry import Polygon2D


@dataclass(frozen=True)
class Grid:
    """A raster over the xy plane: cell (row, col) covers origin + cell * (col, row)."""

    origin: NDArray[np.float64]
    cell: float
    shape: tuple[int, int]

    @classmethod
    def around(cls, xy: NDArray[np.float64], cell: float, margin: float) -> Grid:
        low = xy.min(axis=0) - margin
        high = xy.max(axis=0) + margin
        cols, rows = np.ceil((high - low) / cell).astype(int) + 1
        return cls(low, cell, (int(rows), int(cols)))

    def index(self, xy: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
        cells = np.floor((xy - self.origin) / self.cell).astype(int)
        cols = np.clip(cells[:, 0], 0, self.shape[1] - 1)
        rows = np.clip(cells[:, 1], 0, self.shape[0] - 1)
        return rows, cols

    def rasterize(self, xy: NDArray[np.float64]) -> NDArray[np.bool_]:
        mask = np.zeros(self.shape, dtype=bool)
        if len(xy):
            rows, cols = self.index(xy)
            mask[rows, cols] = True
        return mask

    def centres(self, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        rows, cols = np.nonzero(mask)
        return self.origin + (np.column_stack([cols, rows]) + 0.5) * self.cell


def mask_to_geometry(grid: Grid, mask: NDArray[np.bool_]) -> Polygon | MultiPolygon:
    """Exact union of the cells of ``mask`` (row runs merged first)."""
    boxes = []
    for row in range(grid.shape[0]):
        line = mask[row]
        if not line.any():
            continue
        padded = np.concatenate([[False], line, [False]])
        edges = np.nonzero(np.diff(padded.astype(np.int8)))[0]
        y0 = grid.origin[1] + row * grid.cell
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            x0 = grid.origin[0] + start * grid.cell
            boxes.append(box(x0, y0, grid.origin[0] + stop * grid.cell, y0 + grid.cell))
    return unary_union(boxes) if boxes else Polygon()


@dataclass(frozen=True)
class WallLine:
    """An observed wall's footprint: a line with observed support intervals."""

    wall_id: str
    point: NDArray[np.float64]  # a point on the line (xy)
    direction: NDArray[np.float64]  # unit (xy)
    intervals: tuple[tuple[float, float], ...]  # observed spans along direction

    def segments(self) -> list[LineString]:
        return [
            LineString(
                [self.point + a * self.direction, self.point + b * self.direction]
            )
            for a, b in self.intervals
        ]

    def infinite(self, reach: float) -> LineString:
        return LineString(
            [self.point - reach * self.direction, self.point + reach * self.direction]
        )


def arrangement_outline(
    region: Polygon | MultiPolygon,
    walls: list[WallLine],
    closure: list[LineString],
    min_cover: float,
) -> Polygon | MultiPolygon:
    """Union of arrangement faces mostly covered by ``region``.

    The arrangement is cut by every wall line (extended) and by ``closure``
    lines that bound the region where no wall was observed. A face is kept
    when at least ``min_cover`` of its area is observed region.
    """
    if region.is_empty:
        return Polygon()
    minx, miny, maxx, maxy = region.bounds
    reach = 2.0 * float(np.hypot(maxx - minx, maxy - miny)) + 1.0
    frame = box(minx - 1.0, miny - 1.0, maxx + 1.0, maxy + 1.0)
    # Closure lines are extended like wall lines, so every face is bounded on
    # all sides and a strip along an observed wall cannot leak to the frame.
    lines = (
        [w.infinite(reach) for w in walls]
        + [_extend(line, reach) for line in closure]
        + [frame.exterior]
    )
    noded = unary_union([shapely.intersection(line, frame) for line in lines])
    kept = []
    for face in polygonize(noded):
        if face.area <= 0:
            continue
        if region.intersection(face).area >= min_cover * face.area:
            kept.append(face)
    return unary_union(kept) if kept else Polygon()


def _extend(line: LineString, reach: float) -> LineString:
    a, b = (np.asarray(p, dtype=float) for p in line.coords[:2])
    direction = (b - a) / np.linalg.norm(b - a)
    return LineString([a - reach * direction, b + reach * direction])


def oriented_closure(region: Polygon | MultiPolygon, angle: float) -> list[LineString]:
    """Edges of the region's bounding rectangle in a frame rotated by ``angle``."""
    if region.is_empty:
        return []
    rotated = shapely.affinity.rotate(region, -angle, origin=(0, 0), use_radians=True)
    rectangle = box(*rotated.bounds)
    back = shapely.affinity.rotate(rectangle, angle, origin=(0, 0), use_radians=True)
    coords = list(back.exterior.coords)
    return [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]


@dataclass(frozen=True)
class EdgeEvidence:
    start: tuple[float, float]
    end: tuple[float, float]
    length_m: float
    supported_share: float
    wall_ids: tuple[str, ...]
    status: str  # "observed_wall" or "unknown"
    # Unobserved spans (from, to) metres along the edge from ``start``, at
    # least ``min_gap`` long: openings or unseen parts of an observed wall.
    gaps: tuple[tuple[float, float], ...] = ()


def edge_evidence(
    polygon: Polygon,
    walls: list[WallLine],
    tolerance: float,
    max_angle_deg: float,
    min_share: float,
    min_gap: float,
    step: float = 0.05,
) -> list[EdgeEvidence]:
    """For each outline edge, the share of its length lying on observed wall."""
    rings = [polygon.exterior, *polygon.interiors]
    cos_limit = float(np.cos(np.radians(max_angle_deg)))
    result = []
    for ring in rings:
        coords = np.asarray(ring.coords)
        for a, b in zip(coords[:-1], coords[1:], strict=True):
            length = float(np.linalg.norm(b - a))
            if length < 1e-9:
                continue
            direction = (b - a) / length
            samples = a + np.outer(np.arange(step / 2, length, step), direction)
            hit = np.zeros(len(samples), dtype=bool)
            ids = []
            for wall in walls:
                if abs(float(direction @ wall.direction)) < cos_limit:
                    continue
                for segment in wall.segments():
                    near = (
                        shapely.distance(shapely.points(samples), segment) <= tolerance
                    )
                    if near.any():
                        hit |= near
                        ids.append(wall.wall_id)
            share = float(hit.mean()) if len(hit) else 0.0
            gaps = []
            if share >= min_share:
                padded = np.concatenate([[True], hit, [True]]).astype(np.int8)
                changes = np.nonzero(np.diff(padded))[0]
                for first, stop in zip(changes[::2], changes[1::2], strict=True):
                    span = (float(first * step), float(min(stop * step, length)))
                    if span[1] - span[0] >= min_gap:
                        gaps.append(span)
            result.append(
                EdgeEvidence(
                    start=(float(a[0]), float(a[1])),
                    end=(float(b[0]), float(b[1])),
                    length_m=length,
                    supported_share=share,
                    wall_ids=tuple(sorted(set(ids))),
                    status="observed_wall" if share >= min_share else "unknown",
                    gaps=tuple(gaps),
                )
            )
    return result


def to_polygon2d(polygon: Polygon) -> Polygon2D:
    """Contract polygon: CCW outer ring and CW holes, closing vertex dropped."""
    oriented = shapely.geometry.polygon.orient(polygon, sign=1.0)
    return Polygon2D(
        outer=[list(map(float, p)) for p in oriented.exterior.coords[:-1]],
        holes=[
            [list(map(float, p)) for p in hole.coords[:-1]]
            for hole in oriented.interiors
        ],
    )


def simplify(polygon: Polygon, tolerance: float) -> Polygon:
    """Drop collinear and near-collinear vertices without changing topology."""
    simplified = polygon.simplify(tolerance, preserve_topology=True)
    return simplified if isinstance(simplified, Polygon) else polygon
