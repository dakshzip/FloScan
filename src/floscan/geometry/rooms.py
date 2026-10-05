"""Room hypotheses from observed planes: floor, ceiling, walls and outlines.

Input is plane evidence in one metric frame (W, +z up): each plane with its
raw support points, as produced by reconstruction. The steps are:

1. Levels. Horizontal planes are grouped into height levels. The floor is
   the lowest up-facing level with enough observed area that lies below the
   cameras (a phone is never held below the floor); higher up-facing levels
   are furniture (tables, counters). The ceiling is the highest down-facing
   level above the cameras; without one, ceilings are absent, not assumed.
2. Walls. Vertical planes that start near the floor and rise high enough
   are wall candidates; low vertical planes are furniture. Faces of the same
   wall from different submaps are merged. Each wall's footprint is its
   intersection with the floor plane, with its observed spans.
3. Rooms. Observed floor minus walls is free space. It is split into rooms
   at narrow passages by eroding it by a doorway half-width and growing the
   remaining cores back (morphological room segmentation).
4. Outlines. Each room's outline is the union of wall-line arrangement
   faces mostly covered by its free space. Every edge carries its evidence:
   observed wall, or unknown (closed by the observed floor extent).

Nothing here assumes a room shape, size or ceiling height. A room with any
unknown edge or thin floor coverage is ``partial``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from pydantic import Field, model_validator
from scipy import ndimage
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from floscan import __version__
from floscan.contracts.base import Contract, Provenance
from floscan.contracts.geometry import HeightSample, Room, Surface, Wall
from floscan.geometry.planes import PlaneFitError, least_squares_plane, robust_refit
from floscan.geometry.polygons import (
    EdgeEvidence,
    Grid,
    WallLine,
    arrangement_outline,
    edge_evidence,
    mask_to_geometry,
    oriented_closure,
    simplify,
    to_polygon2d,
)
from floscan.geometry.surfaces import (
    FittedPlane,
    ceiling_surface,
    floor_surface,
    wall_surface,
)

STAGE = "rooms.build"


class RoomConfig(Contract):
    """Evidence thresholds; none is a room dimension."""

    horizontal_tolerance_deg: float = Field(default=10.0, gt=0, lt=45)
    level_tolerance_m: float = Field(default=0.10, gt=0)
    min_level_area_m2: float = Field(default=1.0, gt=0)
    min_wall_height_m: float = Field(default=1.0, gt=0)
    max_wall_bottom_above_floor_m: float = Field(default=0.4, ge=0)
    wall_merge_angle_deg: float = Field(default=3.0, gt=0)
    wall_merge_offset_m: float = Field(default=0.05, gt=0)
    span_gap_m: float = Field(default=0.3, gt=0)
    min_span_m: float = Field(default=0.10, gt=0)
    grid_cell_m: float = Field(default=0.05, gt=0)
    # Passages narrower than twice this split rooms (a doorway half-width).
    door_half_width_m: float = Field(default=0.45, gt=0)
    min_room_core_area_m2: float = Field(default=0.5, gt=0)
    min_room_area_m2: float = Field(default=1.0, gt=0)
    face_min_cover: float = Field(default=0.5, gt=0, le=1)
    edge_tolerance_m: float = Field(default=0.08, gt=0)
    edge_angle_deg: float = Field(default=10.0, gt=0, lt=90)
    edge_min_share: float = Field(default=0.6, gt=0, le=1)
    min_room_floor_coverage: float = Field(default=0.5, gt=0, le=1)
    simplify_m: float = Field(default=0.02, ge=0)
    # Optional soft orthogonality: wall directions within this many degrees
    # of the dominant perpendicular family are rotated onto it. None: off.
    snap_orthogonal_deg: float | None = Field(default=None, gt=0, lt=10)
    # A room gets a ceiling only from support observed over at least this
    # share of its outline.
    min_ceiling_cover: float = Field(default=0.3, gt=0, le=1)
    # Robust plane fits of floors and ceilings (Tukey cut-off).
    plane_fit_cutoff_m: float = Field(default=0.03, gt=0)
    min_plane_points: int = Field(default=50, ge=3)
    # A tall vertical surface continuing below a candidate floor counts only
    # if its base lies within this distance of the level's observed area,
    # and the level is rejected only if such surfaces carry more than this
    # share of the vertical support over that area.
    below_floor_margin_m: float = Field(default=0.3, ge=0)
    max_below_floor_share: float = Field(default=0.2, gt=0, lt=1)

    @model_validator(mode="after")
    def _coupled(self) -> RoomConfig:
        if self.grid_cell_m * 4 > self.door_half_width_m:
            raise ValueError("grid_cell_m must be well below door_half_width_m")
        return self


def config_hash(config: RoomConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedPlane:
    """One reconstructed plane with its raw support, in W (+z up), metres.

    ``component`` is the tracking component (reconstruction segment) the
    plane was observed in; planes of different components are never
    assembled into one room without registration evidence.
    """

    plane_id: str
    normal: NDArray[np.float64]
    offset: float
    points: NDArray[np.float64]
    component: str = "0"

    @property
    def height(self) -> float:
        return float(np.median(self.points[:, 2]))


@dataclass(frozen=True)
class SceneEvidence:
    planes: list[ObservedPlane]
    cameras: NDArray[np.float64]  # camera centres that observed the planes
    source: str  # where the evidence came from (for provenance)
    source_hashes: list[str] = field(default_factory=list)
    # Tracking component of each camera row; None means a single component.
    camera_components: tuple[str, ...] | None = None
    components: dict[str, dict[str, Any]] = field(default_factory=dict)

    def component_ids(self) -> list[str]:
        ids = {p.component for p in self.planes}
        if self.camera_components is not None:
            ids |= set(self.camera_components)
        return sorted(ids, key=lambda c: (len(c), c)) or ["0"]

    def only(self, component: str) -> SceneEvidence:
        cameras = self.cameras
        if self.camera_components is not None:
            keep = np.array([c == component for c in self.camera_components], bool)
            cameras = self.cameras[keep] if len(keep) else self.cameras
        return SceneEvidence(
            planes=[p for p in self.planes if p.component == component],
            cameras=cameras,
            source=self.source,
            source_hashes=self.source_hashes,
        )


def load_bundle(directory: Path) -> SceneEvidence:
    """Plane evidence from a reconstruction bundle written by P07.

    Each submap belongs to one tracking segment of the bundle (segments
    split at possible tracking resets); that membership is kept per plane
    and per camera.
    """
    manifest = json.loads((directory / "bundle.json").read_text("utf-8"))
    segments = [
        (k, s["first_keyframe"], s["last_keyframe"])
        for k, s in enumerate(manifest.get("segments", []))
    ]

    def component_of(keyframes: list[int]) -> str:
        for k, first, last in segments:
            if first <= keyframes[0] <= last:
                return str(k)
        return "0"

    planes, cameras, camera_components = [], [], []
    components: dict[str, dict[str, Any]] = {}
    for submap in manifest["submaps"]:
        component = component_of(submap["keyframes"])
        entry = components.setdefault(
            component, {"submaps": [], "keyframes": 0, "planes": 0}
        )
        entry["submaps"].append(submap["submap_id"])
        entry["keyframes"] += len(submap["keyframes"])
        entry["planes"] += len(submap["planes"])
        cloud = submap["point_cloud"]
        if cloud["xyz"]["shape"][0] == 0:
            continue
        xyz = np.load(directory / cloud["xyz"]["uri"]).astype(np.float64)
        views = np.unique(
            np.load(directory / f"submaps/{submap['submap_id']}/viewpoints.npy"),
            axis=0,
        )
        cameras.append(views)
        camera_components.extend([component] * len(views))
        for item in submap["planes"]:
            record = item["plane"]
            support = np.load(directory / record["support"]["uri"])
            planes.append(
                ObservedPlane(
                    plane_id=record["id"],
                    normal=np.asarray(record["normal"], dtype=np.float64),
                    offset=float(record["offset"]),
                    points=xyz[support],
                    component=component,
                )
            )
    return SceneEvidence(
        planes=planes,
        cameras=np.concatenate(cameras) if cameras else np.zeros((0, 3)),
        source=str(directory / "bundle.json"),
        source_hashes=[
            hashlib.sha256((directory / "bundle.json").read_bytes()).hexdigest()
        ],
        camera_components=tuple(camera_components),
        components=components,
    )


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    height: float
    area_m2: float
    plane_ids: tuple[str, ...]
    facing: str  # "up" or "down"
    points: NDArray[np.float64]


def _footprint(points: NDArray[np.float64], cell: float) -> Polygon | MultiPolygon:
    """Union of the grid cells holding at least one point (xy)."""
    if len(points) == 0:
        return Polygon()
    cells = np.unique(np.floor(points[:, :2] / cell).astype(np.int64), axis=0)
    low = cells.min(axis=0)
    shape = tuple(int(v) for v in (np.ptp(cells, axis=0) + 1)[::-1])
    grid = Grid(low * cell, cell, shape)
    return mask_to_geometry(grid, grid.rasterize((cells + 0.5) * cell))


def _footprint_area(points: NDArray[np.float64], cell: float) -> float:
    if len(points) == 0:
        return 0.0
    cells = np.unique(np.floor(points[:, :2] / cell).astype(np.int64), axis=0)
    return float(len(cells)) * cell * cell


def find_levels(
    evidence: SceneEvidence, config: RoomConfig, facing: str
) -> list[Level]:
    """Horizontal planes facing ``facing`` grouped by height, lowest first."""
    limit = math.cos(math.radians(config.horizontal_tolerance_deg))
    sign = 1.0 if facing == "up" else -1.0
    planes = sorted(
        (p for p in evidence.planes if sign * p.normal[2] >= limit),
        key=lambda p: p.height,
    )
    groups: list[list[ObservedPlane]] = []
    for plane in planes:
        if groups and plane.height - groups[-1][-1].height <= config.level_tolerance_m:
            groups[-1].append(plane)
        else:
            groups.append([plane])
    levels = []
    for group in groups:
        points = np.concatenate([p.points for p in group])
        levels.append(
            Level(
                height=float(np.median(points[:, 2])),
                area_m2=_footprint_area(points, config.grid_cell_m),
                plane_ids=tuple(p.plane_id for p in group),
                facing=facing,
                points=points,
            )
        )
    return levels


def fit_horizontal(
    points: NDArray[np.float64],
    plane_ids: tuple[str, ...],
    facing: str,
    config: RoomConfig,
) -> FittedPlane | None:
    """Robust plane through floor or ceiling support, normal up or down.

    The fitted tilt is kept; nothing is flattened to the horizontal.
    """
    if len(points) < config.min_plane_points:
        return None
    try:
        normal, offset, _ = least_squares_plane(points)
        fit = robust_refit(points, normal, offset, config.plane_fit_cutoff_m)
    except PlaneFitError:
        return None
    normal, offset = fit.normal, fit.offset
    if (normal[2] > 0) != (facing == "up"):
        normal, offset = -normal, -offset
    return FittedPlane(np.asarray(normal), float(offset), plane_ids)


@dataclass(frozen=True)
class TallVertical:
    """A vertical plane taller than ``min_wall_height_m`` (floor-independent)."""

    plane: ObservedPlane
    bottom_xy: NDArray[np.float64]
    bottom_z: float
    top_z: float


def tall_verticals(evidence: SceneEvidence, config: RoomConfig) -> list[TallVertical]:
    limit = math.sin(math.radians(config.horizontal_tolerance_deg))
    found = []
    for plane in evidence.planes:
        if abs(plane.normal[2]) > limit:
            continue
        z = plane.points[:, 2]
        bottom, top = float(np.percentile(z, 2)), float(np.percentile(z, 98))
        if top - bottom < config.min_wall_height_m:
            continue
        low = plane.points[z <= np.percentile(z, 5)]
        found.append(TallVertical(plane, low[:, :2].mean(axis=0), bottom, top))
    return found


def choose_floor(
    levels: list[Level],
    cameras: NDArray[np.float64],
    tall: list[TallVertical],
    config: RoomConfig,
) -> tuple[Level | None, FittedPlane | None, str]:
    """The floor level, if structural evidence supports the role.

    Candidates are up-facing levels with enough area below the cameras,
    lowest first. The lowest is the floor only if tall vertical surfaces
    (walls) stand on it and none continues below it; otherwise an unseen
    floor and furniture cannot be told apart and no floor is chosen.
    """
    camera_height = float(np.median(cameras[:, 2])) if len(cameras) else math.inf
    tolerance = config.level_tolerance_m
    for level in levels:
        if level.area_m2 < config.min_level_area_m2:
            continue
        if level.height >= camera_height:
            return (
                None,
                None,
                (
                    f"the lowest substantial up-facing level ({level.height:.2f} m) is "
                    f"not below the cameras ({camera_height:.2f} m)"
                ),
            )
        plane = fit_horizontal(level.points, level.plane_ids, "up", config)
        if plane is None:
            return None, None, f"the level at {level.height:.2f} m has no stable plane"
        footprint = _footprint(level.points, config.grid_cell_m).buffer(
            config.below_floor_margin_m
        )
        relative = [
            (t, t.bottom_z - float(plane.z_at(t.bottom_xy[None])[0])) for t in tall
        ]
        near = [(t, d) for t, d in relative if footprint.contains(Point(t.bottom_xy))]
        below = [(t, d) for t, d in near if d < -tolerance]
        standing = [
            t
            for t, d in relative
            if -tolerance <= d <= config.max_wall_bottom_above_floor_m
        ]
        near_support = sum(len(t.plane.points) for t, _ in near)
        share = (
            sum(len(t.plane.points) for t, _ in below) / near_support
            if near_support
            else 0.0
        )
        if share > config.max_below_floor_share:
            deepest = min(d for _, d in below)
            reason = (
                f"ambiguous: {len(below)} tall vertical surface(s) over the up-facing "
                f"level at {level.height:.2f} m continue up to {-deepest:.2f} m "
                f"below it ({share:.0%} of the vertical support there), so the "
                "floor is lower and unseen"
            )
            return None, None, reason
        if not standing:
            reason = (
                f"ambiguous: no wall stands on the up-facing level at "
                f"{level.height:.2f} m; an unseen floor and furniture cannot be "
                "told apart"
            )
            return None, None, reason
        reason = (
            f"lowest up-facing level with {level.area_m2:.1f} m2 of observed area, "
            f"{camera_height - level.height:.2f} m below the median camera, with "
            f"{len(standing)} wall(s) standing on it"
        )
        anomalies = [(t, d) for t, d in relative if d < -tolerance]
        if anomalies:
            reason += (
                f"; {len(anomalies)} tall vertical surface(s) reach below it "
                f"({', '.join(t.plane.plane_id for t, _ in anomalies)}; "
                f"{share:.1%} of the support over the floor area)"
            )
        return level, plane, reason
    return None, None, "no up-facing level with enough observed area"


# --------------------------------------------------------------------------
# Walls
# --------------------------------------------------------------------------


@dataclass
class WallCandidate:
    wall_id: str
    plane_ids: list[str]
    normal_xy: NDArray[np.float64]  # unit, facing the room
    offset_xy: float  # n_xy . p + offset = 0 on the floor plane
    points: NDArray[np.float64]
    bottom: float
    top: float
    line: WallLine | None = None


def _floor_line(
    normal: NDArray, offset: float, floor: FittedPlane
) -> tuple[NDArray, float]:
    """Wall plane intersected with the floor plane, projected to xy.

    On the floor ``z = -(f_xy . xy + f_d) / f_z``; substituting gives the 2D
    line ``n . xy + c = 0`` (unit ``n``) of the floor-wall intersection.
    """
    f = floor.normal
    n_xy = normal[:2] - normal[2] * f[:2] / f[2]
    c = offset - normal[2] * floor.offset / f[2]
    scale = float(np.linalg.norm(n_xy))
    return n_xy / scale, c / scale


def find_walls(
    evidence: SceneEvidence, floor: FittedPlane, config: RoomConfig
) -> tuple[list[WallCandidate], dict[str, int]]:
    """Merged wall candidates and counts of rejected vertical planes.

    Heights are measured from the floor plane below each wall.
    """
    vertical_limit = math.sin(math.radians(config.horizontal_tolerance_deg))
    rejected = {"too_low": 0, "starts_above_floor": 0}
    candidates: list[WallCandidate] = []
    for plane in evidence.planes:
        if abs(plane.normal[2]) > vertical_limit:
            continue
        above = plane.points[:, 2] - floor.z_at(plane.points[:, :2])
        bottom, top = float(np.percentile(above, 2)), float(np.percentile(above, 98))
        if bottom > config.max_wall_bottom_above_floor_m:
            rejected["starts_above_floor"] += 1
            continue
        if top < config.min_wall_height_m:
            rejected["too_low"] += 1
            continue
        n_xy, c = _floor_line(plane.normal, plane.offset, floor)
        candidates.append(
            WallCandidate(
                plane.plane_id, [plane.plane_id], n_xy, c, plane.points, bottom, top
            )
        )
    merged = _merge_walls(candidates, floor, config)
    for k, wall in enumerate(merged):
        wall.wall_id = f"wall:{k:03d}"
        wall.line = _wall_line(wall, config)
    return [w for w in merged if w.line is not None], rejected


def _merge_walls(
    walls: list[WallCandidate], floor: FittedPlane, config: RoomConfig
) -> list[WallCandidate]:
    """Merge faces of one wall seen from several submaps (same facing side)."""
    cos_limit = math.cos(math.radians(config.wall_merge_angle_deg))
    remaining = sorted(walls, key=lambda w: -len(w.points))
    merged: list[WallCandidate] = []
    for wall in remaining:
        for target in merged:
            if float(wall.normal_xy @ target.normal_xy) < cos_limit:
                continue
            # Distance between the lines at the wall's own support.
            distances = wall.points[:, :2] @ target.normal_xy + target.offset_xy
            if abs(float(np.median(distances))) > config.wall_merge_offset_m:
                continue
            target.plane_ids.extend(wall.plane_ids)
            target.points = np.concatenate([target.points, wall.points])
            target.bottom = min(target.bottom, wall.bottom)
            target.top = max(target.top, wall.top)
            _refit_footprint(target, floor)
            break
        else:
            merged.append(wall)
    return merged


def _refit_footprint(wall: WallCandidate, floor: FittedPlane) -> None:
    """Refit a merged wall's plane on all its support; re-intersect the floor."""
    try:
        normal, offset, _ = least_squares_plane(wall.points)
    except PlaneFitError:
        return
    n_xy, c = _floor_line(normal, offset, floor)
    if n_xy @ wall.normal_xy < 0:
        n_xy, c = -n_xy, -c
    wall.normal_xy, wall.offset_xy = n_xy, c


def _wall_line(wall: WallCandidate, config: RoomConfig) -> WallLine | None:
    direction = np.array([-wall.normal_xy[1], wall.normal_xy[0]])
    point = -wall.offset_xy * wall.normal_xy
    s = np.sort((wall.points[:, :2] - point) @ direction)
    breaks = np.nonzero(np.diff(s) > config.span_gap_m)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(s) - 1]])
    spans = tuple(
        (float(s[a]), float(s[b]))
        for a, b in zip(starts, ends, strict=True)
        if s[b] - s[a] >= config.min_span_m
    )
    if not spans:
        return None
    return WallLine(wall.wall_id, point, direction, spans)


def snap_orthogonal(walls: list[WallCandidate], config: RoomConfig) -> float:
    """Dominant direction (radians, mod 90 deg); optionally snap near-axis walls."""
    if not walls:
        return 0.0
    angles = np.array([math.atan2(w.normal_xy[1], w.normal_xy[0]) for w in walls])
    weights = np.array([len(w.points) for w in walls], dtype=float)
    dominant = (
        math.atan2(weights @ np.sin(4 * angles), weights @ np.cos(4 * angles)) / 4
    )
    if config.snap_orthogonal_deg is None:
        return dominant
    for wall, angle in zip(walls, angles, strict=True):
        deviation = (angle - dominant + math.pi / 4) % (math.pi / 2) - math.pi / 4
        if abs(math.degrees(deviation)) <= config.snap_orthogonal_deg:
            centre = np.median(wall.points[:, :2], axis=0)
            new = angle - deviation
            wall.normal_xy = np.array([math.cos(new), math.sin(new)])
            wall.offset_xy = -float(wall.normal_xy @ centre)
            wall.line = _wall_line(wall, config)
    return dominant


# --------------------------------------------------------------------------
# Rooms
# --------------------------------------------------------------------------


@dataclass
class RoomHypothesis:
    room_id: str
    outline: Polygon
    edges: list[EdgeEvidence]
    floor_coverage: float
    wall_ids: list[str]
    fragments_dropped: int
    status: str
    reasons: list[str]


def segment_rooms(
    grid: Grid, free: NDArray[np.bool_], config: RoomConfig
) -> tuple[NDArray[np.int32], dict[str, Any]]:
    """Room labels over free space (0 = unassigned), split at narrow passages."""
    clearance = ndimage.distance_transform_edt(free) * grid.cell
    cores, count = ndimage.label(clearance > config.door_half_width_m)
    sizes = ndimage.sum(np.ones_like(cores), cores, index=np.arange(1, count + 1))
    keep = np.zeros(count + 1, dtype=np.int32)
    next_label = 0
    min_cells = config.min_room_core_area_m2 / grid.cell**2
    for label, size in enumerate(sizes, start=1):
        if size >= min_cells:
            next_label += 1
            keep[label] = next_label
    labels = keep[cores]
    # Grow cores back over free space, one cell ring per pass (geodesic).
    structure = ndimage.generate_binary_structure(2, 1)
    while True:
        grown = ndimage.grey_dilation(labels, footprint=structure)
        update = free & (labels == 0) & (grown > 0)
        if not update.any():
            break
        labels = np.where(update, grown, labels)
    return labels, {
        "cores_found": int(count),
        "rooms": int(next_label),
        "free_cells_unassigned": int((free & (labels == 0)).sum()),
    }


def _room_walls(
    region: Polygon | MultiPolygon, walls: list[WallCandidate], reach: float
) -> list[WallLine]:
    """Walls whose observed spans border the region from its own side."""
    found = []
    for wall in walls:
        line = wall.line
        for segment in line.segments():
            middle = np.asarray(segment.interpolate(0.5, normalized=True).coords[0])
            probe = middle + reach * wall.normal_xy
            if region.distance(Point(probe)) <= reach:
                found.append(line)
                break
    return found


def outline_room(
    room_id: str,
    region: Polygon | MultiPolygon,
    floor_region: Polygon | MultiPolygon,
    walls: list[WallCandidate],
    dominant: float,
    config: RoomConfig,
    neighbours: Polygon | MultiPolygon | None = None,
    taken: Polygon | MultiPolygon | None = None,
) -> RoomHypothesis | None:
    """Outline of one room.

    ``neighbours`` is other rooms' observed floor: faces are judged without
    it, so a room cannot claim its neighbour's space. ``taken`` is outlines
    already assigned; any sliver still shared with them is removed, so rooms
    never overlap and edges stay on wall or closure lines.
    """
    lines = _room_walls(region, walls, 2 * config.edge_tolerance_m + config.grid_cell_m)
    closure = oriented_closure(region, dominant)
    # Free space stops short of each wall (wall cells are barriers), so a face
    # running up to a wall line is judged against the region grown by a few
    # cells; closure lines still sit at the observed extent.
    # The grown region never reaches into other rooms' floor, so a room cannot
    # claim its neighbour's space; whatever is still shared is removed after.
    grown = region.buffer(3 * config.grid_cell_m, join_style="mitre")
    if neighbours is not None and not neighbours.is_empty:
        grown = grown.difference(neighbours)
    shape = arrangement_outline(grown, lines, closure, config.face_min_cover)
    if taken is not None and not taken.is_empty:
        shape = shape.difference(taken)
    if shape.is_empty:
        return None
    fragments = 0
    if isinstance(shape, MultiPolygon):
        parts = sorted(shape.geoms, key=lambda g: g.area, reverse=True)
        shape, fragments = parts[0], len(parts) - 1
    # CCW outer ring, CW holes: the room is on the left of every edge.
    shape = orient(simplify(shape.buffer(0), config.simplify_m), sign=1.0)
    if shape.area < config.min_room_area_m2:
        return None
    edges = edge_evidence(
        shape,
        lines,
        config.edge_tolerance_m,
        config.edge_angle_deg,
        config.edge_min_share,
        config.span_gap_m,
    )
    coverage = float(floor_region.intersection(shape).area / shape.area)
    unknown = [e for e in edges if e.status == "unknown"]
    reasons = []
    if unknown:
        reasons.append(
            f"{len(unknown)} of {len(edges)} outline edges "
            f"({sum(e.length_m for e in unknown):.2f} m) have no observed wall"
        )
    gaps = [gap for e in edges for gap in e.gaps]
    if gaps:
        reasons.append(
            f"{len(gaps)} unobserved span(s) within observed walls "
            f"({sum(b - a for a, b in gaps):.2f} m): openings or unseen wall"
        )
    if coverage < config.min_room_floor_coverage:
        reasons.append(f"observed floor covers only {coverage:.0%} of the outline")
    if fragments:
        reasons.append(f"{fragments} disconnected fragment(s) dropped")
    return RoomHypothesis(
        room_id=room_id,
        outline=shape,
        edges=edges,
        floor_coverage=coverage,
        wall_ids=sorted({w for e in edges for w in e.wall_ids}),
        fragments_dropped=fragments,
        status="partial" if reasons else "ok",
        reasons=reasons,
    )


# --------------------------------------------------------------------------
# Model and records
# --------------------------------------------------------------------------


@dataclass
class RoomModel:
    status: str  # ok, partial or insufficient_evidence
    reasons: list[str]
    rooms: list[Room]
    walls: list[Wall]
    surfaces: list[Surface]
    hypotheses: list[RoomHypothesis]
    diagnostics: dict[str, Any]


def build_rooms(evidence: SceneEvidence, config: RoomConfig | None = None) -> RoomModel:
    """Room hypotheses, walls and versioned surfaces from plane evidence.

    Each tracking component is built on its own; rooms are never assembled
    across a possible tracking reset.
    """
    config = config or RoomConfig()
    components = evidence.component_ids()
    if len(components) == 1:
        model = _build_component(evidence.only(components[0]), config, "")
        model.diagnostics["components"] = {
            components[0]: evidence.components.get(components[0], {})
        }
        model.diagnostics["source"] = evidence.source
        return model
    parts = {
        c: _build_component(evidence.only(c), config, f"c{c}:") for c in components
    }
    rooms = [r for m in parts.values() for r in m.rooms]
    reasons = [
        f"{len(components)} tracking components (split at possible tracking resets) "
        "were built separately; no room combines evidence across them"
    ]
    for component, model in parts.items():
        reasons += [f"component {component}: {reason}" for reason in model.reasons]
    status = "insufficient_evidence" if not rooms else "partial"
    return RoomModel(
        status=status,
        reasons=reasons,
        rooms=rooms,
        walls=[w for m in parts.values() for w in m.walls],
        surfaces=[s for m in parts.values() for s in m.surfaces],
        hypotheses=[h for m in parts.values() for h in m.hypotheses],
        diagnostics={
            "source": evidence.source,
            "config": config.model_dump(mode="json"),
            "config_hash": config_hash(config),
            "components": {
                c: {**evidence.components.get(c, {}), "result": m.diagnostics}
                for c, m in parts.items()
            },
        },
    )


def _build_component(
    evidence: SceneEvidence, config: RoomConfig, prefix: str
) -> RoomModel:
    provenance = Provenance(
        stage=STAGE,
        stage_version=f"floscan {__version__}",
        mode="live",
        config_hash=config_hash(config),
        upstream_artifact_hashes=evidence.source_hashes,
    )
    up = find_levels(evidence, config, "up")
    down = find_levels(evidence, config, "down")
    tall = tall_verticals(evidence, config)
    floor, floor_plane, floor_reason = choose_floor(up, evidence.cameras, tall, config)
    diagnostics: dict[str, Any] = {
        "config": config.model_dump(mode="json"),
        "config_hash": config_hash(config),
        "floor": {"chosen": floor is not None, "reason": floor_reason},
        "up_facing_levels": [_level_summary(level) for level in up],
        "down_facing_levels": [_level_summary(level) for level in down],
    }
    if floor is None or floor_plane is None:
        return RoomModel(
            "insufficient_evidence", [f"floor not established: {floor_reason}"],
            [], [], [], [], diagnostics,
        )  # fmt: skip
    diagnostics["floor"].update(
        height_m=floor.height,
        plane_ids=list(floor.plane_ids),
        normal=floor_plane.normal.tolist(),
        offset=floor_plane.offset,
        tilt_deg=floor_plane.tilt_deg(),
    )
    diagnostics["furniture_levels"] = [
        _level_summary(level) for level in up if level.height > floor.height
    ]
    walls, rejected = find_walls(evidence, floor_plane, config)
    dominant = snap_orthogonal(walls, config)
    diagnostics["walls"] = {
        "candidates": len(walls),
        "rejected_vertical_planes": rejected,
        "dominant_direction_deg": math.degrees(dominant),
        "orthogonal_snap_deg": config.snap_orthogonal_deg,
    }
    floor_xy = floor.points[:, :2]
    wall_xy = [w.points[:, :2] for w in walls]
    grid = Grid.around(
        np.concatenate([floor_xy, *wall_xy]), config.grid_cell_m, 4 * config.grid_cell_m
    )
    floor_mask = ndimage.binary_closing(grid.rasterize(floor_xy), iterations=2)
    wall_mask = np.zeros(grid.shape, dtype=bool)
    for wall in walls:
        for segment in wall.line.segments():
            length = segment.length
            steps = np.linspace(0, length, max(2, int(length / (grid.cell / 2)) + 1))
            samples = np.array([segment.interpolate(t).coords[0] for t in steps])
            wall_mask |= grid.rasterize(samples)
    wall_mask = ndimage.binary_dilation(wall_mask, iterations=1)
    # Furniture-occluded floor inside a room is still room; walls stay walls.
    free = ndimage.binary_fill_holes(floor_mask & ~wall_mask) & ~wall_mask
    labels, segmentation = segment_rooms(grid, free, config)
    diagnostics["segmentation"] = segmentation
    floor_region = mask_to_geometry(grid, floor_mask)
    hypotheses, dropped = [], 0
    regions = [
        mask_to_geometry(grid, labels == label)
        for label in range(1, int(labels.max()) + 1)
    ]
    for index, region in enumerate(regions):
        neighbours = unary_union([r for k, r in enumerate(regions) if k != index])
        taken = unary_union([h.outline for h in hypotheses])
        hypothesis = outline_room(
            f"room:{prefix}{len(hypotheses) + 1:02d}",
            region,
            floor_region,
            walls,
            dominant,
            config,
            neighbours,
            taken,
        )
        if hypothesis is None:
            dropped += 1
            continue
        hypotheses.append(hypothesis)
    diagnostics["rooms_dropped_small_or_empty"] = dropped
    ceiling_candidates = _ceiling_candidates(down, evidence.cameras)
    rooms, wall_records, surfaces, room_diagnostics = _records(
        hypotheses, walls, floor, floor_plane, ceiling_candidates, config, provenance
    )
    diagnostics["rooms"] = room_diagnostics
    reasons = []
    if not hypotheses:
        reasons.append("no room outline could be formed from the observed floor")
    without = [r.id for r in rooms if not r.ceiling_surface_ids]
    if without:
        reasons.append(
            f"no ceiling observed over {len(without)} of {len(rooms)} room(s)"
        )
    partial = [r.id for r in rooms if r.status != "ok"]
    if partial:
        reasons.append(f"{len(partial)} of {len(rooms)} rooms are partial")
    status = (
        "insufficient_evidence" if not hypotheses else "partial" if reasons else "ok"
    )
    return RoomModel(
        status, reasons, rooms, wall_records, surfaces, hypotheses, diagnostics
    )


def _ceiling_candidates(
    down: list[Level], cameras: NDArray[np.float64]
) -> list[ObservedPlane]:
    """Down-facing support above every camera, kept per plane."""
    top = float(cameras[:, 2].max()) if len(cameras) else -math.inf
    planes = []
    for level in down:
        if level.height > top:
            planes.append(
                ObservedPlane(
                    plane_id="+".join(level.plane_ids),
                    normal=np.array([0.0, 0.0, -1.0]),
                    offset=level.height,
                    points=level.points,
                )
            )
    return planes


def _room_plane(
    outline: Polygon,
    points: NDArray[np.float64],
    plane_ids: tuple[str, ...],
    facing: str,
    config: RoomConfig,
) -> tuple[FittedPlane | None, float]:
    """Plane through the support lying over ``outline``, and its coverage."""
    if len(points) == 0:
        return None, 0.0
    inside = shapely.contains_xy(outline, points[:, 0], points[:, 1])
    covered = _footprint_area(points[inside], config.grid_cell_m) / outline.area
    plane = fit_horizontal(points[inside], plane_ids, facing, config)
    return plane, float(covered)


def _level_summary(level: Level) -> dict[str, Any]:
    return {
        "height_m": level.height,
        "observed_area_m2": level.area_m2,
        "plane_ids": list(level.plane_ids),
    }


def _records(
    hypotheses: list[RoomHypothesis],
    walls: list[WallCandidate],
    floor: Level,
    floor_plane: FittedPlane,
    ceilings: list[ObservedPlane],
    config: RoomConfig,
    provenance: Provenance,
) -> tuple[list[Room], list[Wall], list[Surface], dict[str, Any]]:
    """Records per room, using only horizontal evidence observed over it."""
    by_id = {w.wall_id: w for w in walls}
    rooms, wall_records, surfaces = [], [], []
    room_diagnostics: dict[str, Any] = {}
    for hypothesis in hypotheses:
        room_id = hypothesis.room_id
        reasons = list(hypothesis.reasons)
        info: dict[str, Any] = {}
        # Floor: the room's own floor support if enough, else the level plane.
        own_floor, _ = _room_plane(
            hypothesis.outline, floor.points, floor.plane_ids, "up", config
        )
        room_floor = own_floor or floor_plane
        info["floor"] = {
            "plane": "room support" if own_floor else "floor level (shared)",
            "tilt_deg": room_floor.tilt_deg(),
        }
        floor_id = f"surface:{room_id}:floor"
        surfaces.append(
            floor_surface(floor_id, room_id, hypothesis.outline, room_floor, provenance)
        )
        # Ceiling: only support observed over this room, at its own height.
        best, best_cover = None, 0.0
        for candidate in ceilings:
            plane, cover = _room_plane(
                hypothesis.outline,
                candidate.points,
                (candidate.plane_id,),
                "down",
                config,
            )
            if plane is not None and cover > best_cover:
                best, best_cover = plane, cover
        room_ceiling = best if best_cover >= config.min_ceiling_cover else None
        info["ceiling"] = {
            "observed": room_ceiling is not None,
            "best_cover": best_cover,
        }
        ceiling_ids = []
        if room_ceiling is not None:
            ceiling_id = f"surface:{room_id}:ceiling"
            ceiling_ids.append(ceiling_id)
            surfaces.append(
                ceiling_surface(
                    ceiling_id, room_id, hypothesis.outline, room_ceiling, provenance
                )
            )
        else:
            reasons.append(
                "no ceiling observed over this room (best cover "
                f"{best_cover:.0%}); wall heights are observed extents only"
            )
        wall_ids = []
        for k, edge in enumerate(e for e in hypothesis.edges if e.status != "unknown"):
            wall_id = f"{room_id}:W{k + 1}"
            surface_id = f"surface:{wall_id}"
            plane_ids = [p for w in edge.wall_ids for p in by_id[w].plane_ids]
            ends = np.array([edge.start, edge.end])
            floor_z = room_floor.z_at(ends)
            if room_ceiling is not None:
                top_z = room_ceiling.z_at(ends)
            else:
                observed = max(by_id[w].top for w in edge.wall_ids)
                top_z = floor_z + observed
            surfaces.append(
                wall_surface(
                    surface_id, room_id, ends[0], ends[1], room_floor,
                    (float(top_z[0]), float(top_z[1])), plane_ids, provenance,
                )
            )  # fmt: skip
            profile = []
            if room_ceiling is not None:
                profile = [
                    HeightSample(s_m=0.0, height_m=float(top_z[0] - floor_z[0])),
                    HeightSample(
                        s_m=edge.length_m, height_m=float(top_z[1] - floor_z[1])
                    ),
                ]
            wall_records.append(
                Wall(
                    id=wall_id,
                    provenance=provenance,
                    room_id=room_id,
                    surface_id=surface_id,
                    baseline=[
                        [edge.start[0], edge.start[1], float(floor_z[0])],
                        [edge.end[0], edge.end[1], float(floor_z[1])],
                    ],
                    plane_id=plane_ids[0],
                    height_profile=profile,
                    boundary_evidence=plane_ids,
                )
            )
            wall_ids.append(wall_id)
        room_diagnostics[room_id] = info
        rooms.append(
            Room(
                id=room_id,
                provenance=provenance,
                status="partial" if reasons else "ok",
                status_reason="; ".join(reasons) or None,
                label=room_id.replace("room:", "room "),
                level_id="L0",
                local_frame_id="W",
                boundary=to_polygon2d(hypothesis.outline),
                wall_ids=wall_ids,
                floor_surface_ids=[floor_id],
                ceiling_surface_ids=ceiling_ids,
                coverage=min(1.0, hypothesis.floor_coverage),
                hypothesis_id=f"rooms:{provenance.config_hash[:12]}",
                placement_status="unplaced",
            )
        )
        hypothesis.status = "partial" if reasons else "ok"
        hypothesis.reasons = reasons
    return rooms, wall_records, surfaces, room_diagnostics


def model_to_json(model: RoomModel) -> dict[str, Any]:
    """rooms.json content: records, per-edge evidence and diagnostics."""
    return {
        "kind": "floscan-room-hypotheses",
        "floscan_version": __version__,
        "status": model.status,
        "reasons": model.reasons,
        "frame": {
            "id": "W",
            "note": "session world, +z up (documented sign); rooms are not "
            "placed in a property frame",
        },
        "rooms": [r.model_dump(mode="json") for r in model.rooms],
        "walls": [w.model_dump(mode="json") for w in model.walls],
        "surfaces": [s.model_dump(mode="json") for s in model.surfaces],
        "edge_evidence": {
            h.room_id: [
                {
                    "start": e.start,
                    "end": e.end,
                    "length_m": e.length_m,
                    "supported_share": e.supported_share,
                    "wall_ids": list(e.wall_ids),
                    "status": e.status,
                    "unobserved_spans_m": [list(gap) for gap in e.gaps],
                }
                for e in h.edges
            ]
            for h in model.hypotheses
        },
        "diagnostics": model.diagnostics,
        "limitations": [
            "Wall faces end at the observed support top when no ceiling was "
            "observed over their room; that is not a floor-to-ceiling height.",
            "A floor needs walls standing on it; an up-facing level without that "
            "structural support is reported as ambiguous, not as a floor.",
            "Tracking components (split at possible resets) are built separately "
            "and never combined into one room.",
            "Rooms share the session world W and are not drift-corrected or "
            "placed in a property frame.",
            "Unknown outline edges are closed by the observed floor extent; they "
            "are not walls and may be openings or unobserved boundaries.",
            "Openings are not detected yet.",
        ],
    }


def write_rooms(directory: Path, model: RoomModel) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rooms.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(model_to_json(model), indent=2, allow_nan=False) + "\n")
    return path
