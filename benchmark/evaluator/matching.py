"""Scoring views and frozen one-to-one matching of predictions to ground truth.

The scorer compares two ``PlanView`` documents: evaluator ground truth and one
capture's prediction. Matching is decided from geometry and labels only, never
from the lengths, widths or heights being scored, so it cannot select the
accurate predictions and drop the difficult ones.

Matching policy ``matching-v1`` (frozen before any real data is scored):

1. Rooms match one-to-one by normalised label. Duplicate predicted labels are
   resolved by aligned centroid distance; the rest are phantoms.
2. One property-wide rigid alignment (rotation and translation, no scale, no
   mirroring) maps prediction to ground truth, fitted on the centroids of
   label-matched rooms. With fewer than two pairs, wall-pair hypotheses are
   scored by how well all walls and opening centres line up. Rooms still
   unmatched by label then match by aligned polygon IoU >= 0.3.
3. Walls match within matched rooms by aligned midpoint distance and line
   orientation, gated at ``WALL_GATE_M`` and ``WALL_ANGLE_GATE_DEG``.
4. Openings match within matched rooms by aligned centre distance, gated at
   ``OPENING_GATE_M``; when both host walls are matched they must correspond.
5. An evaluator-authored correspondence file, if given, replaces automated
   matching for the kinds it lists.

Unmatched ground truth is "missing" and unmatched predictions are "phantom";
both stay in every denominator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from floscan.contracts.base import Contract, Id, NonNegative, Text, Tier, Vec2

MATCHING_POLICY = "matching-v1"
WALL_GATE_M = 0.75
WALL_ANGLE_GATE_DEG = 20.0
OPENING_GATE_M = 0.5
_UNMATCHABLE = 1e9


class Estimate(Contract):
    """A scalar with an optional predictive interval or reference uncertainty."""

    value: float
    lower: float | None = None
    upper: float | None = None
    nominal_level: float | None = Field(default=None, gt=0.0, lt=1.0)
    uncertainty: NonNegative | None = None  # ground-truth reading uncertainty

    @model_validator(mode="after")
    def _ordered(self) -> Estimate:
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("interval lower bound exceeds upper bound")
        return self


Placement = Literal["placed", "unplaced", "ambiguous"]
Registration = Literal[
    "connected", "disconnected", "ambiguous", "not_attempted", "failed"
]


class RoomView(Contract):
    """A room in the plan frame; unplaced rooms carry room-local coordinates.

    ``holes`` are interior rings (P04A); omitted means none. ``placement``
    (P04A) defaults to ``placed``, which is what pre-P04A views meant. An
    unplaced room's polygon, walls and openings are in its own local frame:
    they are scored for local dimensions only and never enter property-level
    footprint, overlap or adjacency.
    """

    id: Id
    label: Text
    polygon: list[Vec2] | None = Field(default=None, min_length=3)
    holes: list[list[Vec2]] = Field(default_factory=list)
    placement: Placement = "placed"
    ceiling_height: Estimate | None = None
    floor_area: Estimate | None = None

    @model_validator(mode="after")
    def _valid_shape(self) -> RoomView:
        if self.holes and self.polygon is None:
            raise ValueError(f"room {self.id}: holes need an outer polygon")
        if self.polygon is not None:
            shape = Polygon(self.polygon, self.holes)
            if not shape.is_valid or shape.area <= 0:
                raise ValueError(
                    f"room {self.id}: polygon with holes is not a valid simple shape"
                )
        return self

    def shape(self, alignment: RigidAlignment2D | None = None) -> Polygon:
        """The room polygon with holes, optionally moved by a rigid alignment."""
        if self.polygon is None:
            return Polygon()
        move = alignment.apply if alignment else np.asarray
        return Polygon(
            np.asarray(move(self.polygon)).tolist(),
            [np.asarray(move(h)).tolist() for h in self.holes],
        )


class WallView(Contract):
    id: Id
    room_id: Id
    start: Vec2
    end: Vec2
    length: Estimate | None = None


class OpeningView(Contract):
    id: Id
    room_id: Id
    host_wall_id: Id | None = None
    center: Vec2
    width: Estimate | None = None


class PlanView(Contract):
    """Plan-level scoring view, metres.

    Ground truth is one surveyed frame, so every ground-truth room is placed.
    For predictions, ``registration_status`` (P04A) states whether the rooms
    were stitched into one plan; whole-property gates pass only for an
    explicit ``connected`` plan with every room placed. Pre-P04A views without
    it still load but cannot pass those gates.
    """

    kind: Literal["ground_truth", "prediction"]
    case_id: Id
    capture_id: Id | None = None
    tier: Tier | None = None
    rooms: list[RoomView] = Field(default_factory=list)
    walls: list[WallView] = Field(default_factory=list)
    openings: list[OpeningView] = Field(default_factory=list)
    adjacency: list[list[Id]] = Field(default_factory=list)
    connected_components: int | None = Field(default=None, ge=0)
    registration_status: Registration | None = None

    @model_validator(mode="after")
    def _consistent(self) -> PlanView:
        if self.kind == "prediction" and (self.capture_id is None or self.tier is None):
            raise ValueError("a prediction needs capture_id and tier")
        for kind, records in (
            ("room", self.rooms),
            ("wall", self.walls),
            ("opening", self.openings),
        ):
            ids = [r.id for r in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {kind} ids")
        rooms = {r.id for r in self.rooms}
        walls = {w.id: w for w in self.walls}
        for wall in self.walls:
            if wall.room_id not in rooms:
                raise ValueError(f"wall {wall.id}: unknown room {wall.room_id}")
        for opening in self.openings:
            if opening.room_id not in rooms:
                raise ValueError(
                    f"opening {opening.id}: unknown room {opening.room_id}"
                )
            if opening.host_wall_id is not None:
                host = walls.get(opening.host_wall_id)
                if host is None or host.room_id != opening.room_id:
                    raise ValueError(f"opening {opening.id}: bad host wall")
        for pair in self.adjacency:
            if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= rooms:
                raise ValueError(f"adjacency {pair} must join two known rooms")
        if self.kind == "ground_truth":
            labels = [r.label.strip().lower() for r in self.rooms]
            if len(labels) != len(set(labels)):
                raise ValueError("ground-truth room labels must be unique")
            if any(r.placement != "placed" for r in self.rooms):
                raise ValueError("ground-truth rooms are surveyed in one frame")
        if self.registration_status == "connected":
            unplaced = [r.id for r in self.rooms if r.placement != "placed"]
            if unplaced:
                raise ValueError(f"a connected plan cannot contain unplaced {unplaced}")
            if self.connected_components not in (None, 1):
                raise ValueError("a connected plan has exactly one component")
        return self

    def placed_rooms(self) -> list[RoomView]:
        return [r for r in self.rooms if r.placement == "placed"]

    def stitched(self) -> bool:
        """True only for an explicitly connected plan with every room placed."""
        return (
            self.registration_status == "connected"
            and self.connected_components in (None, 1)
            and all(r.placement == "placed" for r in self.rooms)
        )


def load_plan(path: Path) -> PlanView:
    """Load a scoring view from JSON (NaN and Infinity are rejected)."""
    return PlanView.model_validate_json(path.read_text(encoding="utf-8"))


class Correspondence(Contract):
    """Evaluator-authored ground-truth -> prediction IDs, audited separately."""

    rooms: dict[str, str] = Field(default_factory=dict)
    walls: dict[str, str] = Field(default_factory=dict)
    openings: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_to_one(self) -> Correspondence:
        for kind in ("rooms", "walls", "openings"):
            values = list(getattr(self, kind).values())
            if len(values) != len(set(values)):
                raise ValueError(f"correspondence {kind} must be one-to-one")
        return self


# --------------------------------------------------------------------------
# Rigid alignment
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RigidAlignment2D:
    """``p_gt = R p_pred + t``: rotation and translation only, det(R) = +1."""

    rotation: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))
    translation: tuple[float, float] = (0.0, 0.0)
    method: str = "identity"
    pairs: int = 0

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.rotation[1][0], self.rotation[0][0]))

    def apply(self, points: Any) -> np.ndarray:
        xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return xy @ np.asarray(self.rotation).T + np.asarray(self.translation)


def fit_rigid_2d(pred: np.ndarray, gt: np.ndarray) -> RigidAlignment2D:
    """Least-squares rotation + translation (Kabsch, reflections excluded)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if len(pred) < 2:
        return RigidAlignment2D(
            method="identity (fewer than two room pairs)", pairs=len(pred)
        )
    mp, mg = pred.mean(axis=0), gt.mean(axis=0)
    u, _, vt = np.linalg.svd((pred - mp).T @ (gt - mg))
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, d]) @ u.T
    translation = mg - rotation @ mp
    return RigidAlignment2D(
        rotation=tuple(map(tuple, rotation.tolist())),  # type: ignore[arg-type]
        translation=(float(translation[0]), float(translation[1])),
        method="rigid fit on label-matched room centroids",
        pairs=len(pred),
    )


def _centroid(room: RoomView) -> np.ndarray | None:
    if room.polygon is None:
        return None
    c = room.shape().centroid
    return np.array([c.x, c.y])


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

Pair = tuple[str | None, str | None]


@dataclass(frozen=True)
class MatchResult:
    """One-to-one pairs per kind: (gt_id, pred_id); None marks missing/phantom."""

    rooms: list[Pair]
    walls: list[Pair]
    openings: list[Pair]
    alignment: RigidAlignment2D
    policy: str = MATCHING_POLICY
    notes: list[str] = field(default_factory=list)
    local_alignments: dict[str, RigidAlignment2D] = field(default_factory=dict)

    def matched(self, kind: str) -> dict[str, str]:
        return {g: p for g, p in getattr(self, kind) if g is not None and p is not None}


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    return [
        (r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] < _UNMATCHABLE
    ]


def _complete(
    gt_ids: list[str], pred_ids: list[str], matched: dict[str, str]
) -> list[Pair]:
    used = set(matched.values())
    pairs: list[Pair] = [(g, matched.get(g)) for g in gt_ids]
    pairs += [(None, p) for p in pred_ids if p not in used]
    return pairs


@dataclass(frozen=True)
class _Items:
    """Walls and openings that an alignment hypothesis is scored on."""

    walls: list[WallView]
    openings: list[OpeningView]


def _items(plan: PlanView, room_ids: set[str]) -> _Items:
    return _Items(
        [w for w in plan.walls if w.room_id in room_ids],
        [o for o in plan.openings if o.room_id in room_ids],
    )


def _line_angle(start: Any, end: Any) -> float:
    return math.atan2(end[1] - start[1], end[0] - start[0]) % math.pi


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return math.degrees(min(d, math.pi - d))


def _placement_cost(gt: _Items, pred: _Items, alignment: RigidAlignment2D) -> float:
    """How well walls and opening centres line up under ``alignment``.

    Each ground-truth wall contributes its gated best midpoint-and-angle
    residual and each opening its gated best centre distance; unmatched items
    pay the gate. Lengths and widths are never used.
    """
    cost = 0.0
    pred_walls = [(alignment.apply([w.start, w.end])) for w in pred.walls]
    for wall in gt.walls:
        g_mid = (np.asarray(wall.start) + np.asarray(wall.end)) / 2
        g_ang = _line_angle(wall.start, wall.end)
        best = WALL_GATE_M
        for ends in pred_walls:
            angle = _angle_diff_deg(g_ang, _line_angle(ends[0], ends[1]))
            if angle <= WALL_ANGLE_GATE_DEG:
                best = min(best, float(np.linalg.norm(ends.mean(axis=0) - g_mid)))
        cost += best
    centres = [alignment.apply(o.center)[0] for o in pred.openings]
    for opening in gt.openings:
        distances = [
            float(np.linalg.norm(c - np.asarray(opening.center))) for c in centres
        ]
        cost += min([OPENING_GATE_M, *distances])
    return cost


def _wall_hypothesis_alignment(
    gt: _Items, pred: _Items, method: str
) -> RigidAlignment2D | None:
    """Rigid alignment for plans with fewer than two label-matched rooms.

    Every (ground-truth wall, predicted wall, direction) triple proposes the
    rotation between the two lines and the translation between their
    midpoints; the proposal under which all walls and openings line up best
    wins. Ties keep the first proposal, so the result is deterministic.
    """
    best: tuple[float, RigidAlignment2D] | None = None
    for g in gt.walls:
        g_mid = (np.asarray(g.start) + np.asarray(g.end)) / 2
        g_dir = math.atan2(g.end[1] - g.start[1], g.end[0] - g.start[0])
        for w in pred.walls:
            p_mid = (np.asarray(w.start) + np.asarray(w.end)) / 2
            p_dir = math.atan2(w.end[1] - w.start[1], w.end[0] - w.start[0])
            for flip in (0.0, math.pi):
                theta = g_dir - p_dir + flip
                rotation = np.array(
                    [
                        [math.cos(theta), -math.sin(theta)],
                        [math.sin(theta), math.cos(theta)],
                    ]
                )
                translation = g_mid - rotation @ p_mid
                candidate = RigidAlignment2D(
                    rotation=tuple(map(tuple, rotation.tolist())),  # type: ignore[arg-type]
                    translation=(float(translation[0]), float(translation[1])),
                    method=method,
                    pairs=0,
                )
                cost = _placement_cost(gt, pred, candidate)
                if best is None or cost < best[0] - 1e-9:
                    best = (cost, candidate)
    return best[1] if best else None


def _rooms_by_overlap(
    gt: PlanView, pred: PlanView, matched: dict[str, str], alignment: RigidAlignment2D
) -> dict[str, str]:
    """Match rooms left unmatched by label using aligned polygon IoU >= 0.3."""
    used = set(matched.values())
    gts = [r for r in gt.rooms if r.id not in matched and r.polygon]
    preds = [r for r in pred.placed_rooms() if r.id not in used and r.polygon]
    if not gts or not preds:
        return {}
    cost = np.full((len(gts), len(preds)), _UNMATCHABLE)
    for i, g in enumerate(gts):
        gp = g.shape()
        for j, r in enumerate(preds):
            rp = r.shape(alignment)
            union = gp.union(rp).area
            iou = gp.intersection(rp).area / union if union else 0.0
            if iou >= 0.3:
                cost[i, j] = 1.0 - iou
    return {gts[i].id: preds[j].id for i, j in _assign(cost)}


def match_plans(
    gt: PlanView, pred: PlanView, correspondence: Correspondence | None = None
) -> MatchResult:
    """Match ``pred`` to ``gt`` under policy ``matching-v1``."""
    notes: list[str] = []
    gt_rooms = {r.id: r for r in gt.rooms}
    pred_rooms = {r.id: r for r in pred.rooms}

    # 1. Rooms by label (unique labels first; duplicates resolved after alignment).
    by_label: dict[str, list[str]] = {}
    for room in pred.rooms:
        by_label.setdefault(room.label.strip().lower(), []).append(room.id)
    room_match: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for room in gt.rooms:
        candidates = by_label.get(room.label.strip().lower(), [])
        if len(candidates) == 1:
            room_match[room.id] = candidates[0]
        elif len(candidates) > 1:
            duplicates[room.id] = candidates
            notes.append(f"room label {room.label!r} predicted {len(candidates)} times")
    if correspondence and correspondence.rooms:
        room_match = dict(correspondence.rooms)
        duplicates = {}
        notes.append("rooms matched by evaluator correspondence")

    # 2. One property-wide rigid alignment, from placed rooms only.
    placed = {r.id for r in pred.placed_rooms()}
    pairs = [
        (_centroid(pred_rooms[p]), _centroid(gt_rooms[g]))
        for g, p in room_match.items()
        if g in gt_rooms and p in pred_rooms and p in placed
    ]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    alignment = fit_rigid_2d(
        np.array([a for a, _ in pairs]).reshape(-1, 2),
        np.array([b for _, b in pairs]).reshape(-1, 2),
    )
    if alignment.pairs < 2 and placed:
        alignment = (
            _wall_hypothesis_alignment(
                _items(gt, set(gt_rooms)),
                _items(pred, placed),
                "rigid wall-pair hypothesis (fewer than two room pairs)",
            )
            or alignment
        )
    for gid, candidates in duplicates.items():
        target = _centroid(gt_rooms[gid])
        options = [
            (float(np.linalg.norm(alignment.apply(c)[0] - target)), pid)
            for pid in candidates
            if target is not None
            and pid in placed
            and (c := _centroid(pred_rooms[pid])) is not None
        ]
        if options:
            room_match[gid] = min(options)[1]
    if not (correspondence and correspondence.rooms):
        room_match.update(_rooms_by_overlap(gt, pred, room_match, alignment))
    room_pairs = _complete(list(gt_rooms), list(pred_rooms), room_match)

    # Unplaced rooms are matched in their own frame, for local dimensions only.
    local: dict[str, RigidAlignment2D] = {}
    for gid, pid in room_match.items():
        if pid in pred_rooms and pid not in placed:
            local[pid] = _wall_hypothesis_alignment(
                _items(gt, {gid}),
                _items(pred, {pid}),
                "room-local wall-pair hypothesis (room not placed)",
            ) or RigidAlignment2D(method="identity (unplaced room without walls)")
            notes.append(f"room {pid} is not placed: matched in its local frame only")

    def frame_of(pid: str) -> RigidAlignment2D:
        return local.get(pid, alignment)

    # 3. Walls within matched rooms, by aligned midpoint and orientation.
    if correspondence and correspondence.walls:
        wall_match = dict(correspondence.walls)
        notes.append("walls matched by evaluator correspondence")
    else:
        wall_match = {}
        for gid, pid in room_match.items():
            gws = [w for w in gt.walls if w.room_id == gid]
            pws = [w for w in pred.walls if w.room_id == pid]
            cost = np.full((len(gws), len(pws)), _UNMATCHABLE)
            for i, g in enumerate(gws):
                g_mid = (np.asarray(g.start) + np.asarray(g.end)) / 2
                g_ang = _line_angle(g.start, g.end)
                for j, p in enumerate(pws):
                    ends = frame_of(pid).apply([p.start, p.end])
                    distance = float(np.linalg.norm(ends.mean(axis=0) - g_mid))
                    angle = _angle_diff_deg(g_ang, _line_angle(ends[0], ends[1]))
                    if distance <= WALL_GATE_M and angle <= WALL_ANGLE_GATE_DEG:
                        cost[i, j] = distance + angle / WALL_ANGLE_GATE_DEG * 0.1
            for i, j in _assign(cost):
                wall_match[gws[i].id] = pws[j].id
    wall_pairs = _complete(
        [w.id for w in gt.walls], [w.id for w in pred.walls], wall_match
    )

    # 4. Openings within matched rooms, by aligned centre and host wall.
    if correspondence and correspondence.openings:
        opening_match = dict(correspondence.openings)
        notes.append("openings matched by evaluator correspondence")
    else:
        opening_match = {}
        for gid, pid in room_match.items():
            gos = [o for o in gt.openings if o.room_id == gid]
            pos = [o for o in pred.openings if o.room_id == pid]
            cost = np.full((len(gos), len(pos)), _UNMATCHABLE)
            for i, g in enumerate(gos):
                expected_host = (
                    wall_match.get(g.host_wall_id) if g.host_wall_id else None
                )
                for j, p in enumerate(pos):
                    if (
                        expected_host
                        and p.host_wall_id
                        and p.host_wall_id != expected_host
                    ):
                        continue
                    distance = float(
                        np.linalg.norm(
                            frame_of(pid).apply(p.center)[0] - np.asarray(g.center)
                        )
                    )
                    if distance <= OPENING_GATE_M:
                        cost[i, j] = distance
            for i, j in _assign(cost):
                opening_match[gos[i].id] = pos[j].id
    opening_pairs = _complete(
        [o.id for o in gt.openings], [o.id for o in pred.openings], opening_match
    )
    return MatchResult(
        room_pairs,
        wall_pairs,
        opening_pairs,
        alignment,
        notes=notes,
        local_alignments=local,
    )


# --------------------------------------------------------------------------
# Product output -> scoring view
# --------------------------------------------------------------------------


def plan_from_result(result: Any, case_id: str) -> PlanView:
    """Project a validated ``PropertyResult`` (internal-v0) into a scoring view.

    Placed rooms are mapped into the property frame through their placement.
    Unplaced rooms keep their local frame and are marked ``unplaced``, so the
    scorer uses them for local dimensions only. Holes are kept. Registration
    status comes from the property graph (``not_attempted`` without one).
    Values come only from measurements; an unavailable measurement becomes no
    estimate.
    """
    from floscan.geometry.frames import Points

    measurements = {m.id: m for m in result.measurements}
    surfaces = {s.id: s for s in result.surfaces}

    def estimate(mid: str | None) -> Estimate | None:
        if mid is None or mid not in measurements:
            return None
        m = measurements[mid]
        if m.value is None:
            return None
        return Estimate(
            value=m.value,
            lower=m.interval.lower,
            upper=m.interval.upper,
            nominal_level=m.interval.nominal_level,
        )

    def by_quantity(subject: str, quantity: str) -> Estimate | None:
        for m in result.measurements:
            if m.subject_id == subject and m.quantity == quantity:
                return estimate(m.id)
        return None

    rooms, walls, openings = [], [], []
    for room in result.rooms:
        transform = (
            room.T_property_from_room.to_math() if room.T_property_from_room else None
        )

        def to_property_xy(
            xyz: Any, frame: str | None = None, room=room, transform=transform
        ) -> list[float]:
            """Room-frame (default) or already-property-frame point -> property xy."""
            frame = frame or room.local_frame_id
            points = Points(
                np.asarray(xyz, dtype=np.float64).reshape(-1, 3), frame, "m"
            )
            if transform is not None and frame == transform.from_frame:
                points = transform.apply(points)
            return [float(v) for v in points.xyz[0, :2]]

        polygon, holes = None, []
        if room.boundary is not None:
            polygon = [to_property_xy([x, y, 0.0]) for x, y in room.boundary.outer]
            holes = [
                [to_property_xy([x, y, 0.0]) for x, y in hole]
                for hole in room.boundary.holes
            ]
        rooms.append(
            RoomView(
                id=room.id,
                label=room.label,
                polygon=polygon,
                holes=holes,
                placement=room.placement_status,
                ceiling_height=by_quantity(room.id, "ceiling_height"),
                floor_area=by_quantity(room.id, "floor_area"),
            )
        )
        for wall in result.walls:
            if wall.room_id != room.id:
                continue
            walls.append(
                WallView(
                    id=wall.id,
                    room_id=room.id,
                    start=to_property_xy(wall.baseline[0]),
                    end=to_property_xy(wall.baseline[1]),
                    length=by_quantity(wall.id, "wall_length"),
                )
            )
        for opening in result.openings:
            if opening.room_id != room.id:
                continue
            surface = surfaces[opening.surface_id]
            c = Polygon(opening.polygon_uv.outer).centroid
            xyz = (
                np.asarray(surface.origin)
                + c.x * np.asarray(surface.basis_u)
                + c.y * np.asarray(surface.basis_v)
            )
            openings.append(
                OpeningView(
                    id=opening.id,
                    room_id=room.id,
                    host_wall_id=opening.host_wall_id,
                    center=to_property_xy(xyz, surface.frame_id),
                    width=estimate(opening.width_measurement_id),
                )
            )
    graph = result.property_graph
    adjacency = []
    if graph is not None:
        adjacency = [list(c.room_ids) for c in graph.connectors]
    return PlanView(
        kind="prediction",
        case_id=case_id,
        capture_id=result.capture_id or result.run.id,
        tier=result.run.tier,
        rooms=rooms,
        walls=walls,
        openings=openings,
        adjacency=adjacency,
        connected_components=len(graph.components) if graph is not None else None,
        registration_status=(
            graph.registration_status if graph is not None else "not_attempted"
        ),
    )


def load_json(path: Path) -> Any:
    def reject(name: str) -> None:
        raise ValueError(f"non-finite JSON number {name}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
