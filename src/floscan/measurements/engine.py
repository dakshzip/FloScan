"""Measurement records for rooms and walls, with honest availability.

Each quantity is computed by one definition (``definitions.py``) from the
record geometry, in fixed units and never rounded. No empirical error
calibration exists yet, so every interval is ``unavailable`` with that
reason: no interval is invented. A value whose evidence is incomplete is
``degraded`` with the reason; a quantity without evidence (for example a
ceiling height where no ceiling was observed) has no value and is
``unavailable``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from floscan.contracts.base import Provenance
from floscan.contracts.geometry import Room, Surface, Wall
from floscan.contracts.inspection import ConfidenceInterval, Measurement
from floscan.measurements.area import polygon_area
from floscan.measurements.definitions import definition_for
from floscan.measurements.linear import (
    mean_profile_height,
    plan_extents,
    plan_length,
    vertical_gap,
)

STAGE = "measurements.evaluate"
NO_CALIBRATION = (
    "no empirical error calibration exists yet (it needs surveyed ground truth "
    "and a calibration split); no interval is estimated"
)


@dataclass
class Measured:
    measurements: list[Measurement]
    rooms: list[Room]
    walls: list[Wall]
    summary: dict[str, Any]


def _interval(quantity: str, unit: str) -> ConfidenceInterval:
    return ConfidenceInterval(
        lower=None,
        upper=None,
        unit=unit,
        nominal_level=0.9,
        quantity_family=quantity,
        method="none",
        support_n=0,
        independent_unit="property",
        coverage_scope="marginal",
        status="unavailable",
        reason=NO_CALIBRATION,
    )


def _measurement(
    subject_id: str,
    quantity: str,
    value: float | None,
    quality: str,
    method: str,
    evidence_ids: list[str],
    provenance: Provenance,
    reason: str | None = None,
) -> Measurement:
    definition = definition_for(quantity)
    # A value without an interval is a partial record (the contract requires
    # it); a missing value is an unavailable one.
    record_status = "partial" if value is not None else "unavailable"
    return Measurement(
        id=f"m:{subject_id}:{quantity}",
        provenance=provenance,
        status=record_status,
        status_reason=NO_CALIBRATION if value is not None else reason,
        subject_id=subject_id,
        subject_geometry_version=1,
        quantity=quantity,
        definition_id=definition.id,
        value=value,
        unit=definition.unit,
        interval=_interval(quantity, definition.unit),
        method=method,
        evidence_ids=evidence_ids,
        quality_status=quality,
        unavailable_reason=reason,
    )


def _edge_walls(room: Room, walls: dict[str, Wall]) -> dict[int, str]:
    """Outline edge index (exterior then holes, in order) -> wall id."""
    rings = [room.boundary.outer, *room.boundary.holes]
    edges = []
    for ring in rings:
        for k in range(len(ring)):
            edges.append((ring[k], ring[(k + 1) % len(ring)], len(ring), k))
    mapping = {}
    for wall_id in room.wall_ids:
        start, end = (np.asarray(p[:2]) for p in walls[wall_id].baseline)
        for index, (a, b, _, _) in enumerate(edges):
            if np.allclose(a, start, atol=1e-6) and np.allclose(b, end, atol=1e-6):
                mapping[index] = wall_id
                break
    return mapping


def _neighbours(room: Room) -> list[tuple[int, int]]:
    """For each outline edge, its previous and next edge in the same ring."""
    out, base = [], 0
    for ring in [room.boundary.outer, *room.boundary.holes]:
        n = len(ring)
        for k in range(n):
            out.append((base + (k - 1) % n, base + (k + 1) % n))
        base += n
    return out


def measure(
    rooms: list[Room],
    walls: list[Wall],
    surfaces: list[Surface],
    provenance: Provenance,
) -> Measured:
    """Measurements of every room and wall; returns rooms/walls linked to them."""
    by_wall = {w.id: w for w in walls}
    by_surface = {s.id: s for s in surfaces}
    measurements: list[Measurement] = []
    room_links: dict[str, list[str]] = {}
    wall_links: dict[str, list[str]] = {}
    counts = {"ok": 0, "degraded": 0, "unavailable": 0}

    def add(m: Measurement, room_id: str, wall_id: str | None = None) -> None:
        measurements.append(m)
        counts[m.quality_status] += 1
        room_links.setdefault(room_id, []).append(m.id)
        if wall_id is not None:
            wall_links.setdefault(wall_id, []).append(m.id)

    for room in rooms:
        outline = Polygon(room.boundary.outer, room.boundary.holes)
        partial = room.status != "ok"
        quality = "degraded" if partial else "ok"
        why = f"room is partial: {room.status_reason}" if partial else None
        area = polygon_area(room.boundary.outer, room.boundary.holes)
        add(
            _measurement(
                room.id, "floor_area", area, quality,
                "plan area of the room outline (shoelace, holes excluded)",
                room.floor_surface_ids, provenance, why,
            ),
            room.id,
        )  # fmt: skip
        longer, shorter = plan_extents(outline)
        for quantity, value in (("room_length", longer), ("room_width", shorter)):
            add(
                _measurement(
                    room.id, quantity, value, quality,
                    "minimum-area rectangle around the room outline",
                    room.floor_surface_ids, provenance, why,
                ),
                room.id,
            )  # fmt: skip
        floor = by_surface[room.floor_surface_ids[0]]
        centroid = np.asarray(outline.centroid.coords[0])
        floor_plane = (floor.normal, -float(np.dot(floor.normal, floor.origin)))
        if room.ceiling_surface_ids:
            ceiling = by_surface[room.ceiling_surface_ids[0]]
            ceiling_plane = (
                ceiling.normal,
                -float(np.dot(ceiling.normal, ceiling.origin)),
            )
            height = vertical_gap(floor_plane, ceiling_plane, centroid)
            add(
                _measurement(
                    room.id, "ceiling_height", height, "ok",
                    "floor plane to ceiling plane above the outline centroid",
                    [floor.id, ceiling.id], provenance,
                ),
                room.id,
            )  # fmt: skip
        else:
            add(
                _measurement(
                    room.id, "ceiling_height", None, "unavailable",
                    "floor plane to ceiling plane above the outline centroid",
                    [floor.id], provenance,
                    "no ceiling was observed over this room",
                ),
                room.id,
            )  # fmt: skip
        edge_walls = _edge_walls(room, by_wall)
        neighbours = _neighbours(room)
        for index, wall_id in edge_walls.items():
            wall = by_wall[wall_id]
            previous, following = neighbours[index]
            open_ends = [
                side
                for side, neighbour in (("start", previous), ("end", following))
                if neighbour not in edge_walls
            ]
            length = plan_length(*wall.baseline)
            reason = (
                f"the {' and '.join(open_ends)} of this wall meets an outline edge "
                "with no observed wall; the corner is not observed, so the length "
                "may be truncated"
                if open_ends
                else None
            )
            add(
                _measurement(
                    wall_id, "wall_length", length,
                    "degraded" if open_ends else "ok",
                    "plan length of the wall baseline between outline corners",
                    wall.boundary_evidence, provenance, reason,
                ),
                room.id, wall_id,
            )  # fmt: skip
            if wall.height_profile:
                height = mean_profile_height([s.height_m for s in wall.height_profile])
                add(
                    _measurement(
                        wall_id, "wall_height", height, "ok",
                        "mean of the floor-to-ceiling profile at the wall's ends",
                        wall.boundary_evidence, provenance,
                    ),
                    room.id, wall_id,
                )  # fmt: skip
            else:
                add(
                    _measurement(
                        wall_id, "wall_height", None, "unavailable",
                        "mean of the floor-to-ceiling profile at the wall's ends",
                        wall.boundary_evidence, provenance,
                        "no ceiling was observed over this wall's room",
                    ),
                    room.id, wall_id,
                )  # fmt: skip
    linked_rooms = [
        r.model_copy(update={"measurement_ids": room_links.get(r.id, [])})
        for r in rooms
    ]
    linked_walls = [
        w.model_copy(update={"measurement_ids": wall_links.get(w.id, [])})
        for w in walls
    ]
    return Measured(
        measurements,
        linked_rooms,
        linked_walls,
        {
            "measurements": len(measurements),
            "quality": counts,
            "intervals": "unavailable for every quantity: " + NO_CALIBRATION,
        },
    )
