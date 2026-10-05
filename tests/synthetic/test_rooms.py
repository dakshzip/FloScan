"""Room hypotheses from planes (P08).

Scenes are built from points sampled on analytic surfaces (walls, floor,
ceiling, furniture), so the expected outlines are known. A fixture verifies
the geometry logic; it never establishes real-world accuracy.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from floscan.capture import stray
from floscan.contracts.geometry import Room, Surface, Wall
from floscan.geometry.rooms import (
    ObservedPlane,
    RoomConfig,
    SceneEvidence,
    build_rooms,
    load_bundle,
    model_to_json,
)
from floscan.reconstruction import lidar

SPACING = 0.04
CAMERA_Z = 1.4
SAMPLE = Path(__file__).resolve().parents[2] / "example input " / "c00a170fe1"


def _plane(plane_id: str, points: np.ndarray, normal) -> ObservedPlane:
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    return ObservedPlane(plane_id, normal, float(-normal @ points.mean(axis=0)), points)


def wall(plane_id: str, a, b, inside, top: float = 2.5, bottom: float = 0.0, noise=0.0):
    """Points on the vertical face a -> b, normal toward ``inside``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = np.linalg.norm(b - a)
    s = np.arange(SPACING / 2, length, SPACING) / length
    z = np.arange(bottom + SPACING / 2, top, SPACING)
    ss, zz = np.meshgrid(s, z)
    xy = a + np.outer(ss.ravel(), b - a)
    points = np.column_stack([xy, zz.ravel()])
    normal = np.array([-(b - a)[1], (b - a)[0], 0.0])
    if normal[:2] @ (np.asarray(inside) - a) < 0:
        normal = -normal
    if noise:
        points = points + np.random.default_rng(1).normal(0, noise, points.shape) * [
            *normal[:2] / np.linalg.norm(normal),
            0,
        ]
    return _plane(plane_id, points, normal)


def horizontal(plane_id: str, polygon: Polygon, z: float, facing: str, minus=None):
    """Points on a horizontal polygon (optionally minus another area)."""
    minx, miny, maxx, maxy = polygon.bounds
    xs, ys = np.meshgrid(
        np.arange(minx + SPACING / 2, maxx, SPACING),
        np.arange(miny + SPACING / 2, maxy, SPACING),
    )
    candidates = np.column_stack([xs.ravel(), ys.ravel()])
    inside = np.array([polygon.contains(Point(p)) for p in candidates])
    if minus is not None:
        inside &= ~np.array([minus.contains(Point(p)) for p in candidates])
    xy = candidates[inside]
    points = np.column_stack([xy, np.full(len(xy), z)])
    return _plane(plane_id, points, [0, 0, 1 if facing == "up" else -1])


def room_walls(corners, inside, skip=(), prefix="w", **kwargs):
    corners = [np.asarray(c, float) for c in corners]
    return [
        wall(
            f"{prefix}{k}",
            corners[k],
            corners[(k + 1) % len(corners)],
            inside,
            **kwargs,
        )
        for k in range(len(corners))
        if k not in skip
    ]


def scene(planes, cameras) -> SceneEvidence:
    cameras = np.asarray([[x, y, CAMERA_Z] for x, y in cameras], dtype=float)
    return SceneEvidence(planes=planes, cameras=cameras, source="synthetic")


def rectangle_scene(ceiling: bool = True, skip=(), noise=0.0) -> SceneEvidence:
    corners = [(0, 0), (4, 0), (4, 3), (0, 3)]
    footprint = Polygon(corners)
    planes = room_walls(corners, (2, 1.5), skip=skip, noise=noise)
    planes.append(horizontal("floor", footprint, 0.0, "up"))
    if ceiling:
        planes.append(horizontal("ceiling", footprint, 2.5, "down"))
    return scene(planes, [(2, 1.5), (1, 1), (3, 2)])


def _single_room(model):
    assert len(model.hypotheses) == 1, model.diagnostics
    return model.hypotheses[0]


def _same_shape(outline: Polygon, expected: Polygon, tolerance: float = 0.03) -> None:
    assert outline.symmetric_difference(expected).area < tolerance * expected.length
    assert outline.hausdorff_distance(expected) < tolerance


# --------------------------------------------------------------------------
# Known outlines
# --------------------------------------------------------------------------


def test_rectangle_is_recovered_with_every_edge_observed() -> None:
    model = build_rooms(rectangle_scene())
    room = _single_room(model)
    _same_shape(room.outline, Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]))
    assert len(room.outline.exterior.coords) - 1 == 4
    assert all(edge.status == "observed_wall" for edge in room.edges)
    assert all(edge.gaps == () for edge in room.edges)
    assert room.status == "ok" and model.status == "ok"
    assert model.diagnostics["floor"]["height_m"] == pytest.approx(0.0)
    assert model.diagnostics["ceiling"]["height_m"] == pytest.approx(2.5)


def test_concave_l_room_keeps_its_inner_corner() -> None:
    corners = [(0, 0), (5, 0), (5, 2), (2, 2), (2, 4), (0, 4)]
    footprint = Polygon(corners)
    planes = room_walls(corners, (1, 1))
    planes.append(horizontal("floor", footprint, 0.0, "up"))
    model = build_rooms(scene(planes, [(1, 1), (4, 1), (1, 3)]))
    room = _single_room(model)
    _same_shape(room.outline, footprint)
    assert len(room.outline.exterior.coords) - 1 == 6
    assert room.outline.area == pytest.approx(footprint.area, rel=0.01)


def test_diagonal_wall_is_not_squared_off() -> None:
    corners = [(0, 0), (4, 0), (4, 2), (2, 4), (0, 4)]
    footprint = Polygon(corners)
    planes = room_walls(corners, (1.5, 1.5))
    planes.append(horizontal("floor", footprint, 0.0, "up"))
    model = build_rooms(scene(planes, [(1.5, 1.5), (3, 1), (1, 3)]))
    room = _single_room(model)
    _same_shape(room.outline, footprint)
    diagonal = [e for e in room.edges if abs(abs(e.start[0] - e.end[0]) - 2) < 0.05]
    assert diagonal and diagonal[0].status == "observed_wall"


def test_pillar_becomes_a_hole() -> None:
    outer = [(0, 0), (6, 0), (6, 5), (0, 5)]
    pillar = [(2.6, 2.1), (3.4, 2.1), (3.4, 2.9), (2.6, 2.9)]
    footprint = Polygon(outer, [pillar])
    planes = room_walls(outer, (1, 1))
    # Pillar faces look outward, toward the room.
    pillar_points = [np.asarray(p, float) for p in pillar]
    for k in range(4):
        a, b = pillar_points[k], pillar_points[(k + 1) % 4]
        middle = (a + b) / 2
        outward = middle + (middle - np.array([3.0, 2.5])) * 2
        planes.append(wall(f"p{k}", a, b, outward))
    planes.append(horizontal("floor", Polygon(outer), 0.0, "up", minus=Polygon(pillar)))
    model = build_rooms(scene(planes, [(1, 1), (5, 4), (1, 4), (5, 1)]))
    room = _single_room(model)
    assert len(room.outline.interiors) == 1
    hole = Polygon(room.outline.interiors[0])
    _same_shape(hole, Polygon(pillar))
    assert room.outline.area == pytest.approx(footprint.area, rel=0.02)
    record = Room.model_validate(model.rooms[0].model_dump())
    assert len(record.boundary.holes) == 1


# --------------------------------------------------------------------------
# Missing evidence stays visible
# --------------------------------------------------------------------------


def test_missing_wall_leaves_an_unknown_edge() -> None:
    model = build_rooms(rectangle_scene(skip=(1,)))  # the x = 4 wall unseen
    room = _single_room(model)
    unknown = [e for e in room.edges if e.status == "unknown"]
    assert len(unknown) == 1
    edge = unknown[0]
    assert abs(edge.start[0] - 4) < 0.08 and abs(edge.end[0] - 4) < 0.08
    assert room.status == "partial"
    assert len(model.walls) == 3  # no wall record for the unseen side
    assert "no observed wall" in model.rooms[0].status_reason


def test_absent_ceiling_is_not_assumed() -> None:
    model = build_rooms(rectangle_scene(ceiling=False))
    assert model.diagnostics["ceiling"]["chosen"] is False
    assert model.status == "partial"
    assert any("ceiling absent" in reason for reason in model.reasons)
    assert all(not wall.height_profile for wall in model.walls)
    assert not [s for s in model.surfaces if s.kind == "ceiling"]
    assert model.rooms[0].ceiling_surface_ids == []


def test_no_floor_is_insufficient_evidence() -> None:
    corners = [(0, 0), (4, 0), (4, 3), (0, 3)]
    model = build_rooms(scene(room_walls(corners, (2, 1.5)), [(2, 1.5)]))
    assert model.status == "insufficient_evidence"
    assert model.rooms == [] and model.walls == []
    assert "floor not found" in model.reasons[0]


def test_table_is_not_the_floor() -> None:
    corners = [(0, 0), (4, 0), (4, 3), (0, 3)]
    table = Polygon([(1, 1), (3.5, 1), (3.5, 2.5), (1, 2.5)])
    planes = room_walls(corners, (2, 1.5))
    # The floor is mostly hidden under the larger table top.
    planes.append(horizontal("floor", Polygon(corners), 0.0, "up", minus=table))
    planes.append(horizontal("table", table, 0.75, "up"))
    for k, (a, b) in enumerate(
        zip(table.exterior.coords[:-1], table.exterior.coords[1:], strict=True)
    ):
        planes.append(wall(f"table_side{k}", a, b, (0.5, 0.5), top=0.75, bottom=0.0))
    model = build_rooms(scene(planes, [(0.5, 0.5), (3.7, 2.7)]))
    assert model.diagnostics["floor"]["height_m"] == pytest.approx(0.0)
    assert model.diagnostics["floor"]["plane_ids"] == ["floor"]
    assert [lv["plane_ids"] for lv in model.diagnostics["furniture_levels"]] == [
        ["table"]
    ]
    assert model.diagnostics["walls"]["rejected_vertical_planes"]["too_low"] == 4
    room = _single_room(model)
    _same_shape(room.outline, Polygon(corners))  # the table footprint is room


def test_two_rooms_split_at_a_doorway() -> None:
    planes = room_walls(
        [(0, 0), (4, 0), (4, 3), (0, 3)], (2, 1.5), skip=(1,), prefix="a"
    )
    planes += room_walls(
        [(4.1, 0), (8, 0), (8, 3), (4.1, 3)], (6, 1.5), skip=(3,), prefix="b"
    )
    # Partition faces on both sides, with a 0.8 m doorway gap.
    planes.append(wall("a_part_low", (4, 0), (4, 1.1), (2, 1.5)))
    planes.append(wall("a_part_high", (4, 1.9), (4, 3), (2, 1.5)))
    planes.append(wall("b_part_low", (4.1, 0), (4.1, 1.1), (6, 1.5)))
    planes.append(wall("b_part_high", (4.1, 1.9), (4.1, 3), (6, 1.5)))
    floor = (
        Polygon([(0, 0), (8, 0), (8, 3), (0, 3)])
        .difference(Polygon([(4, 0), (4.1, 0), (4.1, 1.1), (4, 1.1)]))
        .difference(Polygon([(4, 1.9), (4.1, 1.9), (4.1, 3), (4, 3)]))
    )
    planes.append(horizontal("floor", floor, 0.0, "up"))
    model = build_rooms(scene(planes, [(2, 1.5), (6, 1.5), (4.05, 1.5)]))
    assert len(model.hypotheses) == 2
    first, second = (h.outline for h in model.hypotheses)
    assert first.intersection(second).area < 1e-6  # rooms never share floor
    areas = sorted(h.outline.area for h in model.hypotheses)
    assert areas[0] == pytest.approx(12.0, rel=0.05)
    assert areas[1] == pytest.approx(11.7, rel=0.05)
    for hypothesis in model.hypotheses:
        # The doorway is an unobserved span inside the observed partition.
        assert hypothesis.status == "partial"
        gaps = [gap for edge in hypothesis.edges for gap in edge.gaps]
        assert len(gaps) == 1
        assert gaps[0][1] - gaps[0][0] == pytest.approx(0.8, abs=0.15)
        assert "unobserved span" in "; ".join(hypothesis.reasons)


def test_orthogonal_snap_is_optional_and_soft() -> None:
    tilt = math.radians(1.0)
    rotation = np.array(
        [[math.cos(tilt), -math.sin(tilt)], [math.sin(tilt), math.cos(tilt)]]
    )
    square = [(0, 0), (4, 0), (4, 3), (0, 3)]
    corners = [tuple(rotation @ np.asarray(c, float)) for c in square[:2]] + square[2:]
    # Two walls tilted by 1 degree, two exact.
    planes = room_walls(corners, (2, 1.5))
    planes.append(horizontal("floor", Polygon(corners), 0.0, "up"))
    evidence = scene(planes, [(2, 1.5)])
    plain = _single_room(build_rooms(evidence)).outline
    snapped = _single_room(
        build_rooms(evidence, RoomConfig(snap_orthogonal_deg=2.0))
    ).outline

    def edge_angles(polygon):
        coords = np.asarray(polygon.exterior.coords)
        d = np.diff(coords, axis=0)
        return np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90

    # Snapping aligns walls to their own dominant perpendicular family, not
    # to the world axes; unsnapped, the 1 degree disagreement remains.
    def spread(polygon):
        angles = edge_angles(polygon)
        return float(np.ptp(np.minimum(angles, 90 - angles)))

    assert spread(snapped) < 0.05
    assert spread(plain) > 0.5


def test_noisy_walls_still_give_the_outline() -> None:
    model = build_rooms(rectangle_scene(noise=0.005))
    _same_shape(_single_room(model).outline, Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]))


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def test_records_are_valid_and_surfaces_face_into_the_room() -> None:
    model = build_rooms(rectangle_scene())
    centre = np.array([2.0, 1.5, 1.2])
    kinds = {s.kind for s in model.surfaces}
    assert kinds == {"floor", "ceiling", "wall"}
    for surface in model.surfaces:
        Surface.model_validate(surface.model_dump())
        origin = np.asarray(surface.origin)
        normal = np.asarray(surface.normal)
        if surface.kind == "wall":
            assert (centre - origin)[:2] @ normal[:2] > 0
        else:
            assert normal[2] == (1.0 if surface.kind == "floor" else -1.0)
        assert Polygon(surface.boundary_uv.outer).area > 0
    for record in model.walls:
        Wall.model_validate(record.model_dump())
        assert record.height_profile[0].height_m == pytest.approx(2.5)
    room = Room.model_validate(model.rooms[0].model_dump())
    assert room.placement_status == "unplaced" and room.local_frame_id == "W"
    assert room.status == "ok" and room.coverage == pytest.approx(1.0, abs=0.02)
    data = model_to_json(model)
    assert data["edge_evidence"]["room:01"][0]["status"] == "observed_wall"


# --------------------------------------------------------------------------
# Supplied sample: honest partial output (skipped where absent)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="sample session not present")
def test_real_sample_gives_partial_rooms_without_a_ceiling(tmp_path: Path) -> None:
    inspection = stray.inspect_session(stray.open_session(SAMPLE))
    lidar.reconstruct(inspection.session, inspection, tmp_path / "r")
    model = build_rooms(load_bundle(tmp_path / "r"))
    assert model.status in ("partial", "insufficient_evidence")
    assert model.diagnostics["ceiling"]["chosen"] is False
    assert model.diagnostics["floor"]["chosen"] is True
    assert model.rooms, model.reasons
    outlines = []
    for room in model.rooms:
        Room.model_validate(room.model_dump())
        assert room.status in ("ok", "partial")
        outlines.append(Polygon(room.boundary.outer, room.boundary.holes))
    for k, a in enumerate(outlines):
        for b in outlines[k + 1 :]:
            assert a.intersection(b).area < 1e-4  # float noise only
