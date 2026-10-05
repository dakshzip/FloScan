"""Measurement definitions and evaluation (P09): analytic values and honesty."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from floscan.contracts.base import Provenance
from floscan.geometry.rooms import ObservedPlane, SceneEvidence, build_rooms
from floscan.measurements.area import polygon_area, ring_area
from floscan.measurements.definitions import DEFINITIONS, definition_for
from floscan.measurements.engine import NO_CALIBRATION, measure
from floscan.measurements.linear import plan_extents, plan_length, vertical_gap

ROOMS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "synthetic" / "test_rooms.py")
)
PROVENANCE = Provenance(
    stage="measurements.evaluate", stage_version="test", mode="live"
)


def _measured(evidence: SceneEvidence):
    model = build_rooms(evidence)
    return measure(model.rooms, model.walls, model.surfaces, PROVENANCE)


def _value(measured, subject_suffix: str, quantity: str):
    return next(
        m
        for m in measured.measurements
        if m.quantity == quantity and m.subject_id.endswith(subject_suffix)
    )


# --------------------------------------------------------------------------
# Analytic quantities
# --------------------------------------------------------------------------


def test_area_with_a_hole_is_exact_and_orientation_free() -> None:
    outer = [(0, 0), (4, 0), (4, 3), (0, 3)]
    hole = [(1, 1), (1, 2), (2, 2), (2, 1)]
    assert polygon_area(outer, [hole]) == pytest.approx(11.0, abs=1e-12)
    assert polygon_area(outer[::-1], [hole[::-1]]) == pytest.approx(11.0, abs=1e-12)
    assert ring_area(outer) == pytest.approx(12.0) and ring_area(outer[::-1]) < 0
    assert polygon_area(outer, [hole]) == pytest.approx(
        Polygon(outer, [hole]).area, abs=1e-12
    )


def test_lengths_extents_and_heights_are_exact() -> None:
    assert plan_length((0, 0, 5.0), (3, 4, -2.0)) == pytest.approx(5.0, abs=1e-12)
    angle = math.radians(30)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    rectangle = Polygon([tuple(rotation @ p) for p in [(0, 0), (4, 0), (4, 3), (0, 3)]])
    longer, shorter = plan_extents(rectangle)
    assert (longer, shorter) == (pytest.approx(4.0), pytest.approx(3.0))
    tilt = np.array([-0.05, 0.0, 1.0]) / math.hypot(0.05, 1.0)
    floor = (tilt, 0.0)  # z = 0.05 x
    ceiling = (-tilt, 2.5 * tilt[2])  # z = 2.5 + 0.05 x
    assert vertical_gap(floor, ceiling, (3.0, 1.0)) == pytest.approx(2.5, abs=1e-12)


def test_values_are_never_rounded() -> None:
    width = 4.123456789
    corners = [(0, 0), (width, 0), (width, 3), (0, 3)]
    planes = ROOMS["room_walls"](corners, (2, 1.5))
    planes.append(ROOMS["horizontal"]("floor", Polygon(corners), 0.0, "up"))
    planes.append(ROOMS["horizontal"]("ceiling", Polygon(corners), 2.5, "down"))
    measured = _measured(ROOMS["scene"](planes, [(2, 1.5)]))
    lengths = sorted(
        m.value for m in measured.measurements if m.quantity == "wall_length"
    )
    # The outline comes from fitted wall planes; within fit tolerance, and not
    # rounded to centimetres.
    assert lengths[-1] == pytest.approx(width, abs=0.005)
    assert lengths[-1] != round(lengths[-1], 2)


# --------------------------------------------------------------------------
# Transform invariance
# --------------------------------------------------------------------------


def _moved(evidence: SceneEvidence, angle_deg: float, shift) -> SceneEvidence:
    angle = math.radians(angle_deg)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shift = np.asarray(shift, dtype=float)
    planes = []
    for plane in evidence.planes:
        points = plane.points @ rotation.T + shift
        normal = rotation @ plane.normal
        planes.append(
            ObservedPlane(
                plane.plane_id, normal, float(-normal @ points.mean(0)), points
            )
        )
    return SceneEvidence(planes, evidence.cameras @ rotation.T + shift, "moved")


def test_measurements_do_not_depend_on_the_frame() -> None:
    base = ROOMS["rectangle_scene"]()
    reference = {
        (m.quantity, m.subject_id): m.value for m in _measured(base).measurements
    }
    moved = _measured(_moved(base, 37.0, (5.0, -2.0, 0.0)))
    values = {(m.quantity, m.subject_id): m.value for m in moved.measurements}
    assert sorted(k[0] for k in values) == sorted(k[0] for k in reference)
    for quantity in ("floor_area", "room_length", "room_width", "ceiling_height"):
        assert values[(quantity, "room:01")] == pytest.approx(
            reference[(quantity, "room:01")], abs=1e-3
        )
    assert sorted(v for (q, _), v in values.items() if q == "wall_length") == (
        pytest.approx(
            sorted(v for (q, _), v in reference.items() if q == "wall_length"),
            abs=1e-3,
        )
    )


# --------------------------------------------------------------------------
# Units, intervals and availability
# --------------------------------------------------------------------------


def test_every_measurement_has_its_definition_unit_and_an_honest_interval() -> None:
    measured = _measured(ROOMS["rectangle_scene"](skip=(1,), ceiling=False))
    assert measured.measurements
    for m in measured.measurements:
        definition = DEFINITIONS[m.definition_id]
        assert definition.quantity == m.quantity
        assert m.unit == definition.unit == m.interval.unit
        assert m.interval.status == "unavailable"
        assert m.interval.lower is None and m.interval.upper is None
        assert m.interval.reason == NO_CALIBRATION
        if m.value is None:
            assert m.quality_status == "unavailable" and m.unavailable_reason
            assert m.status == "unavailable"
        else:
            assert m.status == "partial" and m.status_reason == NO_CALIBRATION


def test_missing_evidence_is_unavailable_or_degraded_never_guessed() -> None:
    measured = _measured(ROOMS["rectangle_scene"](skip=(1,), ceiling=False))
    ceiling = _value(measured, "room:01", "ceiling_height")
    assert ceiling.value is None
    assert "no ceiling was observed" in ceiling.unavailable_reason
    assert all(
        m.value is None for m in measured.measurements if m.quantity == "wall_height"
    )
    area = _value(measured, "room:01", "floor_area")
    assert area.quality_status == "degraded"
    assert "room is partial" in area.unavailable_reason
    walls = [m for m in measured.measurements if m.quantity == "wall_length"]
    degraded = [m for m in walls if m.quality_status == "degraded"]
    assert len(degraded) == 2  # the two walls meeting the unseen side
    assert all("not observed" in m.unavailable_reason for m in degraded)


def test_complete_room_measures_ok_and_links_records() -> None:
    measured = _measured(ROOMS["rectangle_scene"]())
    assert all(m.quality_status == "ok" for m in measured.measurements)
    assert _value(measured, "room:01", "floor_area").value == pytest.approx(
        12.0, abs=0.01
    )
    assert _value(measured, "room:01", "ceiling_height").value == pytest.approx(2.5)
    room = measured.rooms[0]
    assert set(room.measurement_ids) == {m.id for m in measured.measurements}
    for wall in measured.walls:
        assert {q.split(":")[-1] for q in wall.measurement_ids} == {
            "wall_length",
            "wall_height",
        }


def test_definition_lookup() -> None:
    assert definition_for("floor_area").unit == "m2"
    with pytest.raises(KeyError):
        definition_for("damage_area")
