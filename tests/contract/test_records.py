"""Contract tests for the internal records (schemas internal-v0, capture-v0)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from floscan.contracts.capture import Capture
from floscan.contracts.geometry import (
    DepthFrame,
    Plane,
    Polygon2D,
    Pose,
    PropertyGraph,
    RigidTransformRecord,
    ScaleEvidence,
)
from floscan.contracts.inspection import ConcealedDamageFlag, Measurement
from floscan.contracts.result import (
    SECTIONS,
    MetricReconstruction,
    PropertyResult,
    ReconstructionBundle,
    schema_documents,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = {"stage": "test.fixture", "stage_version": "0", "mode": "live"}
HASH = "ab" * 32


def _mutate(data: Any, path: tuple[Any, ...], value: Any) -> Any:
    data = copy.deepcopy(data)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return data


def _interval(lower: float, upper: float, unit: str = "m", **extra: Any) -> dict:
    return {
        "lower": lower,
        "upper": upper,
        "unit": unit,
        "nominal_level": 0.9,
        "quantity_family": "fixture",
        "method": "fixture",
        "support_n": 0,
        "independent_unit": "property",
        "coverage_scope": "marginal",
        "status": "provisional",
        **extra,
    }


def _measurement(
    mid: str, subject: str, quantity: str, value: float, unit: str = "m"
) -> dict:
    return {
        "id": mid,
        "provenance": PROVENANCE,
        "subject_id": subject,
        "subject_geometry_version": 1,
        "quantity": quantity,
        "definition_id": f"def.{quantity}",
        "value": value,
        "unit": unit,
        "interval": _interval(value - 0.05, value + 0.05, unit),
        "method": "fixture",
        "quality_status": "ok",
    }


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0, 0, 0, 1],
]


def _result() -> dict:
    """A small valid partial result (asymmetric room 3.1 x 4.7 m).

    Its coverage describes exactly what it contains: the room plan is partial
    because no ceiling height or floor area is measured.
    """
    coverage = {
        name: {"status": "unavailable", "reason": "fixture"} for name in SECTIONS
    }
    coverage["capture"] = {"status": "available"}
    coverage["coordinate_frames"] = {"status": "available"}
    coverage["per_room_plan"] = {
        "status": "partial",
        "reason": "no ceiling height or floor area measured",
    }
    coverage["measurements"] = {"status": "available"}
    return {
        "run": {
            "id": "run-1",
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "tier": "lidar",
            "mode": "live",
            "input_manifest_hashes": [HASH],
            "config_hash": HASH,
            "schema_versions": {"internal": "internal-v0"},
            "package_lock_hash": HASH,
            "platform": {"os": "test"},
            "deterministic": True,
            "started_utc": "2026-10-04T00:00:00Z",
        },
        "capture_id": "capture-1",
        "status": "partial",
        "status_reason": "fixture covers geometry only",
        "coverage": coverage,
        "coordinate_frames": [
            {"id": "P", "kind": "property", "unit": "m", "description": "property"},
            {"id": "R1", "kind": "room", "unit": "m", "description": "room 1"},
            {
                "id": "U",
                "kind": "sfm_world",
                "unit": "reconstruction_unit",
                "description": "unscaled SfM world",
            },
        ],
        "rooms": [
            {
                "id": "room-1",
                "provenance": PROVENANCE,
                "label": "living room",
                "level_id": "level-0",
                "local_frame_id": "R1",
                "T_property_from_room": {
                    "to_frame_id": "P",
                    "from_frame_id": "R1",
                    "matrix": IDENTITY,
                    "unit": "m",
                },
                "boundary": {"outer": [[0, 0], [3.1, 0], [3.1, 4.7], [0, 4.7]]},
                "wall_ids": ["wall-1"],
                "opening_ids": ["door-1"],
                "measurement_ids": ["m-wall"],
                "hypothesis_id": "h-1",
                "placement_status": "placed",
            }
        ],
        "surfaces": [
            {
                "id": "surface-1",
                "provenance": PROVENANCE,
                "room_id": "room-1",
                "kind": "wall",
                "geometry_version": 1,
                "frame_id": "R1",
                "origin": [0.0, 0.0, 0.0],
                "basis_u": [1.0, 0.0, 0.0],
                "basis_v": [0.0, 0.0, 1.0],
                "normal": [0.0, -1.0, 0.0],
                "boundary_uv": {"outer": [[0, 0], [3.1, 0], [3.1, 2.45], [0, 2.45]]},
                "material": {"label": "painted plaster", "source": "fixture"},
            }
        ],
        "walls": [
            {
                "id": "wall-1",
                "provenance": PROVENANCE,
                "room_id": "room-1",
                "surface_id": "surface-1",
                "baseline": [[0.0, 0.0, 0.0], [3.1, 0.0, 0.0]],
                "opening_ids": ["door-1"],
                "measurement_ids": ["m-wall"],
            }
        ],
        "openings": [
            {
                "id": "door-1",
                "provenance": PROVENANCE,
                "room_id": "room-1",
                "host_wall_id": "wall-1",
                "surface_id": "surface-1",
                "opening_class": "door",
                "dimension_definition": "clear_aperture",
                "polygon_uv": {
                    "outer": [[1.0, 0], [1.86, 0], [1.86, 2.03], [1.0, 2.03]]
                },
                "width_measurement_id": "m-door",
                "detection_score": 0.9,
                "visibility_state": "visible",
            }
        ],
        "measurements": [
            _measurement("m-wall", "wall-1", "wall_length", 3.1),
            _measurement("m-door", "door-1", "opening_width", 0.86),
        ],
    }


def test_valid_result_round_trips_through_json() -> None:
    result = PropertyResult.model_validate(_result())
    text = result.to_json()
    assert "NaN" not in text and "Infinity" not in text
    again = PropertyResult.model_validate_json(text)
    assert again == result
    assert again.rooms[0].boundary is not None
    assert again.rooms[0].boundary.area() == pytest.approx(3.1 * 4.7)


# --------------------------------------------------------------------------
# Strictness and non-finite values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("measurements", 0, "value"), float("nan")),
        (("measurements", 0, "value"), float("inf")),
        (("measurements", 0, "value"), "3.1"),
        (("measurements", 0, "value"), True),
        (("measurements", 0, "surprise"), 1),
        (("rooms", 0, "label"), ""),
    ],
    ids=["nan", "inf", "numeric-string", "bool", "unknown-field", "empty-text"],
)
def test_strict_records_reject_bad_scalars(path: tuple, value: Any) -> None:
    with pytest.raises(ValidationError):
        PropertyResult.model_validate(_mutate(_result(), path, value))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_is_rejected(literal: str) -> None:
    text = json.dumps(_result()).replace('"value": 3.1', f'"value": {literal}', 1)
    assert literal in text
    with pytest.raises(ValidationError):
        PropertyResult.model_validate_json(text)


def test_non_ok_record_needs_reason() -> None:
    bad = _mutate(_result(), ("walls", 0, "status"), "partial")
    with pytest.raises(ValidationError, match="status_reason"):
        PropertyResult.model_validate(bad)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_quantity_unit_is_fixed() -> None:
    bad = _mutate(_result(), ("measurements", 0, "unit"), "m2")
    bad = _mutate(bad, ("measurements", 0, "interval", "unit"), "m2")
    with pytest.raises(ValidationError, match="mixed units"):
        PropertyResult.model_validate(bad)


def test_interval_unit_must_equal_measurement_unit() -> None:
    bad = _mutate(_result(), ("measurements", 0, "interval", "unit"), "ratio")
    with pytest.raises(ValidationError, match="interval unit ratio differs"):
        PropertyResult.model_validate(bad)


def test_measured_geometry_must_be_metric() -> None:
    bad = _mutate(_result(), ("surfaces", 0, "frame_id"), "U")
    with pytest.raises(ValidationError, match="up-to-scale frame U"):
        PropertyResult.model_validate(bad)


def _pose(pid: str, unit: str) -> dict:
    return {
        "id": pid,
        "provenance": PROVENANCE,
        "to_frame_id": "W",
        "from_frame_id": f"C-{pid}",
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "translation": [0.1, 0.2, 0.3],
        "translation_unit": unit,
        "covariance_status": "unknown",
        "method": "fixture",
        "tracking_state": "normal",
    }


def _bundle(**overrides: Any) -> dict:
    return {
        "id": "bundle-1",
        "provenance": PROVENANCE,
        "capture_id": "capture-1",
        "capture_hash": HASH,
        "world_frame_id": "U",
        "geometry_unit": "reconstruction_unit",
        "poses": [_pose("p1", "reconstruction_unit")],
        **overrides,
    }


def test_bundle_rejects_mixed_geometry_units() -> None:
    ReconstructionBundle.model_validate(_bundle())
    bad = _bundle(poses=[_pose("p1", "reconstruction_unit"), _pose("p2", "m")])
    with pytest.raises(ValidationError, match="pose p2 is m"):
        ReconstructionBundle.model_validate(bad)


def _evidence(role: str, kind: str = "learned_metric_depth") -> dict:
    return {
        "id": "ev-1",
        "type": kind,
        "observation": 1.02,
        "unit": "ratio",
        "source_ids": ["frame-1"],
        "permitted_tiers": ["photo"],
        "likelihood_assumptions": "fixture",
        "role": role,
    }


def test_metric_reconstruction_needs_resolved_scale() -> None:
    unresolved = {
        "id": "s1",
        "component_id": "c1",
        "multiplier": None,
        "status": "unresolved",
    }
    with pytest.raises(ValidationError, match="every scale resolved"):
        MetricReconstruction.model_validate(
            _bundle(
                geometry_unit="m",
                poses=[_pose("p1", "m")],
                scale_estimates=[unresolved],
            )
        )


def test_learned_scale_is_only_ever_a_prior() -> None:
    with pytest.raises(ValidationError, match="prior, not a measurement"):
        ScaleEvidence.model_validate(_evidence("measurement"))
    estimate = {
        "id": "s1",
        "component_id": "c1",
        "multiplier": 1.02,
        "evidence_ids": ["ev-1"],
        "status": "sensor_metric",
    }
    with pytest.raises(ValidationError, match="prior_metric"):
        ReconstructionBundle.model_validate(
            _bundle(scale_evidence=[_evidence("prior")], scale_estimates=[estimate])
        )


def test_depth_unit_converts_once_with_matching_scale() -> None:
    depth = {
        "id": "d1",
        "provenance": PROVENANCE,
        "frame_id": "f1",
        "camera_id": "cam",
        "timestamp_s": 0.0,
        "depth": {
            "uri": "d.npy",
            "sha256": HASH,
            "byte_count": 4,
            "mime_type": "x",
            "shape": [192, 256],
            "dtype": "<f4",
        },
        "depth_kind": "optical_z",
        "valid_mask": {
            "uri": "m.npy",
            "sha256": HASH,
            "byte_count": 4,
            "mime_type": "x",
            "shape": [192, 256],
            "dtype": "|b1",
        },
        "source_unit": "mm",
        "scale_to_m": 0.001,
        "alignment": "aligned_rgb",
        "sync_residual_s": 0.0,
    }
    DepthFrame.model_validate(depth)
    with pytest.raises(ValidationError, match="scale_to_m 0.001"):
        DepthFrame.model_validate(_mutate(depth, ("scale_to_m",), 1.0))


# --------------------------------------------------------------------------
# Rotations and covariance
# --------------------------------------------------------------------------


def test_pose_quaternion_must_be_unit_and_canonical() -> None:
    Pose.model_validate(_pose("p", "m"))
    with pytest.raises(ValidationError, match="unit vector"):
        Pose.model_validate(
            _mutate(_pose("p", "m"), ("rotation_xyzw",), [0, 0, 0, 2.0])
        )
    with pytest.raises(ValidationError, match="canonical"):
        Pose.model_validate(
            _mutate(_pose("p", "m"), ("rotation_xyzw",), [0, 0, 0, -1.0])
        )


@pytest.mark.parametrize(
    "rotation",
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[1.02, 0, 0], [0, 1.02, 0], [0, 0, 1.02]],
        [[1, 0.2, 0], [0, 1, 0], [0, 0, 1]],
    ],
    ids=["reflection", "hidden-scale", "shear"],
)
def test_transform_records_reject_invalid_rotations(rotation: list) -> None:
    matrix = [row + [0.0] for row in rotation] + [[0.0, 0.0, 0.0, 1.0]]
    with pytest.raises(ValidationError, match="rotation"):
        RigidTransformRecord.model_validate(
            {"to_frame_id": "A", "from_frame_id": "B", "matrix": matrix, "unit": "m"}
        )


def test_unknown_covariance_stays_unknown() -> None:
    zeros = np.zeros((6, 6)).tolist()
    diagonal = (np.eye(6) * 1e-4).tolist()
    base = _pose("p", "m")
    with pytest.raises(ValidationError, match="must not carry"):
        Pose.model_validate({**base, "covariance": diagonal})
    with pytest.raises(ValidationError, match="needs a 6x6"):
        Pose.model_validate({**base, "covariance_status": "known"})
    with pytest.raises(ValidationError, match="all zeros"):
        Pose.model_validate({**base, "covariance_status": "known", "covariance": zeros})
    known = Pose.model_validate(
        {**base, "covariance_status": "known", "covariance": diagonal}
    )
    assert known.covariance is not None


def test_pose_record_converts_to_math_transform() -> None:
    half = np.sqrt(0.5)
    pose = Pose.model_validate(
        _mutate(_pose("p", "m"), ("rotation_xyzw",), [0.0, 0.0, half, half])
    )
    transform = pose.to_math()
    np.testing.assert_allclose(
        transform.apply_direction([1, 0, 0]), [0, 1, 0], atol=1e-12
    )
    assert (transform.to_frame, transform.from_frame) == ("W", "C-p")


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("walls", 0, "surface_id"), "surface-missing", "unknown surface"),
        (
            ("openings", 0, "width_measurement_id"),
            "m-missing",
            "unknown measurement m-missing",
        ),
        (("measurements", 1, "id"), "m-wall", "duplicate record id"),
    ],
    ids=[
        "wall-surface",
        "opening-width",
        "dup-id",
    ],
)
def test_broken_references_are_rejected(path: tuple, value: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PropertyResult.model_validate(_mutate(_result(), path, value))


def test_stale_surface_version_is_rejected() -> None:
    data = _result()
    data["measurements"].append(
        {
            **_measurement("m-surf", "surface-1", "wall_height", 2.45),
            "subject_geometry_version": 2,
        }
    )
    with pytest.raises(ValidationError, match="geometry version 2 of surface-1"):
        PropertyResult.model_validate(data)


# --------------------------------------------------------------------------
# Intervals and availability
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("interval", "message"),
    [
        (_interval(3.1, 3.1), "zero-width"),
        (_interval(3.2, 3.0), "exceeds"),
        (_interval(3.0, 3.2, status="calibrated"), "calibrator"),
        (_interval(3.0, 3.2, status="unavailable", reason="x"), "no bounds"),
        ({**_interval(3.0, 3.2), "upper": None}, "unbounded side"),
    ],
    ids=["zero-width", "inverted", "uncalibrated", "unavailable-bounds", "unbounded"],
)
def test_interval_invariants(interval: dict, message: str) -> None:
    measurement = {**_measurement("m", "s", "wall_length", 3.1), "interval": interval}
    with pytest.raises(ValidationError, match=message):
        Measurement.model_validate(measurement)


def test_missing_value_is_null_with_reason_not_zero() -> None:
    missing = {
        **_measurement("m", "s", "ceiling_height", 0.0),
        "value": None,
        "quality_status": "unavailable",
        "unavailable_reason": "ceiling not observed",
        "interval": {
            **_interval(0, 1),
            "lower": None,
            "upper": None,
            "status": "unavailable",
            "reason": "no estimate",
        },
    }
    assert Measurement.model_validate(missing).value is None
    with pytest.raises(ValidationError, match="cannot have an available interval"):
        Measurement.model_validate({**missing, "interval": _interval(2.3, 2.5)})


def test_lengths_cannot_be_negative() -> None:
    with pytest.raises(ValidationError, match="cannot be negative"):
        Measurement.model_validate(_measurement("m", "s", "wall_length", -0.2))


# --------------------------------------------------------------------------
# Geometry semantics
# --------------------------------------------------------------------------


def test_polygon_orientation_and_simplicity() -> None:
    with pytest.raises(ValidationError, match="counter-clockwise"):
        Polygon2D.model_validate({"outer": [[0, 0], [0, 1], [1, 1], [1, 0]]})
    with pytest.raises(ValidationError, match="not simple"):
        # Positive signed area (6 m2) but two edges cross the bottom edge.
        Polygon2D.model_validate({"outer": [[0, 0], [4, 0], [4, 4], [2, -1], [0, 4]]})


def test_room_placement_has_no_arbitrary_origin() -> None:
    placed = _result()
    with pytest.raises(ValidationError, match="placed room needs"):
        PropertyResult.model_validate(
            _mutate(placed, ("rooms", 0, "T_property_from_room"), None)
        )
    with pytest.raises(ValidationError, match="no arbitrary origin"):
        PropertyResult.model_validate(
            _mutate(placed, ("rooms", 0, "placement_status"), "unplaced")
        )


def test_window_is_not_a_connector() -> None:
    bad = _mutate(_result(), ("openings", 0, "opening_class"), "window")
    bad = _mutate(bad, ("openings", 0, "connector_id"), "conn-1")
    with pytest.raises(ValidationError, match="window"):
        PropertyResult.model_validate(bad)


def test_surface_basis_must_be_right_handed() -> None:
    bad = _mutate(_result(), ("surfaces", 0, "normal"), [0.0, 1.0, 0.0])
    with pytest.raises(ValidationError, match="right-handed"):
        PropertyResult.model_validate(bad)


def test_graph_components_partition_rooms() -> None:
    graph = {
        "id": "graph-1",
        "provenance": PROVENANCE,
        "property_id": "prop-1",
        "property_frame_id": "P",
        "level_ids": ["level-0"],
        "room_ids": ["room-1", "room-2"],
        "nodes": [{"id": "n1", "frame_id": "P", "scale_state": "metric"}],
        "components": [["room-1"], ["room-2"]],
        "anchor_id": "n1",
        "registration_status": "disconnected",
    }
    PropertyGraph.model_validate(graph)
    with pytest.raises(ValidationError, match="exactly one component"):
        PropertyGraph.model_validate({**graph, "registration_status": "connected"})
    with pytest.raises(ValidationError, match="partition"):
        PropertyGraph.model_validate(
            {**graph, "components": [["room-1", "room-2"], ["room-1"]]}
        )


def test_concealed_flag_must_follow_its_predicates() -> None:
    flag = {
        "id": "flag-1",
        "provenance": PROVENANCE,
        "room_id": "room-1",
        "flag_status": "inspect",
        "rule_id": "rule.stain_near_wet_room",
        "rule_version": "1",
        "predicates": [
            {
                "name": "stain_area_m2",
                "observed": 0.3,
                "comparator": ">=",
                "threshold": 0.1,
                "satisfied": True,
            },
            {
                "name": "adjacent_wet_room",
                "observed": None,
                "comparator": "present",
                "threshold": True,
                "satisfied": None,
            },
        ],
        "explanation": "stain near a wet room",
    }
    with pytest.raises(ValidationError, match="every predicate satisfied"):
        ConcealedDamageFlag.model_validate(flag)
    assert (
        ConcealedDamageFlag.model_validate(
            {**flag, "flag_status": "insufficient_evidence"}
        ).flag_status
        == "insufficient_evidence"
    )


# --------------------------------------------------------------------------
# Capture tier isolation
# --------------------------------------------------------------------------


def _capture(tier: str, modalities: list[str], photos: int = 3) -> dict:
    return {
        "id": "capture-1",
        "provenance": PROVENANCE,
        "property_session_id": "session-1",
        "tier": tier,
        "assets": [
            {
                "uri": "a.jpg",
                "sha256": HASH,
                "byte_count": 10,
                "mime_type": "image/jpeg",
            }
        ],
        "frames_uri": "frames.jsonl",
        "room_groups": [
            {
                "id": "g1",
                "label": "room_001",
                "frame_ids": [f"f{i}" for i in range(photos)],
            }
        ],
        "allowed_modalities": modalities,
        "capture_profile_id": "native_camera",
        "raw_manifest_hash": HASH,
    }


@pytest.mark.parametrize(
    ("tier", "modalities"),
    [
        ("photo", ["rgb_image", "depth"]),
        ("photo", ["rgb_image", "pose"]),
        ("video", ["rgb_video", "imu"]),
    ],
    ids=["photo-depth", "photo-pose", "video-imu"],
)
def test_strict_tiers_cannot_use_sensor_data(tier: str, modalities: list[str]) -> None:
    with pytest.raises(ValidationError, match=f"strict {tier} tier"):
        Capture.model_validate(_capture(tier, modalities))


@pytest.mark.parametrize(
    ("photos", "ok"), [(1, False), (2, True), (8, True), (9, False)]
)
def test_photo_tier_takes_two_to_eight_photos_per_room(photos: int, ok: bool) -> None:
    data = _capture("photo", ["rgb_image"], photos)
    if ok:
        assert Capture.model_validate(data).schema_version == "capture-v0"
    else:
        with pytest.raises(ValidationError, match="2 to 8"):
            Capture.model_validate(data)


def test_asset_uri_cannot_escape_root() -> None:
    bad = _mutate(_capture("lidar", ["depth"]), ("assets", 0, "uri"), "../outside.jpg")
    with pytest.raises(ValidationError, match="relative"):
        Capture.model_validate(bad)


# --------------------------------------------------------------------------
# Published project schemas
# --------------------------------------------------------------------------


def test_committed_schemas_match_models() -> None:
    for filename, document in schema_documents().items():
        committed = json.loads((PROJECT_ROOT / "schemas" / filename).read_text("utf-8"))
        assert committed == document, (
            f"schemas/{filename} is stale; regenerate with "
            "`uv run python -m floscan.contracts.result --write-schemas schemas`"
        )
        assert "Project-owned" in document["$comment"]


def test_unknown_frame_is_rejected() -> None:
    bad = _mutate(_result(), ("rooms", 0, "local_frame_id"), "R9")
    bad = _mutate(bad, ("rooms", 0, "T_property_from_room", "from_frame_id"), "R9")
    with pytest.raises(ValidationError, match="unknown frame R9"):
        PropertyResult.model_validate(bad)


# --------------------------------------------------------------------------
# P03A regressions (docs/reviews/P03-review.md)
# --------------------------------------------------------------------------

SVG_PLAN = {
    "uri": "plan.svg",
    "sha256": HASH,
    "byte_count": 2048,
    "mime_type": "image/svg+xml",
}


def _complete_result() -> dict:
    """Every internal section available and backed by records; no damage found.

    The negative damage finding is explicit: surface-1 is listed as inspected,
    and a 'not_triggered' rule evaluation records the concealed-damage check.
    """
    d = _result()
    d["status"], d["status_reason"] = "ok", None
    d["measurements"] += [
        _measurement("m-ceiling", "room-1", "ceiling_height", 2.45),
        _measurement("m-floor", "room-1", "floor_area", 14.57, "m2"),
        _measurement("m-footprint", "graph-1", "footprint_area", 14.57, "m2"),
    ]
    d["rooms"][0]["measurement_ids"] += ["m-ceiling", "m-floor"]
    d["scale"] = [
        {
            "id": "scale-1",
            "component_id": "comp-1",
            "multiplier": 1.0,
            "evidence_ids": ["ev-lidar-depth"],
            "status": "sensor_metric",
        }
    ]
    d["property_graph"] = {
        "id": "graph-1",
        "provenance": PROVENANCE,
        "property_id": "prop-1",
        "property_frame_id": "P",
        "level_ids": ["level-0"],
        "room_ids": ["room-1"],
        "nodes": [{"id": "node-1", "frame_id": "P", "scale_state": "metric"}],
        "components": [["room-1"]],
        "anchor_id": "node-1",
        "footprint_measurement_ids": ["m-footprint"],
        "registration_status": "connected",
    }
    d["concealed_damage_flags"] = [
        {
            "id": "flag-1",
            "provenance": PROVENANCE,
            "room_id": "room-1",
            "surface_id": "surface-1",
            "flag_status": "not_triggered",
            "rule_id": "rule.stain_near_wet_room",
            "rule_version": "1",
            "predicates": [
                {
                    "name": "stain_area_m2",
                    "observed": 0.0,
                    "comparator": ">=",
                    "threshold": 0.1,
                    "satisfied": False,
                }
            ],
            "explanation": "no stain observed on the inspected wall",
        }
    ]
    d["artifacts"] = [SVG_PLAN]
    d["coverage"] = {name: {"status": "available"} for name in SECTIONS}
    d["coverage"]["damage_regions"] = {
        "status": "available",
        "evidence_ids": ["surface-1"],
    }
    d["coverage"]["public_schema_export"] = {
        "status": "blocked_external",
        "reason": "the assignment's published schema is unavailable",
    }
    return d


def test_empty_result_cannot_claim_success() -> None:
    d = _result()
    for key in (
        "rooms",
        "surfaces",
        "walls",
        "openings",
        "measurements",
        "coordinate_frames",
    ):
        d[key] = []
    d["capture_id"] = None
    d["coverage"] = {name: {"status": "available"} for name in SECTIONS}
    d["status"], d["status_reason"] = "ok", None
    with pytest.raises(ValidationError, match="'available' but"):
        PropertyResult.model_validate(d)


def test_complete_result_with_negative_damage_finding_is_ok() -> None:
    result = PropertyResult.model_validate(_complete_result())
    assert result.status == "ok"
    assert result.damage_regions == [] and result.scope_items == []
    assert PropertyResult.model_validate_json(result.to_json()) == result


def test_partial_result_stays_valid() -> None:
    result = PropertyResult.model_validate(_result())
    assert result.coverage["per_room_plan"].status == "partial"


def test_negative_damage_finding_needs_inspection_evidence() -> None:
    d = _complete_result()
    d["coverage"]["damage_regions"] = {"status": "available"}
    with pytest.raises(ValidationError, match="no inspected surfaces"):
        PropertyResult.model_validate(d)


def _inspect_flag(d: dict) -> dict:
    d["concealed_damage_flags"][0]["flag_status"] = "inspect"
    d["concealed_damage_flags"][0]["predicates"][0]["satisfied"] = True
    return d


def _drop_ceiling(d: dict) -> dict:
    d["measurements"] = [m for m in d["measurements"] if m["id"] != "m-ceiling"]
    d["rooms"][0]["measurement_ids"].remove("m-ceiling")
    return d


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_ceiling, "per_room_plan is 'available'"),
        (
            lambda d: (
                d["property_graph"].update(registration_status="disconnected") or d
            ),
            "stitched_plan is 'available'",
        ),
        (lambda d: d.update(artifacts=[]) or d, "rendered_plan is 'available'"),
        (
            lambda d: (
                d.update(
                    scale=[
                        {
                            "id": "scale-1",
                            "component_id": "c",
                            "multiplier": None,
                            "status": "unresolved",
                        }
                    ]
                )
                or d
            ),
            "scale is 'available'",
        ),
        (_inspect_flag, "scope_items is 'available'"),
        (
            lambda d: (
                d["coverage"].update(public_schema_export={"status": "available"}) or d
            ),
            "public_schema_export cannot be 'available'",
        ),
        (
            lambda d: (
                d["coverage"].update(rendered_plan={"status": "partial", "reason": "x"})
                or d
            ),
            "status 'ok' needs every internal coverage section",
        ),
    ],
    ids=[
        "no-ceiling",
        "disconnected",
        "no-plan",
        "unresolved-scale",
        "inspect-without-scope",
        "export-available",
        "ok-with-gap",
    ],
)
def test_available_coverage_is_checked_against_records(mutate, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PropertyResult.model_validate(mutate(_complete_result()))


def test_unavailable_section_cannot_hide_content() -> None:
    d = _mutate(
        _result(),
        ("coverage", "measurements"),
        {"status": "unavailable", "reason": "x"},
    )
    with pytest.raises(ValidationError, match="contains it"):
        PropertyResult.model_validate(d)


def test_coverage_evidence_must_resolve() -> None:
    d = _mutate(
        _complete_result(),
        ("coverage", "damage_regions", "evidence_ids"),
        ["surface-ghost"],
    )
    with pytest.raises(ValidationError, match="unknown evidence surface-ghost"):
        PropertyResult.model_validate(d)


def _with_height(d: dict, measurement: dict) -> dict:
    d["measurements"].append(measurement)
    d["openings"][0]["height_measurement_id"] = measurement["id"]
    return d


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: _mutate(
                d, ("openings", 0, "height_measurement_id"), "nonexistent"
            ),
            "unknown measurement nonexistent",
        ),
        (
            lambda d: _mutate(
                d, ("walls", 0, "thickness_measurement_id"), "nonexistent"
            ),
            "unknown measurement nonexistent",
        ),
        (
            lambda d: _mutate(d, ("openings", 0, "width_measurement_id"), "m-wall"),
            "measures wall-1, not door-1",
        ),
        (
            lambda d: _with_height(
                d, _measurement("m-h", "door-1", "opening_width", 2.0)
            ),
            "is a opening_width, expected opening_height",
        ),
        (
            lambda d: (
                d.update(
                    coordinate_frames=d["coordinate_frames"]
                    + [copy.deepcopy(d["coordinate_frames"][1])]
                )
                or d
            ),
            "duplicate record id 'R1'",
        ),
        (
            lambda d: (
                d.update(
                    measurements=d["measurements"]
                    + [_measurement("m-x", "ghost", "wall_length", 1.0)]
                )
                or d
            ),
            "unknown subject ghost",
        ),
        (
            lambda d: _mutate(d, ("openings", 0, "connector_id"), "conn-9"),
            "unknown connector conn-9",
        ),
    ],
    ids=[
        "height-missing",
        "thickness-missing",
        "width-bound-to-wall",
        "height-wrong-quantity",
        "duplicate-frame",
        "unknown-subject",
        "unknown-connector",
    ],
)
def test_references_resolve_to_the_right_record(mutate, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PropertyResult.model_validate(mutate(_result()))


def test_room_cannot_claim_another_rooms_wall() -> None:
    d = _result()
    d["rooms"].append(
        {
            "id": "room-2",
            "provenance": PROVENANCE,
            "status": "partial",
            "status_reason": "boundary not closed",
            "label": "hall",
            "level_id": "level-0",
            "local_frame_id": "R1",
            "wall_ids": ["wall-1"],
            "hypothesis_id": "h-2",
            "placement_status": "unplaced",
        }
    )
    with pytest.raises(ValidationError, match="wall wall-1 belongs to room room-1"):
        PropertyResult.model_validate(d)


def test_valid_optional_measurement_bindings() -> None:
    bound = _with_height(
        _result(), _measurement("m-h", "door-1", "opening_height", 2.03)
    )
    assert (
        PropertyResult.model_validate(bound).openings[0].height_measurement_id == "m-h"
    )
    missing = {
        **_measurement("m-h", "door-1", "opening_height", 0.0),
        "value": None,
        "quality_status": "unavailable",
        "unavailable_reason": "head jamb occluded",
        "interval": {
            **_interval(0, 1),
            "lower": None,
            "upper": None,
            "status": "unavailable",
            "reason": "no estimate",
        },
    }
    d = _with_height(_result(), missing)
    d["coverage"]["measurements"] = {
        "status": "partial",
        "reason": "door height missing",
    }
    result = PropertyResult.model_validate(d)
    assert result.measurements[-1].value is None


def test_indefinite_covariance_is_rejected_and_singular_psd_kept() -> None:
    indefinite = np.eye(6)
    indefinite[0, 1] = indefinite[1, 0] = 2.0
    base = {**_pose("p", "m"), "covariance_status": "known"}
    with pytest.raises(ValidationError, match="not positive semidefinite"):
        Pose.model_validate({**base, "covariance": indefinite.tolist()})
    singular = np.diag([1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0]).tolist()
    assert Pose.model_validate({**base, "covariance": singular}).covariance is not None


def test_plane_parameter_covariance_is_validated() -> None:
    plane = {
        "id": "plane-1",
        "provenance": PROVENANCE,
        "frame_id": "R1",
        "normal": [0.0, -1.0, 0.0],
        "offset": 0.0,
        "unit": "m",
        "basis_u": [1.0, 0.0, 0.0],
        "basis_v": [0.0, 0.0, 1.0],
        "residual_stats": {"rms": 0.002, "max": 0.01, "count": 500},
        "kind": "wall",
        "observability": "full",
        "orientation_is_prior": False,
    }
    assert Plane.model_validate(plane).parameter_covariance is None
    bad = (np.eye(4) - 2 * np.outer([1, 0, 0, 0], [1, 0, 0, 0])).tolist()
    with pytest.raises(ValidationError, match="not positive semidefinite"):
        Plane.model_validate({**plane, "parameter_covariance": bad})
    with pytest.raises(ValidationError, match="4x4"):
        Plane.model_validate({**plane, "parameter_covariance": np.eye(3).tolist()})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.surfaces[0].origin.__setitem__(0, float("nan")),
        lambda r: r.surfaces[0].origin.append(1.0),
        lambda r: r.rooms[0].wall_ids.extend(["wall-x"]),
        lambda r: r.rooms[0].T_property_from_room.matrix[0].__setitem__(0, 2.0),
        lambda r: r.coverage.__setitem__("measurements", r.coverage["capture"]),
        lambda r: r.coverage.pop("capture"),
        lambda r: r.run.platform.update(os="other"),
    ],
    ids=[
        "nan-item",
        "append",
        "extend",
        "matrix-row",
        "dict-set",
        "dict-pop",
        "dict-update",
    ],
)
def test_validated_records_refuse_nested_mutation(mutate) -> None:
    result = PropertyResult.model_validate(_result())
    with pytest.raises(TypeError, match="immutable"):
        mutate(result)


def test_bypassed_mutation_never_serialises_silently() -> None:
    result = PropertyResult.model_validate(_result())
    list.__setitem__(result.surfaces[0].origin, 0, float("nan"))  # bypass on purpose
    with pytest.raises(ValidationError):
        result.to_json()
    raw = result.model_dump_json()
    assert "NaN" in raw  # never silently turned into null


def test_copies_stay_equal_and_frozen() -> None:
    import pickle

    result = PropertyResult.model_validate(_result())
    for clone in (copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
        assert clone == result
        with pytest.raises(TypeError, match="immutable"):
            clone.surfaces[0].origin.append(1.0)


def _camera() -> dict:
    identity = {"kind": "identity", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    return {
        "id": "cam-1",
        "provenance": PROVENANCE,
        "sensor_id": "wide",
        "width_px": 640,
        "height_px": 480,
        "model": "pinhole",
        "fx": 500.0,
        "fy": 498.0,
        "cx": 320.5,
        "cy": 239.0,
        "distortion": {"model": "none"},
        "intrinsics_origin": "metadata",
        "pixel_transform": identity,
    }


def _frame(capture: str = "capture-1", depth: str | None = None) -> dict:
    return {
        "id": "frame-1",
        "provenance": PROVENANCE,
        "capture_id": capture,
        "source_index": 0,
        "camera_id": "cam-1",
        "depth_id": depth,
        "image": {
            "uri": "f.jpg",
            "sha256": HASH,
            "byte_count": 1,
            "mime_type": "image/jpeg",
        },
        "orientation": {
            "kind": "identity",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }


def test_bundle_frames_belong_to_their_capture_and_depths_link_back() -> None:
    ReconstructionBundle.model_validate(_bundle(frames=[_frame()], cameras=[_camera()]))
    with pytest.raises(ValidationError, match="belongs to capture capture-2"):
        ReconstructionBundle.model_validate(
            _bundle(frames=[_frame("capture-2")], cameras=[_camera()])
        )
    depth = {
        "id": "depth-1",
        "provenance": PROVENANCE,
        "frame_id": "frame-9",
        "camera_id": "cam-1",
        "timestamp_s": 0.0,
        "depth": {
            "uri": "d.npy",
            "sha256": HASH,
            "byte_count": 4,
            "mime_type": "x",
            "shape": [192, 256],
            "dtype": "<f4",
        },
        "depth_kind": "optical_z",
        "valid_mask": {
            "uri": "m.npy",
            "sha256": HASH,
            "byte_count": 4,
            "mime_type": "x",
            "shape": [192, 256],
            "dtype": "|b1",
        },
        "source_unit": "mm",
        "scale_to_m": 0.001,
        "alignment": "aligned_rgb",
        "sync_residual_s": 0.0,
    }
    with pytest.raises(ValidationError, match="depth depth-1: unknown frame"):
        ReconstructionBundle.model_validate(
            _bundle(
                frames=[_frame(depth="depth-1")],
                cameras=[_camera()],
                depth_frames=[depth],
            )
        )
    linked = {**depth, "frame_id": "frame-1"}
    ReconstructionBundle.model_validate(
        _bundle(
            frames=[_frame(depth="depth-1")],
            cameras=[_camera()],
            depth_frames=[linked],
        )
    )
    with pytest.raises(ValidationError, match="does not link back"):
        ReconstructionBundle.model_validate(
            _bundle(frames=[_frame()], cameras=[_camera()], depth_frames=[linked])
        )
    with pytest.raises(ValidationError, match="depth depth-1: unknown camera"):
        ReconstructionBundle.model_validate(
            _bundle(
                frames=[_frame(depth="depth-1")],
                cameras=[_camera()],
                depth_frames=[{**linked, "camera_id": "cam-9"}],
            )
        )
