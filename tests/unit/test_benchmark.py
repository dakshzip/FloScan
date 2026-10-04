"""Hand-calculated scorer tests (P04).

Expected numbers are derived by hand in each test, never copied from the
scorer. Plans are axis-aligned rectangles; predictions are the same plan moved
by a known rigid transform, with specific values perturbed or entities removed.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark.evaluator.matching import (
    Correspondence,
    PlanView,
    fit_rigid_2d,
    match_plans,
    plan_from_result,
)
from benchmark.evaluator.metrics import (
    IncumbentExport,
    calibration,
    interval_score,
    item_rows,
)
from benchmark.evaluator.report import score_case, write_outputs
from floscan.pipeline import load_gate_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "scoring"
REGISTRY = load_gate_registry()

# (id, label, x0, y0, x1, y1, ceiling height)
ROOMS = [
    ("living", "Living", 0.0, 0.0, 4.0, 5.0, 2.40),
    ("kitchen", "Kitchen", 4.0, 0.0, 7.0, 5.0, 2.50),
    ("hall", "Hall", 7.0, 0.0, 8.5, 5.0, 2.45),
]
ADJACENCY = [["living", "kitchen"], ["kitchen", "hall"]]
# (id, room, host wall, centre, clear width): kitchen door and hall door.
OPENINGS = [
    ("door-lk", "living", "living-E", (4.0, 2.0), 0.86),
    ("door-kh", "kitchen", "kitchen-E", (7.0, 3.0), 0.81),
]


def _walls(rid: str, x0: float, y0: float, x1: float, y1: float) -> list[dict]:
    corners = {
        "S": ((x0, y0), (x1, y0)),
        "E": ((x1, y0), (x1, y1)),
        "N": ((x1, y1), (x0, y1)),
        "W": ((x0, y1), (x0, y0)),
    }
    return [
        {
            "id": f"{rid}-{side}",
            "room_id": rid,
            "start": list(a),
            "end": list(b),
            "length": {"value": math.dist(a, b)},
        }
        for side, (a, b) in corners.items()
    ]


def ground_truth(
    rooms: list = ROOMS, openings: list = OPENINGS, adjacency: list = ADJACENCY
) -> dict:
    data: dict[str, Any] = {
        "kind": "ground_truth",
        "case_id": "case-1",
        "rooms": [],
        "walls": [],
        "openings": [],
        "adjacency": adjacency,
    }
    for rid, label, x0, y0, x1, y1, ceiling in rooms:
        data["rooms"].append(
            {
                "id": rid,
                "label": label,
                "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "ceiling_height": {"value": ceiling},
                "floor_area": {"value": (x1 - x0) * (y1 - y0)},
            }
        )
        data["walls"] += _walls(rid, x0, y0, x1, y1)
    for oid, room, host, centre, width in openings:
        data["openings"].append(
            {
                "id": oid,
                "room_id": room,
                "host_wall_id": host,
                "center": list(centre),
                "width": {"value": width},
            }
        )
    return data


def prediction(
    gt: dict,
    capture: str = "cap-a",
    tier: str = "photo",
    angle_deg: float = 30.0,
    shift: tuple[float, float] = (10.0, -3.0),
    scale: float = 1.0,
    mirror: bool = False,
    values: dict[str, float] | None = None,
    drop: frozenset[str] | set[str] = frozenset(),
) -> dict:
    """The ground-truth plan seen through a known pose, renamed with p- prefixes.

    ``values`` overrides predicted scalars by ground-truth item id; ``drop``
    removes predicted rooms, walls or openings by ground-truth id.
    """
    values = values or {}
    a = math.radians(angle_deg)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def move(xy: list[float]) -> list[float]:
        p = np.array(xy, dtype=float) * scale
        if mirror:
            p[0] = -p[0]
        return (rotation @ p + np.array(shift)).tolist()

    def est(key: str, truth: dict) -> dict:
        """Predicted scalar (lengths scale with the plan) with a +-5 cm interval."""
        v = values.get(key, truth["value"] * scale)
        return {"value": v, "lower": v - 0.05, "upper": v + 0.05, "nominal_level": 0.9}

    out: dict[str, Any] = {
        "kind": "prediction",
        "case_id": gt["case_id"],
        "capture_id": capture,
        "tier": tier,
        "rooms": [],
        "walls": [],
        "openings": [],
        "adjacency": [],
        "connected_components": 1,
    }
    kept = set()
    for room in gt["rooms"]:
        if room["id"] in drop:
            continue
        kept.add(room["id"])
        out["rooms"].append(
            {
                "id": f"p-{room['id']}",
                "label": room["label"],
                "polygon": [move(v) for v in room["polygon"]],
                "ceiling_height": est(f"{room['id']}:ceiling", room["ceiling_height"]),
                "floor_area": {
                    "value": values.get(
                        f"{room['id']}:area", room["floor_area"]["value"] * scale**2
                    )
                },
            }
        )
    for wall in gt["walls"]:
        if wall["id"] in drop or wall["room_id"] not in kept:
            continue
        out["walls"].append(
            {
                "id": f"p-{wall['id']}",
                "room_id": f"p-{wall['room_id']}",
                "start": move(wall["start"]),
                "end": move(wall["end"]),
                "length": est(wall["id"], wall["length"]),
            }
        )
    walls = {w["id"] for w in out["walls"]}
    for o in gt["openings"]:
        if o["id"] in drop or o["room_id"] not in kept:
            continue
        host = f"p-{o['host_wall_id']}"
        out["openings"].append(
            {
                "id": f"p-{o['id']}",
                "room_id": f"p-{o['room_id']}",
                "host_wall_id": host if host in walls else None,
                "center": move(o["center"]),
                "width": est(o["id"], o["width"]),
            }
        )
    out["adjacency"] = [
        [f"p-{a}", f"p-{b}"] for a, b in gt["adjacency"] if a in kept and b in kept
    ]
    return out


def _score(gt: dict, *preds: dict, **kwargs: Any) -> dict:
    return score_case(
        PlanView.model_validate(gt),
        [PlanView.model_validate(p) for p in preds],
        REGISTRY,
        **kwargs,
    )


def _gate(scored: dict, gate_id: str) -> dict:
    return next(v for v in scored["gates"]["verdicts"] if v["id"] == gate_id)


# --------------------------------------------------------------------------
# Matching and alignment
# --------------------------------------------------------------------------


def test_rigid_alignment_recovers_pose_without_scale_or_mirror() -> None:
    gt = ground_truth()
    scored = _score(gt, prediction(gt))
    alignment = scored["captures"]["cap-a"]["matching"]["alignment"]
    # Prediction = R(30 deg) gt + t, so gt = R(-30 deg) (pred - t).
    assert alignment["angle_deg"] == pytest.approx(-30.0)
    assert alignment["pairs"] == 3
    for gate_id in (
        "photo_wall_length",
        "photo_footprint",
        "photo_stitch_adjacency",
        "photo_stitch_no_overlap",
        "photo_stitch_single_plan",
        "opening_width",
        "ceiling_height_error",
    ):
        assert _gate(scored, gate_id)["status"] == "measured_pass", gate_id
    footprint = scored["captures"]["cap-a"]["footprint"]
    assert footprint["area_rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert footprint["iou"] == pytest.approx(1.0)


def test_alignment_never_absorbs_reflection() -> None:
    src = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    mirrored = src * [-1.0, 1.0]
    alignment = fit_rigid_2d(mirrored, src)
    r = np.asarray(alignment.rotation)
    assert np.linalg.det(r) == pytest.approx(1.0)
    assert not np.allclose(alignment.apply(mirrored), src, atol=0.1)


def test_scale_error_stays_visible() -> None:
    gt = ground_truth()
    scored = _score(gt, prediction(gt, scale=1.1))
    # Area scales by 1.1^2 = 1.21 exactly; no similarity alignment may hide it.
    assert scored["captures"]["cap-a"]["footprint"]["area_rel_error"] == pytest.approx(
        0.21
    )
    assert _gate(scored, "photo_footprint")["status"] == "measured_fail"


def test_matching_ignores_the_values_being_scored() -> None:
    gt = PlanView.model_validate(ground_truth())
    honest = PlanView.model_validate(prediction(ground_truth()))
    skewed = PlanView.model_validate(
        prediction(ground_truth(), values={"door-lk": 3.0, "living-N": 0.1})
    )
    a, b = match_plans(gt, honest), match_plans(gt, skewed)
    assert a.walls == b.walls and a.openings == b.openings and a.rooms == b.rooms


def test_correspondence_file_overrides_automatic_rooms() -> None:
    gt = PlanView.model_validate(ground_truth())
    pred = PlanView.model_validate(prediction(ground_truth()))
    swapped = Correspondence(
        rooms={"living": "p-kitchen", "kitchen": "p-living", "hall": "p-hall"}
    )
    match = match_plans(gt, pred, swapped)
    assert ("living", "p-kitchen") in match.rooms
    assert "rooms matched by evaluator correspondence" in match.notes


# --------------------------------------------------------------------------
# Openings: the 8/11 example
# --------------------------------------------------------------------------


def _studio_openings() -> tuple[dict, dict]:
    """10 GT openings 1 m apart; 9 detected, 1 missed, 1 phantom, 1 bad width."""
    room = [("studio", "Studio", 0.0, 0.0, 10.0, 6.0, 2.5)]
    openings = [
        (f"o{i}", "studio", "studio-S", (0.5 + i, 0.0), 0.80) for i in range(10)
    ]
    gt = ground_truth(room, openings, [])
    values = {f"o{i}": 0.81 for i in range(8)}  # within 2 cm
    values["o8"] = 0.85  # detected, width 5 cm off
    pred = prediction(gt, values=values, drop={"o9"})  # o9 missed
    pred["openings"].append(
        {
            "id": "p-phantom",
            "room_id": "p-studio",
            "center": prediction(gt)["walls"][2]["start"],
            "width": {"value": 0.9},
        }
    )  # on the north wall, no GT
    return gt, pred


def test_opening_score_is_8_of_11() -> None:
    gt, pred = _studio_openings()
    scored = _score(gt, pred)
    o = scored["captures"]["cap-a"]["openings"]
    assert (o["n_gt"], o["tp"], o["fn"], o["fp"], o["g_within_tolerance"]) == (
        10,
        9,
        1,
        1,
        8,
    )
    assert o["score"] == pytest.approx(8 / 11)  # 72.73 %
    assert o["precision"] == pytest.approx(9 / 10)
    assert o["recall"] == pytest.approx(9 / 10)
    assert _gate(scored, "opening_width")["status"] == "measured_fail"


def test_deleting_the_bad_opening_cannot_pass() -> None:
    gt, pred = _studio_openings()
    pred["openings"] = [o for o in pred["openings"] if o["id"] != "p-o8"]
    o = _score(gt, pred)["captures"]["cap-a"]["openings"]
    # o8 becomes a miss instead of a bad width: G stays 8, N + FP stays 11.
    assert (o["fn"], o["g_within_tolerance"], o["score"]) == (
        2,
        8,
        pytest.approx(8 / 11),
    )


def test_duplicate_detection_is_a_false_positive() -> None:
    gt = ground_truth()
    pred = prediction(gt)
    twin = dict(pred["openings"][0], id="p-door-lk-twin")
    pred["openings"].append(twin)
    o = _score(gt, pred)["captures"]["cap-a"]["openings"]
    assert (o["tp"], o["fp"], o["score"]) == (2, 1, pytest.approx(2 / 3))


# --------------------------------------------------------------------------
# Missing rooms and deleted predictions
# --------------------------------------------------------------------------


def test_missing_room_fails_every_dependent_gate() -> None:
    gt = ground_truth()
    scored = _score(gt, prediction(gt, drop={"hall"}))
    errors = scored["captures"]["cap-a"]["errors"]
    assert errors["wall_length"]["n_expected"] == 12
    assert errors["wall_length"]["n_missing"] == 4
    assert errors["ceiling_height"]["n_missing"] == 1
    footprint = scored["captures"]["cap-a"]["footprint"]
    assert footprint["area_rel_error"] == pytest.approx(7.5 / 42.5)
    for gate_id in (
        "photo_wall_length",
        "ceiling_height_error",
        "photo_footprint",
        "photo_stitch_adjacency",
        "photo_stitch_single_plan",
    ):
        assert _gate(scored, gate_id)["status"] == "measured_fail", gate_id
    adjacency = scored["captures"]["cap-a"]["adjacency"]
    assert adjacency["missed_edges"] == [["hall", "kitchen"]]


def test_deleting_a_bad_wall_cannot_pass() -> None:
    gt = ground_truth()
    bad = _score(gt, prediction(gt, values={"living-N": 4.4}))  # 10 % > 8 %
    assert _gate(bad, "photo_wall_length")["status"] == "measured_fail"
    deleted = _score(gt, prediction(gt, drop={"living-N"}))
    assert _gate(deleted, "photo_wall_length")["status"] == "measured_fail"
    assert deleted["captures"]["cap-a"]["errors"]["wall_length"]["n_missing"] == 1


def test_unmeasured_value_is_not_a_pass() -> None:
    gt = ground_truth()
    pred = prediction(gt)
    pred["rooms"][0]["ceiling_height"] = None
    scored = _score(gt, pred)
    assert scored["captures"]["cap-a"]["errors"]["ceiling_height"]["n_unmeasured"] == 1
    assert _gate(scored, "ceiling_height_error")["status"] == "measured_fail"


# --------------------------------------------------------------------------
# Footprint shape
# --------------------------------------------------------------------------


def test_equal_area_wrong_shape_is_exposed() -> None:
    square = ground_truth([("room", "Room", 0.0, 0.0, 4.0, 4.0, 2.4)], [], [])
    strip = ground_truth([("room", "Room", 0.0, 0.0, 2.0, 8.0, 2.4)], [], [])
    pred = prediction(strip, angle_deg=0.0, shift=(0.0, 0.0))
    scored = _score(square, pred)
    f = scored["captures"]["cap-a"]["footprint"]
    assert f["area_rel_error"] == pytest.approx(0.0)  # 16 m2 both
    assert f["iou"] == pytest.approx(8 / 24)  # overlap 2x4, union 24
    assert f["extent_rel_errors"] == pytest.approx([0.5, 1.0])
    assert f["hausdorff_m"] == pytest.approx(4.0)  # strip tip at y=8 vs edge y=4
    # Area alone passes the provisional footprint gate; the walls do not.
    assert _gate(scored, "photo_footprint")["status"] == "measured_pass"
    assert _gate(scored, "photo_wall_length")["status"] == "measured_fail"


def test_overlapping_rooms_fail_and_shared_edges_do_not() -> None:
    gt = ground_truth()
    clean = _score(gt, prediction(gt, angle_deg=0.0, shift=(0.0, 0.0)))
    assert _gate(clean, "photo_stitch_no_overlap")["status"] == "measured_pass"
    pred = prediction(gt, angle_deg=0.0, shift=(0.0, 0.0))
    pred["rooms"][1]["polygon"] = [[3.5, 0.0], [7.0, 0.0], [7.0, 5.0], [3.5, 5.0]]
    overlap = _score(gt, pred)
    pairs = overlap["captures"]["cap-a"]["overlaps"]["overlapping_pairs"]
    assert pairs == [
        {"rooms": ["p-kitchen", "p-living"], "area_m2": pytest.approx(2.5)}
    ]
    assert _gate(overlap, "photo_stitch_no_overlap")["status"] == "measured_fail"


def test_extra_adjacency_edge_fails() -> None:
    gt = ground_truth()
    pred = prediction(gt)
    pred["adjacency"].append(["p-living", "p-hall"])
    scored = _score(gt, pred)
    assert scored["captures"]["cap-a"]["adjacency"]["fp"] == 1
    assert _gate(scored, "photo_stitch_adjacency")["status"] == "measured_fail"


# --------------------------------------------------------------------------
# Repeatability
# --------------------------------------------------------------------------

ONE_ROOM = [("room", "Room", 0.0, 0.0, 5.0, 1.0, 2.40)]


def test_biased_but_repeatable_ceiling() -> None:
    gt = ground_truth(ONE_ROOM, [], [])
    a = prediction(gt, "cap-a", "lidar", values={"room:ceiling": 2.418})
    b = prediction(gt, "cap-b", "lidar", values={"room:ceiling": 2.421})
    scored = _score(gt, a, b)
    (room,) = scored["repeatability"]["ceiling"]
    assert room["errors"] == {
        "cap-a": pytest.approx(0.018),
        "cap-b": pytest.approx(0.021),
    }
    assert room["spread"] == pytest.approx(0.003)
    assert room["classification"] == "repeatable_but_biased"
    assert _gate(scored, "ceiling_height_error")["status"] == "measured_fail"
    assert _gate(scored, "ceiling_height_repeat_spread")["status"] == "measured_pass"


def test_accurate_but_unrepeatable_ceiling() -> None:
    gt = ground_truth(ONE_ROOM, [], [])
    a = prediction(gt, "cap-a", "lidar", values={"room:ceiling": 2.395})
    b = prediction(gt, "cap-b", "lidar", values={"room:ceiling": 2.412})
    scored = _score(gt, a, b)
    assert scored["repeatability"]["ceiling"][0]["classification"] == "unrepeatable"
    assert _gate(scored, "ceiling_height_error")["status"] == "measured_pass"
    assert _gate(scored, "ceiling_height_repeat_spread")["status"] == "measured_fail"


def test_wall_repeatability_or_and_readings() -> None:
    gt = ground_truth(ONE_ROOM, [], [])  # walls: S 5 m, E 1 m, N 5 m, W 1 m
    a = prediction(gt, "cap-a", "lidar")
    b = prediction(
        gt, "cap-b", "lidar", values={"room-S": 5.020, "room-E": 1.008, "room-N": 5.004}
    )
    scored = _score(gt, a, b)
    walls = {w["gt_id"]: w for w in scored["repeatability"]["walls"]}
    # S: d = 0.020; OR limit max(0.010, 0.025) = 0.025 pass; AND min = 0.010 fail.
    assert walls["room-S"]["d"] == pytest.approx(0.020)
    assert (walls["room-S"]["or_pass"], walls["room-S"]["and_pass"]) == (True, False)
    # E: d = 0.008; OR max(0.010, 0.005) pass; AND min = 0.005 fail.
    assert (walls["room-E"]["or_pass"], walls["room-E"]["and_pass"]) == (True, False)
    # N: d = 0.004 passes both; W: d = 0 passes both.
    assert walls["room-N"]["and_pass"] and walls["room-W"]["and_pass"]
    verdict = _gate(scored, "wall_repeatability")
    assert verdict["status"] == "measured_pass"
    assert verdict["and_reading"] == "fail"


def test_wall_missing_in_one_capture_fails_repeatability() -> None:
    gt = ground_truth(ONE_ROOM, [], [])
    scored = _score(
        gt,
        prediction(gt, "cap-a", "lidar"),
        prediction(gt, "cap-b", "lidar", drop={"room-E"}),
    )
    walls = {w["gt_id"]: w for w in scored["repeatability"]["walls"]}
    assert walls["room-E"]["status"] == "not_measured_in_every_capture"
    assert _gate(scored, "wall_repeatability")["status"] == "measured_fail"


def test_single_capture_repeatability_is_unverified() -> None:
    gt = ground_truth(ONE_ROOM, [], [])
    scored = _score(gt, prediction(gt, "cap-a", "lidar"))
    assert _gate(scored, "wall_repeatability")["status"] == "unverified"
    assert _gate(scored, "lidar_wall_length")["status"] == "unspecified_source"


def test_repeat_captures_must_be_distinct() -> None:
    gt = ground_truth(ONE_ROOM, [], [])
    with pytest.raises(ValueError, match="distinct"):
        _score(gt, prediction(gt, "cap-a", "lidar"), prediction(gt, "cap-a", "lidar"))


# --------------------------------------------------------------------------
# Calibration and incumbent
# --------------------------------------------------------------------------


def test_interval_score_and_coverage_by_hand() -> None:
    assert interval_score(2.38, 2.42, 2.40, 0.9) == pytest.approx(0.04)
    # Below the interval by 0.03 at 90 %: 0.04 + (2 / 0.1) * 0.03 = 0.64.
    assert interval_score(5.03, 5.07, 5.00, 0.9) == pytest.approx(0.64)
    gt = ground_truth(ONE_ROOM, [], [])
    pred = prediction(gt, values={"room-S": 5.05})  # interval [5.00, 5.10] covers 5
    rows = item_rows(
        PlanView.model_validate(gt),
        PlanView.model_validate(pred),
        match_plans(PlanView.model_validate(gt), PlanView.model_validate(pred)),
        "wall_length",
    )
    c = calibration(rows)
    assert (c["n_expected"], c["n_with_interval"], c["coverage"]) == (4, 4, 1.0)
    assert c["mean_width"] == pytest.approx(0.10)


def _incumbent(dims: list[dict]) -> IncumbentExport:
    return IncumbentExport(
        app_name="magicplan",
        app_version="9.9-test",
        export_file="magicplan_export.pdf",
        room_ids=["living", "kitchen"],
        dimensions=dims,
    )


def test_incumbent_missing_output_counts_as_our_loss() -> None:
    gt = ground_truth()
    pred = prediction(
        gt,
        tier="lidar",
        values={"living-S": 4.01, "living-E": 5.02, "living:ceiling": 2.41},
        drop={"kitchen-S"},
    )
    export = _incumbent(
        [
            {"gt_id": "living-S", "kind": "wall_length", "app_value": 4.03},  # win
            {"gt_id": "living-E", "kind": "wall_length", "app_value": 5.00},  # loss
            {"gt_id": "living-N", "kind": "wall_length", "app_value": None},  # excluded
            {
                "gt_id": "kitchen-S",
                "kind": "wall_length",
                "app_value": 3.02,
            },  # ours missing
            {"gt_id": "living", "kind": "ceiling_height", "app_value": 2.42},  # win
            {
                "gt_id": "kitchen-E",
                "kind": "wall_length",
                "app_value": 5.00,
            },  # tie (both 0)
        ]
    )
    scored = _score(gt, pred, incumbent=export)
    inc = scored["incumbent"]
    assert (inc["n_shared"], inc["wins"], inc["ties"], inc["losses"]) == (5, 2, 1, 2)
    assert inc["beat_or_tie_fraction"] == pytest.approx(3 / 5)
    outcomes = {row["gt_id"]: row["outcome"] for row in inc["table"]}
    assert outcomes["kitchen-S"] == "loss" and outcomes["living-N"] == "excluded"
    assert _gate(scored, "incumbent_head_to_head")["status"] == "measured_fail"


# --------------------------------------------------------------------------
# Product output, isolation and the command line
# --------------------------------------------------------------------------


def test_internal_result_projects_into_a_scoring_view() -> None:
    import runpy

    from floscan.contracts.result import PropertyResult

    fixtures = runpy.run_path(
        str(PROJECT_ROOT / "tests" / "contract" / "test_records.py")
    )
    result = PropertyResult.model_validate(fixtures["_complete_result"]())
    plan = plan_from_result(result, "case-1")
    (door,) = plan.openings
    # Door polygon u in [1.0, 1.86], v in [0, 2.03] on wall u = +x from origin.
    assert door.center == pytest.approx([1.43, 0.0])
    assert door.width.value == pytest.approx(0.86)
    assert plan.walls[0].length.value == pytest.approx(3.1)
    assert plan.rooms[0].ceiling_height.value == pytest.approx(2.45)
    assert plan.tier == "lidar" and plan.connected_components == 1


def test_inference_code_never_imports_the_evaluator() -> None:
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "src").rglob("*.py")
        if "import benchmark" in path.read_text()
        or "from benchmark" in path.read_text()
    ]
    assert offenders == []


def test_score_command_writes_csv_json_and_report(tmp_path: Path) -> None:
    output = tmp_path / "score"
    command = [
        str(PROJECT_ROOT / "run_benchmark.sh"),
        "score",
        "--ground-truth",
        str(FIXTURES / "ground_truth.json"),
        "--predictions",
        str(FIXTURES / "pred_photo_missing_hall.json"),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    capture = metrics["captures"]["cap-a"]
    assert capture["errors"]["wall_length"]["n_missing"] == 4
    assert capture["footprint"]["area_rel_error"] == pytest.approx(7.5 / 42.5)
    statuses = {v["id"]: v["status"] for v in metrics["gates"]["verdicts"]}
    assert statuses["photo_stitch_single_plan"] == "measured_fail"
    lines = (output / "items.csv").read_text().splitlines()
    assert lines[0].startswith("capture_id,kind,gt_id,pred_id,status")
    # The hall's 4 walls, its ceiling height and its floor area.
    assert sum(",missing," in line for line in lines) == 4 + 1 + 1
    assert (
        "| photo_stitch_single_plan | measured_fail |"
        in (output / "report.md").read_text()
    )
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode == 1 and "append-only" in again.stderr


def test_written_fixtures_match_the_builders() -> None:
    gt = ground_truth()
    assert json.loads((FIXTURES / "ground_truth.json").read_text()) == gt
    assert json.loads(
        (FIXTURES / "pred_photo_missing_hall.json").read_text()
    ) == prediction(gt, drop={"hall"})


def test_write_outputs_refuses_a_non_empty_directory(tmp_path: Path) -> None:
    gt = ground_truth()
    scored = _score(gt, prediction(gt))
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError, match="append-only"):
        write_outputs(scored, tmp_path)


def test_unlabelled_rooms_match_by_geometry() -> None:
    gt = ground_truth()
    pred = prediction(gt)
    for i, room in enumerate(pred["rooms"]):
        room["label"] = f"room_{i + 1:03d}"  # photo-folder names, not GT labels
    scored = _score(gt, pred)
    capture = scored["captures"]["cap-a"]
    assert capture["matching"]["alignment"]["angle_deg"] == pytest.approx(-30.0)
    assert capture["placement"]["all_rooms_matched"]
    assert capture["errors"]["wall_length"]["n_missing"] == 0
    assert _gate(scored, "photo_wall_length")["status"] == "measured_pass"


def test_per_capture_gate_needs_every_capture() -> None:
    gt = ground_truth()
    good = prediction(gt, "cap-a")
    bad = prediction(gt, "cap-b", values={"living-N": 4.4})  # 10 % > 8 %
    scored = _score(gt, good, bad)
    verdict = _gate(scored, "photo_wall_length")
    assert (
        verdict["observed"]["cap-a"]["pass"]
        and not verdict["observed"]["cap-b"]["pass"]
    )
    assert verdict["status"] == "measured_fail"


def test_centimetre_overlap_is_not_hidden_by_tolerance() -> None:
    gt = ground_truth()
    pred = prediction(gt, angle_deg=0.0, shift=(0.0, 0.0))
    pred["rooms"][1]["polygon"] = [[3.99, 0.0], [7.0, 0.0], [7.0, 5.0], [3.99, 5.0]]
    overlap = _score(gt, pred)["captures"]["cap-a"]["overlaps"]
    # A 1 cm strip along a 5 m wall: 0.05 m2, far above the 1e-6 m2 round-off.
    assert overlap["total_overlap_m2"] == pytest.approx(0.05)
