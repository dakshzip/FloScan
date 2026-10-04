"""Reproduce the three P04 review findings against the committed scorer."""

import copy
import runpy
import sys
from pathlib import Path

# The evaluator lives outside the installed package; import it from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.evaluator.matching import PlanView, plan_from_result
from benchmark.evaluator.metrics import IncumbentExport
from benchmark.evaluator.report import score_case
from floscan.contracts.result import PropertyResult
from floscan.pipeline import load_gate_registry

REGISTRY = load_gate_registry()
records = runpy.run_path("tests/contract/test_records.py")
bench = runpy.run_path("tests/unit/test_benchmark.py")


def gate(scored, gate_id):
    return next(v for v in scored["gates"]["verdicts"] if v["id"] == gate_id)["status"]


def as_ground_truth(plan):
    data = plan.model_dump()
    data.update(kind="ground_truth", capture_id=None, tier=None)
    return PlanView.model_validate(data)


print("## Finding 1: unplaced local geometry and whole-property gates")
placed = records["_complete_result"]()
placed["run"]["tier"] = "photo"
gt = as_ground_truth(plan_from_result(PropertyResult.model_validate(placed), "case-1"))
unplaced = copy.deepcopy(placed)
unplaced["status"], unplaced["status_reason"] = "partial", "room not placed"
unplaced["rooms"][0]["placement_status"] = "unplaced"
unplaced["rooms"][0]["T_property_from_room"] = None
unplaced["property_graph"]["registration_status"] = "not_attempted"
unplaced["coverage"]["stitched_plan"] = {"status": "partial", "reason": "not stitched"}
result = PropertyResult.model_validate(unplaced)
scored = score_case(gt, [plan_from_result(result, "case-1")], REGISTRY)
for gate_id in ("photo_stitch_single_plan", "photo_footprint"):
    print(f"  {gate_id}: {gate(scored, gate_id)}")

print("## Finding 2: incumbent two-room set and duplicates")
gt3 = PlanView.model_validate(bench["ground_truth"]())
pred = PlanView.model_validate(
    bench["prediction"](bench["ground_truth"](), tier="lidar")
)
one = {"gt_id": "living-S", "kind": "wall_length", "app_value": 4.05}
for name, dims in (("only living-S", [one]), ("living-S five times", [one] * 5)):
    try:
        export = IncumbentExport(
            app_name="x",
            app_version="1",
            export_file="x.pdf",
            room_ids=["living", "kitchen"],
            dimensions=dims,
        )
        s = score_case(gt3, [pred], REGISTRY, incumbent=export)
        print(
            f"  {name}: {gate(s, 'incumbent_head_to_head')} "
            f"(n_shared={s['incumbent']['n_shared']})"
        )
    except ValueError as error:  # pydantic ValidationError is a ValueError
        print(f"  {name}: rejected ({str(error).splitlines()[1].strip()[:80]})")

print("## Finding 3: polygon holes")
holed = records["_result"]()
holed["rooms"][0]["boundary"]["holes"] = [[[1, 1], [1, 2], [2, 2], [2, 1]]]
r = PropertyResult.model_validate(holed)
plan = plan_from_result(r, "case-1")
from shapely.geometry import Polygon  # noqa: E402

print(
    f"  internal area {r.rooms[0].boundary.area():.2f} m2, scoring shape area "
    f"{plan.rooms[0].shape().area:.2f} m2 (outer ring only: "
    f"{Polygon(plan.rooms[0].polygon).area:.2f} m2), holes kept: "
    f"{len(plan.rooms[0].holes)}"
)
