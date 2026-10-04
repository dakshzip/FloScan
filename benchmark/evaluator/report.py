"""Score a benchmark case and write per-item CSV, metrics JSON and a report.

Usage (scoring existing predictions; live inference is not built yet):

    ./run_benchmark.sh score --ground-truth GT.json \\
        --predictions P.json [P2.json ...] \\
        --output runs/<run-id>/score [--correspondence C.json] [--incumbent I.json]

Predictions may be scoring views (``PlanView`` JSON) or FloScan results
(``internal-v0`` JSON), which are projected with ``plan_from_result``. Ground
truth is opened only here, by the evaluator, never by inference code.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmark.evaluator.gates import OVERLAP_TOLERANCE_M2, evaluate_gates
from benchmark.evaluator.matching import (
    MATCHING_POLICY,
    Correspondence,
    MatchResult,
    PlanView,
    load_json,
    match_plans,
    plan_from_result,
)
from benchmark.evaluator.metrics import (
    IncumbentExport,
    ItemRow,
    adjacency,
    calibration,
    ceiling_repeatability,
    error_summary,
    footprint,
    incumbent_comparison,
    item_rows,
    opening_detection,
    room_overlaps,
    tolerance_fraction,
    wall_repeatability,
)
from floscan.pipeline import DEFAULT_GATES_PATH, GateRegistry, load_gate_registry

KINDS = ("wall_length", "opening_width", "ceiling_height", "floor_area")
SCORER_VERSION = "scorer-v1"


def _threshold(registry: GateRegistry, gate_id: str, index: int = 0) -> float:
    gate = next(g for g in registry.gates if g["id"] == gate_id)
    return gate["criteria"][index]["value"]


def _wall_tolerance(registry: GateRegistry, tier: str) -> float | None:
    gate_id = {"photo": "photo_wall_length", "video": "video_wall_length"}.get(tier)
    return _threshold(registry, gate_id) if gate_id else None


def score_capture(
    gt: PlanView,
    pred: PlanView,
    registry: GateRegistry,
    correspondence: Correspondence | None = None,
) -> tuple[dict[str, Any], dict[str, list[ItemRow]], MatchResult]:
    """Match one capture and compute every per-capture metric."""
    match = match_plans(gt, pred, correspondence)
    rows = {kind: item_rows(gt, pred, match, kind) for kind in KINDS}
    wall_tolerance = _wall_tolerance(registry, pred.tier or "")
    tolerances = {
        "wall_length": tolerance_fraction(rows["wall_length"], True, wall_tolerance)
        if wall_tolerance is not None
        else {
            "n_expected": len(rows["wall_length"]),
            "fraction": None,
            "all_within": None,
            "note": f"no {pred.tier} wall tolerance in sources",
        },
        "ceiling_height": tolerance_fraction(
            rows["ceiling_height"], False, _threshold(registry, "ceiling_height_error")
        ),
    }
    pred_rooms = {r.id: r for r in pred.rooms}
    metrics = {
        "capture_id": pred.capture_id,
        "matching": {
            "policy": match.policy,
            "alignment": {
                "method": match.alignment.method,
                "pairs": match.alignment.pairs,
                "angle_deg": match.alignment.angle_deg,
                "translation_m": list(match.alignment.translation),
            },
            "notes": match.notes,
        },
        "errors": {kind: error_summary(rows[kind]) for kind in KINDS},
        "tolerances": tolerances,
        "openings": opening_detection(
            rows["opening_width"], _threshold(registry, "opening_width", 0)
        ),
        "footprint": footprint(gt, pred, match.alignment),
        "adjacency": adjacency(gt, pred, match),
        "overlaps": room_overlaps(pred, OVERLAP_TOLERANCE_M2),
        "placement": {
            "all_rooms_matched": all(
                p is not None and pred_rooms[p].polygon is not None
                for g, p in match.rooms
                if g is not None
            ),
            # Every GT room matched to a room placed in the property frame.
            "all_matched_rooms_placed": all(
                p is not None and pred_rooms[p].placement == "placed"
                for g, p in match.rooms
                if g is not None
            ),
            "unplaced_rooms": [r.id for r in pred.rooms if r.placement != "placed"],
            "registration_status": pred.registration_status,
            "connected_components": pred.connected_components,
            "stitched": pred.stitched(),
        },
        "calibration": {kind: calibration(rows[kind]) for kind in KINDS},
    }
    return metrics, rows, match


def score_case(
    gt: PlanView,
    predictions: Sequence[PlanView],
    registry: GateRegistry,
    correspondence: Correspondence | None = None,
    incumbent: IncumbentExport | None = None,
    incumbent_capture: str | None = None,
) -> dict[str, Any]:
    """Score every capture of one case at one tier, then evaluate the gates."""
    if gt.kind != "ground_truth":
        raise ValueError("the first document must be ground truth")
    if not predictions:
        raise ValueError("at least one prediction is required")
    tiers = {p.tier for p in predictions}
    if len(tiers) != 1:
        raise ValueError(f"one benchmark run scores one tier, got {sorted(tiers)}")
    capture_ids = [p.capture_id for p in predictions]
    if len(set(capture_ids)) != len(capture_ids):
        raise ValueError("capture ids must be distinct (a re-run is not a new capture)")
    for pred in predictions:
        if pred.kind != "prediction" or pred.case_id != gt.case_id:
            raise ValueError(
                f"prediction {pred.capture_id} is not for case {gt.case_id}"
            )
    tier = tiers.pop() or ""
    captures, all_rows = {}, {}
    for pred in predictions:
        metrics, rows, _ = score_capture(gt, pred, registry, correspondence)
        captures[pred.capture_id] = metrics
        all_rows[pred.capture_id] = rows
    scored: dict[str, Any] = {
        "scorer": SCORER_VERSION,
        "matching_policy": MATCHING_POLICY,
        "case_id": gt.case_id,
        "tier": tier,
        "gates_registry_sha256": registry.sha256,
        "captures": captures,
        "repeatability": {
            "ceiling": ceiling_repeatability(
                {c: r["ceiling_height"] for c, r in all_rows.items()},
                _threshold(registry, "ceiling_height_error"),
                _threshold(registry, "ceiling_height_repeat_spread"),
            ),
            "walls": wall_repeatability(
                {c: r["wall_length"] for c, r in all_rows.items()},
                _threshold(registry, "wall_repeatability", 0),
                _threshold(registry, "wall_repeatability", 1),
            )
            if len(all_rows) >= 2
            else [],
        },
        "incumbent": None,
    }
    if incumbent is not None:
        capture = incumbent_capture or sorted(all_rows)[0]
        rows = [row for kind_rows in all_rows[capture].values() for row in kind_rows]
        scored["incumbent"] = {
            "capture_id": capture,
            **incumbent_comparison(rows, incumbent, gt),
        }
    scored["gates"] = evaluate_gates(registry, tier, scored)
    scored["_rows"] = all_rows
    return scored


def write_outputs(scored: dict[str, Any], output: Path) -> list[Path]:
    """Write items.csv, metrics.json and report.md into a new directory."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; score runs are append-only")
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        row.as_dict()
        for per in scored["_rows"].values()
        for kind_rows in per.values()
        for row in kind_rows
    ]
    items = output / "items.csv"
    with items.open("x", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["capture_id"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metrics = output / "metrics.json"
    public = {k: v for k, v in scored.items() if not k.startswith("_")}
    with metrics.open("x", encoding="utf-8") as handle:
        json.dump(public, handle, indent=2, allow_nan=False, default=str)
        handle.write("\n")
    report = output / "report.md"
    with report.open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(public))
    return [items, metrics, report]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "n/a" if value is None else str(value)


def render_markdown(scored: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark score: {scored['case_id']} ({scored['tier']})",
        "",
        f"Scorer {scored['scorer']}, matching {scored['matching_policy']}, gate "
        f"registry sha256 {scored['gates_registry_sha256'][:12]}.",
        "Thresholds come from configs/gates.yaml; provisional interpretations are "
        "listed there.",
        "",
        "## Gates",
        "",
        "| Gate | Status | Reason |",
        "|---|---|---|",
    ]
    for verdict in scored["gates"]["verdicts"]:
        lines.append(f"| {verdict['id']} | {verdict['status']} | {verdict['reason']} |")
    lines += [
        "",
        f"Not applicable to the {scored['tier']} tier: "
        + (", ".join(scored["gates"]["excluded_for_tier"]) or "none")
        + ".",
        "",
    ]
    for capture, metrics in scored["captures"].items():
        lines += [
            f"## Capture {capture}",
            "",
            "| Quantity | Expected | Matched | Missing | Unmeasured | Phantom "
            "| MAE | Max rel |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for kind, s in metrics["errors"].items():
            lines.append(
                f"| {kind} | {s['n_expected']} | {s['n_matched']} | {s['n_missing']} | "
                f"{s['n_unmeasured']} | {s['n_phantom']} | {_fmt(s['mae'])} | "
                f"{_fmt(s['max_rel'])} |"
            )
        o, f = metrics["openings"], metrics["footprint"]
        lines += [
            "",
            f"Openings: N={o['n_gt']} TP={o['tp']} FN={o['fn']} FP={o['fp']} "
            f"G={o['g_within_tolerance']} score={_fmt(o['score'])}.",
            f"Footprint: area error {_fmt(f['area_rel_error'])}, IoU "
            f"{_fmt(f['iou'])}, Hausdorff {_fmt(f.get('hausdorff_m'))} m.",
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _load_prediction(path: Path, case_id: str) -> PlanView:
    data = load_json(path)
    if isinstance(data, dict) and data.get("schema_version") == "internal-v0":
        from floscan.contracts.result import PropertyResult

        return plan_from_result(PropertyResult.model_validate(data), case_id)
    return PlanView.model_validate(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_benchmark.sh score",
        description="Score existing predictions against sealed ground truth.",
    )
    parser.add_argument("command", choices=["score"])
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--incumbent-capture")
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES_PATH)
    args = parser.parse_args(argv)
    try:
        gt = PlanView.model_validate(load_json(args.ground_truth))
        predictions = [_load_prediction(p, gt.case_id) for p in args.predictions]
        correspondence = (
            Correspondence.model_validate(load_json(args.correspondence))
            if args.correspondence
            else None
        )
        incumbent = (
            IncumbentExport.model_validate(load_json(args.incumbent))
            if args.incumbent
            else None
        )
        scored = score_case(
            gt,
            predictions,
            load_gate_registry(args.gates),
            correspondence,
            incumbent,
            args.incumbent_capture,
        )
        paths = write_outputs(scored, args.output)
    except (OSError, ValueError) as error:
        print(f"run_benchmark.sh score: error: {error}", file=sys.stderr)
        return 1
    counts = scored["gates"]["counts"]
    print(
        f"scored {gt.case_id} ({scored['tier']}, {len(predictions)} capture(s)): "
        + ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    )
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
