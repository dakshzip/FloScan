"""Gate verdicts from scored metrics, using thresholds from configs/gates.yaml.

Semantics are conservative and frozen before real data:

- A per-capture gate passes only if it passes in every scored capture.
- Missing, unmeasured or phantom items count against a gate; they are never
  dropped to improve a score.
- A gate with no measurable evidence in this run is ``unverified`` with a
  reason; a gate whose threshold the sources do not give is
  ``unspecified_source`` and still carries its metrics.
- Gates that do not apply to the run's tier are listed as excluded.

Thresholds are read from the validated gate registry; nothing here changes
them. The only numeric value defined here is the room-overlap numerical
tolerance that the registry leaves for P04 to freeze.
"""

from __future__ import annotations

from typing import Any

from floscan.pipeline import GateRegistry

# Frozen by P04 before any scoring: pairwise room overlap below this area is
# floating-point round-off on shared boundaries, not physical overlap. It is
# deliberately not a centimetre-scale buffer that could hide real overlap.
OVERLAP_TOLERANCE_M2 = 1e-6


def _criteria(gate: dict[str, Any]) -> list[float]:
    return [c["value"] for c in gate["criteria"]]


def _verdict(
    gate_id: str, passed: bool, observed: Any, threshold: Any, reason: str
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "status": "measured_pass" if passed else "measured_fail",
        "observed": observed,
        "threshold": threshold,
        "reason": reason,
    }


def _unverified(gate_id: str, reason: str, observed: Any = None) -> dict[str, Any]:
    return {
        "id": gate_id,
        "status": "unverified",
        "observed": observed,
        "threshold": None,
        "reason": reason,
    }


def evaluate_gates(
    registry: GateRegistry, tier: str, scored: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate every registry gate applicable to ``tier``.

    ``scored`` is the output of ``report.score_case``: per-capture metrics
    under ``captures`` plus cross-capture ``repeatability`` and ``incumbent``.
    """
    captures: dict[str, dict[str, Any]] = scored["captures"]
    verdicts, excluded = [], []
    for gate in registry.gates:
        gid = gate["id"]
        if tier not in gate["tiers"]:
            excluded.append(gid)
            continue
        if gate["source_status"] == "unspecified_source":
            verdicts.append(
                {
                    "id": gid,
                    "status": "unspecified_source",
                    "threshold": None,
                    "observed": _observed_for_unspecified(gid, captures),
                    "reason": "no threshold in the supplied sources; reported without "
                    "a verdict",
                }
            )
            continue
        verdicts.append(_evaluate(gid, _criteria(gate), tier, captures, scored))
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict["status"]] = counts.get(verdict["status"], 0) + 1
    return {
        "tier": tier,
        "verdicts": verdicts,
        "excluded_for_tier": excluded,
        "counts": counts,
    }


def _observed_for_unspecified(gid: str, captures: dict[str, Any]) -> Any:
    key = {"lidar_wall_length": "wall_length", "floor_area": "floor_area"}.get(gid)
    if key is None:
        return None
    return {c: m["errors"][key] for c, m in captures.items()}


def _every_capture(captures: dict[str, Any], check: Any) -> tuple[bool, dict]:
    observed = {c: check(m) for c, m in captures.items()}
    return all(o["pass"] for o in observed.values()), observed


def _evaluate(
    gid: str,
    criteria: list[float],
    tier: str,
    captures: dict[str, Any],
    scored: dict[str, Any],
) -> dict[str, Any]:
    if not captures:
        return _unverified(gid, "no capture was scored")

    if gid == "opening_width":
        tolerance, fraction = criteria
        if all(m["openings"]["n_gt"] == 0 for m in captures.values()):
            return _unverified(gid, "ground truth has no openings")
        passed, observed = _every_capture(
            captures,
            lambda m: {
                "score": m["openings"]["score"],
                "n_gt": m["openings"]["n_gt"],
                "fp": m["openings"]["fp"],
                "g": m["openings"]["g_within_tolerance"],
                "pass": m["openings"]["score"] is not None
                and m["openings"]["score"] >= fraction,
            },
        )
        return _verdict(
            gid,
            passed,
            observed,
            {"width_m": tolerance, "fraction": fraction},
            "G / (N + FP) in every capture (provisional denominator)",
        )

    if gid == "ceiling_height_error":
        (tolerance,) = criteria
        if all(
            m["tolerances"]["ceiling_height"]["n_expected"] == 0
            for m in captures.values()
        ):
            return _unverified(gid, "ground truth has no ceiling heights")
        passed, observed = _every_capture(
            captures,
            lambda m: {
                **m["tolerances"]["ceiling_height"],
                "pass": m["tolerances"]["ceiling_height"]["all_within"],
            },
        )
        return _verdict(
            gid,
            passed,
            observed,
            {"abs_m": tolerance},
            "every room in every capture; missing heights fail",
        )

    if gid == "ceiling_height_repeat_spread":
        rooms = scored["repeatability"]["ceiling"]
        if len(captures) < 2:
            return _unverified(gid, "fewer than two captures of the same tier")
        passed = bool(rooms) and all(
            r["spread"] is not None
            and r["spread"] <= criteria[0]
            and r["classification"] != "missing_in_capture"
            for r in rooms
        )
        return _verdict(
            gid,
            passed,
            rooms,
            {"spread_m": criteria[0]},
            "every repeated room; classification separates bias from variability",
        )

    if gid == "wall_repeatability":
        walls = scored["repeatability"]["walls"]
        if len(captures) < 2:
            return _unverified(gid, "fewer than two captures of the same tier")
        passed = bool(walls) and all(w["or_pass"] for w in walls)
        and_passed = bool(walls) and all(w["and_pass"] for w in walls)
        verdict = _verdict(
            gid,
            passed,
            walls,
            {"abs_m": criteria[0], "rel": criteria[1]},
            "OR reading (primary, provisional); "
            f"AND reading would be {'pass' if and_passed else 'fail'}",
        )
        verdict["and_reading"] = "pass" if and_passed else "fail"
        return verdict

    if gid in ("photo_wall_length", "video_wall_length"):
        (tolerance,) = criteria
        passed, observed = _every_capture(
            captures,
            lambda m: {
                **m["tolerances"]["wall_length"],
                "pass": m["tolerances"]["wall_length"]["all_within"],
            },
        )
        return _verdict(
            gid,
            passed,
            observed,
            {"rel": tolerance},
            "every expected wall in every capture; missing walls fail",
        )

    if gid == "photo_footprint":
        (tolerance,) = criteria
        passed, observed = _every_capture(
            captures,
            lambda m: {
                "area_rel_error": m["footprint"]["area_rel_error"],
                "iou": m["footprint"]["iou"],
                "extent_rel_errors": m["footprint"].get("extent_rel_errors"),
                "pass": m["footprint"]["area_rel_error"] is not None
                and m["footprint"]["area_rel_error"] <= tolerance,
            },
        )
        return _verdict(
            gid,
            passed,
            observed,
            {"area_rel": tolerance},
            "area of the union of rooms (provisional definition); "
            "extents and IoU reported alongside",
        )

    if gid in ("photo_stitch_adjacency", "stitched_plan_adjacency"):
        passed, observed = _every_capture(
            captures,
            lambda m: {
                **{k: m["adjacency"][k] for k in ("tp", "fn", "fp", "missed_edges")},
                "pass": m["adjacency"]["exact"],
            },
        )
        return _verdict(
            gid,
            passed,
            observed,
            "exact edge set",
            "missed, extra or unplaced edges fail",
        )

    if gid == "photo_stitch_no_overlap":
        passed, observed = _every_capture(
            captures,
            lambda m: {
                **m["overlaps"],
                "pass": not m["overlaps"]["overlapping_pairs"],
            },
        )
        return _verdict(
            gid,
            passed,
            observed,
            {"tolerance_m2": OVERLAP_TOLERANCE_M2},
            "no pairwise interior overlap beyond numerical tolerance",
        )

    if gid == "photo_stitch_single_plan":
        passed, observed = _every_capture(
            captures,
            lambda m: {
                **m["placement"],
                "pass": m["placement"]["all_rooms_matched"]
                and m["placement"]["connected_components"] == 1,
            },
        )
        return _verdict(
            gid,
            passed,
            observed,
            "one connected plan with every room",
            "a disconnected or single-room output fails",
        )

    if gid == "incumbent_head_to_head":
        comparison = scored.get("incumbent")
        if comparison is None or not comparison["n_shared"]:
            return _unverified(gid, "no incumbent export with shared dimensions")
        fraction = comparison["beat_or_tie_fraction"]
        return _verdict(
            gid,
            fraction >= criteria[0],
            comparison,
            {"fraction": criteria[0]},
            "missing outputs of ours count as losses",
        )

    if gid == "drift_accountability":
        return _unverified(gid, "needs the drift-correction ON/OFF ablation (P13)")
    if gid == "confidence_calibration":
        observed = {c: m["calibration"] for c, m in captures.items()}
        return _unverified(
            gid,
            "no numeric calibration band is supplied; empirical coverage is reported",
            observed,
        )
    return _unverified(gid, "not a benchmark measurement (deliverable gate)")
