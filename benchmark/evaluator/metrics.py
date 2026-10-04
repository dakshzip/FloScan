"""Metric formulas of 04-benchmarks.md over matched scoring items.

Every expected ground-truth item yields a row, matched or not, so missing and
phantom entities stay in the denominators. Errors are ``e = p - y``,
``a = |e|`` and ``r = a / y``. Conditional error statistics use available
matches only, but coverage and gates always count every expected item.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator
from shapely.geometry import Polygon
from shapely.ops import unary_union

from benchmark.evaluator.matching import (
    Estimate,
    MatchResult,
    PlanView,
    RigidAlignment2D,
)
from floscan.contracts.base import Contract, Id, Text

ItemKind = Literal["wall_length", "opening_width", "ceiling_height", "floor_area"]
ItemStatus = Literal["matched", "unmeasured", "missing", "phantom"]
BOUNDARY_SAMPLE_SPACING_M = 0.05


@dataclass(frozen=True)
class ItemRow:
    """One expected or predicted item in one capture."""

    capture_id: str
    kind: str
    gt_id: str | None
    pred_id: str | None
    status: str
    gt_value: float | None
    pred_value: float | None
    lower: float | None
    upper: float | None
    nominal_level: float | None
    gt_uncertainty: float | None

    @property
    def error(self) -> float | None:
        if self.status != "matched" or self.gt_value is None or self.pred_value is None:
            return None
        return self.pred_value - self.gt_value

    @property
    def abs_error(self) -> float | None:
        return None if self.error is None else abs(self.error)

    @property
    def rel_error(self) -> float | None:
        if self.abs_error is None or not self.gt_value:
            return None
        return self.abs_error / self.gt_value

    @property
    def covered(self) -> bool | None:
        if self.gt_value is None or self.lower is None or self.upper is None:
            return None
        return self.lower <= self.gt_value <= self.upper

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.update(
            error=self.error,
            abs_error=self.abs_error,
            rel_error=self.rel_error,
            covered=self.covered,
        )
        return row


def summarize(errors: list[float]) -> dict[str, Any]:
    """N, mean signed, MAE, RMSE, median, P90, P95 and max of signed errors."""
    if not errors:
        return {
            "n": 0,
            "mean_signed": None,
            "mae": None,
            "rmse": None,
            "median_abs": None,
            "p90_abs": None,
            "p95_abs": None,
            "max_abs": None,
        }
    e = np.asarray(errors, dtype=np.float64)
    a = np.abs(e)
    return {
        "n": int(len(e)),
        "mean_signed": float(e.mean()),
        "mae": float(a.mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90)),
        "p95_abs": float(np.percentile(a, 95)),
        "max_abs": float(a.max()),
    }


# --------------------------------------------------------------------------
# Item rows
# --------------------------------------------------------------------------


def _estimates(plan: PlanView, kind: str) -> dict[str, Estimate | None]:
    if kind == "wall_length":
        return {w.id: w.length for w in plan.walls}
    if kind == "opening_width":
        return {o.id: o.width for o in plan.openings}
    if kind == "ceiling_height":
        return {r.id: r.ceiling_height for r in plan.rooms}
    if kind == "floor_area":
        return {r.id: r.floor_area for r in plan.rooms}
    raise ValueError(f"unknown item kind {kind!r}")


_PAIRS = {
    "wall_length": "walls",
    "opening_width": "openings",
    "ceiling_height": "rooms",
    "floor_area": "rooms",
}


def item_rows(
    gt: PlanView, pred: PlanView, match: MatchResult, kind: str
) -> list[ItemRow]:
    """Rows for every ground-truth item of ``kind`` and every phantom prediction."""
    truth = _estimates(gt, kind)
    guess = _estimates(pred, kind)
    rows = []
    for gid, pid in getattr(match, _PAIRS[kind]):
        g = truth.get(gid) if gid is not None else None
        p = guess.get(pid) if pid is not None else None
        if gid is not None and g is None:
            continue  # ground truth did not measure this quantity for the entity
        if gid is None:
            status = "phantom"
        elif pid is None:
            status = "missing"
        elif p is None:
            status = "unmeasured"
        else:
            status = "matched"
        if (
            status == "phantom"
            and kind in ("ceiling_height", "floor_area")
            and p is None
        ):
            continue
        rows.append(
            ItemRow(
                capture_id=pred.capture_id or "",
                kind=kind,
                gt_id=gid,
                pred_id=pid,
                status=status,
                gt_value=g.value if g else None,
                pred_value=p.value if p else None,
                lower=p.lower if p else None,
                upper=p.upper if p else None,
                nominal_level=p.nominal_level if p else None,
                gt_uncertainty=g.uncertainty if g else None,
            )
        )
    return rows


def expected(rows: list[ItemRow]) -> list[ItemRow]:
    return [r for r in rows if r.gt_id is not None]


def error_summary(rows: list[ItemRow]) -> dict[str, Any]:
    """Conditional error statistics plus coverage of every expected item."""
    exp = expected(rows)
    matched = [r for r in exp if r.status == "matched"]
    summary = summarize([r.error for r in matched if r.error is not None])
    rel = [r.rel_error for r in matched if r.rel_error is not None]
    summary.update(
        n_expected=len(exp),
        n_matched=len(matched),
        n_unmeasured=sum(r.status == "unmeasured" for r in exp),
        n_missing=sum(r.status == "missing" for r in exp),
        n_phantom=sum(r.status == "phantom" for r in rows),
        max_rel=float(max(rel)) if rel else None,
        median_rel=float(np.median(rel)) if rel else None,
    )
    return summary


def tolerance_fraction(rows: list[ItemRow], relative: bool, tolerance: float) -> dict:
    """Share of expected items within tolerance; missing or unmeasured never pass."""
    exp = expected(rows)
    within = 0
    for row in exp:
        value = row.rel_error if relative else row.abs_error
        if value is not None and value <= tolerance:
            within += 1
    return {
        "n_expected": len(exp),
        "n_within": within,
        "fraction": within / len(exp) if exp else None,
        "all_within": bool(exp) and within == len(exp),
    }


# --------------------------------------------------------------------------
# Openings
# --------------------------------------------------------------------------


def opening_detection(rows: list[ItemRow], width_tolerance_m: float) -> dict[str, Any]:
    """Detection and width score ``G / (N + FP)`` (provisional official policy).

    N counts ground-truth openings, so misses stay in the denominator; FP
    counts phantom detections, including duplicates left after one-to-one
    matching; G counts matched openings whose width is within tolerance.
    """
    n = len(expected(rows))
    tp = sum(r.status in ("matched", "unmeasured") for r in rows)
    fn = sum(r.status == "missing" for r in rows)
    fp = sum(r.status == "phantom" for r in rows)
    good = sum(
        r.status == "matched"
        and r.abs_error is not None
        and r.abs_error <= width_tolerance_m
        for r in rows
    )
    return {
        "n_gt": n,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "g_within_tolerance": good,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / n if n else None,
        "score": good / (n + fp) if n + fp else None,
        "width_errors": summarize([r.error for r in rows if r.error is not None]),
    }


# --------------------------------------------------------------------------
# Repeatability (several captures of the same tier)
# --------------------------------------------------------------------------


def _by_gt(rows_by_capture: dict[str, list[ItemRow]]) -> dict[str, dict[str, ItemRow]]:
    table: dict[str, dict[str, ItemRow]] = {}
    for capture, rows in rows_by_capture.items():
        for row in expected(rows):
            table.setdefault(row.gt_id or "", {})[capture] = row
    return table


def ceiling_repeatability(
    rows_by_capture: dict[str, list[ItemRow]],
    error_tolerance_m: float,
    spread_tolerance_m: float,
) -> list[dict[str, Any]]:
    """Per room: every capture's error, the spread, and bias versus variability."""
    captures = sorted(rows_by_capture)
    result = []
    for gid, per in sorted(_by_gt(rows_by_capture).items()):
        values = {
            c: per[c].pred_value if c in per and per[c].status == "matched" else None
            for c in captures
        }
        errors = {c: per[c].error if c in per else None for c in captures}
        measured = [v for v in values.values() if v is not None]
        accurate = all(
            e is not None and abs(e) <= error_tolerance_m for e in errors.values()
        )
        spread = max(measured) - min(measured) if len(measured) >= 2 else None
        if len(captures) < 2:
            label = "single_capture"
        elif len(measured) < len(captures):
            label = "missing_in_capture"
        else:
            repeatable = spread is not None and spread <= spread_tolerance_m
            label = {
                (True, True): "accurate_and_repeatable",
                (False, True): "repeatable_but_biased",
                (True, False): "unrepeatable",
                (False, False): "biased_and_unrepeatable",
            }[(accurate, repeatable)]
        signed = [e for e in errors.values() if e is not None]
        result.append(
            {
                "gt_id": gid,
                "values": values,
                "errors": errors,
                "mean_signed_error": float(np.mean(signed)) if signed else None,
                "spread": spread,
                "classification": label,
            }
        )
    return result


def wall_repeatability(
    rows_by_capture: dict[str, list[ItemRow]],
    abs_tolerance_m: float,
    rel_tolerance: float,
) -> list[dict[str, Any]]:
    """Per physical wall: spread across captures with OR and AND readings.

    OR (primary, provisional): d <= max(abs, rel * y_gt).
    AND (sensitivity): d <= min(abs, rel * y_gt).
    A wall not measured in every capture fails both readings.
    """
    captures = sorted(rows_by_capture)
    result = []
    for gid, per in sorted(_by_gt(rows_by_capture).items()):
        values = [
            per[c].pred_value if c in per and per[c].status == "matched" else None
            for c in captures
        ]
        y = next((r.gt_value for r in per.values() if r.gt_value is not None), None)
        if any(v is None for v in values) or y is None or len(values) < 2:
            result.append(
                {
                    "gt_id": gid,
                    "values": values,
                    "gt_value": y,
                    "d": None,
                    "abs_pass": False,
                    "rel_pass": False,
                    "or_pass": False,
                    "and_pass": False,
                    "status": "not_measured_in_every_capture",
                }
            )
            continue
        measured = [v for v in values if v is not None]
        d = max(measured) - min(measured)
        abs_pass = d <= abs_tolerance_m
        rel_pass = d <= rel_tolerance * y
        result.append(
            {
                "gt_id": gid,
                "values": values,
                "gt_value": y,
                "d": d,
                "relative_d": d / y,
                "abs_pass": abs_pass,
                "rel_pass": rel_pass,
                "or_pass": abs_pass or rel_pass,
                "and_pass": abs_pass and rel_pass,
                "status": "measured",
            }
        )
    return result


# --------------------------------------------------------------------------
# Footprint, shape, adjacency and overlap
# --------------------------------------------------------------------------


def _union(shapes: list[Any]) -> Any:
    return unary_union(shapes) if shapes else Polygon()


def _rings(shape: Any) -> list[Any]:
    """Exterior and interior (hole) rings of a polygon or multipolygon."""
    polygons = (
        [shape] if shape.geom_type == "Polygon" else list(getattr(shape, "geoms", []))
    )
    rings = []
    for polygon in polygons:
        rings.append(polygon.exterior)
        rings.extend(polygon.interiors)
    return rings


def _boundary_samples(shape: Any, spacing: float) -> np.ndarray:
    lines = _rings(shape)
    points = []
    for line in lines:
        n = max(int(np.ceil(line.length / spacing)), 1)
        points += [line.interpolate(i * line.length / n).coords[0] for i in range(n)]
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def footprint(
    gt: PlanView, pred: PlanView, alignment: RigidAlignment2D
) -> dict[str, Any]:
    """Union-of-rooms area, extents, IoU and boundary distances (rigid only).

    Only placed predicted rooms enter the union; unplaced rooms are in local
    frames and cannot contribute to a property footprint. Holes are kept. The
    prediction is moved by the single property-wide rigid alignment; it is
    never scaled or mirrored, so scale error stays visible.
    """
    g = _union([r.shape() for r in gt.rooms if r.polygon])
    p = _union([r.shape(alignment) for r in pred.placed_rooms() if r.polygon])
    area_gt, area_pred = float(g.area), float(p.area)
    result: dict[str, Any] = {
        "unplaced_rooms_excluded": [
            r.id for r in pred.rooms if r.placement != "placed"
        ],
        "area_gt_m2": area_gt,
        "area_pred_m2": area_pred,
        "area_rel_error": abs(area_pred - area_gt) / area_gt if area_gt else None,
        "iou": float(g.intersection(p).area / g.union(p).area)
        if not p.is_empty
        else 0.0,
    }
    if p.is_empty:
        result.update(
            extent_rel_errors=None,
            boundary_mean_m=None,
            boundary_p95_m=None,
            hausdorff_m=None,
        )
        return result
    gx0, gy0, gx1, gy1 = g.bounds
    px0, py0, px1, py1 = p.bounds
    result["extents_gt_m"] = [gx1 - gx0, gy1 - gy0]
    result["extents_pred_m"] = [px1 - px0, py1 - py0]
    result["extent_rel_errors"] = [
        abs((px1 - px0) - (gx1 - gx0)) / (gx1 - gx0),
        abs((py1 - py0) - (gy1 - gy0)) / (gy1 - gy0),
    ]
    gs = _boundary_samples(g, BOUNDARY_SAMPLE_SPACING_M)
    ps = _boundary_samples(p, BOUNDARY_SAMPLE_SPACING_M)
    d_gp = np.array([g.boundary.distance(_pt(x)) for x in ps])
    d_pg = np.array([p.boundary.distance(_pt(x)) for x in gs])
    both = np.concatenate([d_gp, d_pg])
    result.update(
        boundary_mean_m=float(both.mean()),
        boundary_p95_m=float(np.percentile(both, 95)),
        hausdorff_m=float(both.max()),
    )
    return result


def _pt(xy: Any) -> Any:
    from shapely.geometry import Point

    return Point(float(xy[0]), float(xy[1]))


def adjacency(gt: PlanView, pred: PlanView, match: MatchResult) -> dict[str, Any]:
    """Undirected traversable room-edge sets compared after identity matching.

    An edge counts only between placed rooms; an edge touching an unplaced
    room is unknown, so it is an extra edge and its true edge stays missed.
    """
    placed = {r.id for r in pred.placed_rooms()}
    to_gt = {
        p: g for g, p in match.rooms if g is not None and p is not None and p in placed
    }
    truth = {frozenset(e) for e in gt.adjacency}
    mapped, unmapped = set(), 0
    for edge in pred.adjacency:
        ends = [to_gt.get(r) for r in edge]
        if None in ends:
            unmapped += 1
        else:
            mapped.add(frozenset(ends))
    tp = len(truth & mapped)
    fn = len(truth - mapped)
    fp = len(mapped - truth) + unmapped
    return {
        "n_gt_edges": len(truth),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / len(truth) if truth else None,
        "exact": fn == 0
        and fp == 0
        and all(g is None or (p is not None and p in placed) for g, p in match.rooms),
        "missed_edges": sorted(sorted(e) for e in truth - mapped),
    }


def room_overlaps(pred: PlanView, tolerance_m2: float) -> dict[str, Any]:
    """Pairwise interior overlap of placed rooms (shared edges are not overlap).

    Holes are kept, so a room sitting inside another room's hole does not
    overlap it.
    """
    polygons = {r.id: r.shape() for r in pred.placed_rooms() if r.polygon}
    pairs = []
    for a, b in combinations(sorted(polygons), 2):
        area = float(polygons[a].intersection(polygons[b]).area)
        if area > tolerance_m2:
            pairs.append({"rooms": [a, b], "area_m2": area})
    return {
        "tolerance_m2": tolerance_m2,
        "overlapping_pairs": pairs,
        "total_overlap_m2": float(sum(p["area_m2"] for p in pairs)),
    }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def interval_score(lower: float, upper: float, y: float, nominal_level: float) -> float:
    """Winkler interval score for a central interval at ``nominal_level``."""
    alpha = 1.0 - nominal_level
    score = upper - lower
    if y < lower:
        score += 2.0 / alpha * (lower - y)
    if y > upper:
        score += 2.0 / alpha * (y - upper)
    return score


def calibration(rows: list[ItemRow]) -> dict[str, Any]:
    """Empirical coverage, width and interval score; unavailable stays counted."""
    exp = expected(rows)
    # (lower, upper, truth, level, covered) for items with a finite interval.
    finite = [
        (r.lower, r.upper, r.gt_value, r.nominal_level, bool(r.covered))
        for r in exp
        if r.lower is not None and r.upper is not None and r.gt_value is not None
    ]
    scored = [(lo, up, y, level) for lo, up, y, level, _ in finite if level is not None]
    return {
        "n_expected": len(exp),
        "n_with_interval": len(finite),
        "n_unavailable": len(exp) - len(finite),
        "coverage": sum(c for *_, c in finite) / len(finite) if finite else None,
        "nominal_levels": sorted({level for *_, level in scored}),
        "mean_width": float(np.mean([up - lo for lo, up, *_ in finite]))
        if finite
        else None,
        "mean_interval_score": float(
            np.mean([interval_score(lo, up, y, level) for lo, up, y, level in scored])
        )
        if scored
        else None,
    }


# --------------------------------------------------------------------------
# Incumbent comparison
# --------------------------------------------------------------------------


class IncumbentDimension(Contract):
    gt_id: Id
    kind: ItemKind
    app_value: float | None


class IncumbentExport(Contract):
    """Shared dimension set, frozen before any of our errors are inspected.

    ``room_ids`` names the two distinct benchmark rooms compared; every
    dimension must belong to one of them and appear once.
    """

    app_name: Text
    app_version: Text
    export_file: Text
    room_ids: list[Id] = Field(min_length=2, max_length=2)
    dimensions: list[IncumbentDimension] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> IncumbentExport:
        if self.room_ids[0] == self.room_ids[1]:
            raise ValueError("the incumbent comparison needs two distinct rooms")
        keys = [(d.kind, d.gt_id) for d in self.dimensions]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"duplicate incumbent dimensions {duplicates}")
        return self


def _owner_room(gt: PlanView, kind: str, gt_id: str) -> str:
    """The ground-truth room that owns a compared dimension."""
    if kind in ("ceiling_height", "floor_area"):
        if gt_id not in {r.id for r in gt.rooms}:
            raise ValueError(f"incumbent dimension {kind} {gt_id}: unknown room")
        return gt_id
    table = gt.walls if kind == "wall_length" else gt.openings
    for item in table:
        if item.id == gt_id:
            return item.room_id
    raise ValueError(f"incumbent dimension {kind} {gt_id}: not in ground truth")


def incumbent_comparison(
    rows: list[ItemRow], export: IncumbentExport, gt: PlanView
) -> dict[str, Any]:
    """Win or tie when |ours - gt| <= |theirs - gt| at unrounded precision.

    Dimensions the app does not report are excluded from the shared set S; a
    dimension the app reports but we do not counts as our loss. Both declared
    rooms must be ground-truth rooms and every dimension must belong to one of
    them; ``covers_both_rooms`` says whether each room has a shared dimension.
    """
    gt_rooms = {r.id for r in gt.rooms}
    unknown = [r for r in export.room_ids if r not in gt_rooms]
    if unknown:
        raise ValueError(f"incumbent rooms {unknown} are not ground-truth rooms")
    owners = {}
    for dim in export.dimensions:
        owner = _owner_room(gt, dim.kind, dim.gt_id)
        if owner not in export.room_ids:
            raise ValueError(
                f"incumbent dimension {dim.kind} {dim.gt_id} belongs to room {owner}, "
                f"outside the declared rooms {export.room_ids}"
            )
        owners[(dim.kind, dim.gt_id)] = owner
    ours = {(r.kind, r.gt_id): r for r in expected(rows)}
    shared_per_room = dict.fromkeys(export.room_ids, 0)
    table, wins, ties, losses = [], 0, 0, 0
    for dim in export.dimensions:
        row = ours.get((dim.kind, dim.gt_id))
        gt_value = row.gt_value if row else None
        if dim.app_value is None:
            table.append(
                {
                    "gt_id": dim.gt_id,
                    "kind": dim.kind,
                    "outcome": "excluded",
                    "reason": "not reported by the incumbent",
                }
            )
            continue
        if gt_value is None:
            raise ValueError(f"incumbent dimension {dim.gt_id} has no ground truth")
        their_abs = abs(dim.app_value - gt_value)
        our_abs = row.abs_error if row and row.status == "matched" else None
        if our_abs is None:
            outcome = "loss"
        elif our_abs < their_abs:
            outcome = "win"
        elif our_abs == their_abs:
            outcome = "tie"
        else:
            outcome = "loss"
        wins += outcome == "win"
        ties += outcome == "tie"
        losses += outcome == "loss"
        shared_per_room[owners[(dim.kind, dim.gt_id)]] += 1
        table.append(
            {
                "gt_id": dim.gt_id,
                "kind": dim.kind,
                "gt": gt_value,
                "ours": row.pred_value if row else None,
                "theirs": dim.app_value,
                "our_abs_error": our_abs,
                "their_abs_error": their_abs,
                "outcome": outcome,
            }
        )
    shared = wins + ties + losses
    return {
        "app": f"{export.app_name} {export.app_version}",
        "export_file": export.export_file,
        "rooms": export.room_ids,
        "n_shared": shared,
        "shared_per_room": shared_per_room,
        "covers_both_rooms": all(n > 0 for n in shared_per_room.values()),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "beat_or_tie_fraction": (wins + ties) / shared if shared else None,
        "table": table,
    }
