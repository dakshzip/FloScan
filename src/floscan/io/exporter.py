"""Local LiDAR result: rooms, measurements and plan as a validated PropertyResult.

``finish_lidar`` runs after reconstruction: rooms.build (P08), then
measurements.evaluate, render.plan and export.serialize. The result is the
project-owned ``internal-v0`` PropertyResult, validated record by record
and section by section before it is written; the run envelope points to it.
Coverage labels describe exactly what the records contain: partial where
evidence is incomplete, unavailable where no stage exists.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from floscan import __version__
from floscan.contracts.base import (
    CAPTURE_SCHEMA_VERSION,
    INTERNAL_SCHEMA_VERSION,
    Provenance,
)
from floscan.contracts.capture import Capture
from floscan.contracts.geometry import CoordinateFrame, ScaleEstimate
from floscan.contracts.result import (
    Diagnostic,
    PropertyResult,
    RunManifest,
    SectionCoverage,
)
from floscan.geometry.rooms import build_rooms, load_bundle, write_rooms
from floscan.io.assets import asset_ref, sha256_file
from floscan.measurements.engine import measure
from floscan.render.svg import build_drawing, to_png, to_svg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECORDS_FILENAME = "property_result.json"
EXTERNAL_SCHEMA_REASON = (
    "The assignment's published JSON schema was not supplied; no exporter to it "
    "exists and no conformance is claimed."
)


class ExportError(RuntimeError):
    """The result cannot be identified or assembled."""


@dataclass
class LocalResult:
    stages: list[dict[str, str]]
    diagnostics: list[dict[str, str]]
    status: str
    status_reason: str
    records: dict[str, str] | None
    sections: dict[str, dict[str, str]] = field(default_factory=dict)


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None


def run_manifest(
    run_id: str,
    mode: str,
    input_hashes: list[str],
    config_hash: str,
    started_utc: str,
    stage_times: dict[str, float],
    outputs: list,
) -> RunManifest:
    """Identify the code, inputs and configuration of this run.

    Raises:
        ExportError: if the code is not a git checkout (its identity would
            otherwise be invented).
    """
    commit = (_git("rev-parse", "HEAD") or "").strip()
    if len(commit) != 40:
        raise ExportError(
            "the code is not a git checkout; the run cannot name its code version"
        )
    diff = _git("diff", "HEAD") or ""
    lock = PROJECT_ROOT / "uv.lock"
    return RunManifest(
        id=run_id,
        git_commit=commit,
        dirty_diff_hash=hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        tier="lidar",
        mode=mode,
        input_manifest_hashes=input_hashes,
        config_hash=config_hash,
        schema_versions={
            "result": INTERNAL_SCHEMA_VERSION,
            "capture": CAPTURE_SCHEMA_VERSION,
        },
        package_lock_hash=sha256_file(lock),
        platform={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "floscan": __version__,
        },
        seeds={"ransac": 0},
        deterministic=True,
        started_utc=started_utc,
        ended_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        stage_times_s=stage_times,
        outputs=outputs,
    )


def _combined_hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def finish_lidar(output_dir: Path, run_id: str, mode: str) -> LocalResult:
    """rooms.build, measurements.evaluate, render.plan and export.serialize."""
    started = datetime.now(UTC)
    stages: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []
    times: dict[str, float] = {}

    def stage(name: str, status: str, reason: str) -> None:
        stages.append({"name": name, "status": status, "reason": reason})

    def timed(name: str, function, *args):
        tick = datetime.now(UTC)
        value = function(*args)
        times[name] = (datetime.now(UTC) - tick).total_seconds()
        return value

    capture = Capture.model_validate_json(
        (output_dir / "capture" / "capture.json").read_text("utf-8")
    )
    bundle_dir = output_dir / "reconstruction"
    bundle = json.loads((bundle_dir / "bundle.json").read_text("utf-8"))
    model = timed("rooms.build", lambda: build_rooms(load_bundle(bundle_dir)))
    write_rooms(output_dir / "rooms", model)
    if not model.rooms:
        reason = "no room could be built: " + "; ".join(model.reasons)
        stage("rooms.build", "insufficient_evidence", reason)
        return LocalResult(stages, diagnostics, "insufficient_evidence", reason, None)
    stage(
        "rooms.build",
        "ok",
        f"{len(model.rooms)} room hypotheses ({model.status}); see rooms/rooms.json",
    )
    provenance = Provenance(
        stage="measurements.evaluate",
        stage_version=f"floscan {__version__}",
        mode=mode,
        upstream_artifact_hashes=[sha256_file(output_dir / "rooms" / "rooms.json")],
    )
    measured = timed(
        "measurements.evaluate",
        measure,
        model.rooms,
        model.walls,
        model.surfaces,
        provenance,
    )
    stage(
        "measurements.evaluate",
        "ok",
        f"{measured.summary['measurements']} measurements "
        f"{measured.summary['quality']}; intervals unavailable (no calibration)",
    )
    partial = [r for r in model.rooms if r.status != "ok"]
    title = (
        f"FloScan plan, session world W (metres): {len(model.rooms)} room(s), "
        f"{len(partial)} partial; accuracy not verified"
    )
    drawing = build_drawing(
        measured.rooms, measured.walls, measured.measurements, title
    )
    plan_dir = output_dir / "plan"
    plan_dir.mkdir(exist_ok=False)
    (plan_dir / "plan.svg").write_text(to_svg(drawing), "utf-8")
    to_png(drawing).save(plan_dir / "plan.png")
    artifacts = [
        asset_ref(output_dir, "plan/plan.svg"),
        asset_ref(output_dir, "plan/plan.png"),
    ]
    artifacts = [
        a.model_copy(update={"mime_type": "image/svg+xml"})
        if a.uri.endswith(".svg")
        else a
        for a in artifacts
    ]
    stage("render.plan", "ok", "plan/plan.svg and plan/plan.png drawn from the records")
    result = _assemble(
        output_dir,
        run_id,
        mode,
        capture,
        bundle,
        model,
        measured,
        artifacts,
        times,
        started,
    )
    path = output_dir / RECORDS_FILENAME
    text = result.to_json()
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
    stage(
        "export.serialize",
        "ok",
        f"{RECORDS_FILENAME} ({INTERNAL_SCHEMA_VERSION}, project-owned) validated "
        "and written",
    )
    for item in result.diagnostics:
        diagnostics.append({"code": item.code, "message": item.message})
    sections = {
        name: {
            "status": section.status,
            "reason": section.reason or f"available: see {RECORDS_FILENAME}",
        }
        for name, section in result.coverage.items()
    }
    return LocalResult(
        stages,
        diagnostics,
        "partial",
        result.status_reason or "",
        {
            "path": RECORDS_FILENAME,
            "sha256": sha256_file(path),
            "schema_version": INTERNAL_SCHEMA_VERSION,
        },
        sections,
    )


def _assemble(
    output_dir: Path,
    run_id: str,
    mode: str,
    capture: Capture,
    bundle: dict[str, Any],
    model,
    measured,
    artifacts,
    times,
    started: datetime,
) -> PropertyResult:
    frame = CoordinateFrame(
        id="W",
        kind="session_world",
        unit="m",
        description="LiDAR session world: source ARKit world rotated to +z up "
        "(documented up sign, unverified); rooms are not placed in a property frame",
    )
    scale = ScaleEstimate(
        id="scale:session",
        component_id="W",
        multiplier=1.0,
        evidence_ids=[capture.id],
        status="sensor_metric",
    )
    rooms = measured.rooms
    partial_rooms = [r.id for r in rooms if r.status != "ok"]
    quality = measured.summary["quality"]
    coverage = {
        "capture": SectionCoverage(status="available", evidence_ids=[]),
        "coordinate_frames": SectionCoverage(status="available"),
        "scale": SectionCoverage(status="available"),
        "per_room_plan": SectionCoverage(
            status="partial",
            reason=(
                f"{len(partial_rooms)} of {len(rooms)} rooms are partial (unknown "
                "edges, unobserved ceilings or thin floor coverage); openings are "
                "not detected"
            ),
            evidence_ids=[r.id for r in rooms],
        ),
        "stitched_plan": SectionCoverage(
            status="unavailable",
            reason="no property graph: rooms are not registered or placed in a "
            "property frame (later packets)",
        ),
        "measurements": SectionCoverage(
            status="partial",
            reason=(
                f"values: {quality['ok']} ok, {quality['degraded']} degraded, "
                f"{quality['unavailable']} unavailable; no measurement has an "
                "accuracy interval (no calibration exists)"
            ),
        ),
        "damage_regions": SectionCoverage(
            status="unavailable", reason="damage detection is not implemented"
        ),
        "concealed_damage_flags": SectionCoverage(
            status="unavailable", reason="concealed-damage rules are not implemented"
        ),
        "scope_items": SectionCoverage(
            status="unavailable", reason="scope generation is not implemented"
        ),
        "rendered_plan": SectionCoverage(
            status="available", evidence_ids=[r.id for r in rooms]
        ),
        "public_schema_export": SectionCoverage(
            status="blocked_external", reason=EXTERNAL_SCHEMA_REASON
        ),
    }
    config_hash = _combined_hash(
        capture.provenance.config_hash or "",
        bundle.get("config_hash", ""),
        model.diagnostics.get("config_hash", "")
        or json.dumps(model.diagnostics.get("config"), sort_keys=True),
    )
    manifest = run_manifest(
        run_id,
        mode,
        [capture.raw_manifest_hash],
        config_hash,
        started.isoformat(timespec="seconds"),
        times,
        artifacts,
    )
    diagnostics = [
        Diagnostic(code="result_reason", message=reason) for reason in model.reasons
    ]
    return PropertyResult(
        run=manifest,
        capture_id=capture.id,
        status="partial",
        status_reason=(
            "local LiDAR plan only: rooms are partial, measurements have no "
            "intervals, and no openings, stitched plan, damage, flags or scope exist"
        ),
        coverage=coverage,
        coordinate_frames=[frame],
        scale=[scale],
        rooms=rooms,
        surfaces=model.surfaces,
        walls=measured.walls,
        measurements=measured.measurements,
        diagnostics=diagnostics,
        artifacts=artifacts,
    )
