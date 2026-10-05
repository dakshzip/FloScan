"""Local LiDAR result through ./run.sh (P09): records, plan, envelope rules."""

from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from floscan.contracts.result import PropertyResult
from floscan.pipeline import (
    EnvelopeError,
    read_envelope,
    validate_envelope,
    validate_records,
)
from floscan.render.svg import build_drawing, to_svg
from tests.integration import test_stray as stray_fixtures

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = PROJECT_ROOT / "example input " / "c00a170fe1"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(PROJECT_ROOT / "run.sh"), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )


def _sweep(index: int) -> np.ndarray:
    """A full turn, the camera sweeping between level and 40 degrees down."""
    yaw = np.radians(3.0 * index)
    pitch = np.radians(-20.0 + 20.0 * math.sin(index / 6.0))
    forward = np.array(
        [np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)]
    )
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(forward, right), forward])


def make_capture(root: Path) -> Path:
    """Synthetic 4 x 3 m room capture with enough floor and wall coverage."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stray_fixtures, "_camera_rotation", _sweep)
        patch.setattr(stray_fixtures, "RGB", (256, 192))
        patch.setattr(stray_fixtures, "DEPTH", (128, 96))
        patch.setattr(stray_fixtures, "FX", 200.0)
        return stray_fixtures.make_session(root, 120)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    session = make_capture(tmp_path_factory.mktemp("room") / "session")
    out = tmp_path_factory.mktemp("runs") / "run"
    completed = _run(
        "--input", str(session), "--tier", "lidar", "--output", str(out),
        "--mode", "live",
    )  # fmt: skip
    assert completed.returncode == 4, completed.stderr
    return out


def test_one_command_writes_a_validated_partial_result(run_dir: Path) -> None:
    envelope = read_envelope(run_dir / "result.json")
    validate_envelope(envelope)
    validate_records(envelope, run_dir)
    assert envelope["schema_version"] == "floscan-result/0.2.0"
    assert envelope["status"] == "partial"
    assert envelope["coverage"]["contract_complete"] is False
    stages = {s["name"]: s["status"] for s in envelope["stages"]}
    for name in (
        "capture.normalize",
        "reconstruction.reconstruct",
        "rooms.build",
        "measurements.evaluate",
        "render.plan",
        "export.serialize",
    ):
        assert stages[name] == "ok", name
    assert stages["damage.detect"] == "not_implemented"
    sections = envelope["coverage"]["sections"]
    assert sections["public_schema_export"]["status"] == "blocked_external"
    assert sections["stitched_plan"]["status"] == "unavailable"
    assert sections["measurements"]["status"] == "partial"
    # Records live in the validated file, never inline in the envelope.
    assert envelope["rooms"] == [] and envelope["measurements"] == []


def test_records_are_a_valid_property_result(run_dir: Path) -> None:
    result = PropertyResult.model_validate_json(
        (run_dir / "property_result.json").read_text()
    )
    assert result.rooms and result.walls and result.measurements
    assert result.scale[0].status == "sensor_metric"
    assert {a.mime_type for a in result.artifacts} == {"image/svg+xml", "image/png"}
    assert all(m.interval.status == "unavailable" for m in result.measurements)
    assert result.run.input_manifest_hashes
    assert re.fullmatch(r"[0-9a-f]{40}", result.run.git_commit)
    area = next(m for m in result.measurements if m.quantity == "floor_area")
    assert area.value == pytest.approx(12.0, abs=0.05)  # synthetic 4 x 3 m box
    lengths = sorted(
        m.value for m in result.measurements if m.quantity == "wall_length"
    )
    assert lengths == pytest.approx([3.0, 3.0, 4.0, 4.0], abs=0.02)
    ceiling = next(m for m in result.measurements if m.quantity == "ceiling_height")
    assert ceiling.value is None  # the sweep never looks up: not observed


def test_validate_command_checks_the_records(run_dir: Path, tmp_path: Path) -> None:
    ok = subprocess.run(
        [sys.executable, "-m", "floscan.cli", "validate", str(run_dir / "result.json")],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )
    assert ok.returncode == 0, ok.stderr
    copied = tmp_path / "copy"
    copied.mkdir()
    for name in ("result.json", "property_result.json"):
        (copied / name).write_bytes((run_dir / name).read_bytes())
    tampered = json.loads((copied / "property_result.json").read_text())
    tampered["measurements"][0]["value"] = 99.0
    (copied / "property_result.json").write_text(json.dumps(tampered))
    bad = subprocess.run(
        [sys.executable, "-m", "floscan.cli", "validate", str(copied / "result.json")],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )
    assert bad.returncode == 1 and "does not match its recorded hash" in bad.stderr


def test_envelope_rules_for_records(run_dir: Path) -> None:
    envelope = read_envelope(run_dir / "result.json")
    without = copy.deepcopy(envelope)
    without["records"] = None
    with pytest.raises(EnvelopeError, match="needs a records file"):
        validate_envelope(without)
    disagree = copy.deepcopy(envelope)
    disagree["coverage"]["sections"]["measurements"]["status"] = "available"
    validate_envelope(disagree)  # structurally fine ...
    with pytest.raises(EnvelopeError, match="records say 'partial'"):
        validate_records(disagree, run_dir)  # ... but contradicts the records
    inline = copy.deepcopy(envelope)
    inline["rooms"] = [{"id": "room:01"}]
    with pytest.raises(EnvelopeError, match="carries no entity records"):
        validate_envelope(inline)


# --------------------------------------------------------------------------
# Plan rendering
# --------------------------------------------------------------------------


def test_svg_geometry_maps_back_to_the_records(run_dir: Path) -> None:
    result = PropertyResult.model_validate_json(
        (run_dir / "property_result.json").read_text()
    )
    svg = (run_dir / "plan" / "plan.svg").read_text()
    attributes = dict(re.findall(r'data-(x0|y1|scale|margin)="([^"]+)"', svg))
    x0, y1, scale, margin = (
        float(attributes[k]) for k in ("x0", "y1", "scale", "margin")
    )
    for room in result.rooms:
        path = re.search(rf'data-room="{re.escape(room.id)}" d="([^"]+)"', svg)
        assert path, room.id
        first_ring = path.group(1).split("Z")[0]
        canvas = [
            tuple(map(float, pair.split(",")))
            for pair in re.findall(r"[-0-9.e]+,[-0-9.e]+", first_ring)
        ]
        world = [
            ((px - margin) / scale + x0, y1 - (py - margin) / scale)
            for px, py in canvas
        ]
        np.testing.assert_allclose(world, room.boundary.outer, atol=1e-9)
    assert 'data-scale-bar="' in svg
    png = Image.open(run_dir / "plan" / "plan.png")
    width = float(re.search(r'<svg[^>]* width="([0-9.]+)"', svg).group(1))
    assert abs(png.width - width) <= 1


def test_renderer_does_not_change_records_and_marks_unknowns(run_dir: Path) -> None:
    result = PropertyResult.model_validate_json(
        (run_dir / "property_result.json").read_text()
    )
    before = [r.model_dump() for r in result.rooms]
    rooms = copy.deepcopy(result.rooms)
    for room in rooms:
        room_ceiling = [m for m in result.measurements if m.subject_id == room.id]
        assert room_ceiling
    drawing = build_drawing(rooms, result.walls, result.measurements, "t")
    svg = to_svg(drawing)
    assert [r.model_dump() for r in rooms] == before
    # Dashed edges exactly where an outline edge has no wall record.
    edges = sum(
        len(ring)
        for r in result.rooms
        for ring in [r.boundary.outer, *r.boundary.holes]
    )
    unknown = edges - len(result.walls)
    assert svg.count('data-edge="unknown"') == unknown
    assert svg.count('data-edge="wall"') == len(result.walls)
    if any(
        m.quantity == "ceiling_height" and m.value is None for m in result.measurements
    ):
        assert "ceiling n/a" in svg


# --------------------------------------------------------------------------
# Supplied sample (skipped where absent)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="sample session not present")
def test_sample_capture_gives_a_reviewable_plan(tmp_path: Path) -> None:
    out = tmp_path / "run"
    completed = _run(
        "--input", str(SAMPLE), "--tier", "lidar", "--output", str(out),
        "--mode", "live",
    )  # fmt: skip
    assert completed.returncode == 4, completed.stderr
    envelope = read_envelope(out / "result.json")
    validate_envelope(envelope)
    validate_records(envelope, out)
    assert envelope["status"] == "partial"
    assert (out / "plan" / "plan.svg").stat().st_size > 0
    result = PropertyResult.model_validate_json(
        (out / "property_result.json").read_text()
    )
    assert result.rooms
    assert all(m.interval.lower is None for m in result.measurements)  # no narrow CIs
