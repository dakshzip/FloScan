"""Stray Scanner ingestion and synchronization audit (P06).

Synthetic sessions render exact depth of a box room from known poses, so the
expected association, conventions and per-frame problems are known. The
supplied sample sessions are inspected too when they are present (they are
not in Git); those tests skip, saying so, on machines without them.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from floscan.capture import stray
from floscan.capture.sync import associate
from floscan.geometry.frames import (
    R_APPLE_CAMERA_FROM_OPTICAL,
    R_WORLD_FROM_APPLE_WORLD,
    PinholeCamera,
    quaternion_xyzw_from_rotation,
)
from floscan.io.manifest import file_manifest, media_digests, raw_manifest_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = PROJECT_ROOT / "example input "
TICK_S = 1.0 / 60.0
CLOCK_PPM = 150.0
RGB = (128, 96)
DEPTH = (64, 48)
FX = 100.0
ROOM_MIN = np.array([-2.0, 0.0, -1.5])  # source world, +y up
ROOM_MAX = np.array([2.0, 2.5, 1.5])
HEADER = (
    "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, "
    "distortion_center_x, distortion_center_y"
)


# --------------------------------------------------------------------------
# Synthetic session writer (a test fixture: geometry known exactly)
# --------------------------------------------------------------------------


def _camera_rotation(index: int) -> np.ndarray:
    """``R_world_from_optical`` turning about +y with a slight downward tilt."""
    yaw = np.radians(1.5 * index)
    pitch = np.radians(-10.0)
    forward = np.array(
        [np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)]
    )
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack([right, down, forward])
    assert np.isclose(np.linalg.det(rotation), 1.0)
    return rotation


def _camera_centre(index: int) -> np.ndarray:
    angle = np.radians(2.0 * index)
    return np.array([0.6 * np.cos(angle), 1.4, 0.4 * np.sin(angle)])


def _render_depth_m(rotation: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Optical z of the box room seen from one camera, at depth resolution."""
    camera = PinholeCamera(*RGB, FX, FX, (RGB[0] - 1) / 2, (RGB[1] - 1) / 2).resized(
        *DEPTH
    )
    v, u = np.mgrid[0 : DEPTH[1], 0 : DEPTH[0]]
    rays = camera.rays(np.column_stack([u.ravel(), v.ravel()]))  # optical z = 1
    world = rays @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        hits = np.concatenate(
            [(ROOM_MIN - centre) / world, (ROOM_MAX - centre) / world], axis=1
        )
    hits[~np.isfinite(hits) | (hits <= 0)] = np.inf
    return hits.min(axis=1).reshape(DEPTH[1], DEPTH[0])


def _gaps(count: int, uniform: bool) -> list[int]:
    rng = np.random.default_rng(7)
    return [1] * count if uniform else rng.choice([1, 1, 2, 3], size=count).tolist()


def make_session(
    root: Path,
    frames: int = 90,
    *,
    uniform: bool = False,
    no_video_rows: tuple[int, ...] = (0,),
    camera_axes: str = "optical",
    depth_scale_to_mm: float = 1000.0,
) -> Path:
    """Write a Stray-format session; returns its folder."""
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    ticks = np.concatenate([[0], np.cumsum(_gaps(frames - 1, uniform))])
    times = 1000.0 + ticks * TICK_S * (1 + CLOCK_PPM * 1e-6)
    cx, cy = (RGB[0] - 1) / 2, (RGB[1] - 1) / 2
    lines = [HEADER]
    for index in range(frames):
        rotation, centre = _camera_rotation(index), _camera_centre(index)
        depth_m = _render_depth_m(rotation, centre)
        stored = (
            rotation @ R_APPLE_CAMERA_FROM_OPTICAL
            if camera_axes == "apple"
            else rotation
        )
        q = quaternion_xyzw_from_rotation(stored)
        lines.append(
            f"{float(times[index])!r}, {index:06d}, "
            + ", ".join(f"{float(v)!r}" for v in (*centre, *q, FX, FX, cx, cy))
            + ", , "
        )
        raw = np.rint(depth_m * depth_scale_to_mm).astype(np.uint16)
        Image.fromarray(raw).save(root / "depth" / f"{index:06d}.png")
        Image.fromarray(np.full(DEPTH[::-1], 2, dtype=np.uint8), mode="L").save(
            root / "confidence" / f"{index:06d}.png"
        )
    (root / "odometry.csv").write_text("\n".join(lines) + "\n")
    (root / "camera_matrix.csv").write_text(
        f"{FX}, 0.0, {cx}\n0.0, {FX}, {cy}\n0.0, 0.0, 1.0"
    )
    imu = ["timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z"]
    imu += [f"{float(t)!r}, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0" for t in times[::2]]
    (root / "imu.csv").write_text("\n".join(imu) + "\n")
    _write_video(root / "rgb.mp4", ticks, set(no_video_rows))
    return root


def _write_video(path: Path, ticks: np.ndarray, skip: set[int]) -> None:
    """Constant-colour frames whose PTS are the sensor ticks (VFR)."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=60)
        stream.width, stream.height = RGB
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 60)
        first = min(int(t) for i, t in enumerate(ticks) if i not in skip)
        for index, tick in enumerate(ticks):
            if index in skip:
                continue
            image = np.full((RGB[1], RGB[0], 3), (index * 3) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = int(tick) - first
            frame.time_base = Fraction(1, 60)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _inspect(root: Path) -> stray.Inspection:
    return stray.inspect_session(stray.open_session(root))


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "floscan.cli", "inspect", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )


@pytest.fixture(scope="module")
def good_session(tmp_path_factory) -> Path:
    return make_session(tmp_path_factory.mktemp("stray") / "session")


def _copy(session: Path, tmp_path: Path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(session, target)
    return target


# --------------------------------------------------------------------------
# Positive path
# --------------------------------------------------------------------------


def test_synthetic_session_is_verified(good_session: Path) -> None:
    inspection = _inspect(good_session)
    assert inspection.status == "verified_with_frame_issues", inspection.reasons
    assert [(i.source_index, i.code) for i in inspection.frame_issues] == [
        (0, "no_rgb_frame")
    ]
    assert inspection.association.status == "verified"
    assert inspection.association.chosen.sensor_start == 1
    assert inspection.conventions["status"] == "verified"
    best = inspection.conventions["hypotheses"][0]
    assert (best["direction"], best["camera_axes"], best["depth_kind"]) == (
        "world_from_camera",
        "optical",
        "optical_z",
    )
    # Nearest-pixel lookup at 64x48 costs ~5 mm; it halves with each doubling
    # of resolution (runs/p06-evidence/residual-vs-resolution.log).
    assert best["median_error_m"] < 0.01
    runner_up = inspection.conventions["hypotheses"][1]["median_error_m"]
    assert runner_up > 10 * best["median_error_m"]
    assert inspection.association.clock.rate_ppm == pytest.approx(CLOCK_PPM, abs=1.0)


def test_loader_gives_metric_depth_and_session_world_poses(good_session: Path) -> None:
    session = stray.open_session(good_session)
    depth = session.depth(5)
    expected = _render_depth_m(_camera_rotation(5), _camera_centre(5))
    assert depth.depth_m.dtype == np.float32
    np.testing.assert_allclose(depth.depth_m, expected, atol=0.0006)
    assert depth.valid.all()
    assert (depth.confidence == 2).all()
    pose = session.pose(5)
    assert (pose.to_frame, pose.from_frame, pose.unit) == ("W", "C", "m")
    np.testing.assert_allclose(
        pose.translation, R_WORLD_FROM_APPLE_WORLD @ _camera_centre(5), atol=1e-9
    )
    # The optical axes are kept (no second Apple flip): forward stays forward.
    np.testing.assert_allclose(
        pose.apply_direction([0.0, 0.0, 1.0]),
        R_WORLD_FROM_APPLE_WORLD @ _camera_rotation(5)[:, 2],
        atol=1e-9,
    )
    camera = session.depth_camera(5, DEPTH)
    assert (camera.width, camera.height) == DEPTH
    assert camera.fx == pytest.approx(FX * DEPTH[0] / RGB[0])


def test_records_and_outputs(good_session: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    completed = _cli(
        "--input", str(good_session), "--tier", "lidar", "--output", str(out)
    )
    assert completed.returncode == 0, completed.stderr
    assert "association verified" in completed.stdout
    capture = json.loads((out / "capture.json").read_text())
    assert capture["tier"] == "lidar"
    assert "imu" not in capture["allowed_modalities"]
    assert capture["provenance"]["config_hash"] == stray.profile_hash(
        stray.load_profile()
    )
    assert capture["raw_manifest_hash"] == raw_manifest_hash(
        file_manifest(good_session)
    )
    rows = [
        json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 90
    assert rows[0]["video"] is None and rows[0]["issues"] == ["no_rgb_frame"]
    assert rows[1]["video"]["decoded_index"] == 0
    assert rows[1]["pose"]["to_frame_id"] == "W"
    assert rows[1]["camera"]["status"] == "partial"  # lens distortion not recorded
    report = json.loads((out / "inspection.json").read_text())
    assert report["association"]["method"].endswith("no nominal frame rate used")
    assert any("world up sign" in gap for gap in report["convention_gaps"])


# --------------------------------------------------------------------------
# Timing: VFR, dropped frames, ambiguity
# --------------------------------------------------------------------------


def test_vfr_timing_uses_timestamps_not_nominal_rate(good_session: Path) -> None:
    inspection = _inspect(good_session)
    session = inspection.session
    pts = inspection.video_pts_s
    matched = [(k, j) for k, j in enumerate(inspection.association.stream_to_sensor)]
    assert all(j is not None for _, j in matched)
    gaps = np.diff(session.sensor_time_s)
    assert gaps.max() > 2.5 * gaps.min()  # genuinely variable rate
    for k, j in matched:
        sensor_elapsed = session.sensor_time_s[j] - session.sensor_time_s[1]
        assert abs(sensor_elapsed - (pts[k] - pts[0])) < 0.002
    # Index times a nominal rate would be wrong for most frames.
    nominal = np.arange(len(pts)) / 60.0
    assert np.abs(nominal - (pts - pts[0])).max() > 0.2


def test_dropped_video_frame_mid_stream(tmp_path: Path) -> None:
    session = make_session(tmp_path / "s", no_video_rows=(0, 40))
    inspection = _inspect(session)
    assert inspection.association.status == "verified"
    issues = {(i.source_index, i.code) for i in inspection.frame_issues}
    assert issues == {(0, "no_rgb_frame"), (40, "no_rgb_frame")}
    mapping = inspection.association.sensor_to_stream
    assert (mapping[39], mapping[40], mapping[41]) == (38, None, 39)


def test_uniform_timing_cannot_resolve_a_missing_frame(tmp_path: Path) -> None:
    session = make_session(tmp_path / "s", uniform=True)
    inspection = _inspect(session)
    assert inspection.association.status == "ambiguous"
    assert inspection.status == "unverified"
    assert all(k is None for k in inspection.association.sensor_to_stream)
    completed = _cli("--input", str(session), "--tier", "lidar")
    assert completed.returncode == 4
    assert "timing alone cannot decide" in completed.stdout


def test_associate_follows_clock_drift_and_reports_failure() -> None:
    sensor = 10.0 + np.cumsum([0.0] + [TICK_S * g for g in _gaps(299, False)])
    stream = (sensor - 10.0) / (1 + 400e-6)
    result = associate(stream, sensor, 0.004, 3, 0.99, 0.05)
    assert result.status == "verified"
    assert result.clock.rate_ppm == pytest.approx(400.0, abs=1.0)
    assert result.stream_to_sensor == tuple(range(300))
    unrelated = associate(stream * 1.37, sensor, 0.004, 3, 0.99, 0.05)
    assert unrelated.status == "failed"
    with pytest.raises(ValueError, match="not strictly increasing"):
        associate([0.0, 0.0, 1.0], sensor, 0.004, 3, 0.99, 0.05)


# --------------------------------------------------------------------------
# Per-frame problems are reported, never skipped silently
# --------------------------------------------------------------------------


def test_missing_confidence(good_session: Path, tmp_path: Path) -> None:
    session = _copy(good_session, tmp_path)
    (session / "confidence" / "000010.png").unlink()
    inspection = _inspect(session)
    assert inspection.status == "verified_with_frame_issues"
    assert (10, "confidence_missing") in {
        (i.source_index, i.code) for i in inspection.frame_issues
    }
    assert inspection.session.depth(10).confidence is None


def test_corrupt_depth(good_session: Path, tmp_path: Path) -> None:
    session = _copy(good_session, tmp_path)
    (session / "depth" / "000020.png").write_bytes(b"\x89PNG\r\n\x1a\n not a png")
    inspection = _inspect(session)
    assert (20, "depth_corrupt") in {
        (i.source_index, i.code) for i in inspection.frame_issues
    }
    assert inspection.depth["usable_frames"] == 89
    with pytest.raises(stray.StrayInputError, match="unreadable PNG"):
        inspection.session.depth(20)


def test_mismatched_depth_dimensions(good_session: Path, tmp_path: Path) -> None:
    session = _copy(good_session, tmp_path)
    Image.fromarray(np.full((10, 10), 1000, dtype=np.uint16)).save(
        session / "depth" / "000030.png"
    )
    inspection = _inspect(session)
    assert (30, "depth_wrong_size") in {
        (i.source_index, i.code) for i in inspection.frame_issues
    }


def test_depth_with_another_aspect_ratio_is_unverified(tmp_path: Path) -> None:
    session = make_session(tmp_path / "s", frames=12)
    for name in (session / "depth").iterdir():
        Image.fromarray(np.full((48, 48), 1500, dtype=np.uint16)).save(name)
    for name in (session / "confidence").iterdir():
        Image.fromarray(np.full((48, 48), 2, dtype=np.uint8), mode="L").save(name)
    inspection = _inspect(session)
    assert inspection.status == "unverified"
    assert any("aspect ratio" in reason for reason in inspection.reasons)


def test_path_with_spaces_and_trailing_space(
    good_session: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "example input " / "my capture 2"
    shutil.copytree(good_session, folder)
    out = tmp_path / "out dir"
    completed = _cli("--input", str(folder), "--tier", "lidar", "--output", str(out))
    assert completed.returncode == 0, completed.stderr
    assert (out / "frames.jsonl").is_file()


# --------------------------------------------------------------------------
# No silent convention or unit guess
# --------------------------------------------------------------------------


def test_raw_apple_camera_axes_are_detected(tmp_path: Path) -> None:
    session = make_session(tmp_path / "s", camera_axes="apple")
    inspection = _inspect(session)
    assert inspection.status == "unverified"
    assert inspection.conventions["status"] == "unverified"
    assert "'apple'" in inspection.conventions["reason"]  # the data say Apple axes


def test_wrong_depth_unit_is_detected(tmp_path: Path) -> None:
    # Stored in units of 0.1 mm but declared millimetres: ten times too deep.
    session = make_session(tmp_path / "s", depth_scale_to_mm=10000.0)
    inspection = _inspect(session)
    assert inspection.status == "unverified"
    assert inspection.conventions["status"] == "unverified"
    assert any("median depth" in reason for reason in inspection.reasons)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda text: text.replace("qw", "w", 1), "header"),
        (
            lambda text: text.replace("1000.0, 000000", "1000.5, 000000", 1),
            "not strictly increasing",
        ),
        (lambda text: text.replace(", , \n", ", , , 5\n", 1), "columns"),
    ],
    ids=["header", "timestamps", "row-shape"],
)
def test_malformed_odometry_is_invalid_input(
    good_session: Path, tmp_path: Path, edit, message: str
) -> None:
    session = _copy(good_session, tmp_path)
    path = session / "odometry.csv"
    path.write_text(edit(path.read_text()))
    with pytest.raises(stray.StrayInputError, match=message):
        stray.open_session(session)
    assert _cli("--input", str(session), "--tier", "lidar").returncode == 3


def test_non_rotation_quaternion_is_invalid_input(
    good_session: Path, tmp_path: Path
) -> None:
    session = _copy(good_session, tmp_path)
    lines = (session / "odometry.csv").read_text().splitlines()
    cells = lines[3].split(", ")
    cells[5] = repr(float(cells[5]) + 0.05)  # qx changed: not unit length
    lines[3] = ", ".join(cells)
    (session / "odometry.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(stray.StrayInputError, match="row 2 quaternion norm"):
        stray.open_session(session)


# --------------------------------------------------------------------------
# Inputs are read only; CLI guards
# --------------------------------------------------------------------------


def test_inspection_never_writes_to_the_input(
    good_session: Path, tmp_path: Path
) -> None:
    session = _copy(good_session, tmp_path)
    before = file_manifest(session)
    paths = [session, *session.rglob("*")]
    for path in paths:
        path.chmod(stat.S_IREAD | (stat.S_IEXEC if path.is_dir() else 0))
    try:
        inspection = _inspect(session)
    finally:
        for path in paths:
            path.chmod(stat.S_IRWXU)
    assert inspection.status == "verified_with_frame_issues"
    assert file_manifest(session) == before


def test_cli_guards(good_session: Path, tmp_path: Path) -> None:
    photo = _cli("--input", str(good_session), "--tier", "photo")
    assert photo.returncode == 4 and "not supported yet" in photo.stderr
    inside = _cli(
        "--input",
        str(good_session),
        "--tier",
        "lidar",
        "--output",
        str(good_session / "o"),
    )
    assert inside.returncode == 2 and "inside the input" in inside.stderr
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("x")
    taken = _cli(
        "--input", str(good_session), "--tier", "lidar", "--output", str(existing)
    )
    assert taken.returncode == 2 and "already exists" in taken.stderr
    missing = _cli("--input", str(tmp_path / "nowhere"), "--tier", "lidar")
    assert missing.returncode == 3


# --------------------------------------------------------------------------
# Supplied sample sessions (skipped where the samples are not present)
# --------------------------------------------------------------------------


def _fixture_records() -> dict[str, dict]:
    manifest = json.loads(
        (PROJECT_ROOT / "benchmark/manifests/development.json").read_text()
    )
    return {Path(f["path"]).name: f for f in manifest["fixtures"]}


needs_samples = pytest.mark.skipif(
    not SAMPLES.is_dir(), reason="sample sessions under 'example input /' not present"
)


@needs_samples
@pytest.mark.parametrize(
    ("name", "rows", "presented"),
    [("1a8384c3f6", 5251, 5250), ("c00a170fe1", 1715, 1714)],
)
def test_sample_sessions_parse_with_explained_association(
    name: str, rows: int, presented: int
) -> None:
    inspection = _inspect(SAMPLES / name)
    assert inspection.session.frame_count == rows
    assert inspection.video["presented_frames"] == presented
    assert inspection.video["packets_not_presented"] == 1
    association = inspection.association
    assert association.status == "verified"
    assert (association.chosen.stream_start, association.chosen.sensor_start) == (0, 1)
    assert association.matched == presented
    assert association.clock.max_abs_residual_s < 0.001
    assert inspection.conventions["status"] == "verified"
    assert [(i.source_index, i.code) for i in inspection.frame_issues] == [
        (0, "no_rgb_frame")
    ]
    assert inspection.status == "verified_with_frame_issues"
    record = _fixture_records()[name]
    assert inspection.raw_manifest_hash == record["raw_manifest_hash"]
    assert media_digests(inspection.files) == record["media_sha256"]


@needs_samples
def test_duplicate_sample_is_byte_identical() -> None:
    original = file_manifest(SAMPLES / "1a8384c3f6")
    copy_ = file_manifest(SAMPLES / "1a8384c3f6 2")
    assert original == copy_
    assert len(original) == _fixture_records()["1a8384c3f6"]["file_count"]
