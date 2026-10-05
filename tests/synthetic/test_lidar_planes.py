"""LiDAR reconstruction: metric points and planes (P07).

The synthetic box room has exact depth (rounded to the 1 mm source unit)
from known poses, so every recovered plane can be compared with its analytic
face. A fixture verifies mathematics; it never establishes real accuracy.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from floscan.capture import stray
from floscan.contracts.geometry import Plane, PointCloud
from floscan.geometry.frames import PinholeCamera
from floscan.geometry.planes import (
    PlaneFitError,
    least_squares_plane,
    orient_towards,
    plane_basis,
    ransac_plane,
    robust_refit,
)
from floscan.pipeline import EnvelopeError, read_envelope, validate_envelope
from floscan.reconstruction import lidar
from tests.integration import test_stray as synthetic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = PROJECT_ROOT / "example input " / "c00a170fe1"
# Inward faces of the box in W (+z up): (normal, offset) with n . p + d = 0.
FACES = {
    "x_min": ((1.0, 0.0, 0.0), 2.0),
    "x_max": ((-1.0, 0.0, 0.0), 2.0),
    "y_min": ((0.0, 1.0, 0.0), 1.5),
    "y_max": ((0.0, -1.0, 0.0), 1.5),
    "floor": ((0.0, 0.0, 1.0), 0.0),
    "ceiling": ((0.0, 0.0, -1.0), 2.5),
}
QUANTIZATION_M = 0.002  # 1 mm depth rounding seen through the plane fit
ANGLE_TOLERANCE_DEG = 0.15


def _rotation(index: int) -> np.ndarray:
    """Three-quarter turn, tilting between floor and ceiling (about 250 deg/s)."""
    yaw = np.radians(3.0 * index)
    pitch = np.radians(30.0 * math.sin(index / 10.0))
    forward = np.array(
        [np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)]
    )
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(forward, right), forward])


def make_room(root: Path, frames: int = 90, **kwargs) -> Path:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(synthetic, "_camera_rotation", _rotation)
        return synthetic.make_session(root, frames, **kwargs)


def _inspect(root: Path) -> stray.Inspection:
    return stray.inspect_session(stray.open_session(root))


def _planes(manifest: dict) -> list[dict]:
    return [plane for submap in manifest["submaps"] for plane in submap["planes"]]


def _match(plane: dict) -> tuple[str, float, float]:
    """Nearest analytic face: (name, angle error deg, offset error m)."""
    normal = np.asarray(plane["plane"]["normal"])
    best = None
    for name, (face_normal, face_offset) in FACES.items():
        angle = math.degrees(math.acos(min(1.0, float(normal @ face_normal))))
        error = (name, angle, abs(plane["plane"]["offset"] - face_offset))
        if best is None or error[1] < best[1]:
            best = error
    return best


@pytest.fixture(scope="module")
def room(tmp_path_factory) -> tuple[Path, dict]:
    root = make_room(tmp_path_factory.mktemp("room") / "session")
    inspection = _inspect(root)
    out = tmp_path_factory.mktemp("recon") / "reconstruction"
    return root, lidar.reconstruct(inspection.session, inspection, out) | {
        "_dir": str(out)
    }


# --------------------------------------------------------------------------
# Plane mathematics (noise-free: numerical tolerance)
# --------------------------------------------------------------------------


def test_noise_free_plane_is_exact() -> None:
    rng = np.random.default_rng(1)
    normal = np.array([0.3, -0.5, 0.8])
    normal /= np.linalg.norm(normal)
    u, v = plane_basis(normal)
    points = rng.uniform(-2, 2, (500, 2)) @ np.stack([u, v]) - 1.25 * normal
    fitted, offset, _ = least_squares_plane(points)
    if fitted @ normal < 0:
        fitted, offset = -fitted, -offset
    np.testing.assert_allclose(fitted, normal, atol=1e-12)
    assert offset == pytest.approx(1.25, abs=1e-12)
    fit = robust_refit(points, fitted, offset, cutoff=0.01)
    assert fit.max_abs < 1e-12
    assert fit.covariance is None
    assert fit.covariance_status.startswith("unavailable: zero residual")


def test_robust_refit_ignores_outliers_and_reports_covariance() -> None:
    rng = np.random.default_rng(2)
    inliers = np.column_stack(
        [rng.uniform(-2, 2, (800, 2)), rng.normal(0.0, 0.002, 800)]
    )
    outliers = rng.uniform(-2, 2, (400, 3))
    points = np.concatenate([inliers, outliers])
    start_normal = np.array([0.05, 0.0, 1.0]) / np.linalg.norm([0.05, 0.0, 1.0])
    fit = robust_refit(points, start_normal, 0.02, cutoff=0.01)
    assert math.degrees(math.acos(abs(fit.normal[2]))) < 0.2
    assert abs(fit.offset) < 0.001
    assert fit.covariance_status.startswith("estimated")
    covariance = np.asarray(fit.covariance)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > -1e-15
    along = np.append(fit.normal, 0.0)
    assert abs(along @ covariance @ along) < 1e-15  # the normal's length is fixed


def test_seeded_ransac_finds_the_plane_and_is_reproducible() -> None:
    rng = np.random.default_rng(4)
    plane = np.column_stack([rng.uniform(-2, 2, (600, 2)), np.full(600, 0.7)])
    clutter = rng.uniform(-2, 2, (400, 3))
    points = np.concatenate([plane, clutter])
    first = ransac_plane(points, 0.01, 500, np.random.default_rng(9))
    second = ransac_plane(points, 0.01, 500, np.random.default_rng(9))
    normal, offset, inliers = first
    assert abs(abs(normal[2]) - 1.0) < 1e-9
    assert abs(abs(offset) - 0.7) < 1e-9
    assert set(range(600)) <= set(inliers.tolist())
    np.testing.assert_array_equal(first[2], second[2])
    assert first[1] == second[1]


def test_collinear_support_is_not_a_plane() -> None:
    line = np.column_stack([np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)])
    with pytest.raises(PlaneFitError, match="collinear"):
        least_squares_plane(line)


def test_normal_points_toward_the_observing_cameras() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cameras = np.array([[0.0, 0.0, 1.5]] * 3)
    normal, offset, agree = orient_towards(
        np.array([0.0, 0.0, -1.0]), 0.0, points, cameras
    )
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0])
    assert agree == 1.0


# --------------------------------------------------------------------------
# Synthetic room: known planes, normals, covariance, provenance
# --------------------------------------------------------------------------


def test_every_face_is_recovered_and_nothing_else(room) -> None:
    _, manifest = room
    planes = _planes(manifest)
    matches = [_match(plane) for plane in planes]
    for (name, angle, offset), plane in zip(matches, planes, strict=True):
        assert angle < ANGLE_TOLERANCE_DEG, (name, angle)
        assert offset < QUANTIZATION_M, (name, offset)
        assert plane["plane"]["residual_stats"]["rms"] < 0.001
    assert {name for name, _, _ in matches} == set(FACES)


def test_planes_are_unknown_kind_with_orientation_evidence(room) -> None:
    _, manifest = room
    for plane in _planes(manifest):
        record = Plane.model_validate(plane["plane"])
        assert record.kind == "unknown"
        assert record.orientation_is_prior is True
        assert record.unit == "m" and record.frame_id == "W"
        name = _match(plane)[0]
        expected = {"floor": "up_facing", "ceiling": "down_facing"}.get(
            name, "vertical"
        )
        assert plane["diagnostics"]["orientation_class"] == expected
        assert plane["diagnostics"]["share_of_views_in_front"] == 1.0
        assert plane["diagnostics"]["covariance_status"].startswith("estimated")
        covariance = np.asarray(record.parameter_covariance)
        assert np.linalg.eigvalsh(covariance).min() > -1e-15


def test_points_keep_their_frame_and_pixel(room) -> None:
    root, manifest = room
    directory = Path(manifest["_dir"])
    submap = manifest["submaps"][0]
    cloud = PointCloud.model_validate(submap["point_cloud"])
    xyz = np.load(directory / cloud.xyz.uri)
    observations = np.load(directory / cloud.observation_index.uri)
    assert list(xyz.shape) == cloud.xyz.shape
    session = stray.open_session(root)
    for row in (0, len(xyz) // 2, len(xyz) - 1):
        frame, u, v = (int(x) for x in observations[row])
        depth = session.depth(frame)
        camera = session.depth_camera(
            frame, (depth.depth_m.shape[1], depth.depth_m.shape[0])
        )
        ray = camera.rays([[u, v]])[0]
        expected = session.pose(frame).apply_direction(ray * depth.depth_m[v, u])
        expected = expected + session.pose(frame).translation
        np.testing.assert_allclose(xyz[row], expected, atol=1e-5)


def test_plane_support_files_index_the_submap(room) -> None:
    _, manifest = room
    directory = Path(manifest["_dir"])
    submap = manifest["submaps"][0]
    xyz = np.load(directory / submap["point_cloud"]["xyz"]["uri"])
    for plane in submap["planes"]:
        support = np.load(directory / plane["plane"]["support"]["uri"])
        distances = xyz[support] @ plane["plane"]["normal"] + plane["plane"]["offset"]
        assert np.abs(distances).max() < 0.03
        assert len(support) == plane["plane"]["residual_stats"]["count"]


def test_reconstruction_is_deterministic(room, tmp_path: Path) -> None:
    root, manifest = room
    inspection = _inspect(root)
    again = lidar.reconstruct(inspection.session, inspection, tmp_path / "again")
    strip = lambda planes: [  # noqa: E731
        (p["plane"]["normal"], p["plane"]["offset"]) for p in planes
    ]
    assert strip(_planes(again)) == strip(_planes(manifest))


# --------------------------------------------------------------------------
# Depth semantics, missing depth, outliers
# --------------------------------------------------------------------------


def test_depth_is_optical_z_not_range(room) -> None:
    root, _ = room
    session = stray.open_session(root)
    depth = session.depth(10)
    camera = session.depth_camera(10, (depth.depth_m.shape[1], depth.depth_m.shape[0]))
    v, u = np.nonzero(depth.valid)
    pixels = np.column_stack([u, v])
    values = depth.depth_m[v, u]
    pose = session.pose(10)
    world = {
        kind: camera.unproject(pixels, values, kind, "m").xyz @ pose.rotation.T
        + pose.translation
        for kind in ("optical_z", "range")
    }
    # The face holding most z-read points; read as range, the same pixels bow.
    distances = {
        name: world["optical_z"] @ normal + offset
        for name, (normal, offset) in FACES.items()
    }
    name = max(distances, key=lambda f: int((np.abs(distances[f]) < 0.003).sum()))
    on_face = np.abs(distances[name]) < 0.003
    assert on_face.sum() > 200
    normal, offset = FACES[name]
    bowed = world["range"][on_face] @ normal + offset
    assert np.abs(distances[name][on_face]).max() < 0.003
    assert np.ptp(bowed) > 0.05  # centimetres off a plane that z keeps exact


def test_range_depth_profiles_are_refused(room, tmp_path: Path) -> None:
    root, _ = room
    inspection = _inspect(root)
    session = inspection.session
    profile = session.profile.model_copy(
        update={
            "depth": session.profile.depth.model_copy(update={"depth_kind": "range"})
        }
    )
    ranged = stray.StraySession(**{**session.__dict__, "profile": profile})
    with pytest.raises(stray.StrayInputError, match="optical-z"):
        lidar.frame_points(ranged, 0, (64, 48), lidar.LidarConfig())


def test_missing_depth_is_not_free_space(tmp_path: Path) -> None:
    root = make_room(tmp_path / "s", frames=60)
    for path in sorted((root / "depth").iterdir()):
        depth = np.array(Image.open(path))
        depth[:, :20] = 0  # a band with no return in every frame
        Image.fromarray(depth).save(path)
    inspection = _inspect(root)
    config = lidar.LidarConfig()
    points = lidar.frame_points(inspection.session, 4, (64, 48), config)
    band = points.observations[:, 1] < 20
    assert not band.any()  # no point from missing depth
    assert points.counts["invalid_depth"] == 10 * 24  # stride-2 grid in the band
    assert np.linalg.norm(points.xyz - points.centre, axis=1).min() > 0.15
    manifest = lidar.reconstruct(inspection.session, inspection, tmp_path / "r")
    assert manifest["filter_counts"]["invalid_depth"] > 0
    for plane in _planes(manifest):
        _, angle, offset = _match(plane)
        assert angle < ANGLE_TOLERANCE_DEG and offset < QUANTIZATION_M


def test_outlier_returns_do_not_move_planes(tmp_path: Path) -> None:
    root = make_room(tmp_path / "s", frames=60)
    rng = np.random.default_rng(3)
    for path in sorted((root / "depth").iterdir()):
        depth = np.array(Image.open(path))
        spikes = rng.random(depth.shape) < 0.03
        depth[spikes] = rng.integers(300, 4000, spikes.sum())
        Image.fromarray(depth.astype(np.uint16)).save(path)
    inspection = _inspect(root)
    manifest = lidar.reconstruct(inspection.session, inspection, tmp_path / "r")
    assert manifest["filter_counts"]["depth_edge"] > 0  # isolated spikes dropped
    planes = _planes(manifest)
    assert len(planes) >= 4
    for plane in planes:
        _, angle, offset = _match(plane)
        assert angle < 0.5 and offset < 0.005


# --------------------------------------------------------------------------
# Tracking discontinuities and unverified captures
# --------------------------------------------------------------------------


def test_no_submap_spans_a_pose_jump(tmp_path: Path) -> None:
    root = make_room(tmp_path / "s", frames=60)
    lines = (root / "odometry.csv").read_text().splitlines()
    for row in range(31, len(lines)):
        cells = lines[row].split(", ")
        cells[2] = repr(float(cells[2]) + 0.5)
        lines[row] = ", ".join(cells)
    (root / "odometry.csv").write_text("\n".join(lines) + "\n")
    inspection = _inspect(root)
    keyframes, counts = lidar.select_keyframes(
        inspection.session, inspection, lidar.LidarConfig()
    )
    assert counts["segments"] == 2
    assert keyframes[1][0] == 30
    for submap in lidar.plan_submaps(
        inspection.session, keyframes, lidar.LidarConfig()
    ):
        assert max(submap) < 30 or min(submap) >= 30


def test_unverified_capture_is_not_reconstructed(tmp_path: Path) -> None:
    root = make_room(tmp_path / "s", camera_axes="apple")
    inspection = _inspect(root)
    with pytest.raises(stray.StrayInputError, match="verified capture"):
        lidar.reconstruct(inspection.session, inspection, tmp_path / "r")
    outcome = lidar.run_live(root, tmp_path / "run")
    assert outcome.status == "insufficient_evidence"
    assert [s["status"] for s in outcome.stages] == ["insufficient_evidence", "skipped"]
    assert not (tmp_path / "run" / "reconstruction").exists()


# --------------------------------------------------------------------------
# ./run.sh end to end and the envelope rules
# --------------------------------------------------------------------------


def _run_sh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(PROJECT_ROOT / "run.sh"), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )


def test_run_sh_writes_points_planes_and_an_incomplete_result(
    room, tmp_path: Path
) -> None:
    root, _ = room
    folder = tmp_path / "capture with space "
    shutil.copytree(root, folder)
    out = tmp_path / "run"
    completed = _run_sh(
        "--input",
        str(folder),
        "--tier",
        "lidar",
        "--output",
        str(out),
        "--mode",
        "live",
    )
    assert completed.returncode == 4, completed.stderr
    assert "stage:    reconstruction.reconstruct ok" in completed.stderr
    envelope = read_envelope(out / "result.json")
    validate_envelope(envelope)
    stages = {s["name"]: s["status"] for s in envelope["stages"]}
    assert stages["capture.normalize"] == "ok"
    assert stages["reconstruction.reconstruct"] == "ok"
    assert stages["rooms.build"] == "not_implemented"
    assert envelope["status"] == "unsupported"
    assert envelope["rooms"] == [] and envelope["measurements"] == []
    assert all(
        section["status"] != "available"
        for section in envelope["coverage"]["sections"].values()
    )
    bundle_text = (out / "reconstruction" / "bundle.json").read_text()
    json.loads(bundle_text, parse_constant=lambda c: pytest.fail(f"non-finite {c}"))
    outputs = [
        d["message"] for d in envelope["diagnostics"] if d["code"] == "stage_output"
    ]
    assert any(m.startswith("reconstruction/bundle.json sha256=") for m in outputs)
    assert (out / "reconstruction" / "preview.ply").stat().st_size > 0


def test_invalid_lidar_input_is_reported(tmp_path: Path) -> None:
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "photo.jpg").write_bytes(b"x")
    completed = _run_sh(
        "--input", str(junk), "--tier", "lidar", "--output", str(tmp_path / "o"),
        "--mode", "live",
    )  # fmt: skip
    assert completed.returncode == 3
    envelope = read_envelope(tmp_path / "o" / "result.json")
    stages = {s["name"]: s["status"] for s in envelope["stages"]}
    assert stages["capture.normalize"] == "invalid_input"
    assert stages["reconstruction.reconstruct"] == "skipped"


def test_validator_allows_ok_only_for_implemented_stages(room, tmp_path: Path) -> None:
    root, _ = room
    out = tmp_path / "run"
    _run_sh(
        "--input", str(root), "--tier", "lidar", "--output", str(out), "--mode", "live"
    )
    envelope = read_envelope(out / "result.json")
    rooms = copy.deepcopy(envelope)
    next(s for s in rooms["stages"] if s["name"] == "rooms.build")["status"] = "ok"
    with pytest.raises(EnvelopeError, match="reports 'ok'"):
        validate_envelope(rooms)
    photo = copy.deepcopy(envelope)
    photo["run"]["tier"] = "photo"
    with pytest.raises(EnvelopeError, match="live photo run"):
        validate_envelope(photo)


# --------------------------------------------------------------------------
# Supplied sample (skipped where it is not present)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="sample session not present")
def test_real_cloud_is_plausibly_oriented_and_scaled(tmp_path: Path) -> None:
    inspection = _inspect(SAMPLE)
    manifest = lidar.reconstruct(inspection.session, inspection, tmp_path / "r")
    summary = manifest["summary"]
    walls = summary["vertical_plane_directions"]
    # A rectilinear interior: wall support shares one perpendicular family.
    assert walls["support_share_within_5deg_of_perpendicular_family"] > 0.9
    # The best-supported up-facing level lies a handheld height below the
    # cameras (a plausibility band for metric scale, not a dimension).
    levels = summary["horizontal_levels"]["up_facing"]
    main = max(levels, key=lambda level: level["support_points"])
    cameras = inspection.session.translation_m[:, 1]  # source +y is up
    assert 0.8 < float(np.median(cameras)) - main["height_m"] < 2.0
    assert all(p["plane"]["kind"] == "unknown" for p in _planes(manifest))
    assert manifest["filter_counts"]["kept"] > 0.5 * manifest["filter_counts"]["pixels"]


def test_camera_helper_matches_depth_resolution() -> None:
    camera = PinholeCamera(128, 96, 100.0, 100.0, 63.5, 47.5).resized(64, 48)
    assert (camera.cx, camera.cy) == (31.5, 23.5)
