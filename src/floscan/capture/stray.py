"""Stray Scanner export folders: parse, validate, associate and audit.

A session is opened read-only. ``open_session`` checks the layout and parses
the odometry log strictly; ``inspect_session`` then hashes every file,
decodes every depth and confidence image, decodes the video's presentation
timestamps, associates them with the sensor clock (``capture.sync``) and
checks the declared pose and depth conventions on the data itself
(reprojection consistency between frame pairs). Nothing is inferred from a
nominal frame rate, and a convention that the data does not confirm leaves
the inspection unverified instead of being assumed.

The declared conventions live in ``configs/capture_profiles/stray.yaml``.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import av
import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from floscan import __version__
from floscan.capture.sync import Association, associate
from floscan.contracts.base import Contract, Provenance
from floscan.contracts.capture import Capture, Clock, SourceApp
from floscan.contracts.geometry import Camera, Distortion, PixelTransform, Pose
from floscan.geometry.frames import (
    PinholeCamera,
    RigidTransform,
    depth_to_metres,
    optical_pose_from_apple,
    quaternion_xyzw_from_rotation,
    rotation_from_quaternion_xyzw,
)
from floscan.io.assets import asset_ref
from floscan.io.manifest import (
    FileEntry,
    file_manifest,
    media_digests,
    raw_manifest_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATH = PROJECT_ROOT / "configs" / "capture_profiles" / "stray.yaml"
SOURCE_WORLD = "stray_world"
SESSION_WORLD = "W"
# The CSV prints quaternions with about eight significant digits.
QUATERNION_PRINT_TOLERANCE = 1e-4
STAGE = "capture.normalize"

InspectionStatus = Literal["verified", "verified_with_frame_issues", "unverified"]


class StrayInputError(ValueError):
    """The folder is not a readable Stray Scanner session."""


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------


class LayoutProfile(Contract):
    video: str
    odometry: str
    camera_matrix: str
    imu: str
    depth_dir: str
    confidence_dir: str
    frame_name_digits: int
    odometry_header: list[str]
    imu_header: list[str]


class DepthProfile(Contract):
    png_mode: str
    source_unit: Literal["mm", "m"]
    scale_to_m: float
    depth_kind: Literal["optical_z", "range"]
    alignment: Literal["aligned_rgb", "separate_camera"]
    invalid_value: int
    plausible_median_m: list[float]


class ConfidenceProfile(Contract):
    png_mode: str
    values: list[int]
    encoding: str


class PoseProfile(Contract):
    translation_unit: Literal["m"]
    quaternion_order: Literal["xyzw"]
    direction: Literal["world_from_camera", "camera_from_world"]
    camera_axes: Literal["apple", "optical"]
    world_axes: Literal["arkit_gravity_y_up"]


class TrackingProfile(Contract):
    max_speed_m_s: float
    max_angular_speed_deg_s: float


class IntrinsicsProfile(Contract):
    per_frame: bool
    image: Literal["rgb"]
    distortion: Literal["not_recorded"]


class VideoProfile(Contract):
    required_rotation_deg: int


class SyncProfile(Contract):
    tolerance_fraction_of_min_interval: float
    max_start_offset: int
    min_matched_fraction: float
    min_evidence_gap: float


class ConventionCheckProfile(Contract):
    pairs: int
    frame_gap: int
    min_confidence: int
    min_rotation_deg: float
    min_translation_m: float
    min_pairs: int
    min_overlap_pixels: int
    max_median_error_m: float
    min_error_ratio: float


class ImuProfile(Contract):
    use: Literal["forbidden_until_verified"]


class StrayProfile(Contract):
    profile_id: str
    source_app: str
    recorder_version: str
    layout: LayoutProfile
    clocks: dict[str, str]
    depth: DepthProfile
    confidence: ConfidenceProfile
    pose: PoseProfile
    tracking: TrackingProfile
    intrinsics: IntrinsicsProfile
    video: VideoProfile
    sync: SyncProfile
    convention_check: ConventionCheckProfile
    imu: ImuProfile


def profile_hash(profile: StrayProfile) -> str:
    """SHA-256 of the profile's canonical JSON (the config every output used)."""
    return hashlib.sha256(profile.model_dump_json().encode("utf-8")).hexdigest()


def load_profile(path: Path = PROFILE_PATH) -> StrayProfile:
    return StrayProfile.model_validate(yaml.safe_load(path.read_text("utf-8")))


# --------------------------------------------------------------------------
# Session (strict parse of the layout and the odometry log)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedDepth:
    """Canonical depth of one frame: float32 metres, validity, raw confidence."""

    depth_m: NDArray[np.float32]
    valid: NDArray[np.bool_]
    confidence: NDArray[np.uint8] | None


@dataclass(frozen=True)
class FrameIssue:
    source_index: int
    code: str
    detail: str


@dataclass(frozen=True)
class StraySession:
    """A parsed session. Arrays are indexed by odometry row (source index)."""

    root: Path
    profile: StrayProfile
    frame_ids: NDArray[np.int64]
    sensor_time_s: NDArray[np.float64]
    translation_m: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    intrinsics: NDArray[np.float64]  # fx, fy, cx, cy at the RGB resolution
    max_quaternion_norm_error: float
    distortion_columns_empty: bool
    rgb_size: tuple[int, int]
    has_imu: bool

    @property
    def frame_count(self) -> int:
        return len(self.frame_ids)

    def frame_name(self, source_index: int) -> str:
        digits = self.profile.layout.frame_name_digits
        return f"{int(self.frame_ids[source_index]):0{digits}d}.png"

    def depth_path(self, source_index: int) -> Path:
        return self.root / self.profile.layout.depth_dir / self.frame_name(source_index)

    def confidence_path(self, source_index: int) -> Path:
        layout = self.profile.layout
        return self.root / layout.confidence_dir / self.frame_name(source_index)

    def rgb_camera(self, source_index: int) -> PinholeCamera:
        fx, fy, cx, cy = self.intrinsics[source_index]
        width, height = self.rgb_size
        return PinholeCamera(width, height, fx, fy, cx, cy)

    def depth_camera(
        self, source_index: int, depth_size: tuple[int, int]
    ) -> PinholeCamera:
        """RGB intrinsics resampled to the depth image (same aspect required)."""
        width, height = depth_size
        rgb_width, rgb_height = self.rgb_size
        if width * rgb_height != height * rgb_width:
            raise StrayInputError(
                f"depth {width}x{height} and RGB {rgb_width}x{rgb_height} differ in "
                "aspect ratio; the crop between them is unknown"
            )
        return self.rgb_camera(source_index).resized(width, height)

    def source_pose(self, source_index: int) -> RigidTransform:
        """``T_stray_world_from_C`` exactly as recorded (optical camera axes)."""
        quaternion = self.quaternion_xyzw[source_index]
        transform = RigidTransform.from_quaternion_xyzw(
            quaternion / np.linalg.norm(quaternion),
            self.translation_m[source_index],
            SOURCE_WORLD,
            "C",
            "m",
        )
        if self.profile.pose.direction == "camera_from_world":
            transform = transform.inverse()
        return transform

    def pose(self, source_index: int) -> RigidTransform:
        """``T_W_from_C``: optical camera in the +z-up session world, metres."""
        return optical_pose_from_apple(
            self.source_pose(source_index),
            SESSION_WORLD,
            "C",
            camera_axes=self.profile.pose.camera_axes,
        )

    def depth(self, source_index: int) -> NormalizedDepth:
        """Depth of one frame in metres; zero source values are invalid.

        Raises:
            StrayInputError: if the depth image is missing, unreadable or of
                the wrong pixel format.
        """
        raw = _read_png(self.depth_path(source_index), self.profile.depth.png_mode)
        if raw is None:
            raise StrayInputError(f"no depth image for frame {source_index}")
        metres, valid = depth_to_metres(raw, self.profile.depth.scale_to_m)
        confidence = _read_png(
            self.confidence_path(source_index), self.profile.confidence.png_mode
        )
        if confidence is not None and confidence.shape != raw.shape:
            confidence = None
        return NormalizedDepth(metres, valid, confidence)


def _read_png(path: Path, mode: str) -> NDArray | None:
    """Pixels of a PNG in the expected mode; None if it does not exist."""
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            if image.mode != mode:
                raise StrayInputError(f"{path.name}: PNG mode {image.mode}, not {mode}")
            image.load()
            return np.array(image)
    except (OSError, UnidentifiedImageError, SyntaxError) as error:
        raise StrayInputError(f"{path.name}: unreadable PNG ({error})") from error


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [cell.strip() for cell in next(csv.reader(handle), [])]


def _parse_odometry(path: Path, profile: StrayProfile) -> dict[str, Any]:
    expected = profile.layout.odometry_header
    header = _read_header(path)
    if header != expected:
        raise StrayInputError(
            f"{path.name}: header {header} does not match the profile {expected}"
        )
    rows = []
    distortion_empty = True
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for line, row in enumerate(reader, start=2):
            if not row:
                continue
            cells = [cell.strip() for cell in row]
            if len(cells) != len(expected):
                raise StrayInputError(
                    f"{path.name}:{line}: {len(cells)} columns, "
                    f"expected {len(expected)}"
                )
            if any(cells[13:]):
                distortion_empty = False
            try:
                values = [float(cells[0]), int(cells[1])] + [
                    float(cell) for cell in cells[2:13]
                ]
            except ValueError as error:
                raise StrayInputError(f"{path.name}:{line}: {error}") from error
            if not all(math.isfinite(v) for v in values):
                raise StrayInputError(f"{path.name}:{line}: non-finite value")
            rows.append(values)
    if len(rows) < 2:
        raise StrayInputError(f"{path.name}: needs at least two frames")
    table = np.array(rows, dtype=np.float64)
    return {"table": table, "distortion_empty": distortion_empty}


def _video_size(path: Path) -> tuple[int, int]:
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise StrayInputError(f"{path.name}: no video stream")
            stream = container.streams.video[0]
            return int(stream.width), int(stream.height)
    except av.FFmpegError as error:
        raise StrayInputError(f"{path.name}: cannot open video ({error})") from error


def open_session(root: Path, profile: StrayProfile | None = None) -> StraySession:
    """Parse a Stray Scanner folder strictly; raise StrayInputError if invalid."""
    profile = profile or load_profile()
    layout = profile.layout
    if not root.is_dir():
        raise StrayInputError(f"{root} is not a directory")
    missing = [
        name
        for name in (layout.video, layout.odometry, layout.camera_matrix)
        if not (root / name).is_file()
    ] + [
        name
        for name in (layout.depth_dir, layout.confidence_dir)
        if not (root / name).is_dir()
    ]
    if missing:
        raise StrayInputError(
            f"{root} is not a Stray Scanner session: missing {missing}"
        )
    parsed = _parse_odometry(root / layout.odometry, profile)
    table = parsed["table"]
    times, frame_ids = table[:, 0], table[:, 1].astype(np.int64)
    if (np.diff(times) <= 0).any():
        index = int(np.nonzero(np.diff(times) <= 0)[0][0]) + 1
        raise StrayInputError(
            f"{layout.odometry}: timestamps not strictly increasing at row {index}"
        )
    if (np.diff(frame_ids) <= 0).any() or frame_ids[0] < 0:
        raise StrayInputError(f"{layout.odometry}: frame ids not strictly increasing")
    quaternions = table[:, 5:9]
    norm_error = np.abs(np.linalg.norm(quaternions, axis=1) - 1.0)
    if norm_error.max() > QUATERNION_PRINT_TOLERANCE:
        index = int(norm_error.argmax())
        raise StrayInputError(
            f"{layout.odometry}: row {index} quaternion norm off by "
            f"{norm_error[index]:.2e}; not a rotation"
        )
    rgb_size = _video_size(root / layout.video)
    intrinsics = table[:, 9:13]
    width, height = rgb_size
    bad = (
        (intrinsics[:, 0] <= 0)
        | (intrinsics[:, 1] <= 0)
        | (intrinsics[:, 2] < -0.5)
        | (intrinsics[:, 2] > width - 0.5)
        | (intrinsics[:, 3] < -0.5)
        | (intrinsics[:, 3] > height - 0.5)
    )
    if bad.any():
        raise StrayInputError(
            f"{layout.odometry}: row {int(np.nonzero(bad)[0][0])} intrinsics do not "
            f"fit the {width}x{height} video"
        )
    return StraySession(
        root=root,
        profile=profile,
        frame_ids=frame_ids,
        sensor_time_s=times,
        translation_m=table[:, 2:5],
        quaternion_xyzw=quaternions,
        intrinsics=intrinsics,
        max_quaternion_norm_error=float(norm_error.max()),
        distortion_columns_empty=parsed["distortion_empty"],
        rgb_size=rgb_size,
        has_imu=(root / layout.imu).is_file(),
    )


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------


@dataclass
class Inspection:
    """Everything ``inspect_session`` established, including what it could not."""

    status: InspectionStatus
    reasons: list[str]
    session: StraySession
    files: list[FileEntry]
    video: dict[str, Any]
    odometry: dict[str, Any]
    intrinsics: dict[str, Any]
    depth: dict[str, Any]
    confidence: dict[str, Any]
    association: Association
    video_pts_s: NDArray[np.float64]
    conventions: dict[str, Any]
    imu: dict[str, Any]
    convention_gaps: list[str]
    frame_issues: list[FrameIssue] = field(default_factory=list)

    @property
    def raw_manifest_hash(self) -> str:
        return raw_manifest_hash(self.files)


def _decode_presentation_times(path: Path) -> dict[str, Any]:
    """Presented frames' PTS (as a decoder delivers them) and stream facts."""
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base
            packets = sum(
                1 for packet in container.demux(stream) if packet.pts is not None
            )
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            pts, rotations = [], set()
            for frame in container.decode(stream):
                pts.append(frame.pts)
                rotations.add(int(frame.rotation))
            info = {
                "codec": stream.codec_context.name,
                "width": int(stream.width),
                "height": int(stream.height),
                "time_base": str(time_base),
                "nominal_rate_not_used": str(stream.average_rate),
                "packets": packets,
                "presented_frames": len(pts),
                "packets_not_presented": packets - len(pts),
                "rotation_deg": sorted(rotations),
            }
    except av.FFmpegError as error:
        raise StrayInputError(f"{path.name}: cannot decode video ({error})") from error
    if time_base is None or not pts or any(p is None for p in pts):
        raise StrayInputError(f"{path.name}: frames without presentation timestamps")
    seconds = np.array(pts, dtype=np.float64) * float(time_base)
    info["first_pts_s"] = float(seconds[0])
    info["span_s"] = float(seconds[-1] - seconds[0])
    return {"info": info, "pts_s": seconds}


def _scan_depth(
    session: StraySession, issues: list[FrameIssue]
) -> tuple[dict[str, Any], dict[str, Any], list[bool]]:
    """Decode every depth and confidence image; record per-frame problems."""
    profile = session.profile
    sizes: Counter[tuple[int, int]] = Counter()
    raw: dict[int, NDArray] = {}
    usable = [False] * session.frame_count
    medians, zero_fractions = [], []
    confidence_counts = np.zeros(256, dtype=np.int64)
    allowed = set(profile.confidence.values)
    first_size: tuple[int, int] | None = None
    for index in range(session.frame_count):
        try:
            depth = _read_png(session.depth_path(index), profile.depth.png_mode)
        except StrayInputError as error:
            issues.append(FrameIssue(index, "depth_corrupt", str(error)))
            continue
        if depth is None:
            issues.append(FrameIssue(index, "depth_missing", "no depth image"))
            continue
        size = (depth.shape[1], depth.shape[0])
        first_size = first_size or size
        sizes[size] += 1
        if size != first_size:
            issues.append(
                FrameIssue(
                    index,
                    "depth_wrong_size",
                    f"{size[0]}x{size[1]}, session uses "
                    f"{first_size[0]}x{first_size[1]}",
                )
            )
            continue
        valid = depth != profile.depth.invalid_value
        zero_fractions.append(1.0 - float(valid.mean()))
        if valid.any():
            medians.append(float(np.median(depth[valid])) * profile.depth.scale_to_m)
        usable[index] = True
        raw[index] = depth
        try:
            confidence = _read_png(
                session.confidence_path(index), profile.confidence.png_mode
            )
        except StrayInputError as error:
            issues.append(FrameIssue(index, "confidence_corrupt", str(error)))
            continue
        if confidence is None:
            issues.append(
                FrameIssue(index, "confidence_missing", "no confidence image")
            )
        elif confidence.shape != depth.shape:
            issues.append(
                FrameIssue(index, "confidence_wrong_size", f"{confidence.shape}")
            )
        else:
            counts = np.bincount(confidence.ravel(), minlength=256)
            confidence_counts += counts
            unexpected = set(np.nonzero(counts)[0].tolist()) - allowed
            if unexpected:
                issues.append(
                    FrameIssue(
                        index,
                        "confidence_unexpected_values",
                        f"values {sorted(unexpected)} outside {sorted(allowed)}",
                    )
                )
    extra = _unreferenced_images(session)
    depth_summary = {
        "size": list(first_size) if first_size else None,
        "sizes_seen": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "png_mode": profile.depth.png_mode,
        "source_unit": profile.depth.source_unit,
        "usable_frames": sum(usable),
        "invalid_fraction_median": float(np.median(zero_fractions))
        if zero_fractions
        else None,
        "median_depth_m": {
            "min": min(medians),
            "median": float(np.median(medians)),
            "max": max(medians),
        }
        if medians
        else None,
        "unreferenced_files": extra,
    }
    present = {
        str(v): int(confidence_counts[v]) for v in np.nonzero(confidence_counts)[0]
    }
    confidence_summary = {
        "encoding": profile.confidence.encoding,
        "pixel_counts": present,
    }
    return depth_summary, confidence_summary, usable


def _unreferenced_images(session: StraySession) -> dict[str, list[str]]:
    names = {session.frame_name(i) for i in range(session.frame_count)}
    layout = session.profile.layout
    extra = {}
    for directory in (layout.depth_dir, layout.confidence_dir):
        found = {p.name for p in (session.root / directory).iterdir() if p.is_file()}
        unexpected = sorted(found - names - {".DS_Store"})
        if unexpected:
            extra[directory] = unexpected[:20]
    return extra


def _timing_summary(times: NDArray[np.float64]) -> dict[str, Any]:
    intervals = np.diff(times)
    smallest = float(intervals.min())
    ticks = Counter(np.rint(intervals / smallest).astype(int).tolist())
    return {
        "rows": len(times),
        "span_s": float(times[-1] - times[0]),
        "smallest_interval_s": smallest,
        "median_interval_s": float(np.median(intervals)),
        "intervals_in_smallest_interval_units": {
            str(k): v for k, v in sorted(ticks.items())
        },
        "mean_rate_hz": float((len(times) - 1) / (times[-1] - times[0])),
    }


def _intrinsics_summary(session: StraySession) -> dict[str, Any]:
    k = session.intrinsics
    summary = {
        "per_frame": True,
        "image_size": list(session.rgb_size),
        "fx_range": [float(k[:, 0].min()), float(k[:, 0].max())],
        "fy_range": [float(k[:, 1].min()), float(k[:, 1].max())],
        "cx_range": [float(k[:, 2].min()), float(k[:, 2].max())],
        "cy_range": [float(k[:, 3].min()), float(k[:, 3].max())],
        "max_abs_fx_minus_fy": float(np.abs(k[:, 0] - k[:, 1]).max()),
        "distortion": "not recorded (columns empty)"
        if session.distortion_columns_empty
        else "distortion columns are filled but unused",
    }
    path = session.root / session.profile.layout.camera_matrix
    try:
        matrix = np.loadtxt(path, delimiter=",")
        last = k[-1]
        expected = np.array(
            [[last[0], 0, last[2]], [0, last[1], last[3]], [0, 0, 1]], dtype=float
        )
        summary["camera_matrix_csv"] = (
            "equals the last frame's intrinsics (unused; per-frame values are used)"
            if matrix.shape == (3, 3) and np.allclose(matrix, expected, atol=1e-4)
            else "differs from the per-frame intrinsics (unused)"
        )
    except ValueError as error:
        summary["camera_matrix_csv"] = f"unreadable ({error}); unused"
    return summary


def _imu_summary(session: StraySession) -> dict[str, Any]:
    layout = session.profile.layout
    if not session.has_imu:
        return {"present": False}
    path = session.root / layout.imu
    header = _read_header(path)
    if header != layout.imu_header:
        return {"present": True, "status": f"unexpected header {header}; not parsed"}
    try:
        table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except ValueError as error:
        return {"present": True, "status": f"unreadable ({error})"}
    times = table[:, 0]
    magnitude = np.linalg.norm(table[:, 1:4], axis=1)
    return {
        "present": True,
        "rows": len(table),
        "median_interval_s": float(np.median(np.diff(times)))
        if len(times) > 1
        else None,
        "time_range_s": [float(times.min()), float(times.max())],
        "overlaps_odometry_clock": bool(
            times.min() < session.sensor_time_s[-1]
            and times.max() > session.sensor_time_s[0]
        ),
        "accel_magnitude_median": float(np.median(magnitude)),
        "status": "parsed; units, axes and gravity sign unverified, so not used",
    }


def _rotation_angle_deg(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    cosine = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _convention_check(
    session: StraySession, usable: list[bool], depth_size: tuple[int, int]
) -> dict[str, Any]:
    """Which pose/depth convention makes depth maps agree across frame pairs.

    Each hypothesis (pose direction, camera axes, depth kind) unprojects the
    high-confidence depth of frame i, moves it into frame j with the poses
    and compares the predicted depth with frame j's measured depth (both in
    the hypothesis's depth kind). The median absolute disagreement is the
    score; only the right convention is consistent for moving cameras.
    """
    cfg = session.profile.convention_check
    n = session.frame_count
    gap = max(1, min(cfg.frame_gap, (n - 1) // 4))
    width, height = depth_size
    v, u = np.mgrid[0:height, 0:width]
    pixels = np.column_stack([u.ravel(), v.ravel()]).astype(np.float64)
    quats = session.quaternion_xyzw / np.linalg.norm(
        session.quaternion_xyzw, axis=1, keepdims=True
    )
    pairs, skipped_static = [], 0
    for i in np.linspace(0, n - gap - 1, cfg.pairs).astype(int):
        j = int(i) + gap
        if not (usable[i] and usable[j]):
            continue
        rotation = _rotation_angle_deg(
            rotation_from_quaternion_xyzw(quats[i]),
            rotation_from_quaternion_xyzw(quats[j]),
        )
        moved = float(
            np.linalg.norm(session.translation_m[j] - session.translation_m[i])
        )
        if rotation < cfg.min_rotation_deg and moved < cfg.min_translation_m:
            skipped_static += 1
            continue
        pairs.append((int(i), j))
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    hypotheses = list(
        itertools.product(
            ("world_from_camera", "camera_from_world"),
            ("optical", "apple"),
            ("optical_z", "range"),
        )
    )

    def world_from_optical(index: int, direction: str, axes: str) -> NDArray:
        transform = np.eye(4)
        transform[:3, :3] = rotation_from_quaternion_xyzw(quats[index])
        transform[:3, 3] = session.translation_m[index]
        if direction == "camera_from_world":
            transform = np.linalg.inv(transform)
        return transform @ flip if axes == "apple" else transform

    errors: dict[tuple[str, str, str], list[float]] = {h: [] for h in hypotheses}
    for i, j in pairs:
        di, dj = session.depth(i), session.depth(j)
        if di.confidence is None or dj.confidence is None:
            continue
        cam_i = session.depth_camera(i, depth_size)
        cam_j = session.depth_camera(j, depth_size)
        keep = di.valid.ravel() & (di.confidence.ravel() >= cfg.min_confidence)
        rays = cam_i.rays(pixels[keep])
        depth_i = di.depth_m.ravel()[keep].astype(np.float64)
        target_ok = dj.valid & (dj.confidence >= cfg.min_confidence)
        for hypothesis in hypotheses:
            direction, axes, kind = hypothesis
            if kind == "optical_z":
                points = rays * depth_i[:, None]
            else:
                unit = rays / np.linalg.norm(rays, axis=1, keepdims=True)
                points = unit * depth_i[:, None]
            relative = np.linalg.inv(
                world_from_optical(j, direction, axes)
            ) @ world_from_optical(i, direction, axes)
            moved_points = points @ relative[:3, :3].T + relative[:3, 3]
            front = moved_points[:, 2] > 1e-3
            moved_points = moved_points[front]
            col = np.rint(
                cam_j.fx * moved_points[:, 0] / moved_points[:, 2] + cam_j.cx
            ).astype(int)
            row = np.rint(
                cam_j.fy * moved_points[:, 1] / moved_points[:, 2] + cam_j.cy
            ).astype(int)
            inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
            col, row, moved_points = col[inside], row[inside], moved_points[inside]
            measured_ok = target_ok[row, col]
            if measured_ok.sum() < cfg.min_overlap_pixels:
                continue
            measured = dj.depth_m[row[measured_ok], col[measured_ok]]
            if kind == "optical_z":
                predicted = moved_points[measured_ok, 2]
            else:
                predicted = np.linalg.norm(moved_points[measured_ok], axis=1)
            errors[hypothesis].append(float(np.median(np.abs(predicted - measured))))
    declared = (
        session.profile.pose.direction,
        session.profile.pose.camera_axes,
        session.profile.depth.depth_kind,
    )
    table = []
    for hypothesis in hypotheses:
        values = errors[hypothesis]
        table.append(
            {
                "direction": hypothesis[0],
                "camera_axes": hypothesis[1],
                "depth_kind": hypothesis[2],
                "pairs_scored": len(values),
                "median_error_m": float(np.median(values)) if values else None,
            }
        )
    table.sort(
        key=lambda r: math.inf if r["median_error_m"] is None else r["median_error_m"]
    )
    result: dict[str, Any] = {
        "frame_gap": gap,
        "pairs_considered": len(pairs) + skipped_static,
        "pairs_without_motion": skipped_static,
        "declared": dict(
            zip(("direction", "camera_axes", "depth_kind"), declared, strict=True)
        ),
        "depth_unit": session.profile.depth.source_unit,
        "hypotheses": table,
    }
    best = table[0]
    best_key = (best["direction"], best["camera_axes"], best["depth_kind"])
    scored = [r for r in table if r["median_error_m"] is not None]
    if best["pairs_scored"] < cfg.min_pairs:
        result.update(
            status="unverified",
            reason=(
                f"only {best['pairs_scored']} frame pairs with motion and overlap; "
                f"{cfg.min_pairs} needed"
            ),
        )
    elif best_key != declared:
        result.update(
            status="unverified",
            reason=(
                f"the data favour {best_key} ({best['median_error_m'] * 1e3:.1f} mm), "
                f"not the declared {declared}"
            ),
        )
    elif best["median_error_m"] > cfg.max_median_error_m:
        result.update(
            status="unverified",
            reason=(
                "best convention still disagrees by "
                f"{best['median_error_m'] * 1e3:.1f} mm "
                f"(limit {cfg.max_median_error_m * 1e3:.0f} mm); depth unit, "
                "intrinsics or poses are suspect"
            ),
        )
    elif len(scored) > 1 and scored[1]["median_error_m"] < cfg.min_error_ratio * max(
        best["median_error_m"], 1e-4
    ):
        result.update(
            status="unverified",
            reason=(
                f"runner-up convention is within a factor {cfg.min_error_ratio} "
                "of the declared one; the data cannot separate them"
            ),
        )
    else:
        runner = scored[1]["median_error_m"] if len(scored) > 1 else None
        result.update(
            status="verified",
            reason=(
                f"declared convention agrees to {best['median_error_m'] * 1e3:.1f} mm "
                f"over {best['pairs_scored']} pairs; next best "
                + (f"{runner * 1e3:.1f} mm" if runner is not None else "not scored")
            ),
        )
    return result


def _vertical_axis_check(session: StraySession) -> dict[str, Any]:
    """Is the source world's +y the vertical axis? (sign not testable here)."""
    spread = session.translation_m.std(axis=0)
    horizontal = float(max(spread[0], spread[2]))
    return {
        "trajectory_std_m": [float(s) for s in spread],
        "y_is_least_varying_axis": bool(spread[1] < min(spread[0], spread[2])),
        "status": "consistent"
        if spread[1] < 0.25 * horizontal
        else "not established (trajectory too short or not level)",
    }


def _pose_jumps(session: StraySession, issues: list[FrameIssue]) -> dict[str, Any]:
    """Flag steps faster than a handheld walk: likely tracking resets."""
    limits = session.profile.tracking
    dt = np.diff(session.sensor_time_s)
    step_m = np.linalg.norm(np.diff(session.translation_m, axis=0), axis=1)
    quats = session.quaternion_xyzw / np.linalg.norm(
        session.quaternion_xyzw, axis=1, keepdims=True
    )
    dots = np.clip(np.abs((quats[1:] * quats[:-1]).sum(axis=1)), 0.0, 1.0)
    step_deg = np.degrees(2.0 * np.arccos(dots))
    speed, angular = step_m / dt, step_deg / dt
    jumps = np.nonzero(
        (speed > limits.max_speed_m_s) | (angular > limits.max_angular_speed_deg_s)
    )[0]
    for k in jumps:
        issues.append(
            FrameIssue(
                int(k) + 1,
                "pose_jump",
                f"{step_m[k] * 100:.1f} cm and {step_deg[k]:.2f} deg in "
                f"{dt[k] * 1e3:.1f} ms after row {int(k)} (possible tracking reset)",
            )
        )
    return {
        "speed_m_s_p99": float(np.percentile(speed, 99)),
        "angular_speed_deg_s_p99": float(np.percentile(angular, 99)),
        "pose_jump_rows": [int(k) + 1 for k in jumps],
    }


def inspect_session(session: StraySession) -> Inspection:
    """Validate, associate and audit a parsed session (reads only)."""
    profile = session.profile
    issues: list[FrameIssue] = []
    reasons: list[str] = []
    files = file_manifest(session.root)
    video = _decode_presentation_times(session.root / profile.layout.video)
    video_info, pts_s = video["info"], video["pts_s"]
    if video_info["rotation_deg"] != [profile.video.required_rotation_deg]:
        reasons.append(
            f"video frames carry rotation {video_info['rotation_deg']}; the "
            "intrinsics refer to unrotated frames"
        )
    odometry = _timing_summary(session.sensor_time_s)
    odometry["frame_ids_contiguous"] = bool(
        np.array_equal(session.frame_ids, np.arange(session.frame_count))
    )
    odometry["max_quaternion_norm_error"] = session.max_quaternion_norm_error
    odometry["motion"] = _pose_jumps(session, issues)
    sync = profile.sync
    tolerance = (
        sync.tolerance_fraction_of_min_interval * odometry["smallest_interval_s"]
    )
    if len(pts_s) < 2:
        raise StrayInputError("the video has fewer than two presented frames")
    association = associate(
        pts_s,
        session.sensor_time_s,
        tolerance,
        sync.max_start_offset,
        sync.min_matched_fraction,
        sync.min_evidence_gap,
    )
    if association.status != "verified":
        reasons.append(
            f"video-sensor association {association.status}: {association.reason}"
        )
    else:
        for index, stream in enumerate(association.sensor_to_stream):
            if stream is None:
                issues.append(
                    FrameIssue(index, "no_rgb_frame", "no presented video frame")
                )
        unmatched = [k for k, j in enumerate(association.stream_to_sensor) if j is None]
        if unmatched:
            reasons.append(
                f"{len(unmatched)} presented video frames have no sensor row"
            )
    depth, confidence, usable = _scan_depth(session, issues)
    depth_size = tuple(depth["size"]) if depth["size"] else None
    if depth_size is None:
        reasons.append("no readable depth image")
        conventions: dict[str, Any] = {"status": "unverified", "reason": "no depth"}
    else:
        rgb_w, rgb_h = session.rgb_size
        if depth_size[0] * rgb_h != depth_size[1] * rgb_w:
            reasons.append(
                f"depth {depth_size[0]}x{depth_size[1]} and RGB {rgb_w}x{rgb_h} "
                "differ in aspect ratio; registration between them is unknown"
            )
            conventions = {"status": "unverified", "reason": "aspect ratio differs"}
        else:
            conventions = _convention_check(session, usable, depth_size)
        low, high = profile.depth.plausible_median_m
        median = depth["median_depth_m"]
        if median is not None and not low <= median["median"] <= high:
            reasons.append(
                f"median depth {median['median']:.3f} m outside {low}-{high} m; the "
                f"declared unit ({profile.depth.source_unit}) is suspect"
            )
    if conventions["status"] != "verified":
        reasons.append(f"conventions unverified: {conventions['reason']}")
    conventions["vertical_axis"] = _vertical_axis_check(session)
    gaps = [
        "recorder/app version: not present in the export",
        "device model: not present in the export",
        "lens distortion: not recorded; pinhole intrinsics only",
        "world up sign: ARKit gravity alignment (+y up) per source documentation; "
        "the vertical axis is checked from the trajectory, the sign is not",
        "tracking state: not recorded per frame; resets are inferred only from "
        "pose jumps (frame issue pose_jump)",
        "IMU: units, axes and gravity sign undocumented; parsed but not used",
        "display orientation: not recorded; frames are stored sensor-native "
        "(landscape) and may need rotating for display",
    ]
    if not session.distortion_columns_empty:
        gaps.append("distortion columns are filled; their model is undocumented")
    status: InspectionStatus = (
        "unverified"
        if reasons
        else "verified_with_frame_issues"
        if issues
        else "verified"
    )
    return Inspection(
        status=status,
        reasons=reasons,
        session=session,
        files=files,
        video=video_info,
        odometry=odometry,
        intrinsics=_intrinsics_summary(session),
        depth=depth,
        confidence=confidence,
        association=association,
        video_pts_s=pts_s,
        conventions=conventions,
        imu=_imu_summary(session),
        convention_gaps=gaps,
        frame_issues=issues,
    )


# --------------------------------------------------------------------------
# Records and report
# --------------------------------------------------------------------------


def _provenance(inspection: Inspection, sources: list[str]) -> Provenance:
    return Provenance(
        stage=STAGE,
        stage_version=f"floscan {__version__}; {inspection.session.profile.profile_id}",
        mode="live",
        config_hash=profile_hash(inspection.session.profile),
        source_asset_hashes=sources,
    )


def association_summary(inspection: Inspection) -> dict[str, Any]:
    a = inspection.association
    summary: dict[str, Any] = {
        "status": a.status,
        "reason": a.reason,
        "method": "timestamp association of presented video PTS with sensor "
        "timestamps; no nominal frame rate used",
        "tolerance_s": a.tolerance_s,
        "stream_frames": len(a.stream_to_sensor),
        "sensor_rows": len(a.sensor_to_stream),
        "matched": a.matched,
        "sensor_rows_without_video": [
            i for i, k in enumerate(a.sensor_to_stream) if k is None
        ][:50]
        if a.status == "verified"
        else None,
    }
    for name, hypothesis in (("chosen", a.chosen), ("runner_up", a.runner_up)):
        if hypothesis is not None:
            summary[name] = {
                "stream_start": hypothesis.stream_start,
                "sensor_start": hypothesis.sensor_start,
                "matched": hypothesis.matched,
            }
    if a.clock is not None:
        summary["clock_fit"] = {
            "sensor_minus_stream_offset_s": a.clock.offset_s,
            "rate_difference_ppm": a.clock.rate_ppm,
            "max_abs_residual_ms": a.clock.max_abs_residual_s * 1e3,
            "rms_residual_ms": a.clock.rms_residual_s * 1e3,
        }
    return summary


def report(inspection: Inspection) -> dict[str, Any]:
    """The inspection as a JSON-ready dictionary (no per-frame rows)."""
    issue_counts = Counter(issue.code for issue in inspection.frame_issues)
    return {
        "kind": "floscan-capture-inspection",
        "floscan_version": __version__,
        "profile_id": inspection.session.profile.profile_id,
        "profile_sha256": profile_hash(inspection.session.profile),
        "status": inspection.status,
        "reasons": inspection.reasons,
        "input": {
            "file_count": len(inspection.files),
            "byte_count": sum(e.byte_count for e in inspection.files),
            "raw_manifest_hash": inspection.raw_manifest_hash,
            "media_sha256": media_digests(inspection.files),
        },
        "video": inspection.video,
        "odometry": inspection.odometry,
        "intrinsics": inspection.intrinsics,
        "depth": inspection.depth,
        "confidence": inspection.confidence,
        "association": association_summary(inspection),
        "conventions": inspection.conventions,
        "imu": inspection.imu,
        "convention_gaps": inspection.convention_gaps,
        "frame_issues": {
            "counts": dict(sorted(issue_counts.items())),
            "first": [
                {"source_index": i.source_index, "code": i.code, "detail": i.detail}
                for i in inspection.frame_issues[:50]
            ],
        },
    }


def capture_record(inspection: Inspection) -> Capture:
    """The Capture manifest for this session (strict LiDAR tier)."""
    session = inspection.session
    layout = session.profile.layout
    by_name = {e.relative: e for e in inspection.files}
    assets = [
        asset_ref(session.root, name, by_name[name].sha256, by_name[name].byte_count)
        for name in (layout.video, layout.odometry, layout.camera_matrix, layout.imu)
        if name in by_name
    ]
    raw_hash = inspection.raw_manifest_hash
    verified = inspection.status != "unverified"
    return Capture(
        id=f"capture:{raw_hash[:16]}",
        provenance=_provenance(inspection, [a.sha256 for a in assets]),
        status="ok" if verified else "insufficient_evidence",
        status_reason=None if verified else "; ".join(inspection.reasons),
        property_session_id=f"stray:{raw_hash[:16]}",
        tier="lidar",
        source_app=SourceApp(name=session.profile.source_app),
        assets=assets,
        frames_uri="frames.jsonl",
        clock_domains=[
            Clock(
                id="stray_sensor",
                description=session.profile.clocks["sensor"],
                unit="s",
            ),
            Clock(
                id="video_pts", description=session.profile.clocks["video"], unit="s"
            ),
        ],
        # IMU is excluded until its units and axes are verified.
        allowed_modalities=[
            "rgb_video",
            "depth",
            "depth_confidence",
            "pose",
            "intrinsics",
        ],
        capture_profile_id=session.profile.profile_id,
        raw_manifest_hash=raw_hash,
    )


def frame_rows(inspection: Inspection):
    """Per-frame records: timing, association, camera, pose and depth files."""
    session = inspection.session
    by_name = {e.relative: e for e in inspection.files}
    issues: dict[int, list[str]] = {}
    for issue in inspection.frame_issues:
        issues.setdefault(issue.source_index, []).append(issue.code)
    odometry_hash = by_name[session.profile.layout.odometry].sha256
    association = inspection.association
    clock = association.clock
    identity = PixelTransform(kind="identity", matrix=np.eye(3).tolist())
    width, height = session.rgb_size
    for index in range(session.frame_count):
        stream = association.sensor_to_stream[index]
        video = None
        if stream is not None:
            pts = float(inspection.video_pts_s[stream])
            predicted = (
                clock.offset_s + (1 + clock.rate_ppm * 1e-6) * pts if clock else None
            )
            video = {
                "decoded_index": stream,
                "pts_s": pts,
                "fit_residual_ms": (session.sensor_time_s[index] - predicted) * 1e3
                if predicted is not None
                else None,
            }
        fx, fy, cx, cy = (float(v) for v in session.intrinsics[index])
        camera = Camera(
            id=f"camera:{index:06d}",
            provenance=_provenance(inspection, [odometry_hash]),
            status="partial",
            status_reason="lens distortion not recorded by the source",
            sensor_id="rgb",
            width_px=width,
            height_px=height,
            model="pinhole",
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion=Distortion(model="none"),
            intrinsics_origin="metadata",
            pixel_transform=identity,
        )
        transform = session.pose(index)
        pose = Pose(
            id=f"pose:{index:06d}",
            provenance=_provenance(inspection, [odometry_hash]),
            to_frame_id=SESSION_WORLD,
            from_frame_id=f"camera:{index:06d}",
            timestamp_s=float(session.sensor_time_s[index]),
            rotation_xyzw=quaternion_xyzw_from_rotation(transform.rotation).tolist(),
            translation=transform.translation.tolist(),
            translation_unit="m",
            covariance_status="unknown",
            method="Stray Scanner odometry (ARKit visual-inertial tracking)",
            tracking_state="not_available",
        )
        files = {}
        for key, directory in (
            ("depth", session.profile.layout.depth_dir),
            ("confidence", session.profile.layout.confidence_dir),
        ):
            relative = f"{directory}/{session.frame_name(index)}"
            entry = by_name.get(relative)
            files[key] = (
                {"uri": relative, "sha256": entry.sha256} if entry is not None else None
            )
        yield {
            "source_index": index,
            "frame_id": int(session.frame_ids[index]),
            "sensor_time_s": float(session.sensor_time_s[index]),
            "clock_id": "stray_sensor",
            "video": video,
            **files,
            "issues": issues.get(index, []),
            "camera": camera.model_dump(mode="json"),
            "pose": pose.model_dump(mode="json"),
        }


def summary_lines(inspection: Inspection) -> list[str]:
    """Short human-readable account of an inspection."""
    data = report(inspection)
    video, odometry = data["video"], data["odometry"]
    association, conventions = data["association"], data["conventions"]
    lines = [
        f"session: {inspection.session.root}",
        f"  files {data['input']['file_count']}, raw manifest "
        f"{data['input']['raw_manifest_hash']}",
        f"  sensor rows {odometry['rows']} over {odometry['span_s']:.2f} s (mean "
        f"{odometry['mean_rate_hz']:.1f} Hz; intervals in smallest-interval units "
        f"{odometry['intervals_in_smallest_interval_units']})",
        f"  video {video['codec']} {video['width']}x{video['height']}: "
        f"{video['presented_frames']} presented of {video['packets']} packets, "
        f"time base {video['time_base']}",
        f"  association {association['status']}: {association['reason']}",
    ]
    if "clock_fit" in association:
        fit = association["clock_fit"]
        lines.append(
            f"    clock rate difference {fit['rate_difference_ppm']:.1f} ppm, max "
            f"residual {fit['max_abs_residual_ms']:.3f} ms"
        )
    lines.append(f"  conventions {conventions['status']}: {conventions['reason']}")
    for row in conventions.get("hypotheses", [])[:3]:
        error = row["median_error_m"]
        lines.append(
            f"    {row['direction']}, {row['camera_axes']} camera, "
            f"{row['depth_kind']}: "
            + ("not scored" if error is None else f"{error * 1e3:.1f} mm")
        )
    depth = data["depth"]
    lines.append(
        f"  depth {depth['size']} {depth['source_unit']}: "
        f"{depth['usable_frames']}/{odometry['rows']} frames usable"
    )
    counts = data["frame_issues"]["counts"]
    lines.append(f"  frame issues: {counts or 'none'}")
    lines.append(f"  convention gaps: {len(data['convention_gaps'])} (see report)")
    lines.append(f"status: {data['status']}")
    lines.extend(f"  - {reason}" for reason in data["reasons"])
    return lines
