"""LiDAR (RGB-D) reconstruction: metric point observations and planes.

Input is a Stray session whose timing and conventions ``capture.stray``
verified. Depth of selected keyframes is unprojected with the per-frame
depth intrinsics as optical z (the verified depth kind), filtered, and moved
into the session world W (+z up) by the verified poses, exactly once.

Filtering keeps the reason for every dropped pixel. Invalid depth (zero in
the source) is missing evidence, never free space: it produces no point and
no visibility claim. Points are grouped into short, overlapping submaps along
the trajectory, never across a pose jump, because poses drift and the full
property is not fused here. Planes are discovered per submap with seeded
RANSAC on an Open3D voxel grid, split into connected patches, and refit robustly on
the raw points. Orientation classes rely on the documented +z-up world
(``orientation_is_prior``); a plane is not yet a floor, wall or ceiling of a
room.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from pydantic import Field, model_validator

from floscan import __version__
from floscan.capture.stray import Inspection, StrayInputError, StraySession
from floscan.contracts.base import Contract, Provenance
from floscan.contracts.geometry import Plane, ResidualStats
from floscan.geometry.planes import (
    PlaneFitError,
    orient_towards,
    plane_basis,
    ransac_plane,
    robust_refit,
)
from floscan.reconstruction.base import (
    BUNDLE_KIND,
    BUNDLE_VERSION,
    PlaneResult,
    SubmapPoints,
    SubmapResult,
    save_submap,
    save_support,
    submap_entry,
    write_manifest,
)

STAGE = "reconstruction.reconstruct"
FRAME_ISSUES_EXCLUDING_DEPTH = frozenset(
    {
        "depth_missing",
        "depth_corrupt",
        "depth_wrong_size",
        "confidence_missing",
        "confidence_corrupt",
        "confidence_wrong_size",
        "confidence_unexpected_values",
    }
)
FILTER_REASONS = (
    "invalid_depth",
    "low_confidence",
    "out_of_range",
    "depth_edge",
    "grazing_angle",
    "no_neighbours",
)


class LidarConfig(Contract):
    """Every filter and threshold of the stage (hashed into the bundle)."""

    keyframe_min_translation_m: float = Field(default=0.10, gt=0)
    keyframe_min_rotation_deg: float = Field(default=8.0, gt=0)
    pixel_stride: int = Field(default=2, ge=1)
    min_confidence: int = Field(default=2, ge=0)
    min_depth_m: float = Field(default=0.15, gt=0)
    max_depth_m: float = Field(default=4.0, gt=0)
    # A pixel whose depth differs from a 4-neighbour by more than this share
    # lies on a depth edge (mixed "flying" pixels); it is dropped.
    max_edge_jump_ratio: float = Field(default=0.05, gt=0)
    max_incidence_deg: float = Field(default=75.0, gt=0, lt=90)
    submap_max_travel_m: float = Field(default=2.5, gt=0)
    submap_max_keyframes: int = Field(default=40, ge=2)
    submap_overlap_keyframes: int = Field(default=3, ge=0)
    voxel_m: float = Field(default=0.05, gt=0)
    ransac_distance_m: float = Field(default=0.03, gt=0)
    ransac_iterations: int = Field(default=1000, ge=10)
    max_planes_per_submap: int = Field(default=20, ge=1)
    min_plane_voxels: int = Field(default=150, ge=3)
    cluster_eps_voxels: float = Field(default=2.5, gt=0)
    refit_cutoff_m: float = Field(default=0.03, gt=0)
    # Raw points join a plane's refit only if their own surface normal (from
    # the depth image) agrees, so the adjoining surface at a corner does not.
    max_point_normal_deg: float = Field(default=30.0, gt=0, lt=90)
    min_support_points: int = Field(default=400, ge=3)
    horizontal_tolerance_deg: float = Field(default=10.0, gt=0, lt=45)
    preview_points_per_submap: int = Field(default=50_000, ge=1)
    seed: int = 0

    @model_validator(mode="after")
    def _coupled(self) -> LidarConfig:
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("min_depth_m must be below max_depth_m")
        # Overlap keyframes are carried into the next submap; at or above the
        # cap, submaps would grow without bound instead of streaming.
        if self.submap_overlap_keyframes >= self.submap_max_keyframes:
            raise ValueError(
                "submap_overlap_keyframes must be below submap_max_keyframes"
            )
        return self


def config_hash(config: LidarConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Keyframes and submaps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """Rows between pose jumps: one continuous tracking world."""

    first: int
    last: int


def _segments(session: StraySession, inspection: Inspection) -> list[Segment]:
    jumps = sorted(
        {i.source_index for i in inspection.frame_issues if i.code == "pose_jump"}
    )
    starts = [0, *jumps]
    ends = [j - 1 for j in jumps] + [session.frame_count - 1]
    return [Segment(a, b) for a, b in zip(starts, ends, strict=True) if b >= a]


def select_keyframes(
    session: StraySession, inspection: Inspection, config: LidarConfig
) -> tuple[list[list[int]], dict[str, int]]:
    """Keyframes per segment, chosen by motion; unusable depth is skipped."""
    excluded = {
        i.source_index
        for i in inspection.frame_issues
        if i.code in FRAME_ISSUES_EXCLUDING_DEPTH
    }
    per_segment: list[list[int]] = []
    for segment in _segments(session, inspection):
        chosen: list[int] = []
        last_pose = None
        for index in range(segment.first, segment.last + 1):
            if index in excluded:
                continue
            pose = session.pose(index)
            if last_pose is not None:
                moved = float(np.linalg.norm(pose.translation - last_pose.translation))
                relative = last_pose.rotation.T @ pose.rotation
                turned = math.degrees(
                    math.acos(max(-1.0, min(1.0, (np.trace(relative) - 1) / 2)))
                )
                if (
                    moved < config.keyframe_min_translation_m
                    and turned < config.keyframe_min_rotation_deg
                ):
                    continue
            chosen.append(index)
            last_pose = pose
        per_segment.append(chosen)
    counts = {
        "rows": session.frame_count,
        "rows_without_usable_depth_or_confidence": len(excluded),
        "segments": len(per_segment),
        "keyframes": sum(len(s) for s in per_segment),
    }
    return per_segment, counts


def plan_submaps(
    session: StraySession, keyframes: list[list[int]], config: LidarConfig
) -> list[list[int]]:
    """Consecutive keyframes with bounded travel; overlap only inside a segment."""
    submaps: list[list[int]] = []
    for segment in keyframes:
        current: list[int] = []
        travel = 0.0
        for index in segment:
            if current:
                step = session.translation_m[index] - session.translation_m[current[-1]]
                travel += float(np.linalg.norm(step))
            current.append(index)
            if (
                travel >= config.submap_max_travel_m
                or len(current) >= config.submap_max_keyframes
            ):
                submaps.append(current)
                overlap = config.submap_overlap_keyframes
                current = current[-overlap:] if overlap else []
                travel = 0.0
                for a, b in zip(current, current[1:], strict=False):
                    step = session.translation_m[b] - session.translation_m[a]
                    travel += float(np.linalg.norm(step))
        if current and (not submaps or current != submaps[-1][-len(current) :]):
            submaps.append(current)
    return [s for s in submaps if s]


# --------------------------------------------------------------------------
# Per-frame observations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FramePoints:
    xyz: NDArray[np.float64]
    observations: NDArray[np.int32]
    normals: NDArray[np.float64]
    centre: NDArray[np.float64]
    counts: dict[str, int]


def frame_points(
    session: StraySession, index: int, depth_size: tuple[int, int], config: LidarConfig
) -> FramePoints:
    """Filtered metric points of one keyframe in W, with pixel provenance.

    Every pixel of the strided grid is either kept or counted under exactly
    one reason in ``FILTER_REASONS`` (first failing test wins, in order).
    """
    depth = session.depth(index)
    if depth.confidence is None:
        raise StrayInputError(f"frame {index}: no usable confidence")
    if session.profile.depth.depth_kind != "optical_z":
        raise StrayInputError("only optical-z depth is supported by this stage")
    camera = session.depth_camera(index, depth_size)
    height, width = depth.depth_m.shape
    z = depth.depth_m.astype(np.float64)
    v, u = np.mgrid[0:height, 0:width]
    rays = camera.rays(np.column_stack([u.ravel(), v.ravel()])).reshape(
        height, width, 3
    )
    camera_points = rays * z[..., None]  # optical z: p_C = z K^-1 [u, v, 1]

    # Normals and edges from 4-neighbours (central differences).
    def shifted(array: NDArray, dv: int, du: int) -> NDArray:
        out = np.full_like(array, np.nan)
        src = array[max(dv, 0) : height + min(dv, 0), max(du, 0) : width + min(du, 0)]
        out[max(-dv, 0) : height + min(-dv, 0), max(-du, 0) : width + min(-du, 0)] = src
        return out

    zf = np.where(depth.valid, z, np.nan)
    neighbours = [shifted(zf, dv, du) for dv, du in ((0, 1), (0, -1), (1, 0), (-1, 0))]
    stacked = np.stack(neighbours)
    has_neighbours = np.isfinite(stacked).all(axis=0)
    with np.errstate(invalid="ignore"):
        # NaN where a neighbour is missing; those pixels fail no_neighbours first.
        jump = np.max(np.abs(stacked - zf[None]), axis=0) / zf
    pf = np.where(depth.valid[..., None], camera_points, np.nan)
    du_vec = shifted(pf, 0, -1) - shifted(pf, 0, 1)  # p(u+1) - p(u-1)
    dv_vec = shifted(pf, -1, 0) - shifted(pf, 1, 0)  # p(v+1) - p(v-1)
    normals = np.cross(du_vec, dv_vec)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        view = -rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        normals *= np.sign(np.sum(normals * view, axis=-1, keepdims=True))
        cosine = np.sum(normals * view, axis=-1)

    grid = np.zeros((height, width), dtype=bool)
    grid[:: config.pixel_stride, :: config.pixel_stride] = True
    counts = {"pixels": int(grid.sum())}
    remaining = grid.copy()
    tests = (
        ("invalid_depth", depth.valid),
        ("low_confidence", depth.confidence >= config.min_confidence),
        ("out_of_range", (z >= config.min_depth_m) & (z <= config.max_depth_m)),
        ("no_neighbours", has_neighbours),
        ("depth_edge", jump <= config.max_edge_jump_ratio),
        (
            "grazing_angle",
            cosine >= math.cos(math.radians(config.max_incidence_deg)),
        ),
    )
    for reason, passes in tests:
        failing = remaining & ~passes  # NaN comparisons are already False
        counts[reason] = int(failing.sum())
        remaining &= ~failing
    counts["kept"] = int(remaining.sum())
    rows, cols = np.nonzero(remaining)
    pose = session.pose(index)
    xyz = camera_points[rows, cols] @ pose.rotation.T + pose.translation
    world_normals = normals[rows, cols] @ pose.rotation.T
    observations = np.column_stack([np.full(len(rows), index), cols, rows]).astype(
        np.int32
    )
    return FramePoints(xyz, observations, world_normals, pose.translation, counts)


# --------------------------------------------------------------------------
# Plane discovery
# --------------------------------------------------------------------------


def _voxel_keys(xyz: NDArray[np.float64], voxel: float) -> NDArray[np.int64]:
    return np.floor(xyz / voxel).astype(np.int64)


def _orientation_class(normal: NDArray[np.float64], config: LidarConfig) -> str:
    vertical = abs(float(normal[2]))
    tolerance = math.radians(config.horizontal_tolerance_deg)
    if vertical >= math.cos(tolerance):
        return "up_facing" if normal[2] > 0 else "down_facing"
    if vertical <= math.sin(tolerance):
        return "vertical"
    return "oblique"


def discover_planes(
    points: SubmapPoints, config: LidarConfig
) -> tuple[list[tuple[NDArray[np.int64], Any, dict[str, Any]]], dict[str, int]]:
    """Plane candidates of one submap: (support indices, fit, diagnostics).

    Seeded RANSAC (``geometry.planes.ransac_plane``) runs on an Open3D voxel
    grid for discovery only. Each RANSAC plane is split
    into spatially connected patches (DBSCAN), and every patch is refit with
    Tukey IRLS on the raw points of its voxels, never on voxel centroids.
    """
    if len(points.xyz) == 0:
        return [], {"no_points": 1}
    xyz = points.xyz.astype(np.float64)
    # Seeded per submap: the same input and config always give the same planes.
    rng = np.random.default_rng([config.seed, int(points.submap_id[1:])])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    voxels = cloud.voxel_down_sample(config.voxel_m)
    remaining = np.asarray(voxels.points)
    raw_keys = _voxel_keys(xyz, config.voxel_m)
    candidates = []
    rejected: dict[str, int] = {}
    for _ in range(config.max_planes_per_submap):
        if len(remaining) < config.min_plane_voxels:
            break
        try:
            ransac_normal, ransac_offset, inliers = ransac_plane(
                remaining, config.ransac_distance_m, config.ransac_iterations, rng
            )
        except PlaneFitError:
            break
        if len(inliers) < config.min_plane_voxels:
            break
        patch_points = remaining[inliers]
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]
        labels = np.asarray(
            o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(patch_points)
            ).cluster_dbscan(config.cluster_eps_voxels * config.voxel_m, 5)
        )
        for label in sorted(set(labels.tolist()) - {-1}):
            component = patch_points[labels == label]
            if len(component) < config.min_plane_voxels:
                rejected["small_patch"] = rejected.get("small_patch", 0) + 1
                continue
            # Raw points in the patch's voxels or their 26 neighbours.
            keys = _voxel_keys(component, config.voxel_m)
            offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * 3)).T.reshape(-1, 3)
            dilated = np.unique(
                (keys[:, None, :] + offsets[None]).reshape(-1, 3), axis=0
            )
            in_patch = _rows_in(raw_keys, dilated)
            normal, offset = ransac_normal, ransac_offset
            agrees = np.abs(points.normals.astype(np.float64) @ normal) >= math.cos(
                math.radians(config.max_point_normal_deg)
            )
            near = (
                in_patch
                & agrees
                & (np.abs(xyz @ normal + offset) < 2 * config.refit_cutoff_m)
            )
            index = np.nonzero(near)[0]
            try:
                fit = robust_refit(xyz[index], normal, offset, config.refit_cutoff_m)
            except PlaneFitError as error:
                key = f"refit_failed: {error}"
                rejected[key] = rejected.get(key, 0) + 1
                continue
            support = index[fit.inliers]
            if len(support) < config.min_support_points:
                rejected["too_few_raw_points"] = (
                    rejected.get("too_few_raw_points", 0) + 1
                )
                continue
            candidates.append((support, fit, {"ransac_voxels": int(len(component))}))
    return candidates, rejected


def _rows_in(keys: NDArray[np.int64], members: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Which rows of ``keys`` (N, 3) appear among ``members`` (M, 3)."""
    dtype = np.dtype((np.void, keys.dtype.itemsize * 3))
    flat = np.ascontiguousarray(keys).view(dtype).ravel()
    pool = np.ascontiguousarray(members).view(dtype).ravel()
    return np.isin(flat, pool)


def _plane_result(
    points: SubmapPoints,
    support: NDArray[np.int64],
    fit: Any,
    extra: dict[str, Any],
    plane_id: str,
    provenance: Provenance,
    config: LidarConfig,
) -> PlaneResult:
    xyz = points.xyz[support].astype(np.float64)
    normal, offset, agree = orient_towards(
        fit.normal, fit.offset, xyz, points.viewpoints[support]
    )
    orientation = _orientation_class(normal, config)
    centres = np.unique(points.viewpoints[support], axis=0)
    height = float(np.median(xyz[:, 2]))
    camera_low, camera_high = float(centres[:, 2].min()), float(centres[:, 2].max())
    u, v = plane_basis(normal)
    coords = np.column_stack([xyz @ u, xyz @ v])
    extent = coords.max(axis=0) - coords.min(axis=0)
    spread = np.sqrt(np.maximum(fit.eigenvalues[1:], 0.0))
    observability = "full" if spread.min() > 2 * config.voxel_m else "partial"
    covariance = None
    if fit.covariance is not None:
        flip = 1.0 if normal @ fit.normal > 0 else -1.0
        jacobian = np.diag([flip] * 4)
        covariance = (jacobian @ fit.covariance @ jacobian.T).tolist()
    record = Plane(
        id=plane_id,
        provenance=provenance,
        frame_id="W",
        normal=normal.tolist(),
        offset=float(offset),
        unit="m",
        support=None,
        basis_u=u.tolist(),
        basis_v=v.tolist(),
        parameter_covariance=covariance,
        residual_stats=ResidualStats(
            rms=fit.rms, max=fit.max_abs, count=int(len(support))
        ),
        # Room-surface roles (floor, ceiling, wall) need whole-room evidence;
        # they are assigned by room building, not here.
        kind="unknown",
        observability=observability,
        orientation_is_prior=True,
    )
    diagnostics = {
        **extra,
        "support_points": int(len(support)),
        "support_frames": int(len(np.unique(points.observations[support, 0]))),
        "orientation_class": orientation,
        "angle_to_vertical_deg": math.degrees(math.acos(min(1.0, abs(normal[2])))),
        "median_height_m": height,
        "height_below_lowest_camera_m": camera_low - height,
        "height_above_highest_camera_m": height - camera_high,
        "camera_height_range_m": [camera_low, camera_high],
        "extent_in_plane_m": [float(extent[0]), float(extent[1])],
        "share_of_views_in_front": agree,
        "refit_iterations": fit.iterations,
        "covariance_status": fit.covariance_status,
        "orientation_basis": "documented +z-up world (sign not verified)",
    }
    return PlaneResult(record, support, diagnostics)


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------


def reconstruct(
    session: StraySession,
    inspection: Inspection,
    output_dir: Path,
    config: LidarConfig | None = None,
) -> dict[str, Any]:
    """Run the stage, streaming submaps to ``output_dir``; returns the manifest.

    Raises:
        StrayInputError: if the inspection did not verify the session.
    """
    config = config or LidarConfig()
    if inspection.status == "unverified":
        raise StrayInputError(
            "reconstruction needs a verified capture: " + "; ".join(inspection.reasons)
        )
    if inspection.depth["size"] is None:
        raise StrayInputError("no readable depth")
    depth_size = tuple(inspection.depth["size"])
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = Provenance(
        stage=STAGE,
        stage_version=f"floscan {__version__}",
        mode="live",
        config_hash=config_hash(config),
        source_asset_hashes=[inspection.raw_manifest_hash],
    )
    keyframes, keyframe_counts = select_keyframes(session, inspection, config)
    submap_plan = plan_submaps(session, keyframes, config)
    cache: dict[int, FramePoints] = {}
    totals = dict.fromkeys(("pixels", *FILTER_REASONS, "kept"), 0)
    counted: set[int] = set()
    entries, all_planes = [], []
    preview_xyz, preview_rgb = [], []
    for number, frames in enumerate(submap_plan):
        submap_id = f"S{number:03d}"
        parts = []
        for index in frames:
            if index not in cache:
                cache[index] = frame_points(session, index, depth_size, config)
            part = cache[index]
            parts.append(part)
            if index not in counted:
                counted.add(index)
                for key, value in part.counts.items():
                    totals[key] += value
        keep = set(frames[-config.submap_overlap_keyframes :]) if frames else set()
        cache = {k: v for k, v in cache.items() if k in keep}
        points = SubmapPoints(
            submap_id=submap_id,
            xyz=np.concatenate([p.xyz for p in parts]).astype(np.float32),
            observations=np.concatenate([p.observations for p in parts]),
            normals=np.concatenate([p.normals for p in parts]).astype(np.float32),
            viewpoints=np.concatenate(
                [np.repeat(p.centre[None], len(p.xyz), axis=0) for p in parts]
            ).astype(np.float32),
        )
        record = save_submap(output_dir, points, provenance)
        candidates, rejected = discover_planes(points, config)
        travel = float(
            sum(
                np.linalg.norm(session.translation_m[b] - session.translation_m[a])
                for a, b in zip(frames, frames[1:], strict=False)
            )
        )
        result = SubmapResult(
            submap_id=submap_id,
            record=record,
            keyframes=frames,
            diagnostics={
                "points": int(len(points.xyz)),
                "travel_m": travel,
                "time_span_s": float(
                    session.sensor_time_s[frames[-1]] - session.sensor_time_s[frames[0]]
                ),
                "rejected_plane_candidates": rejected,
            },
        )
        for k, (support, fit, extra) in enumerate(candidates):
            plane_id = f"plane:{submap_id}:P{k:02d}"
            plane = _plane_result(
                points, support, fit, extra, plane_id, provenance, config
            )
            plane.record = plane.record.model_copy(
                update={
                    "support": save_support(output_dir, submap_id, f"P{k:02d}", support)
                }
            )
            Plane.model_validate(plane.record.model_dump())
            result.planes.append(plane)
            all_planes.append((submap_id, plane))
        entries.append(submap_entry(result))
        colours = np.full((len(points.xyz), 3), 0.6)
        for k, plane in enumerate(result.planes):
            colours[plane.support] = PALETTE[(number * 7 + k) % len(PALETTE)]
        step = max(1, len(points.xyz) // config.preview_points_per_submap)
        preview_xyz.append(points.xyz[::step].astype(np.float64))
        preview_rgb.append(colours[::step])
    manifest = {
        "kind": BUNDLE_KIND,
        "bundle_version": BUNDLE_VERSION,
        "floscan_version": __version__,
        "tier": "lidar",
        "scale_status": "metric (LiDAR depth in mm, verified by reprojection)",
        "frame": {
            "id": "W",
            "description": "session world: source ARKit world rotated to +z up; "
            "the up sign is the source's documented convention (unverified)",
            "unit": "m",
        },
        "capture_raw_manifest_hash": inspection.raw_manifest_hash,
        "config": config.model_dump(mode="json"),
        "config_hash": config_hash(config),
        "keyframes": keyframe_counts,
        "segments": [
            {"first_keyframe": s[0], "last_keyframe": s[-1], "keyframes": len(s)}
            for s in keyframes
            if s
        ],
        "filter_counts": totals,
        "submaps": entries,
        "summary": _summary(all_planes),
        "limitations": [
            "Submaps share the session world but are not registered or "
            "drift-corrected against each other; planes are per submap.",
            "Every plane's kind is 'unknown': floor, ceiling and wall roles need "
            "room-level evidence. Orientation classes use the documented +z-up "
            "world (sign unverified); tables and shelves are up-facing too.",
            "Plane covariance assumes independent residuals and is optimistic.",
            "Mirrors and glass are not detected; depth confidence alone cannot "
            "identify them.",
            "Invalid depth produces no point and no free-space evidence.",
            "Points carry no colour yet (video frames are not decoded here).",
        ],
    }
    manifest["evidence_status"] = (
        "no_points"
        if totals["kept"] == 0
        else "points_without_planes"
        if not all_planes
        else "points_and_planes"
    )
    manifest["preview"] = _write_preview(
        output_dir / "preview.ply", preview_xyz, preview_rgb
    )
    write_manifest(output_dir, manifest)
    return manifest


PALETTE = np.array(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.60, 0.90],
        [0.20, 0.75, 0.20],
        [0.95, 0.60, 0.05],
        [0.60, 0.20, 0.80],
        [0.05, 0.75, 0.70],
        [0.85, 0.35, 0.60],
        [0.55, 0.45, 0.10],
    ]
)


def _write_preview(path: Path, xyz: list, rgb: list) -> dict[str, str]:
    """Write the viewing aid if there is anything to show; never decisive.

    Returns the manifest entry: the file, or why it is unavailable.
    """
    points = np.concatenate(xyz) if xyz else np.zeros((0, 3))
    if len(points) == 0:
        return {"status": "unavailable", "reason": "no points survived the filters"}
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(rgb))
    if not o3d.io.write_point_cloud(str(path), cloud):
        return {
            "status": "unavailable",
            "reason": f"Open3D could not write {path.name}",
        }
    return {
        "status": "available",
        "path": path.name,
        "description": "sampled points of every submap; each plane's support "
        "in one colour, unassigned points grey (for viewing only)",
    }


def _summary(planes: list[tuple[str, PlaneResult]]) -> dict[str, Any]:
    """Plausibility evidence across submaps; nothing here is a measurement."""
    classes: dict[str, int] = {}
    levels: dict[str, list[dict[str, float]]] = {"up_facing": [], "down_facing": []}
    angles, weights = [], []
    for _, plane in planes:
        diagnostics = plane.diagnostics
        orientation = diagnostics["orientation_class"]
        classes[orientation] = classes.get(orientation, 0) + 1
        if orientation in levels:
            levels[orientation].append(
                {
                    "height_m": diagnostics["median_height_m"],
                    "support_points": diagnostics["support_points"],
                }
            )
        if orientation == "vertical":
            nx, ny = plane.record.normal[:2]
            angles.append(math.atan2(ny, nx))
            weights.append(diagnostics["support_points"])
    walls: dict[str, Any] = {"vertical_planes": len(angles)}
    if angles:
        # Directions modulo 90 degrees: a rectangular interior has one family.
        a, w = np.asarray(angles), np.asarray(weights, dtype=float)
        mean = math.atan2(float(w @ np.sin(4 * a)), float(w @ np.cos(4 * a))) / 4
        deviation = np.degrees(np.abs((a - mean + np.pi / 4) % (np.pi / 2) - np.pi / 4))
        walls.update(
            dominant_direction_deg=math.degrees(mean),
            support_share_within_5deg_of_perpendicular_family=float(
                w[deviation <= 5.0].sum() / w.sum()
            ),
        )
    return {
        "planes": len(planes),
        "orientation_classes": classes,
        "vertical_plane_directions": walls,
        "horizontal_levels": {
            name: sorted(items, key=lambda item: item["height_m"])
            for name, items in levels.items()
        },
        "note": "plausibility evidence only: no floor, ceiling, wall or "
        "dimension is identified or measured here",
    }


# --------------------------------------------------------------------------
# Live run (called by floscan.pipeline for --tier lidar --mode live)
# --------------------------------------------------------------------------


@dataclass
class LiveOutcome:
    """Stage results and artefacts of one live LiDAR run."""

    stages: list[dict[str, str]]
    diagnostics: list[dict[str, str]]
    status: str  # a pipeline result status
    status_reason: str


def run_live(
    input_dir: Path, output_dir: Path, config: LidarConfig | None = None
) -> LiveOutcome:
    """capture.normalize then reconstruction.reconstruct, writing their outputs.

    Outputs go to ``output_dir/capture`` and ``output_dir/reconstruction``.
    A capture that cannot be parsed is invalid input; one whose timing or
    conventions are unverified stops before reconstruction. Nothing after
    reconstruction exists yet, so a successful run is still incomplete.
    """
    from floscan.capture import stray
    from floscan.io.assets import sha256_file
    from floscan.io.manifest import write_capture_outputs

    stages: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []

    def stage(name: str, status: str, reason: str) -> None:
        stages.append({"name": name, "status": status, "reason": reason})

    def output(path: Path) -> None:
        relative = path.relative_to(output_dir).as_posix()
        diagnostics.append(
            {
                "code": "stage_output",
                "message": f"{relative} sha256={sha256_file(path)}",
            }
        )

    try:
        session = stray.open_session(input_dir)
        inspection = stray.inspect_session(session)
    except stray.StrayInputError as error:
        reason = f"not a readable Stray Scanner LiDAR session: {error}"
        stage("capture.normalize", "invalid_input", reason)
        stage(STAGE, "skipped", "Not attempted: the capture is invalid.")
        return LiveOutcome(stages, diagnostics, "invalid_input", reason)
    written = write_capture_outputs(
        output_dir / "capture",
        stray.capture_record(inspection),
        stray.report(inspection),
        stray.frame_rows(inspection),
    )
    for path in written[:2]:
        output(path)
    issues = _issue_counts(inspection)
    if inspection.status == "unverified":
        reason = "capture timing or conventions unverified: " + "; ".join(
            inspection.reasons
        )
        stage("capture.normalize", "insufficient_evidence", reason)
        stage(STAGE, "skipped", "Not attempted: the capture is not verified.")
        return LiveOutcome(stages, diagnostics, "insufficient_evidence", reason)
    stage(
        "capture.normalize",
        "ok",
        f"{session.frame_count} sensor rows; association and conventions verified"
        + (f"; frame issues {issues}" if issues else "")
        + "; see capture/inspection.json",
    )
    manifest = reconstruct(session, inspection, output_dir / "reconstruction", config)
    output(output_dir / "reconstruction" / "bundle.json")
    summary = manifest["summary"]
    if manifest["evidence_status"] == "no_points":
        reason = (
            "no point survived the depth filters "
            f"{manifest['filter_counts']}; see reconstruction/bundle.json"
        )
        stage(STAGE, "insufficient_evidence", reason)
        return LiveOutcome(stages, diagnostics, "insufficient_evidence", reason)
    stage(
        STAGE,
        "ok",
        f"{manifest['keyframes']['keyframes']} keyframes in {len(manifest['submaps'])} "
        f"submaps; {summary['planes']} plane candidates "
        f"{summary['orientation_classes']} ({manifest['evidence_status']}); "
        "metric points and planes in reconstruction/ (no rooms, surfaces or "
        "measurements)",
    )
    for limitation in manifest["limitations"]:
        diagnostics.append({"code": "reconstruction_limitation", "message": limitation})
    reason = (
        "The output contract is incomplete: the LiDAR capture was normalized and "
        "reconstructed into metric points and plane candidates, but no room, "
        "surface, measurement, damage, rule, export or rendering stage exists "
        "in this build."
    )
    return LiveOutcome(stages, diagnostics, "unsupported", reason)


def _issue_counts(inspection: Inspection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in inspection.frame_issues:
        counts[issue.code] = counts.get(issue.code, 0) + 1
    return counts
