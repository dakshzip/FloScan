"""Coordinate frames, rigid and similarity transforms, cameras and vendor conversions.

Conventions (docs/implementation-strategy/02-data-contracts.md):

- Right-handed frames, column vectors. ``T_A_from_B`` maps ``p_B`` to
  ``p_A = R p_B + t``. Every transform carries its destination and source
  frame IDs, and composition refuses mismatched frames.
- Lengths are metres (``"m"``) or, before scale is resolved,
  ``"reconstruction_unit"``. A rigid transform keeps the unit of its input.
  A similarity transform is the only way from ``reconstruction_unit`` to
  metres, and it refuses metric input, so scale is applied exactly once.
- Quaternions are Hamilton, scalar last ``[qx, qy, qz, qw]``, canonical with
  ``qw >= 0`` when produced here.
- Pixel centres sit at integer coordinates, origin at the top-left pixel
  centre. Resizing by ``s`` maps ``c`` to ``s (c + 0.5) - 0.5``.
- Optical camera C: +x right, +y down, +z forward. Raw Apple camera A: +x
  right, +y up, looking along -z. Session world W: +z up (gravity).

All vendor conventions are converted here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

LengthUnit = Literal["m", "reconstruction_unit"]
DepthKind = Literal["optical_z", "range"]
LENGTH_UNITS: tuple[LengthUnit, ...] = ("m", "reconstruction_unit")

ROTATION_TOLERANCE = 1e-6
QUATERNION_TOLERANCE = 1e-6

# Raw Apple camera (A) from optical camera (C): flip y and z.
R_APPLE_CAMERA_FROM_OPTICAL = np.diag([1.0, -1.0, -1.0])
# Session world W (+z up) from an Apple/ARKit world (+y up):
# x_W = x_v, y_W = -z_v, z_W = y_v.
R_WORLD_FROM_APPLE_WORLD = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
)


class FrameError(ValueError):
    """A geometric value violates the frame, unit or rotation conventions."""


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _array(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64)
    if array.shape != shape:
        raise FrameError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise FrameError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _unit(unit: str) -> LengthUnit:
    if unit not in LENGTH_UNITS:
        raise FrameError(f"length unit must be one of {LENGTH_UNITS}, got {unit!r}")
    return unit  # type: ignore[return-value]


def validate_rotation(
    matrix: ArrayLike, tolerance: float = ROTATION_TOLERANCE
) -> NDArray[np.float64]:
    """Return ``matrix`` as a 3x3 rotation, or raise if it is not one.

    Rejects non-orthonormal matrices, reflections (det -1) and matrices that
    hide a scale factor.
    """
    rotation = _array(matrix, (3, 3), "rotation")
    error = np.abs(rotation.T @ rotation - np.eye(3)).max()
    if error > tolerance:
        raise FrameError(f"rotation is not orthonormal (max error {error:.2e})")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > tolerance:
        raise FrameError(f"rotation determinant is {determinant:+.6f}, expected +1")
    return rotation


def rotation_from_quaternion_xyzw(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrix from a Hamilton quaternion ``[qx, qy, qz, qw]``.

    The quaternion must already be unit length within tolerance; silently
    normalising a wrong-order or corrupted quaternion would hide the bug.
    """
    q = _array(quaternion, (4,), "quaternion")
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > QUATERNION_TOLERANCE:
        raise FrameError(f"quaternion norm is {norm:.6f}, expected 1")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_xyzw_from_rotation(matrix: ArrayLike) -> NDArray[np.float64]:
    """Canonical Hamilton quaternion ``[qx, qy, qz, qw]`` with ``qw >= 0``."""
    r = validate_rotation(matrix)
    trace = float(np.trace(r))
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
            s / 4,
        ]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [
            s / 4,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[2, 1] - r[1, 2]) / s,
        ]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [
            (r[0, 1] + r[1, 0]) / s,
            s / 4,
            (r[1, 2] + r[2, 1]) / s,
            (r[0, 2] - r[2, 0]) / s,
        ]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            s / 4,
            (r[1, 0] - r[0, 1]) / s,
        ]
    quaternion = np.array(q, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return -quaternion if quaternion[3] < 0 else quaternion


# --------------------------------------------------------------------------
# Framed geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Points:
    """An (N, 3) array of points expressed in one frame and one length unit."""

    xyz: NDArray[np.float64]
    frame: str
    unit: LengthUnit

    def __post_init__(self) -> None:
        xyz = np.array(self.xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise FrameError(f"points must have shape (N, 3), got {xyz.shape}")
        if not np.isfinite(xyz).all():
            raise FrameError("points contain non-finite values")
        xyz.setflags(write=False)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "unit", _unit(self.unit))
        if not self.frame:
            raise FrameError("points need a frame id")


def require_metric(points: Points) -> Points:
    """Gate for measurement code: refuse geometry whose scale is unresolved."""
    if points.unit != "m":
        raise FrameError(
            f"metric operation on {points.unit} geometry in frame {points.frame!r}; "
            "resolve scale with a SimilarityTransform first"
        )
    return points


@dataclass(frozen=True)
class Plane:
    """Plane ``n . p + d = 0`` with unit normal ``n``, in one frame and unit."""

    normal: NDArray[np.float64]
    offset: float
    frame: str
    unit: LengthUnit

    def __post_init__(self) -> None:
        normal = _array(self.normal, (3,), "plane normal")
        norm = float(np.linalg.norm(normal))
        if abs(norm - 1.0) > ROTATION_TOLERANCE:
            raise FrameError(f"plane normal must be unit length, got norm {norm:.6f}")
        if not np.isfinite(self.offset):
            raise FrameError("plane offset must be finite")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "unit", _unit(self.unit))

    def signed_distance(self, points: Points) -> NDArray[np.float64]:
        """Signed distance of each point to the plane, in the plane's unit."""
        _same_space(points.frame, points.unit, self.frame, self.unit, "plane distance")
        return points.xyz @ self.normal + self.offset


def _same_space(
    frame_a: str, unit_a: str, frame_b: str, unit_b: str, operation: str
) -> None:
    if frame_a != frame_b:
        raise FrameError(f"{operation}: frame {frame_a!r} does not match {frame_b!r}")
    if unit_a != unit_b:
        raise FrameError(f"{operation}: unit {unit_a!r} does not match {unit_b!r}")


def _plane_from_covector(covector: ArrayLike, frame: str, unit: LengthUnit) -> Plane:
    covector = np.asarray(covector, dtype=np.float64)
    norm = float(np.linalg.norm(covector[:3]))
    return Plane(covector[:3] / norm, covector[3] / norm, frame, unit)


@dataclass(frozen=True)
class RigidTransform:
    """``T_to_from``: ``p_to = R p_from + t``, preserving the length unit."""

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    to_frame: str
    from_frame: str
    unit: LengthUnit

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", validate_rotation(self.rotation))
        object.__setattr__(
            self, "translation", _array(self.translation, (3,), "translation")
        )
        object.__setattr__(self, "unit", _unit(self.unit))
        if not self.to_frame or not self.from_frame:
            raise FrameError("transforms need destination and source frame ids")

    @classmethod
    def identity(cls, frame: str, unit: LengthUnit) -> RigidTransform:
        return cls(np.eye(3), np.zeros(3), frame, frame, unit)

    @classmethod
    def from_matrix(
        cls, matrix: ArrayLike, to_frame: str, from_frame: str, unit: LengthUnit
    ) -> RigidTransform:
        """Build from a 4x4 homogeneous matrix (row-major, column-vector math)."""
        m = _array(matrix, (4, 4), "transform matrix")
        if np.abs(m[3] - [0.0, 0.0, 0.0, 1.0]).max() > ROTATION_TOLERANCE:
            raise FrameError("last row of a rigid transform must be [0, 0, 0, 1]")
        return cls(m[:3, :3], m[:3, 3], to_frame, from_frame, unit)

    @classmethod
    def from_quaternion_xyzw(
        cls,
        quaternion: ArrayLike,
        translation: ArrayLike,
        to_frame: str,
        from_frame: str,
        unit: LengthUnit,
    ) -> RigidTransform:
        return cls(
            rotation_from_quaternion_xyzw(quaternion),
            np.asarray(translation, dtype=np.float64),
            to_frame,
            from_frame,
            unit,
        )

    def matrix(self) -> NDArray[np.float64]:
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.translation
        return m

    def quaternion_xyzw(self) -> NDArray[np.float64]:
        return quaternion_xyzw_from_rotation(self.rotation)

    def inverse(self) -> RigidTransform:
        r_t = self.rotation.T
        return RigidTransform(
            r_t, -r_t @ self.translation, self.from_frame, self.to_frame, self.unit
        )

    def __matmul__(self, other: RigidTransform) -> RigidTransform:
        """``T_a_from_b @ T_b_from_c -> T_a_from_c``; frames and units must chain."""
        if not isinstance(other, RigidTransform):
            return NotImplemented
        if self.from_frame != other.to_frame:
            raise FrameError(
                f"cannot compose T_{self.to_frame}_from_{self.from_frame} with "
                f"T_{other.to_frame}_from_{other.from_frame}"
            )
        if self.unit != other.unit:
            raise FrameError(f"cannot compose {self.unit} and {other.unit} transforms")
        return RigidTransform(
            self.rotation @ other.rotation,
            self.rotation @ other.translation + self.translation,
            self.to_frame,
            other.from_frame,
            self.unit,
        )

    def apply(self, points: Points) -> Points:
        _same_space(
            points.frame, points.unit, self.from_frame, self.unit, "apply transform"
        )
        return Points(
            points.xyz @ self.rotation.T + self.translation, self.to_frame, self.unit
        )

    def apply_direction(self, vectors: ArrayLike) -> NDArray[np.float64]:
        """Rotate direction vectors (translation does not apply)."""
        return np.asarray(vectors, dtype=np.float64) @ self.rotation.T

    def apply_plane(self, plane: Plane) -> Plane:
        """Transform a plane with the inverse transpose of the point transform."""
        _same_space(
            plane.frame, plane.unit, self.from_frame, self.unit, "apply to plane"
        )
        covector = np.linalg.inv(self.matrix()).T @ np.append(
            plane.normal, plane.offset
        )
        return _plane_from_covector(covector, self.to_frame, self.unit)


@dataclass(frozen=True)
class SimilarityTransform:
    """``p_to = s R p_from + t`` from up-to-scale geometry into metres.

    This is the single place where scale enters. It accepts only
    ``reconstruction_unit`` input and produces metres, so applying it twice,
    or to geometry that is already metric, raises.
    """

    scale: float
    rotation: NDArray[np.float64]
    translation_m: NDArray[np.float64]
    to_frame: str
    from_frame: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise FrameError(f"scale must be finite and positive, got {self.scale}")
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "rotation", validate_rotation(self.rotation))
        object.__setattr__(
            self, "translation_m", _array(self.translation_m, (3,), "translation")
        )

    def matrix(self) -> NDArray[np.float64]:
        m = np.eye(4)
        m[:3, :3] = self.scale * self.rotation
        m[:3, 3] = self.translation_m
        return m

    def apply(self, points: Points) -> Points:
        if points.unit != "reconstruction_unit":
            raise FrameError(
                "similarity transforms take reconstruction_unit geometry; these "
                f"points are already {points.unit} (scale must be applied once)"
            )
        if points.frame != self.from_frame:
            raise FrameError(
                f"points are in {points.frame!r}, expected {self.from_frame!r}"
            )
        xyz = self.scale * points.xyz @ self.rotation.T + self.translation_m
        return Points(xyz, self.to_frame, "m")

    def apply_plane(self, plane: Plane) -> Plane:
        _same_space(
            plane.frame,
            plane.unit,
            self.from_frame,
            "reconstruction_unit",
            "apply to plane",
        )
        covector = np.linalg.inv(self.matrix()).T @ np.append(
            plane.normal, plane.offset
        )
        return _plane_from_covector(covector, self.to_frame, "m")

    def apply_pose(self, pose: RigidTransform) -> RigidTransform:
        """Map an up-to-scale camera pose ``T_from_frame_from_C`` into metres."""
        if pose.unit != "reconstruction_unit" or pose.to_frame != self.from_frame:
            raise FrameError(
                f"pose T_{pose.to_frame}_from_{pose.from_frame} ({pose.unit}) does not "
                f"start in up-to-scale frame {self.from_frame!r}"
            )
        return RigidTransform(
            self.rotation @ pose.rotation,
            self.scale * self.rotation @ pose.translation + self.translation_m,
            self.to_frame,
            pose.from_frame,
            "m",
        )

    def __matmul__(self, other: RigidTransform) -> SimilarityTransform:
        """``S_w_from_u @ T_u_from_v`` (both up-to-scale) -> ``S_w_from_v``."""
        if not isinstance(other, RigidTransform):
            return NotImplemented
        if other.unit != "reconstruction_unit" or other.to_frame != self.from_frame:
            raise FrameError(
                "similarity composes only with up-to-scale transforms into its source"
            )
        return SimilarityTransform(
            self.scale,
            self.rotation @ other.rotation,
            self.scale * self.rotation @ other.translation + self.translation_m,
            self.to_frame,
            other.from_frame,
        )

    def __rmatmul__(self, other: RigidTransform) -> SimilarityTransform:
        """``T_x_from_w @ S_w_from_u`` (metric rigid) -> ``S_x_from_u``."""
        if not isinstance(other, RigidTransform):
            return NotImplemented
        if other.unit != "m" or other.from_frame != self.to_frame:
            raise FrameError(
                "only metric transforms out of the similarity's target compose"
            )
        return SimilarityTransform(
            self.scale,
            other.rotation @ self.rotation,
            other.rotation @ self.translation_m + other.translation,
            other.to_frame,
            self.from_frame,
        )


# --------------------------------------------------------------------------
# Cameras and pixel conventions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PinholeCamera:
    """Undistorted pinhole intrinsics for one image size and orientation.

    Points live in the optical frame (+x right, +y down, +z forward).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame: str = field(default="C")

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise FrameError("image size must be positive")
        values = np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        if not np.isfinite(values).all():
            raise FrameError("intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise FrameError("focal lengths must be positive")

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def project(self, points: Points) -> NDArray[np.float64]:
        """Pixel coordinates (N, 2) of points in this camera's frame."""
        if points.frame != self.frame:
            raise FrameError(
                f"points are in {points.frame!r}, camera is {self.frame!r}"
            )
        z = points.xyz[:, 2]
        if (z <= 0).any():
            raise FrameError("cannot project points at or behind the camera (z <= 0)")
        u = self.fx * points.xyz[:, 0] / z + self.cx
        v = self.fy * points.xyz[:, 1] / z + self.cy
        return np.column_stack([u, v])

    def rays(self, pixels: ArrayLike) -> NDArray[np.float64]:
        """Direction ``[x, y, 1]`` (optical z = 1) through each pixel centre."""
        uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        return np.column_stack(
            [
                (uv[:, 0] - self.cx) / self.fx,
                (uv[:, 1] - self.cy) / self.fy,
                np.ones(len(uv)),
            ]
        )

    def unproject(
        self,
        pixels: ArrayLike,
        depth: ArrayLike,
        depth_kind: DepthKind,
        unit: LengthUnit,
    ) -> Points:
        """Points in the camera frame from pixels and depth.

        ``optical_z`` depth is the z coordinate; ``range`` depth is the
        distance along the ray, which needs the normalised ray.
        """
        rays = self.rays(pixels)
        d = np.asarray(depth, dtype=np.float64).reshape(-1)
        if len(d) != len(rays):
            raise FrameError("one depth value per pixel is required")
        if not np.isfinite(d).all() or (d <= 0).any():
            raise FrameError(
                "depth must be finite and positive; mask invalid pixels first"
            )
        if depth_kind == "optical_z":
            xyz = rays * d[:, None]
        elif depth_kind == "range":
            xyz = rays / np.linalg.norm(rays, axis=1, keepdims=True) * d[:, None]
        else:
            raise FrameError(f"unknown depth kind {depth_kind!r}")
        return Points(xyz, self.frame, unit)

    def resized(self, width: int, height: int) -> PinholeCamera:
        """Intrinsics for the same view resampled to ``width`` x ``height``.

        Uses the half-pixel convention ``c' = s (c + 0.5) - 0.5``.
        """
        sx, sy = width / self.width, height / self.height
        return PinholeCamera(
            width,
            height,
            self.fx * sx,
            self.fy * sy,
            sx * (self.cx + 0.5) - 0.5,
            sy * (self.cy + 0.5) - 0.5,
            self.frame,
        )

    def cropped(self, x0: int, y0: int, width: int, height: int) -> PinholeCamera:
        """Intrinsics for the sub-image whose top-left pixel is ``(x0, y0)``."""
        if x0 < 0 or y0 < 0 or x0 + width > self.width or y0 + height > self.height:
            raise FrameError("crop must lie inside the image")
        return PinholeCamera(
            width, height, self.fx, self.fy, self.cx - x0, self.cy - y0, self.frame
        )

    def rotated_90_clockwise(
        self, new_frame: str
    ) -> tuple[PinholeCamera, RigidTransform]:
        """Intrinsics after rotating the image 90 degrees clockwise.

        Pixels map ``(u, v) -> (height - 1 - v, u)``. Returns the new camera
        and ``T_new_from_old``, the rotation between the two optical frames,
        so poses can be updated with the same physical meaning.
        """
        camera = PinholeCamera(
            self.height,
            self.width,
            self.fy,
            self.fx,
            self.height - 1 - self.cy,
            self.cx,
            new_frame,
        )
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        return camera, RigidTransform(rotation, np.zeros(3), new_frame, self.frame, "m")


def rotate_pixels_90_clockwise(pixels: ArrayLike, height: int) -> NDArray[np.float64]:
    """Pixel coordinates after a 90 degree clockwise image rotation."""
    uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([height - 1 - uv[:, 1], uv[:, 0]])


def resize_pixels(
    pixels: ArrayLike, scale_x: float, scale_y: float
) -> NDArray[np.float64]:
    """Pixel coordinates after resampling, with the half-pixel convention."""
    uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    return np.column_stack(
        [scale_x * (uv[:, 0] + 0.5) - 0.5, scale_y * (uv[:, 1] + 0.5) - 0.5]
    )


# --------------------------------------------------------------------------
# Vendor conventions
# --------------------------------------------------------------------------


def optical_pose_from_apple(
    world_from_apple_camera: RigidTransform,
    world_frame: str,
    camera_frame: str,
    camera_axes: Literal["apple", "optical"] = "apple",
) -> RigidTransform:
    """Convert an ARKit-style pose to ``T_W_from_C`` in FloScan conventions.

    Input: ``T_v_from_A``, a camera in a +y-up (gravity-aligned) vendor world
    v, in metres. With ``camera_axes="apple"`` the camera is the raw Apple
    camera (A: +y up, looking -z); with ``"optical"`` the source has already
    converted it to the optical frame (+y down, looking +z), as the Stray
    Scanner export does, and only the world axes change. Output: the optical
    camera C in the +z-up session world W. Pass the camera convention that
    was verified for the source; applying the camera flip twice turns the
    camera around.
    """
    if world_from_apple_camera.unit != "m":
        raise FrameError("Apple poses are metric; got " + world_from_apple_camera.unit)
    if camera_axes not in ("apple", "optical"):
        raise FrameError(f"unknown camera axes {camera_axes!r}")
    vendor = world_from_apple_camera.to_frame
    source = world_from_apple_camera.from_frame
    w_from_v = RigidTransform(
        R_WORLD_FROM_APPLE_WORLD, np.zeros(3), world_frame, vendor, "m"
    )
    flip = R_APPLE_CAMERA_FROM_OPTICAL if camera_axes == "apple" else np.eye(3)
    source_from_c = RigidTransform(flip, np.zeros(3), source, camera_frame, "m")
    return w_from_v @ world_from_apple_camera @ source_from_c


def pose_from_colmap(
    qvec_wxyz: ArrayLike, tvec: ArrayLike, world_frame: str, camera_frame: str
) -> RigidTransform:
    """Convert a COLMAP image pose to ``T_U_from_C`` (up-to-scale SfM world U).

    COLMAP stores the world-to-camera transform (``cam_from_world``) with a
    scalar-first quaternion ``[qw, qx, qy, qz]``; both are converted here.
    """
    w, x, y, z = _array(qvec_wxyz, (4,), "COLMAP quaternion")
    camera_from_world = RigidTransform.from_quaternion_xyzw(
        [x, y, z, w], tvec, camera_frame, world_frame, "reconstruction_unit"
    )
    return camera_from_world.inverse()


def depth_to_metres(
    raw: ArrayLike, scale_to_m: float
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Convert a raw depth image to float32 metres plus a validity mask.

    Zero and non-finite source values are invalid, never surfaces at the
    camera centre. This is the only place a depth unit conversion happens.
    """
    if not np.isfinite(scale_to_m) or scale_to_m <= 0:
        raise FrameError("scale_to_m must be finite and positive")
    source = np.asarray(raw)
    values = source.astype(np.float64)
    valid = np.isfinite(values) & (values > 0)
    metres = np.where(valid, values * scale_to_m, 0.0).astype(np.float32)
    return metres, valid
