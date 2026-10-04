"""Frame, transform, camera and vendor-conversion tests on asymmetric fixtures.

Every fixture is deliberately asymmetric (non-square pixels, off-centre
principal point, unequal room sides, non-axis rotations) so that swapped
axes, sign errors or doubled scale cannot cancel out.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from floscan.geometry.frames import (
    R_APPLE_CAMERA_FROM_OPTICAL,
    R_WORLD_FROM_APPLE_WORLD,
    FrameError,
    PinholeCamera,
    Plane,
    Points,
    RigidTransform,
    SimilarityTransform,
    depth_to_metres,
    optical_pose_from_apple,
    pose_from_colmap,
    quaternion_xyzw_from_rotation,
    require_metric,
    resize_pixels,
    rotate_pixels_90_clockwise,
    rotation_from_quaternion_xyzw,
    validate_rotation,
)

RNG = np.random.default_rng(20261004)


def _rotation(axis: list[float], angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1 - np.cos(angle_rad)) * k @ k


def _random_rotation() -> np.ndarray:
    return _rotation(RNG.normal(size=3).tolist(), float(RNG.uniform(0.1, 3.0)))


CAMERA = PinholeCamera(width=640, height=480, fx=512.0, fy=498.5, cx=301.3, cy=247.9)

# An asymmetric room-sized box (metres) in room frame R.
BOX_SIZE = np.array([3.1, 4.7, 2.45])
BOX_R = Points(
    np.array(list(itertools.product(*[(0.0, s) for s in BOX_SIZE]))), "R", "m"
)


def _pairwise(points: Points) -> np.ndarray:
    xyz = points.xyz
    return np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)


# --------------------------------------------------------------------------
# Rotations and quaternions
# --------------------------------------------------------------------------


def test_quaternion_order_is_xyzw_hamilton() -> None:
    # 90 degrees about +z maps +x to +y.
    q = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    r = rotation_from_quaternion_xyzw(q)
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(quaternion_xyzw_from_rotation(r), q, atol=1e-12)
    # The same numbers read as scalar-first [w, x, y, z] mean a different rotation.
    wxyz_misread = rotation_from_quaternion_xyzw([q[3], q[0], q[1], q[2]])
    assert not np.allclose(wxyz_misread, r)


def test_quaternion_round_trip_is_canonical() -> None:
    for _ in range(50):
        r = _random_rotation()
        q = quaternion_xyzw_from_rotation(r)
        assert q[3] >= 0
        np.testing.assert_allclose(rotation_from_quaternion_xyzw(q), r, atol=1e-12)
        np.testing.assert_allclose(rotation_from_quaternion_xyzw(-q), r, atol=1e-12)


def test_non_unit_quaternion_is_rejected_not_normalised() -> None:
    with pytest.raises(FrameError, match="norm"):
        rotation_from_quaternion_xyzw([0.0, 0.0, 0.0, 2.0])


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.diag([1.0, 1.0, -1.0]), "determinant"),
        (1.01 * np.eye(3), "orthonormal"),
        (np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "orthonormal"),
        (np.full((3, 3), np.nan), "non-finite"),
    ],
    ids=["reflection", "hidden-scale", "shear", "nan"],
)
def test_invalid_rotations_are_rejected(matrix: np.ndarray, message: str) -> None:
    with pytest.raises(FrameError, match=message):
        validate_rotation(matrix)


# --------------------------------------------------------------------------
# Rigid transforms
# --------------------------------------------------------------------------


def _transform(to: str, frm: str, unit: str = "m") -> RigidTransform:
    return RigidTransform(_random_rotation(), RNG.normal(size=3) * 3, to, frm, unit)


def test_inverse_and_composition() -> None:
    t_ab, t_bc = _transform("A", "B"), _transform("B", "C")
    points_c = Points(RNG.normal(size=(20, 3)), "C", "m")
    composed = (t_ab @ t_bc).apply(points_c)
    sequential = t_ab.apply(t_bc.apply(points_c))
    np.testing.assert_allclose(composed.xyz, sequential.xyz, atol=1e-12)
    assert composed.frame == "A"
    identity = t_ab @ t_ab.inverse()
    np.testing.assert_allclose(identity.matrix(), np.eye(4), atol=1e-12)
    assert (identity.to_frame, identity.from_frame) == ("A", "A")
    np.testing.assert_allclose(
        t_ab.inverse().apply(t_ab.apply(Points(points_c.xyz, "B", "m"))).xyz,
        points_c.xyz,
        atol=1e-12,
    )


def test_composition_refuses_mismatched_frames_and_units() -> None:
    with pytest.raises(FrameError, match="cannot compose"):
        _ = _transform("A", "B") @ _transform("C", "D")
    with pytest.raises(FrameError, match="cannot compose m and reconstruction_unit"):
        _ = _transform("A", "B") @ _transform("B", "C", "reconstruction_unit")
    with pytest.raises(FrameError, match="frame"):
        _transform("A", "B").apply(Points(np.zeros((1, 3)), "C", "m"))


def test_matrix_round_trip_and_bad_last_row() -> None:
    t = _transform("A", "B")
    back = RigidTransform.from_matrix(t.matrix(), "A", "B", "m")
    np.testing.assert_allclose(back.matrix(), t.matrix(), atol=1e-12)
    bad = t.matrix()
    bad[3] = [0.0, 0.0, 0.1, 1.0]
    with pytest.raises(FrameError, match="last row"):
        RigidTransform.from_matrix(bad, "A", "B", "m")


# --------------------------------------------------------------------------
# Planes: inverse transpose
# --------------------------------------------------------------------------


def _plane_through(points: np.ndarray, frame: str, unit: str) -> Plane:
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal /= np.linalg.norm(normal)
    return Plane(normal, float(-normal @ points[0]), frame, unit)


def test_plane_transform_uses_inverse_transpose() -> None:
    support = RNG.normal(size=(3, 3)) * 2
    plane = _plane_through(support, "B", "m")
    t = _transform("A", "B")
    moved = t.apply_plane(plane)
    # Points on the plane stay on the transformed plane.
    on_plane = t.apply(Points(support, "B", "m"))
    np.testing.assert_allclose(moved.signed_distance(on_plane), 0.0, atol=1e-10)
    # Closed form: n' = R n, d' = d - n'.t
    np.testing.assert_allclose(moved.normal, t.rotation @ plane.normal, atol=1e-12)
    assert moved.offset == pytest.approx(plane.offset - moved.normal @ t.translation)
    # A naive (non-inverse-transpose) covector transform is wrong.
    naive = t.matrix() @ np.append(plane.normal, plane.offset)
    assert not np.allclose(naive[:3] / np.linalg.norm(naive[:3]), moved.normal)


def test_similarity_plane_transform_scales_offset_once() -> None:
    support = RNG.normal(size=(3, 3))
    plane = _plane_through(support, "U", "reconstruction_unit")
    s = SimilarityTransform(2.5, _random_rotation(), RNG.normal(size=3), "W", "U")
    moved = s.apply_plane(plane)
    assert moved.unit == "m"
    on_plane = s.apply(Points(support, "U", "reconstruction_unit"))
    np.testing.assert_allclose(moved.signed_distance(on_plane), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(moved.normal), 1.0)


# --------------------------------------------------------------------------
# Scale exactly once
# --------------------------------------------------------------------------


def test_similarity_applies_scale_exactly_once() -> None:
    s = SimilarityTransform(1.7, _random_rotation(), [0.4, -1.0, 2.0], "W", "U")
    points_u = Points(RNG.normal(size=(5, 3)), "U", "reconstruction_unit")
    metric = s.apply(points_u)
    assert metric.unit == "m"
    np.testing.assert_allclose(_pairwise(metric), 1.7 * _pairwise(points_u), atol=1e-12)
    with pytest.raises(FrameError, match="scale must be applied once"):
        s.apply(Points(metric.xyz, "U", "m"))
    with pytest.raises(FrameError, match="already m"):
        s.apply(metric)


def test_metric_gate_rejects_up_to_scale_geometry() -> None:
    with pytest.raises(FrameError, match="resolve scale"):
        require_metric(Points(np.zeros((1, 3)), "U", "reconstruction_unit"))
    assert require_metric(Points(np.zeros((1, 3)), "W", "m")).unit == "m"


def test_similarity_composition_rules() -> None:
    s = SimilarityTransform(0.8, _random_rotation(), [1.0, 2.0, 3.0], "W", "U")
    t_u_v = _transform("U", "V", "reconstruction_unit")
    t_x_w = _transform("X", "W", "m")
    points_v = Points(RNG.normal(size=(4, 3)), "V", "reconstruction_unit")
    np.testing.assert_allclose(
        (s @ t_u_v).apply(points_v).xyz, s.apply(t_u_v.apply(points_v)).xyz, atol=1e-12
    )
    s_x_u = t_x_w @ s
    points_u = Points(RNG.normal(size=(4, 3)), "U", "reconstruction_unit")
    np.testing.assert_allclose(
        s_x_u.apply(points_u).xyz, t_x_w.apply(s.apply(points_u)).xyz, atol=1e-12
    )
    with pytest.raises(FrameError):
        _ = s @ _transform("U", "V", "m")
    with pytest.raises(FrameError):
        _ = _transform("X", "W", "reconstruction_unit") @ s
    for bad_scale in (0.0, -1.0, float("nan")):
        with pytest.raises(FrameError, match="scale"):
            SimilarityTransform(bad_scale, np.eye(3), np.zeros(3), "W", "U")


# --------------------------------------------------------------------------
# Cameras: unproject, reproject, resize, crop, rotate
# --------------------------------------------------------------------------


def _pixels(n: int = 40) -> np.ndarray:
    return np.column_stack(
        [RNG.uniform(0, CAMERA.width - 1, n), RNG.uniform(0, CAMERA.height - 1, n)]
    )


@pytest.mark.parametrize("depth_kind", ["optical_z", "range"])
def test_unproject_then_project_returns_pixels(depth_kind: str) -> None:
    pixels = _pixels()
    depth = RNG.uniform(0.5, 9.0, len(pixels))
    points = CAMERA.unproject(pixels, depth, depth_kind, "m")
    np.testing.assert_allclose(CAMERA.project(points), pixels, atol=1e-9)
    if depth_kind == "range":
        np.testing.assert_allclose(
            np.linalg.norm(points.xyz, axis=1), depth, atol=1e-12
        )
    else:
        np.testing.assert_allclose(points.xyz[:, 2], depth, atol=1e-12)


def test_range_and_optical_depth_differ_off_axis() -> None:
    corner = np.array([[0.0, 0.0]])
    z = CAMERA.unproject(corner, [4.0], "optical_z", "m").xyz
    r = CAMERA.unproject(corner, [4.0], "range", "m").xyz
    assert np.linalg.norm(z) > np.linalg.norm(r) == pytest.approx(4.0)


def test_unproject_rejects_invalid_depth() -> None:
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(FrameError, match="depth"):
            CAMERA.unproject([[10.0, 10.0]], [bad], "optical_z", "m")


def test_projection_behind_camera_is_refused() -> None:
    with pytest.raises(FrameError, match="behind"):
        CAMERA.project(Points([[0.0, 0.0, -1.0]], "C", "m"))


def _scene_points() -> Points:
    return CAMERA.unproject(_pixels(), RNG.uniform(1.0, 6.0, 40), "optical_z", "m")


def test_resize_uses_half_pixel_convention() -> None:
    points = _scene_points()
    small = CAMERA.resized(256, 192)
    expected = resize_pixels(CAMERA.project(points), 256 / 640, 192 / 480)
    np.testing.assert_allclose(small.project(points), expected, atol=1e-9)
    assert small.cx == pytest.approx(0.4 * (301.3 + 0.5) - 0.5)
    # A naive c' = s c is off by (s - 1) / 2 pixels.
    assert abs(small.cx - 0.4 * 301.3) == pytest.approx(0.3)


def test_crop_shifts_principal_point() -> None:
    points = _scene_points()
    crop = CAMERA.cropped(40, 25, 500, 400)
    np.testing.assert_allclose(
        crop.project(points), CAMERA.project(points) - [40, 25], atol=1e-9
    )
    with pytest.raises(FrameError, match="inside"):
        CAMERA.cropped(200, 0, 500, 400)


def test_rotate_90_updates_intrinsics_and_camera_frame() -> None:
    points = _scene_points()
    rotated, t_new_from_old = CAMERA.rotated_90_clockwise("C_rot")
    assert (rotated.width, rotated.height) == (480, 640)
    expected = rotate_pixels_90_clockwise(CAMERA.project(points), CAMERA.height)
    np.testing.assert_allclose(
        rotated.project(t_new_from_old.apply(points)), expected, atol=1e-9
    )
    assert np.linalg.det(t_new_from_old.rotation) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Vendor conventions and handedness
# --------------------------------------------------------------------------


def test_vendor_basis_changes_are_proper_rotations() -> None:
    for matrix in (R_APPLE_CAMERA_FROM_OPTICAL, R_WORLD_FROM_APPLE_WORLD):
        assert np.linalg.det(matrix) == pytest.approx(1.0)
        validate_rotation(matrix)
    # Right-handed: x cross y = z in every frame.
    for matrix in (R_APPLE_CAMERA_FROM_OPTICAL, R_WORLD_FROM_APPLE_WORLD):
        x, y, z = matrix.T
        np.testing.assert_allclose(np.cross(x, y), z, atol=1e-12)


def test_apple_camera_axes_map_to_optical_and_gravity_up() -> None:
    # A level Apple camera at the vendor origin with identity pose looks along
    # vendor -z, with image-up along vendor +y (gravity up).
    apple_pose = RigidTransform(np.eye(3), np.zeros(3), "arkit_world", "A", "m")
    t_w_c = optical_pose_from_apple(apple_pose, "W", "C")
    forward_w = t_w_c.apply_direction([0.0, 0.0, 1.0])  # optical +z
    down_w = t_w_c.apply_direction([0.0, 1.0, 0.0])  # optical +y (image down)
    np.testing.assert_allclose(
        forward_w, R_WORLD_FROM_APPLE_WORLD @ [0, 0, -1], atol=1e-12
    )
    np.testing.assert_allclose(down_w, [0.0, 0.0, -1.0], atol=1e-12)  # down is -z_W
    assert (t_w_c.to_frame, t_w_c.from_frame) == ("W", "C")


def test_colmap_pose_is_inverted_and_reordered() -> None:
    t_u_c = _transform("U", "C", "reconstruction_unit")
    c_from_u = t_u_c.inverse()
    x, y, z, w = c_from_u.quaternion_xyzw()
    recovered = pose_from_colmap([w, x, y, z], c_from_u.translation, "U", "C")
    np.testing.assert_allclose(recovered.matrix(), t_u_c.matrix(), atol=1e-12)
    assert recovered.unit == "reconstruction_unit"
    # Forgetting COLMAP's scalar-first order gives a different pose.
    misread = pose_from_colmap([x, y, z, w], c_from_u.translation, "U", "C")
    assert not np.allclose(misread.matrix(), t_u_c.matrix(), atol=1e-6)


def test_depth_conversion_marks_zero_invalid() -> None:
    raw = np.array([[0, 1500], [65535, 2]], dtype=np.uint16)
    metres, valid = depth_to_metres(raw, 0.001)
    assert metres.dtype == np.float32
    np.testing.assert_array_equal(valid, [[False, True], [True, True]])
    assert metres[0, 1] == pytest.approx(1.5)
    assert metres[0, 0] == 0.0


# --------------------------------------------------------------------------
# Acceptance: a metric box survives the full frame chain
# --------------------------------------------------------------------------


def _metric_chain(
    apple_from_optical: np.ndarray = R_APPLE_CAMERA_FROM_OPTICAL,
) -> Points:
    """Box in R -> W -> Apple vendor world/camera -> optical C -> pixels and back."""
    t_w_r = RigidTransform(_rotation([0, 0, 1], 0.37), [1.2, -0.8, 0.0], "W", "R", "m")
    # The phone stands at the room's far corner, 1.5 m up, looking at the box
    # centre. The pose is built in the Apple convention (+y up, looking -z).
    target_w = t_w_r.apply(Points(BOX_SIZE[None, :] / 2, "R", "m")).xyz[0]
    eye_w = t_w_r.apply(
        Points([[BOX_SIZE[0] + 4.5, BOX_SIZE[1] + 5.0, 1.5]], "R", "m")
    ).xyz[0]
    forward = (target_w - eye_w) / np.linalg.norm(target_w - eye_w)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    r_w_a = np.column_stack([right, up, -forward])  # Apple camera axes in W
    r_v_w = R_WORLD_FROM_APPLE_WORLD.T
    apple_pose = RigidTransform(r_v_w @ r_w_a, r_v_w @ eye_w, "arkit_world", "A", "m")

    # The converter under test, with an optional corrupted camera flip.
    t_w_v = RigidTransform(
        R_WORLD_FROM_APPLE_WORLD, np.zeros(3), "W", "arkit_world", "m"
    )
    t_a_c = RigidTransform(apple_from_optical, np.zeros(3), "A", "C", "m")
    t_w_c = t_w_v @ apple_pose @ t_a_c
    if apple_from_optical is R_APPLE_CAMERA_FROM_OPTICAL:
        # The uncorrupted chain must equal the production converter.
        reference = optical_pose_from_apple(apple_pose, "W", "C")
        np.testing.assert_allclose(t_w_c.matrix(), reference.matrix(), atol=1e-12)

    box_c = (t_w_c.inverse() @ t_w_r).apply(BOX_R)
    pixels = CAMERA.project(box_c)
    back_c = CAMERA.unproject(pixels, box_c.xyz[:, 2], "optical_z", "m")
    return (t_w_r.inverse() @ t_w_c).apply(back_c)


def test_metric_box_is_identical_after_full_frame_chain() -> None:
    recovered = _metric_chain()
    assert (recovered.frame, recovered.unit) == ("R", "m")
    np.testing.assert_allclose(recovered.xyz, BOX_R.xyz, atol=1e-9)
    np.testing.assert_allclose(_pairwise(recovered), _pairwise(BOX_R), atol=1e-9)


def test_up_to_scale_box_is_metric_after_one_similarity() -> None:
    # An SfM-like world U: arbitrary rotation, origin and scale (0.37 units/m).
    s_true = SimilarityTransform(
        1 / 0.37, _random_rotation(), [3.0, -2.0, 0.5], "R", "U"
    )
    box_u = Points(
        (BOX_R.xyz - s_true.translation_m) @ s_true.rotation / s_true.scale,
        "U",
        "reconstruction_unit",
    )
    recovered = s_true.apply(box_u)
    np.testing.assert_allclose(recovered.xyz, BOX_R.xyz, atol=1e-9)
    edges = np.linalg.norm(np.diff(recovered.xyz[[0, 4]], axis=0))
    assert edges == pytest.approx(BOX_SIZE[0])


def test_corrupted_camera_flip_is_detected() -> None:
    # Forgetting the Apple y/z flip puts the box behind the camera.
    with pytest.raises(FrameError, match="behind"):
        _metric_chain(apple_from_optical=np.eye(3))


def test_mirrored_basis_is_rejected() -> None:
    with pytest.raises(FrameError, match="determinant"):
        _metric_chain(apple_from_optical=np.diag([1.0, -1.0, 1.0]))


def test_corrupted_scale_is_detected() -> None:
    s = SimilarityTransform(1 / 0.37, np.eye(3), np.zeros(3), "R", "U")
    box_u = Points(BOX_R.xyz * 0.37, "U", "reconstruction_unit")
    wrong = SimilarityTransform(1.01 / 0.37, np.eye(3), np.zeros(3), "R", "U")
    assert np.abs(_pairwise(wrong.apply(box_u)) - _pairwise(BOX_R)).max() > 0.04
    np.testing.assert_allclose(_pairwise(s.apply(box_u)), _pairwise(BOX_R), atol=1e-9)
    with pytest.raises(FrameError, match="already m"):
        s.apply(s.apply(box_u))
