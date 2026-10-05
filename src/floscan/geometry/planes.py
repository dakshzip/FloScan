"""Plane fitting on raw points: least squares, robust refit, uncertainty, sign.

Planes are ``n . p + d = 0`` with unit ``n``. Fits use the eigenvector of the
smallest eigenvalue of the (weighted) scatter matrix. The robust refit is
iteratively reweighted least squares with Tukey's biweight, so points beyond
the cut-off stop influencing the plane instead of dragging it.

The parameter covariance assumes independent residuals. Depth noise is
spatially correlated and poses add correlated error, so it is optimistic;
callers must label it as such and never turn it into a measurement interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# The middle scatter eigenvalue must be at least this share of the largest,
# or the support is (nearly) a line and the plane's tilt about it is unknown.
MIN_SPREAD_RATIO = 1e-4
# Residuals at this share of the coordinates' magnitude are round-off, not
# noise: such data are noise-free and carry no estimable covariance.
NUMERICAL_RESIDUAL = 1e-9


class PlaneFitError(ValueError):
    """The points do not determine a plane."""


@dataclass(frozen=True)
class PlaneFit:
    normal: NDArray[np.float64]
    offset: float
    inliers: NDArray[np.bool_]
    rms: float
    max_abs: float
    eigenvalues: NDArray[np.float64]  # weighted scatter, ascending
    iterations: int
    covariance: NDArray[np.float64] | None
    covariance_status: str

    def distances(self, points: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(points, dtype=np.float64) @ self.normal + self.offset


def _points(points: ArrayLike) -> NDArray[np.float64]:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise PlaneFitError(f"points must have shape (N, 3), got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise PlaneFitError("points contain non-finite values")
    return xyz


def least_squares_plane(
    points: ArrayLike, weights: ArrayLike | None = None
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Total least squares plane: (unit normal, offset, scatter eigenvalues).

    Raises:
        PlaneFitError: fewer than three points with weight, or collinear support.
    """
    xyz = _points(points)
    w = np.ones(len(xyz)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(xyz),) or (w < 0).any() or not np.isfinite(w).all():
        raise PlaneFitError("weights must be finite, non-negative, one per point")
    if (w > 0).sum() < 3:
        raise PlaneFitError("a plane needs at least three weighted points")
    total = w.sum()
    centroid = (w[:, None] * xyz).sum(axis=0) / total
    centred = xyz - centroid
    scatter = (w[:, None] * centred).T @ centred / total
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    if eigenvalues[1] <= MIN_SPREAD_RATIO * max(eigenvalues[2], 1e-300):
        raise PlaneFitError("support is (nearly) collinear; the plane is undetermined")
    normal = eigenvectors[:, 0]
    normal /= np.linalg.norm(normal)
    return normal, float(-normal @ centroid), eigenvalues


def plane_covariance(
    points: NDArray[np.float64],
    weights: NDArray[np.float64],
    normal: NDArray[np.float64],
    sigma2: float,
) -> NDArray[np.float64]:
    """4x4 covariance of (n, d) from independent residuals of variance sigma2.

    The information matrix of the residual ``n . p + d`` is projected onto the
    tangent space of the unit-normal constraint, so the covariance carries no
    variance along the normal's own direction (its length is fixed).
    """
    design = np.column_stack([points, np.ones(len(points))])
    information = (weights[:, None] * design).T @ design / sigma2
    along = np.append(normal, 0.0)
    projector = np.eye(4) - np.outer(along, along)
    covariance = np.linalg.pinv(projector @ information @ projector, rcond=1e-12)
    covariance = projector @ covariance @ projector
    return 0.5 * (covariance + covariance.T)


def robust_refit(
    points: ArrayLike,
    normal: ArrayLike,
    offset: float,
    cutoff: float,
    iterations: int = 10,
    tolerance: float = 1e-9,
) -> PlaneFit:
    """Refine a plane on raw points with Tukey-biweight IRLS.

    Args:
        points: raw candidate support (N, 3).
        normal, offset: the starting plane (for example from RANSAC).
        cutoff: residual (in the points' unit) beyond which a point gets zero
            weight; it is also the inlier threshold of the result.
        iterations: maximum reweighting rounds.
        tolerance: stop when the normal and offset change less than this.

    Raises:
        PlaneFitError: if fewer than three points stay within the cut-off or
            the inliers are collinear.
    """
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise PlaneFitError("cutoff must be positive")
    xyz = _points(points)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    d = float(offset)
    used = 0
    for _ in range(iterations):
        used += 1
        residual = xyz @ n + d
        scaled = residual / cutoff
        weights = np.where(np.abs(scaled) < 1.0, (1.0 - scaled**2) ** 2, 0.0)
        new_n, new_d, _ = least_squares_plane(xyz, weights)
        if new_n @ n < 0:
            new_n, new_d = -new_n, -new_d
        change = max(float(np.abs(new_n - n).max()), abs(new_d - d))
        n, d = new_n, new_d
        if change < tolerance:
            break
    residual = xyz @ n + d
    inliers = np.abs(residual) < cutoff
    if inliers.sum() < 3:
        raise PlaneFitError("fewer than three points within the cut-off")
    support = xyz[inliers]
    n, d, eigenvalues = least_squares_plane(support)
    if n @ np.asarray(normal) < 0:
        n, d = -n, -d
    residual = support @ n + d
    count = len(support)
    scale = float(np.abs(support).max())
    dof = count - 3
    sigma2 = float(residual @ residual) / dof if dof > 0 else 0.0
    if dof <= 0:
        covariance, status = None, "unavailable: too few inliers"
    elif math.sqrt(sigma2) <= NUMERICAL_RESIDUAL * max(1.0, scale):
        covariance, status = None, "unavailable: zero residual (noise-free data)"
    else:
        covariance = plane_covariance(support, np.ones(count), n, sigma2)
        status = "estimated: independent residuals assumed (optimistic)"
    return PlaneFit(
        normal=n,
        offset=float(d),
        inliers=inliers,
        rms=float(np.sqrt(np.mean(residual**2))),
        max_abs=float(np.abs(residual).max()),
        eigenvalues=eigenvalues,
        iterations=used,
        covariance=covariance,
        covariance_status=status,
    )


def orient_towards(
    normal: NDArray[np.float64],
    offset: float,
    points: ArrayLike,
    viewpoints: ArrayLike,
) -> tuple[NDArray[np.float64], float, float]:
    """Flip the plane so its normal points toward the cameras that saw it.

    ``viewpoints[i]`` is the camera centre that observed ``points[i]``. Returns
    the oriented (normal, offset) and the share of observations in front of
    it. A share well below 1 (cameras on both sides) cannot come from one
    opaque surface; callers report it rather than hide it.
    """
    rays = np.asarray(viewpoints, dtype=np.float64) - np.asarray(points, float)
    side = rays @ normal
    agree = float((side > 0).mean())
    if agree < 0.5:
        normal, offset, agree = -normal, -offset, 1.0 - agree
    return normal, offset, agree


def plane_basis(
    normal: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Right-handed in-plane axes (u, v) with u x v = normal.

    u is horizontal (perpendicular to +z) unless the plane is horizontal, in
    which case u is +x projected into the plane.
    """
    n = np.asarray(normal, dtype=np.float64)
    reference = np.array([0.0, 0.0, 1.0])
    u = np.cross(reference, n)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v / np.linalg.norm(v)


def ransac_plane(
    points: ArrayLike,
    threshold: float,
    max_iterations: int,
    rng: np.random.Generator,
    confidence: float = 0.999,
    batch: int = 64,
) -> tuple[NDArray[np.float64], float, NDArray[np.int64]]:
    """Seeded RANSAC plane: (unit normal, offset, inlier indices).

    Deterministic for a given generator state, unlike multithreaded library
    RANSAC, so the same capture and configuration always give the same
    planes. Hypotheses are scored in vectorized batches, and sampling stops
    early once ``confidence`` that an all-inlier sample was drawn is reached
    for the best inlier share found so far.

    Raises:
        PlaneFitError: fewer than three points or no non-degenerate sample.
    """
    xyz = _points(points)
    count = len(xyz)
    if count < 3:
        raise PlaneFitError("RANSAC needs at least three points")
    best_inliers = -1
    best: tuple[NDArray[np.float64], float] | None = None
    needed, done = max_iterations, 0
    while done < min(needed, max_iterations):
        size = min(batch, max_iterations - done)
        sample = rng.integers(0, count, size=(size, 3))
        a, b, c = xyz[sample[:, 0]], xyz[sample[:, 1]], xyz[sample[:, 2]]
        normals = np.cross(b - a, c - a)
        lengths = np.linalg.norm(normals, axis=1)
        usable = lengths > 1e-12
        done += size
        if not usable.any():
            continue
        normals = normals[usable] / lengths[usable, None]
        offsets = -np.einsum("ij,ij->i", normals, a[usable])
        scores = (np.abs(xyz @ normals.T + offsets) < threshold).sum(axis=0)
        winner = int(np.argmax(scores))
        if scores[winner] > best_inliers:
            best_inliers = int(scores[winner])
            best = (normals[winner], float(offsets[winner]))
            share = best_inliers / count
            if share >= 1.0:
                needed = done
            elif share > 0.0:
                needed = math.ceil(
                    math.log(1.0 - confidence) / math.log(1.0 - share**3)
                )
    if best is None:
        raise PlaneFitError("every RANSAC sample was degenerate")
    normal, offset = best
    inliers = np.nonzero(np.abs(xyz @ normal + offset) < threshold)[0]
    return normal, offset, inliers
