"""Geometric records: frames, transforms, cameras, poses, planes, rooms, graph.

Validators encode the invariants of 02-data-contracts.md that a JSON Schema
cannot: proper rotations, unit normals, right-handed surface bases, polygon
orientation and simplicity, unknown covariance staying unknown, and placement
without arbitrary origins.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from shapely.geometry import Polygon

from floscan.contracts.base import (
    AssetRef,
    Contract,
    Id,
    LengthUnit,
    Matrix3,
    Matrix4,
    Matrix6,
    NonNegative,
    Positive,
    Probability,
    Record,
    Text,
    Tier,
    Vec2,
    Vec3,
    Vec4,
    check_rotation,
    check_symmetric_covariance,
    check_unit_vector,
)
from floscan.geometry.frames import RigidTransform

FrameKind = Literal[
    "optical_camera",
    "apple_camera",
    "session_world",
    "sfm_world",
    "room",
    "property",
    "surface",
]


class CoordinateFrame(Contract):
    """A named frame. Different sessions never share a frame implicitly."""

    id: Id
    kind: FrameKind
    unit: LengthUnit
    description: Text


class RigidTransformRecord(Contract):
    """``T_to_from`` as a row-major 4x4 matrix (column-vector convention)."""

    to_frame_id: Id
    from_frame_id: Id
    matrix: Matrix4
    unit: LengthUnit

    @model_validator(mode="after")
    def _proper_rigid(self) -> RigidTransformRecord:
        m = np.asarray(self.matrix, dtype=np.float64)
        if np.abs(m[3] - [0.0, 0.0, 0.0, 1.0]).max() > 1e-9:
            raise ValueError("last row of a rigid transform must be [0, 0, 0, 1]")
        check_rotation(m[:3, :3].tolist(), "transform rotation")
        return self

    def to_math(self) -> RigidTransform:
        return RigidTransform.from_matrix(
            self.matrix, self.to_frame_id, self.from_frame_id, self.unit
        )


class PixelTransform(Contract):
    """Affine map from source pixels to stored pixels (3x3, last row 0 0 1)."""

    kind: Literal["identity", "rotation", "crop", "resize", "composite"]
    matrix: Matrix3

    @model_validator(mode="after")
    def _affine(self) -> PixelTransform:
        if np.abs(np.asarray(self.matrix[2]) - [0.0, 0.0, 1.0]).max() > 1e-12:
            raise ValueError("pixel transform last row must be [0, 0, 1]")
        if abs(np.linalg.det(np.asarray(self.matrix))) < 1e-12:
            raise ValueError("pixel transform must be invertible")
        return self


def _signed_area(ring: list[list[float]]) -> float:
    xy = np.asarray(ring, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class Polygon2D(Contract):
    """Closed simple polygon: CCW outer ring, CW holes, metres in its frame."""

    outer: list[Vec2] = Field(min_length=3)
    holes: list[list[Vec2]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _simple_and_oriented(self) -> Polygon2D:
        if _signed_area(self.outer) <= 0:
            raise ValueError("outer ring must be counter-clockwise with positive area")
        for hole in self.holes:
            if len(hole) < 3 or _signed_area(hole) >= 0:
                raise ValueError("holes must be clockwise rings of 3 or more points")
        if not Polygon(self.outer, self.holes).is_valid:
            raise ValueError("polygon is not simple (self-intersecting or bad holes)")
        return self

    def area(self) -> float:
        return float(Polygon(self.outer, self.holes).area)


class Distortion(Contract):
    model: Literal["none", "opencv", "opencv_fisheye", "simple_radial"]
    coefficients: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coefficients(self) -> Distortion:
        if self.model == "none" and self.coefficients:
            raise ValueError("distortion model 'none' takes no coefficients")
        if self.model != "none" and not self.coefficients:
            raise ValueError(f"distortion model {self.model!r} needs coefficients")
        return self


class Camera(Record):
    """Intrinsics valid for exactly one image resolution and orientation."""

    sensor_id: Id
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    model: Literal["pinhole"]
    fx: Positive
    fy: Positive
    cx: float
    cy: float
    distortion: Distortion
    intrinsics_origin: Literal["measured", "metadata", "estimated"]
    intrinsics_covariance: list[list[float]] | None = None
    T_rgb_from_depth: RigidTransformRecord | None = None
    calibration_id: Id | None = None
    pixel_transform: PixelTransform

    @model_validator(mode="after")
    def _principal_point(self) -> Camera:
        if not (-0.5 <= self.cx <= self.width_px - 0.5):
            raise ValueError("cx lies outside the image")
        if not (-0.5 <= self.cy <= self.height_px - 0.5):
            raise ValueError("cy lies outside the image")
        if self.intrinsics_covariance is not None:
            check_symmetric_covariance(
                self.intrinsics_covariance, "intrinsics_covariance (fx, fy, cx, cy)", 4
            )
        if self.T_rgb_from_depth is not None and self.T_rgb_from_depth.unit != "m":
            raise ValueError("depth-to-RGB extrinsics must be metric")
        return self


class Pose(Record):
    """``T_to_from`` with a canonical scalar-last quaternion. SE(3), not Sim(3)."""

    to_frame_id: Id
    from_frame_id: Id
    timestamp_s: float | None = None
    rotation_xyzw: Vec4
    translation: Vec3
    translation_unit: LengthUnit
    covariance: Matrix6 | None = None
    covariance_status: Literal["known", "unknown"]
    method: Text
    tracking_state: Literal["normal", "limited", "not_available", "not_applicable"]
    observation_ids: list[Id] = Field(default_factory=list)

    @model_validator(mode="after")
    def _quaternion_and_covariance(self) -> Pose:
        check_unit_vector(self.rotation_xyzw, "rotation_xyzw")
        if self.rotation_xyzw[3] < 0:
            raise ValueError("rotation_xyzw must be canonical (qw >= 0)")
        if self.covariance_status == "unknown" and self.covariance is not None:
            raise ValueError("covariance_status 'unknown' must not carry a covariance")
        if self.covariance_status == "known":
            if self.covariance is None:
                raise ValueError("covariance_status 'known' needs a 6x6 covariance")
            check_symmetric_covariance(self.covariance, "pose covariance")
        return self

    def to_math(self) -> RigidTransform:
        return RigidTransform.from_quaternion_xyzw(
            self.rotation_xyzw,
            self.translation,
            self.to_frame_id,
            self.from_frame_id,
            self.translation_unit,
        )


class DepthFrame(Record):
    """Canonical float32 metres plus validity mask; the source unit is recorded."""

    frame_id: Id
    camera_id: Id
    timestamp_s: float
    depth: AssetRef
    depth_kind: Literal["optical_z", "range"]
    valid_mask: AssetRef
    confidence: AssetRef | None = None
    confidence_encoding: Text | None = None
    source_unit: Literal["mm", "m"]
    scale_to_m: Positive
    alignment: Literal["aligned_rgb", "separate_camera"]
    sync_residual_s: NonNegative
    noise_model_id: Id | None = None

    @model_validator(mode="after")
    def _units_and_encoding(self) -> DepthFrame:
        expected = {"mm": 0.001, "m": 1.0}[self.source_unit]
        if abs(self.scale_to_m - expected) > 1e-12:
            raise ValueError(
                f"source unit {self.source_unit} needs scale_to_m {expected}, "
                f"got {self.scale_to_m}"
            )
        if self.depth.dtype != "<f4":
            raise ValueError("canonical depth must be little-endian float32 metres")
        if (self.confidence is None) != (self.confidence_encoding is None):
            raise ValueError("confidence and confidence_encoding go together")
        return self


class PointCloud(Record):
    frame_id: Id
    xyz: AssetRef
    unit: LengthUnit
    rgb: AssetRef | None = None
    normals: AssetRef | None = None
    observation_index: AssetRef
    weights: AssetRef | None = None
    voxel_size_m: Positive | None = None
    submap_id: Id
    scale_status: Literal["metric", "up_to_scale"]

    @model_validator(mode="after")
    def _aligned_channels(self) -> PointCloud:
        if (self.unit == "m") != (self.scale_status == "metric"):
            raise ValueError("unit and scale_status disagree")
        if not self.xyz.shape or len(self.xyz.shape) != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz must be an [N, 3] array")
        count = self.xyz.shape[0]
        for name in ("rgb", "normals"):
            channel = getattr(self, name)
            if channel is not None and (
                not channel.shape or channel.shape != [count, 3]
            ):
                raise ValueError(f"{name} must be [{count}, 3] to align with xyz")
        for name in ("observation_index", "weights"):
            channel = getattr(self, name)
            if channel is not None and (not channel.shape or channel.shape[0] != count):
                raise ValueError(f"{name} must have {count} rows to align with xyz")
        return self


class ResidualStats(Contract):
    rms: NonNegative
    max: NonNegative
    count: int = Field(ge=3)


def _check_basis(u: list[float], v: list[float], n: list[float], where: str) -> None:
    for name, vector in (("basis_u", u), ("basis_v", v), ("normal", n)):
        check_unit_vector(vector, f"{where}.{name}")
    if abs(float(np.dot(u, v))) > 1e-6:
        raise ValueError(f"{where}: basis_u and basis_v must be orthogonal")
    if np.abs(np.cross(u, v) - np.asarray(n)).max() > 1e-6:
        raise ValueError(f"{where}: basis must be right-handed (u x v = normal)")


class Plane(Record):
    """``normal . p + offset = 0`` in ``frame_id``; offset in ``unit``."""

    frame_id: Id
    normal: Vec3
    offset: float
    unit: LengthUnit
    support: AssetRef | None = None
    boundary: Polygon2D | None = None
    basis_u: Vec3
    basis_v: Vec3
    parameter_covariance: list[list[float]] | None = None
    residual_stats: ResidualStats
    kind: Literal["floor", "ceiling", "wall", "unknown"]
    observability: Literal["full", "normal_only", "partial"]
    orientation_is_prior: bool

    @model_validator(mode="after")
    def _geometry(self) -> Plane:
        _check_basis(self.basis_u, self.basis_v, self.normal, "plane")
        if self.parameter_covariance is not None:
            check_symmetric_covariance(
                self.parameter_covariance, "plane parameter_covariance (n, d)", 4
            )
        return self


class Material(Contract):
    label: Text
    probability: Probability | None = None
    source: Text


class Surface(Record):
    """A planar surface patch with a metric (u, v) chart; versions track edits."""

    room_id: Id
    kind: Literal["wall", "floor", "ceiling", "other"]
    geometry_version: int = Field(ge=1)
    frame_id: Id
    plane_id: Id | None = None
    mesh: AssetRef | None = None
    origin: Vec3
    basis_u: Vec3
    basis_v: Vec3
    normal: Vec3
    boundary_uv: Polygon2D
    material: Material
    observation_ids: list[Id] = Field(default_factory=list)
    visibility: AssetRef | None = None
    measurement_ids: list[Id] = Field(default_factory=list)
    lineage: list[Id] = Field(default_factory=list)

    @model_validator(mode="after")
    def _geometry(self) -> Surface:
        _check_basis(self.basis_u, self.basis_v, self.normal, "surface")
        return self


class HeightSample(Contract):
    s_m: NonNegative
    height_m: Positive


class Wall(Record):
    """Interior finished face of one wall, as seen from its room."""

    room_id: Id
    surface_id: Id
    baseline: list[Vec3] = Field(min_length=2, max_length=2)
    plane_id: Id | None = None
    opening_ids: list[Id] = Field(default_factory=list)
    opposite_surface_id: Id | None = None
    shared_partition_id: Id | None = None
    height_profile: list[HeightSample] = Field(default_factory=list)
    thickness_measurement_id: Id | None = None
    measurement_ids: list[Id] = Field(default_factory=list)
    boundary_evidence: list[Id] = Field(default_factory=list)

    @model_validator(mode="after")
    def _baseline(self) -> Wall:
        start, end = (np.asarray(point) for point in self.baseline)
        if np.linalg.norm(end - start) <= 1e-6:
            raise ValueError("wall baseline must have nonzero length")
        if self.opposite_surface_id == self.surface_id:
            raise ValueError("the opposite side of a partition is a distinct surface")
        return self


class Opening(Record):
    """Opening on a host wall, dimensioned by one aperture definition only."""

    room_id: Id
    host_wall_id: Id
    surface_id: Id
    opening_class: Literal["door", "window", "passage", "unknown"]
    dimension_definition: Literal["clear_aperture", "outer_trim"]
    polygon_uv: Polygon2D
    jamb_observation_ids: list[Id] = Field(default_factory=list)
    width_measurement_id: Id
    height_measurement_id: Id | None = None
    sill_measurement_id: Id | None = None
    connector_id: Id | None = None
    detection_score: Probability
    score_calibration_id: Id | None = None
    visibility_state: Literal["visible", "partial", "occluded", "inferred"]

    @model_validator(mode="after")
    def _window_not_connector(self) -> Opening:
        if self.opening_class == "window" and self.connector_id is not None:
            raise ValueError("a window does not imply traversable adjacency")
        return self


class Room(Record):
    """A room in its local frame, placed in the property frame only with evidence."""

    label: Text
    level_id: Id
    local_frame_id: Id
    T_property_from_room: RigidTransformRecord | None = None
    boundary: Polygon2D | None = None
    wall_ids: list[Id] = Field(default_factory=list)
    floor_surface_ids: list[Id] = Field(default_factory=list)
    ceiling_surface_ids: list[Id] = Field(default_factory=list)
    opening_ids: list[Id] = Field(default_factory=list)
    measurement_ids: list[Id] = Field(default_factory=list)
    damage_ids: list[Id] = Field(default_factory=list)
    coverage: Probability | None = None
    hypothesis_id: Id
    placement_status: Literal["placed", "unplaced", "ambiguous"]

    @model_validator(mode="after")
    def _placement(self) -> Room:
        if self.placement_status == "placed":
            if self.T_property_from_room is None:
                raise ValueError("a placed room needs T_property_from_room")
            if self.T_property_from_room.from_frame_id != self.local_frame_id:
                raise ValueError("T_property_from_room must start in the room's frame")
            if self.T_property_from_room.unit != "m":
                raise ValueError("room placement must be metric")
        elif self.T_property_from_room is not None:
            raise ValueError(
                "an unplaced room must not carry a transform (no arbitrary origin)"
            )
        if self.status == "ok" and self.boundary is None:
            raise ValueError("a complete room needs a closed boundary")
        return self


class ScaleEvidence(Contract):
    id: Id
    type: Literal[
        "sensor_depth", "known_reference", "learned_metric_depth", "vio", "object_prior"
    ]
    observation: float
    unit: Literal["m", "ratio"]
    uncertainty: NonNegative | None = None
    source_ids: list[Id] = Field(min_length=1)
    permitted_tiers: list[Tier] = Field(min_length=1)
    likelihood_assumptions: Text
    role: Literal["prior", "measurement"]

    @model_validator(mode="after")
    def _learned_is_prior(self) -> ScaleEvidence:
        if (
            self.type in ("learned_metric_depth", "object_prior")
            and self.role != "prior"
        ):
            raise ValueError(
                f"{self.type} scale evidence is a prior, not a measurement"
            )
        return self


class ScaleEstimate(Contract):
    id: Id
    component_id: Id
    multiplier: Positive | None
    log_scale_samples: AssetRef | None = None
    evidence_ids: list[Id] = Field(default_factory=list)
    status: Literal["sensor_metric", "anchored_metric", "prior_metric", "unresolved"]

    @model_validator(mode="after")
    def _resolved(self) -> ScaleEstimate:
        if (self.status == "unresolved") != (self.multiplier is None):
            raise ValueError("only an unresolved scale has no multiplier")
        if self.status != "unresolved" and not self.evidence_ids:
            raise ValueError("a resolved scale needs evidence")
        return self


class RobustKernel(Contract):
    name: Literal["none", "huber", "cauchy", "tukey"]
    scale: Positive | None = None


class Constraint(Contract):
    """One factor between two graph nodes, accepted or rejected with a reason."""

    id: Id
    node_ids: list[Id] = Field(min_length=2, max_length=2)
    type: Literal[
        "odometry", "visual_match", "plane", "portal", "loop", "scale", "gravity"
    ]
    relative_transform: RigidTransformRecord | None = None
    residual_unit: Text
    information: Matrix6 | None = None
    observability_rank: int = Field(ge=0, le=7)
    robust_kernel: RobustKernel
    source_ids: list[Id] = Field(min_length=1)
    evidence_group: Id
    inlier_count: int = Field(ge=0)
    spatial_spread_m: NonNegative
    accepted: bool
    rejection_reason: Text | None = None

    @model_validator(mode="after")
    def _decision(self) -> Constraint:
        if self.accepted == (self.rejection_reason is not None):
            raise ValueError(
                "a rejected constraint needs a reason; an accepted one none"
            )
        if self.information is not None:
            check_symmetric_covariance(self.information, "information")
        return self


class Connector(Contract):
    """Traversable link between two rooms, backed by an opening on each side."""

    id: Id
    room_ids: list[Id] = Field(min_length=2, max_length=2)
    opening_ids: list[Id] = Field(min_length=1, max_length=2)
    traversable_type: Literal["door", "passage", "stair", "unknown"]
    jamb_geometry_ids: list[Id] = Field(default_factory=list)
    wall_thickness_measurement_id: Id | None = None
    association_confidence: Probability
    ambiguity_set: list[Id] = Field(default_factory=list)

    @model_validator(mode="after")
    def _two_rooms(self) -> Connector:
        if self.room_ids[0] == self.room_ids[1]:
            raise ValueError("a connector joins two different rooms")
        return self


class GraphNode(Contract):
    id: Id
    frame_id: Id
    pose: RigidTransformRecord | None = None
    scale_state: Literal["metric", "up_to_scale", "unresolved"]


class PropertyGraph(Record):
    """Rooms, submaps and constraints; components never share an origin silently."""

    property_id: Id
    property_frame_id: Id
    level_ids: list[Id] = Field(min_length=1)
    room_ids: list[Id]
    submap_ids: list[Id] = Field(default_factory=list)
    nodes: list[GraphNode]
    edges: list[Constraint] = Field(default_factory=list)
    connectors: list[Connector] = Field(default_factory=list)
    components: list[list[Id]]
    anchor_id: Id
    optimizer_manifest: AssetRef | None = None
    joint_samples: AssetRef | None = None
    topology_hypotheses: list[Id] = Field(default_factory=list)
    footprint_measurement_ids: list[Id] = Field(default_factory=list)
    registration_status: Literal[
        "connected", "disconnected", "ambiguous", "not_attempted", "failed"
    ]

    @model_validator(mode="after")
    def _consistency(self) -> PropertyGraph:
        node_ids = {node.id for node in self.nodes}
        if self.anchor_id not in node_ids:
            raise ValueError("anchor_id must be a graph node")
        flattened = [room for component in self.components for room in component]
        if sorted(flattened) != sorted(self.room_ids) or len(set(flattened)) != len(
            flattened
        ):
            raise ValueError("components must partition room_ids exactly once each")
        if self.registration_status == "connected" and len(self.components) != 1:
            raise ValueError("a connected property has exactly one component")
        for edge in self.edges:
            if not set(edge.node_ids) <= node_ids:
                raise ValueError(f"edge {edge.id} references unknown nodes")
        rooms = set(self.room_ids)
        for connector in self.connectors:
            if not set(connector.room_ids) <= rooms:
                raise ValueError(f"connector {connector.id} references unknown rooms")
        return self
