"""Reconstruction bundles, run manifests and the internal property result.

``PropertyResult`` checks referential integrity across records: IDs are unique
before anything is indexed; every internally resolvable reference exists,
belongs to the right room, wall or surface, measures the right subject and
quantity and matches the current geometry version; measured geometry is metric;
and every coverage label agrees with the records actually present.

JSON Schema files for the project-owned ``internal-v0`` and ``capture-v0``
schemas are generated from these models:

    uv run python -m floscan.contracts.result --write-schemas schemas
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from floscan.contracts.base import (
    INTERNAL_SCHEMA_VERSION,
    AssetRef,
    Contract,
    Id,
    LengthUnit,
    NonNegative,
    ProcessingMode,
    Record,
    RecordStatus,
    Sha256,
    Text,
    Tier,
)
from floscan.contracts.capture import Capture, Frame
from floscan.contracts.geometry import (
    Camera,
    CoordinateFrame,
    DepthFrame,
    Opening,
    Plane,
    PointCloud,
    Pose,
    PropertyGraph,
    Room,
    ScaleEstimate,
    ScaleEvidence,
    Surface,
    Wall,
)
from floscan.contracts.inspection import (
    QUANTITY_UNITS,
    ConcealedDamageFlag,
    DamageRegion,
    Measurement,
    ScopeItem,
)


class ReconstructionBundle(Record):
    """One tier's reconstruction; up to scale allowed (MetricReconstruction: not)."""

    capture_id: Id
    capture_hash: Sha256
    world_frame_id: Id
    geometry_unit: LengthUnit
    frames: list[Frame] = Field(default_factory=list)
    cameras: list[Camera] = Field(default_factory=list)
    poses: list[Pose] = Field(default_factory=list)
    depth_frames: list[DepthFrame] = Field(default_factory=list)
    point_clouds: list[PointCloud] = Field(default_factory=list)
    planes: list[Plane] = Field(default_factory=list)
    scale_evidence: list[ScaleEvidence] = Field(default_factory=list)
    scale_estimates: list[ScaleEstimate] = Field(default_factory=list)
    feature_tracks: AssetRef | None = None
    roomplan_proposals: AssetRef | None = None
    diagnostic_assets: list[AssetRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_unit_and_valid_references(self) -> ReconstructionBundle:
        units = (
            [("pose", p.id, p.translation_unit) for p in self.poses]
            + [("point cloud", c.id, c.unit) for c in self.point_clouds]
            + [("plane", p.id, p.unit) for p in self.planes]
        )
        mixed = [
            f"{kind} {rid} is {unit}"
            for kind, rid, unit in units
            if unit != self.geometry_unit
        ]
        if mixed:
            raise ValueError(
                f"bundle geometry is {self.geometry_unit} but " + "; ".join(mixed)
            )
        _require_unique(
            [
                r.id
                for group in (
                    self.frames,
                    self.cameras,
                    self.poses,
                    self.depth_frames,
                    self.point_clouds,
                    self.planes,
                )
                for r in group
            ]
        )
        cameras = {c.id for c in self.cameras}
        poses = {p.id for p in self.poses}
        depths = {d.id: d for d in self.depth_frames}
        frames = {f.id: f for f in self.frames}
        for depth in self.depth_frames:
            _require(depth.frame_id in frames, f"depth {depth.id}: unknown frame")
            _require(depth.camera_id in cameras, f"depth {depth.id}: unknown camera")
            _require(
                frames[depth.frame_id].depth_id == depth.id,
                f"depth {depth.id}: frame {depth.frame_id} does not link back to it",
            )
        for frame in self.frames:
            _require(
                frame.capture_id == self.capture_id,
                f"frame {frame.id}: belongs to capture {frame.capture_id}",
            )
            _require(frame.camera_id in cameras, f"frame {frame.id}: unknown camera")
            _require(
                frame.pose_id is None or frame.pose_id in poses,
                f"frame {frame.id}: unknown pose",
            )
            _require(
                frame.depth_id is None or frame.depth_id in depths,
                f"frame {frame.id}: unknown depth",
            )
        evidence = {e.id: e for e in self.scale_evidence}
        for estimate in self.scale_estimates:
            for evidence_id in estimate.evidence_ids:
                _require(
                    evidence_id in evidence,
                    f"scale {estimate.id}: unknown evidence {evidence_id}",
                )
            used = [evidence[e] for e in estimate.evidence_ids]
            if (
                used
                and all(e.role == "prior" for e in used)
                and estimate.status
                not in (
                    "prior_metric",
                    "unresolved",
                )
            ):
                raise ValueError(
                    f"scale {estimate.id} rests only on priors; status must be "
                    "'prior_metric'"
                )
        return self


class MetricReconstruction(ReconstructionBundle):
    """A bundle after a validated metric transformation."""

    geometry_unit: Literal["m"] = "m"

    @model_validator(mode="after")
    def _scale_resolved(self) -> MetricReconstruction:
        if not self.scale_estimates or any(
            s.status == "unresolved" for s in self.scale_estimates
        ):
            raise ValueError("a metric reconstruction needs every scale resolved")
        return self


class RunManifest(Contract):
    """Everything needed to identify and replay a run. Never holds ground truth."""

    id: Id
    git_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    dirty_diff_hash: Sha256 | None = None
    tier: Tier
    mode: ProcessingMode
    input_manifest_hashes: list[Sha256] = Field(min_length=1)
    config_hash: Sha256
    schema_versions: dict[str, Text]
    package_lock_hash: Sha256
    model_hashes: dict[str, Sha256] = Field(default_factory=dict)
    calibrator_hash: Sha256 | None = None
    platform: dict[str, Text]
    seeds: dict[str, int] = Field(default_factory=dict)
    thread_counts: dict[str, int] = Field(default_factory=dict)
    deterministic: bool
    started_utc: Text
    ended_utc: Text | None = None
    stage_times_s: dict[str, NonNegative] = Field(default_factory=dict)
    peak_memory_bytes: dict[str, int] = Field(default_factory=dict)
    outputs: list[AssetRef] = Field(default_factory=list)


class SectionCoverage(Contract):
    """What one output section contains, checked against the result's records.

    ``evidence_ids`` name the records that back the claim. For damage regions
    they make a negative finding explicit: ``available`` with no regions is
    valid only when the inspected surfaces are listed.
    """

    status: Literal["available", "partial", "unavailable", "blocked_external"]
    reason: Text | None = None
    evidence_ids: list[Id] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reason(self) -> SectionCoverage:
        if self.status != "available" and self.reason is None:
            raise ValueError(f"coverage status {self.status!r} needs a reason")
        return self


class Diagnostic(Contract):
    code: Id
    message: Text


SECTIONS = (
    "capture",
    "coordinate_frames",
    "scale",
    "per_room_plan",
    "stitched_plan",
    "measurements",
    "damage_regions",
    "concealed_damage_flags",
    "scope_items",
    "rendered_plan",
    "public_schema_export",
)
PLAN_MIME_TYPES = frozenset({"image/svg+xml", "image/png"})

# ID fields that point outside a PropertyResult and are provenance links, not
# resolvable here: observation, evidence, source and lineage IDs; planes,
# meshes, frames and masks held in reconstruction bundles or assets; rule,
# definition, calibration, hypothesis, level, submap and sample-group IDs.
# Every other reference must resolve to a record inside the result.
EXTERNAL_REFERENCE_FIELDS = (
    "observation_ids",
    "evidence_ids (Measurement)",
    "source_ids",
    "lineage",
    "plane_id",
    "image_masks.frame_id",
    "jamb_observation_ids",
    "boundary_evidence",
    "rule_id",
    "definition_id",
    "calibration_id",
    "score_calibration_id",
    "classification_calibration_id",
    "bias_correction_id",
    "sample_group_id",
    "hypothesis_id",
    "level_id",
    "submap_ids",
    "shared_partition_id",
    "uncertainty_dependency_ids",
)


class PropertyResult(Contract):
    """Internal result of one run (project-owned schema ``internal-v0``).

    Status ``ok`` means every internal section is available and verified
    against the records. The public-schema export can never be ``available``
    here: the evaluator schema is unavailable, so ``ok`` results carry it as
    ``blocked_external`` (or ``unavailable``) with a reason.
    """

    schema_version: Literal["internal-v0"] = INTERNAL_SCHEMA_VERSION
    run: RunManifest
    capture_id: Id | None
    status: RecordStatus
    status_reason: Text | None = None
    coverage: dict[str, SectionCoverage]
    coordinate_frames: list[CoordinateFrame] = Field(default_factory=list)
    scale: list[ScaleEstimate] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)
    surfaces: list[Surface] = Field(default_factory=list)
    walls: list[Wall] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    property_graph: PropertyGraph | None = None
    measurements: list[Measurement] = Field(default_factory=list)
    damage_regions: list[DamageRegion] = Field(default_factory=list)
    concealed_damage_flags: list[ConcealedDamageFlag] = Field(default_factory=list)
    scope_items: list[ScopeItem] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    artifacts: list[AssetRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _integrity(self) -> PropertyResult:
        if set(self.coverage) != set(SECTIONS):
            raise ValueError(f"coverage must list exactly {SECTIONS}")
        if self.status != "ok" and self.status_reason is None:
            raise ValueError(f"status {self.status!r} needs a status_reason")
        index = _check_references(self)
        _check_coverage(self, index)
        return self


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"broken reference: {message}")


def _require_unique(ids: Iterable[str]) -> None:
    seen: set[str] = set()
    for record_id in ids:
        if record_id in seen:
            raise ValueError(f"duplicate record id {record_id!r}")
        seen.add(record_id)


class _Index:
    """Lookup tables built only after IDs are proven unique."""

    def __init__(self, result: PropertyResult) -> None:
        graph = result.property_graph
        connectors = graph.connectors if graph is not None else []
        _require_unique(
            [f.id for f in result.coordinate_frames]
            + [
                r.id
                for group in (
                    result.rooms,
                    result.surfaces,
                    result.walls,
                    result.openings,
                    result.measurements,
                    result.damage_regions,
                    result.concealed_damage_flags,
                    result.scope_items,
                    result.scale,
                    connectors,
                )
                for r in group
            ]
            + ([graph.id] if graph is not None else [])
        )
        self.frames = {f.id: f for f in result.coordinate_frames}
        self.rooms = {r.id: r for r in result.rooms}
        self.surfaces = {s.id: s for s in result.surfaces}
        self.walls = {w.id: w for w in result.walls}
        self.openings = {o.id: o for o in result.openings}
        self.measurements = {m.id: m for m in result.measurements}
        self.damage = {d.id: d for d in result.damage_regions}
        self.flags = {f.id: f for f in result.concealed_damage_flags}
        self.connectors = {c.id: c for c in connectors}
        self.graph = graph
        self.all_ids = (
            set(self.frames)
            | set(self.rooms)
            | set(self.surfaces)
            | set(self.walls)
            | set(self.openings)
            | set(self.measurements)
            | set(self.damage)
            | set(self.flags)
            | {item.id for item in result.scope_items}
            | {estimate.id for estimate in result.scale}
            | set(self.connectors)
            | ({graph.id} if graph is not None else set())
        )

    def owner_room(self, subject_id: str) -> str | None:
        """The room a measurable record belongs to (None for the property graph)."""
        if subject_id in self.rooms:
            return subject_id
        for table in (self.surfaces, self.walls, self.openings, self.damage):
            if subject_id in table:
                return table[subject_id].room_id
        return None

    def geometry_version(self, subject_id: str) -> int | None:
        """Current geometry version of a subject, through its host surface."""
        if subject_id in self.surfaces:
            return self.surfaces[subject_id].geometry_version
        if subject_id in self.walls:
            return self.surfaces[self.walls[subject_id].surface_id].geometry_version
        if subject_id in self.openings:
            return self.surfaces[self.openings[subject_id].surface_id].geometry_version
        if subject_id in self.damage:
            return self.damage[subject_id].surface_version
        return None


def _check_references(result: PropertyResult) -> _Index:
    index = _Index(result)

    def frame(frame_id: str, owner: str, metric: bool = False) -> None:
        _require(frame_id in index.frames, f"{owner}: unknown frame {frame_id}")
        if metric and index.frames[frame_id].unit != "m":
            raise ValueError(
                f"{owner} is in up-to-scale frame {frame_id}; measured geometry must "
                "be metric"
            )

    def owned(
        ids: Iterable[str], table: dict[str, Any], room: str, owner: str, kind: str
    ) -> None:
        for ref in ids:
            _require(ref in table, f"{owner}: unknown {kind} {ref}")
            _require(
                table[ref].room_id == room,
                f"{owner}: {kind} {ref} belongs to room {table[ref].room_id}",
            )

    def bound(
        mid: str | None, owner: str, subjects: set[str], quantities: set[str]
    ) -> None:
        """A measurement reference must exist and measure the right thing."""
        if mid is None:
            return
        _require(mid in index.measurements, f"{owner}: unknown measurement {mid}")
        m = index.measurements[mid]
        if m.subject_id not in subjects:
            raise ValueError(
                f"{owner}: measurement {mid} measures {m.subject_id}, not "
                f"{' or '.join(sorted(subjects))}"
            )
        if m.quantity not in quantities:
            raise ValueError(
                f"{owner}: measurement {mid} is a {m.quantity}, expected "
                f"{' or '.join(sorted(quantities))}"
            )

    property_frame = index.graph.property_frame_id if index.graph else None
    for room in result.rooms:
        owner = f"room {room.id}"
        frame(room.local_frame_id, owner, metric=True)
        if room.T_property_from_room is not None:
            frame(room.T_property_from_room.to_frame_id, owner, metric=True)
            if property_frame is not None:
                _require(
                    room.T_property_from_room.to_frame_id == property_frame,
                    f"{owner}: placed in {room.T_property_from_room.to_frame_id}, "
                    f"not the property frame {property_frame}",
                )
        owned(room.wall_ids, index.walls, room.id, owner, "wall")
        owned(room.opening_ids, index.openings, room.id, owner, "opening")
        owned(room.damage_ids, index.damage, room.id, owner, "damage region")
        for ids, kind in (
            (room.floor_surface_ids, "floor"),
            (room.ceiling_surface_ids, "ceiling"),
        ):
            owned(ids, index.surfaces, room.id, owner, "surface")
            for sid in ids:
                _require(
                    index.surfaces[sid].kind == kind,
                    f"{owner}: surface {sid} is not a {kind}",
                )
        for mid in room.measurement_ids:
            _require(mid in index.measurements, f"{owner}: unknown measurement {mid}")
            subject = index.measurements[mid].subject_id
            _require(
                index.owner_room(subject) == room.id,
                f"{owner}: measurement {mid} measures {subject} of another room",
            )
    for surface in result.surfaces:
        owner = f"surface {surface.id}"
        _require(surface.room_id in index.rooms, f"{owner}: unknown room")
        frame(surface.frame_id, owner, metric=True)
        allowed = {index.rooms[surface.room_id].local_frame_id, property_frame}
        _require(
            surface.frame_id in allowed,
            f"{owner}: frame {surface.frame_id} is neither its room nor property frame",
        )
        for mid in surface.measurement_ids:
            bound(mid, owner, {surface.id}, set(QUANTITY_UNITS))
    for wall in result.walls:
        owner = f"wall {wall.id}"
        _require(wall.room_id in index.rooms, f"{owner}: unknown room")
        owned([wall.surface_id], index.surfaces, wall.room_id, owner, "surface")
        _require(
            index.surfaces[wall.surface_id].kind == "wall",
            f"{owner}: surface {wall.surface_id} is not a wall surface",
        )
        if wall.opposite_surface_id is not None:
            _require(
                wall.opposite_surface_id in index.surfaces,
                f"{owner}: unknown opposite surface {wall.opposite_surface_id}",
            )
        for oid in wall.opening_ids:
            _require(oid in index.openings, f"{owner}: unknown opening {oid}")
            _require(
                index.openings[oid].host_wall_id == wall.id,
                f"{owner}: opening {oid} is hosted by another wall",
            )
        for mid in wall.measurement_ids:
            bound(
                mid, owner, {wall.id}, {"wall_length", "wall_height", "wall_thickness"}
            )
        bound(wall.thickness_measurement_id, owner, {wall.id}, {"wall_thickness"})
    for opening in result.openings:
        owner = f"opening {opening.id}"
        _require(opening.room_id in index.rooms, f"{owner}: unknown room")
        _require(opening.host_wall_id in index.walls, f"{owner}: unknown host wall")
        host = index.walls[opening.host_wall_id]
        _require(
            host.room_id == opening.room_id, f"{owner}: host wall is in another room"
        )
        _require(
            opening.surface_id == host.surface_id,
            f"{owner}: surface differs from its host wall's surface",
        )
        bound(opening.width_measurement_id, owner, {opening.id}, {"opening_width"})
        bound(opening.height_measurement_id, owner, {opening.id}, {"opening_height"})
        bound(opening.sill_measurement_id, owner, {opening.id}, {"sill_height"})
        if opening.connector_id is not None:
            _require(
                opening.connector_id in index.connectors,
                f"{owner}: unknown connector {opening.connector_id}",
            )
            _require(
                opening.id in index.connectors[opening.connector_id].opening_ids,
                f"{owner}: connector {opening.connector_id} does not list it",
            )
    subjects = (
        set(index.rooms)
        | set(index.surfaces)
        | set(index.walls)
        | set(index.openings)
        | set(index.damage)
        | ({index.graph.id} if index.graph else set())
    )
    for m in result.measurements:
        _require(
            m.subject_id in subjects,
            f"measurement {m.id}: unknown subject {m.subject_id}",
        )
        version = index.geometry_version(m.subject_id)
        if version is not None and m.subject_geometry_version != version:
            raise ValueError(
                f"measurement {m.id} refers to geometry version "
                f"{m.subject_geometry_version} of {m.subject_id}, current is {version}"
            )
    for region in result.damage_regions:
        owner = f"damage {region.id}"
        _require(region.room_id in index.rooms, f"{owner}: unknown room")
        if region.surface_id is not None:
            owned([region.surface_id], index.surfaces, region.room_id, owner, "surface")
            _require(
                index.surfaces[region.surface_id].geometry_version
                == region.surface_version,
                f"{owner}: stale surface version",
            )
        for mid in region.extent_measurement_ids:
            bound(
                mid,
                owner,
                {region.id},
                {"damage_area", "damage_length", "damage_width"},
            )
            if region.surface_id is None and index.measurements[mid].value is not None:
                raise ValueError(
                    f"{owner} has no surface, so its extent {mid} must be unavailable"
                )
    for flag in result.concealed_damage_flags:
        owner = f"flag {flag.id}"
        _require(flag.room_id in index.rooms, f"{owner}: unknown room")
        if flag.surface_id is not None:
            owned([flag.surface_id], index.surfaces, flag.room_id, owner, "surface")
    keys: set[str] = set()
    for item in result.scope_items:
        owner = f"scope {item.id}"
        _require(item.room_id in index.rooms, f"{owner}: unknown room")
        owned([item.surface_id], index.surfaces, item.room_id, owner, "surface")
        _require(
            index.surfaces[item.surface_id].geometry_version == item.surface_version,
            f"{owner}: stale surface version",
        )
        for rid in item.observed_region_ids:
            _require(rid in index.damage, f"{owner}: unknown damage region {rid}")
            _require(
                index.damage[rid].surface_id == item.surface_id,
                f"{owner}: damage region {rid} is on another surface",
            )
        owned(
            item.concealed_flag_ids, index.flags, item.room_id, owner, "concealed flag"
        )
        bound(
            item.quantity_measurement_id,
            owner,
            {item.surface_id, *item.observed_region_ids},
            {
                "scope_area",
                "scope_length",
                "scope_count",
                "damage_area",
                "damage_length",
                "damage_width",
            },
        )
        if item.deduplication_key in keys:
            raise ValueError(f"duplicate scope line {item.deduplication_key!r}")
        keys.add(item.deduplication_key)
    graph = index.graph
    if graph is not None:
        frame(graph.property_frame_id, "property graph", metric=True)
        for rid in graph.room_ids:
            _require(rid in index.rooms, f"property graph: unknown room {rid}")
        for mid in graph.footprint_measurement_ids:
            bound(
                mid,
                "property graph",
                {graph.id},
                {"footprint_area", "footprint_extent"},
            )
        for node in graph.nodes:
            frame(node.frame_id, f"graph node {node.id}")
            if node.pose is not None:
                frame(node.pose.to_frame_id, f"graph node {node.id}")
                frame(node.pose.from_frame_id, f"graph node {node.id}")
        for edge in graph.edges:
            if edge.relative_transform is not None:
                frame(edge.relative_transform.to_frame_id, f"edge {edge.id}")
                frame(edge.relative_transform.from_frame_id, f"edge {edge.id}")
        for connector in graph.connectors:
            owner = f"connector {connector.id}"
            for oid in connector.opening_ids:
                _require(oid in index.openings, f"{owner}: unknown opening {oid}")
                _require(
                    index.openings[oid].room_id in connector.room_ids,
                    f"{owner}: opening {oid} is not in a connected room",
                )
            walls_of_rooms = {
                w.id for w in result.walls if w.room_id in connector.room_ids
            }
            bound(
                connector.wall_thickness_measurement_id,
                owner,
                walls_of_rooms,
                {"wall_thickness"},
            )
    return index


def _check_coverage(result: PropertyResult, index: _Index) -> None:
    """Coverage labels must describe the records actually present."""
    coverage = result.coverage
    for name, section in coverage.items():
        for evidence_id in section.evidence_ids:
            _require(
                evidence_id in index.all_ids,
                f"coverage {name}: unknown evidence {evidence_id}",
            )
    has_plan_artifact = any(a.mime_type in PLAN_MIME_TYPES for a in result.artifacts)
    present = {
        "capture": result.capture_id is not None,
        "coordinate_frames": bool(result.coordinate_frames),
        "scale": bool(result.scale),
        "per_room_plan": bool(
            result.rooms or result.surfaces or result.walls or result.openings
        ),
        "stitched_plan": result.property_graph is not None,
        "measurements": bool(result.measurements),
        "damage_regions": bool(result.damage_regions),
        "concealed_damage_flags": bool(result.concealed_damage_flags),
        "scope_items": bool(result.scope_items),
        "rendered_plan": has_plan_artifact,
        "public_schema_export": False,
    }
    for name, has_content in present.items():
        if coverage[name].status in ("unavailable", "blocked_external") and has_content:
            raise ValueError(
                f"coverage {name} is {coverage[name].status} but the result "
                "contains it; label it 'partial' with a reason"
            )

    def complete(name: str, condition: bool, why: str) -> None:
        if coverage[name].status == "available" and not condition:
            raise ValueError(f"coverage {name} is 'available' but {why}")

    complete("capture", result.capture_id is not None, "there is no capture id")
    complete("coordinate_frames", bool(result.coordinate_frames), "there are no frames")
    complete(
        "scale",
        bool(result.scale) and all(s.status != "unresolved" for s in result.scale),
        "scale is missing or unresolved",
    )
    complete(
        "per_room_plan",
        _rooms_complete(result, index),
        "a room lacks walls, boundary, wall lengths, ceiling height or floor area",
    )
    graph = result.property_graph
    complete(
        "stitched_plan",
        graph is not None
        and graph.registration_status == "connected"
        and set(graph.room_ids) == set(index.rooms)
        and bool(index.rooms)
        and all(r.placement_status == "placed" for r in result.rooms),
        "rooms are missing, unplaced or not one connected graph",
    )
    complete(
        "measurements",
        bool(result.measurements)
        and all(
            m.value is not None and m.interval.status in ("calibrated", "provisional")
            for m in result.measurements
        ),
        "measurements are missing or lack values and intervals",
    )
    inspected = [
        e for e in coverage["damage_regions"].evidence_ids if e in index.surfaces
    ]
    complete(
        "damage_regions",
        bool(result.damage_regions) or bool(inspected),
        "it has no regions and no inspected surfaces as evidence of a negative finding",
    )
    complete(
        "concealed_damage_flags",
        bool(result.concealed_damage_flags),
        "no rule evaluation is recorded (negative findings are 'not_triggered' flags)",
    )
    scoped_regions = {
        r for item in result.scope_items for r in item.observed_region_ids
    }
    scoped_flags = {f for item in result.scope_items for f in item.concealed_flag_ids}
    inspect_flags = {
        f.id for f in result.concealed_damage_flags if f.flag_status == "inspect"
    }
    complete(
        "scope_items",
        coverage["damage_regions"].status == "available"
        and coverage["concealed_damage_flags"].status == "available"
        and set(index.damage) <= scoped_regions
        and inspect_flags <= scoped_flags,
        "damage or flags are incomplete, or a region or 'inspect' flag lacks a "
        "scope line",
    )
    complete("rendered_plan", has_plan_artifact, "no SVG or PNG plan artifact exists")
    if coverage["public_schema_export"].status == "available":
        raise ValueError(
            "coverage public_schema_export cannot be 'available': the evaluator "
            "schema is unavailable and no exporter exists"
        )
    if result.status == "ok":
        internal = [name for name in SECTIONS if name != "public_schema_export"]
        if any(coverage[name].status != "available" for name in internal):
            raise ValueError(
                "status 'ok' needs every internal coverage section available"
            )


def _rooms_complete(result: PropertyResult, index: _Index) -> bool:
    if not result.rooms:
        return False
    measured = {
        (m.subject_id, m.quantity) for m in result.measurements if m.value is not None
    }
    for room in result.rooms:
        if room.status != "ok" or room.boundary is None or not room.wall_ids:
            return False
        if not {(room.id, "ceiling_height"), (room.id, "floor_area")} <= measured:
            return False
        if any((wid, "wall_length") not in measured for wid in room.wall_ids):
            return False
        for oid in room.opening_ids:
            if (oid, "opening_width") not in measured:
                return False
    return True


# --------------------------------------------------------------------------
# Schema generation
# --------------------------------------------------------------------------

SCHEMA_FILES = {
    "internal-v0.schema.json": PropertyResult,
    "capture-v0.schema.json": Capture,
}


def schema_documents() -> dict[str, dict[str, Any]]:
    """Generate the project-owned JSON Schemas (Draft 2020-12)."""
    documents = {}
    for filename, model in SCHEMA_FILES.items():
        schema = model.model_json_schema(mode="validation")
        documents[filename] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://floscan.local/schemas/{filename}",
            "$comment": (
                "Project-owned FloScan schema generated from src/floscan/contracts; "
                "not the assignment's published schema, which is unavailable. "
                "Semantic invariants live in the model validators."
            ),
            **schema,
        }
    return documents


def write_schemas(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, document in schema_documents().items():
        path = directory / filename
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate FloScan JSON Schemas.")
    parser.add_argument("--write-schemas", type=Path, required=True, metavar="DIR")
    args = parser.parse_args(argv)
    for path in write_schemas(args.write_schemas):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
