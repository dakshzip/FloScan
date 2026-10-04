"""Reconstruction bundles, run manifests and the internal property result.

``PropertyResult`` checks referential integrity across records: every
referenced ID exists, surface versions match, frames exist and every measured
geometry lives in a metric frame. JSON Schema files for the project-owned
``internal-v0`` and ``capture-v0`` schemas are generated from these models:

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
        depths = {d.id for d in self.depth_frames}
        for frame in self.frames:
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
    status: Literal["available", "partial", "unavailable", "blocked_external"]
    reason: Text | None = None

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


class PropertyResult(Contract):
    """Internal result of one run (project-owned schema ``internal-v0``)."""

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
        if self.status == "ok" and any(
            c.status != "available" for c in self.coverage.values()
        ):
            raise ValueError("status 'ok' needs every coverage section available")
        _check_references(self)
        return self

    def to_json(self) -> str:
        """Serialise with NaN/Infinity impossible (validated on construction)."""
        return self.model_dump_json(indent=2)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"broken reference: {message}")


def _require_unique(ids: Iterable[str]) -> None:
    seen: set[str] = set()
    for record_id in ids:
        if record_id in seen:
            raise ValueError(f"duplicate record id {record_id!r}")
        seen.add(record_id)


def _check_references(result: PropertyResult) -> None:
    frames = {f.id: f for f in result.coordinate_frames}
    rooms = {r.id: r for r in result.rooms}
    surfaces = {s.id: s for s in result.surfaces}
    walls = {w.id: w for w in result.walls}
    openings = {o.id: o for o in result.openings}
    measurements = {m.id: m for m in result.measurements}
    damage = {d.id: d for d in result.damage_regions}
    flags = {f.id: f for f in result.concealed_damage_flags}
    _require_unique(
        list(frames)
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
            )
            for r in group
        ]
    )

    def frame(frame_id: str, owner: str, metric: bool = False) -> None:
        _require(frame_id in frames, f"{owner}: unknown frame {frame_id}")
        if metric and frames[frame_id].unit != "m":
            raise ValueError(
                f"{owner} is in up-to-scale frame {frame_id}; measured geometry must "
                "be metric"
            )

    def each(ids: Iterable[str], table: dict[str, Any], owner: str, kind: str) -> None:
        for ref in ids:
            _require(ref in table, f"{owner}: unknown {kind} {ref}")

    for room in result.rooms:
        frame(room.local_frame_id, f"room {room.id}", metric=True)
        if room.T_property_from_room is not None:
            frame(room.T_property_from_room.to_frame_id, f"room {room.id}", metric=True)
        each(room.wall_ids, walls, f"room {room.id}", "wall")
        each(
            room.floor_surface_ids + room.ceiling_surface_ids,
            surfaces,
            f"room {room.id}",
            "surface",
        )
        each(room.opening_ids, openings, f"room {room.id}", "opening")
        each(room.measurement_ids, measurements, f"room {room.id}", "measurement")
        each(room.damage_ids, damage, f"room {room.id}", "damage region")
    for surface in result.surfaces:
        _require(surface.room_id in rooms, f"surface {surface.id}: unknown room")
        frame(surface.frame_id, f"surface {surface.id}", metric=True)
        each(
            surface.measurement_ids,
            measurements,
            f"surface {surface.id}",
            "measurement",
        )
    for wall in result.walls:
        _require(wall.room_id in rooms, f"wall {wall.id}: unknown room")
        _require(wall.surface_id in surfaces, f"wall {wall.id}: unknown surface")
        each(wall.opening_ids, openings, f"wall {wall.id}", "opening")
        each(wall.measurement_ids, measurements, f"wall {wall.id}", "measurement")
    for opening in result.openings:
        _require(opening.room_id in rooms, f"opening {opening.id}: unknown room")
        _require(
            opening.host_wall_id in walls, f"opening {opening.id}: unknown host wall"
        )
        _require(
            opening.surface_id in surfaces, f"opening {opening.id}: unknown surface"
        )
        _require(
            opening.width_measurement_id in measurements,
            f"opening {opening.id}: unknown width measurement",
        )
    subjects = {**rooms, **surfaces, **walls, **openings, **damage}
    for m in result.measurements:
        _require(
            m.subject_id in subjects,
            f"measurement {m.id}: unknown subject {m.subject_id}",
        )
        subject = subjects[m.subject_id]
        if (
            isinstance(subject, Surface)
            and m.subject_geometry_version != subject.geometry_version
        ):
            raise ValueError(
                f"measurement {m.id} refers to surface version "
                f"{m.subject_geometry_version}, current is {subject.geometry_version}"
            )
    for region in result.damage_regions:
        _require(region.room_id in rooms, f"damage {region.id}: unknown room")
        if region.surface_id is not None:
            _require(
                region.surface_id in surfaces, f"damage {region.id}: unknown surface"
            )
            _require(
                surfaces[region.surface_id].geometry_version == region.surface_version,
                f"damage {region.id}: stale surface version",
            )
        each(
            region.extent_measurement_ids,
            measurements,
            f"damage {region.id}",
            "measurement",
        )
        if region.surface_id is None:
            for mid in region.extent_measurement_ids:
                if measurements[mid].value is not None:
                    raise ValueError(
                        f"damage {region.id} has no surface, so its extent "
                        f"{mid} must be unavailable"
                    )
    for flag in result.concealed_damage_flags:
        _require(flag.room_id in rooms, f"flag {flag.id}: unknown room")
        if flag.surface_id is not None:
            _require(flag.surface_id in surfaces, f"flag {flag.id}: unknown surface")
    keys: set[str] = set()
    for item in result.scope_items:
        _require(item.room_id in rooms, f"scope {item.id}: unknown room")
        _require(item.surface_id in surfaces, f"scope {item.id}: unknown surface")
        _require(
            surfaces[item.surface_id].geometry_version == item.surface_version,
            f"scope {item.id}: stale surface version",
        )
        each(item.observed_region_ids, damage, f"scope {item.id}", "damage region")
        each(item.concealed_flag_ids, flags, f"scope {item.id}", "concealed flag")
        if item.quantity_measurement_id is not None:
            _require(
                item.quantity_measurement_id in measurements,
                f"scope {item.id}: unknown quantity measurement",
            )
        if item.deduplication_key in keys:
            raise ValueError(f"duplicate scope line {item.deduplication_key!r}")
        keys.add(item.deduplication_key)
    graph = result.property_graph
    if graph is not None:
        frame(graph.property_frame_id, "property graph", metric=True)
        each(graph.room_ids, rooms, "property graph", "room")
        each(
            graph.footprint_measurement_ids,
            measurements,
            "property graph",
            "measurement",
        )
        for node in graph.nodes:
            frame(node.frame_id, f"graph node {node.id}")
        for connector in graph.connectors:
            each(
                connector.opening_ids, openings, f"connector {connector.id}", "opening"
            )


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
