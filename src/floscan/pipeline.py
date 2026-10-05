"""Diagnostic pipeline skeleton, gate registry and result-envelope validation.

P01 scope: this module runs no reconstruction, measurement, damage or export
stage. It inventories the declared input, reports every later stage as
``not_implemented`` and writes a validated diagnostic result envelope whose
status honestly says the output contract is incomplete.

The envelope follows the project-owned schema ``floscan-result``. The
assignment's published JSON schema is unavailable, so no conformance to it is
claimed (docs/adr/001-requirements.md).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from floscan import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATES_PATH = PROJECT_ROOT / "configs" / "gates.yaml"

TIERS = ("photo", "video", "lidar")
MODES = ("live", "replay")

RESULT_SCHEMA_VERSION = "floscan-result/0.2.0"
RECORDS_SCHEMA_VERSION = "internal-v0"
RESULT_KIND = "diagnostic_envelope"
RESULT_FILENAME = "result.json"

RESULT_STATUSES = (
    "ok",
    "partial",
    "invalid_input",
    "insufficient_evidence",
    "ambiguous",
    "unsupported",
    "failed",
)
STAGE_STATUSES = (*RESULT_STATUSES, "not_implemented", "skipped")
SECTION_STATUSES = ("available", "partial", "unavailable", "blocked_external")
GATE_STATUSES = ("measured_pass", "measured_fail", "unverified", "unspecified_source")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INVALID_INPUT = 3
EXIT_INCOMPLETE = 4
EXIT_REPLAY_UNAVAILABLE = 5

# Stage names follow the interface table in
# docs/implementation-strategy/01-architecture.md.
PIPELINE_STAGES = (
    "capture.normalize",
    "quality.evaluate",
    "reconstruction.reconstruct",
    "scale.resolve",
    "rooms.build",
    "registration.register",
    "surfaces.finalize",
    "measurements.evaluate",
    "damage.detect",
    "damage.project",
    "uncertainty.calibrate",
    "rules.evaluate",
    "export.serialize",
    "render.plan",
)
# Stages that exist in this build, per tier. Only these may report "ok"; every
# other stage is "not_implemented" and no contract section is produced.
IMPLEMENTED_STAGES: dict[str, tuple[str, ...]] = {
    "lidar": (
        "capture.normalize",
        "reconstruction.reconstruct",
        "rooms.build",
        "measurements.evaluate",
        "render.plan",
        "export.serialize",
    ),
}
INVENTORY_STAGE = "input.inventory"
REPLAY_STAGE = "replay.lookup"

# Output-contract sections and the top-level envelope fields each one owns.
# An empty tuple means the section has no payload field of its own.
CONTRACT_SECTIONS: dict[str, tuple[str, ...]] = {
    "capture": ("capture",),
    "coordinate_frames": ("coordinate_frames",),
    "scale": ("scale",),
    "per_room_plan": ("rooms", "surfaces", "walls", "openings"),
    "stitched_plan": ("property_graph",),
    "measurements": ("measurements",),
    "damage_regions": ("damage_regions",),
    "concealed_damage_flags": ("concealed_damage_flags",),
    "scope_items": ("scope_items",),
    "rendered_plan": ("artifacts",),
    "public_schema_export": (),
}
NULLABLE_FIELDS = frozenset({"capture", "scale", "property_graph"})
# Section statuses that make no export claim.
_NO_EXPORT = ("blocked_external", "unavailable")

ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "schema_authority",
        "result_kind",
        "run",
        "status",
        "status_reason",
        "exit_code",
        "coverage",
        "stages",
        "gates",
        "diagnostics",
        "records",
        *(name for fields in CONTRACT_SECTIONS.values() for name in fields),
    }
)
RUN_KEYS = frozenset(
    {
        "run_id",
        "created_utc",
        "floscan_version",
        "mode",
        "tier",
        "argv",
        "python_version",
        "platform",
        "input",
        "output_dir",
    }
)

EXTERNAL_SCHEMA_REASON = (
    "The assignment's published JSON schema was not supplied and is unavailable. "
    "This file follows the project-owned schema; no exporter mapping exists and "
    "no conformance to the evaluator schema is claimed."
)


class RegistryError(ValueError):
    """The gate registry is malformed or not traceable to the sources."""


class EnvelopeError(ValueError):
    """A result envelope violates the project-owned diagnostic contract."""


class RunIOError(RuntimeError):
    """A filesystem operation at the run boundary failed.

    Raised only around output-location checks, input inventory and result
    writing, wrapping the underlying ``OSError`` with its path context.
    """


# --------------------------------------------------------------------------
# Gate registry
# --------------------------------------------------------------------------

_SOURCE_REF = re.compile(r"^(?P<source>[a-z_]+):p(?P<page>[1-9][0-9]*)$")
_SOURCE_VALUE = re.compile(
    r"^(?P<number>[0-9]+(?:\.[0-9]+)?) ?(?P<unit>cm|mm|m|%|minutes|pages)$"
)
# Source unit -> (SI unit used in the registry, multiplier).
_SOURCE_UNITS = {
    "cm": ("m", 0.01),
    "mm": ("m", 0.001),
    "m": ("m", 1.0),
    "%": ("ratio", 0.01),
    "minutes": ("s", 60.0),
    "pages": ("pages", 1.0),
}
_COMPARATORS = ("<=", "<", ">=", ">")
_REGISTRY_UNITS = ("m", "m2", "ratio", "s", "pages")
_GATE_CATEGORIES = (
    "accuracy",
    "repeatability",
    "topology",
    "calibration",
    "comparison",
    "process",
    "contract",
    "deliverable",
)
_SOURCE_STATUSES = ("specified", "specified_qualitative", "unspecified_source")
_INTERPRETATION_STATUSES = ("exact", "provisional", "unspecified")


def _check_keys(
    obj: Any, required: set[str], where: str, optional: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Require ``obj`` to be a mapping with exactly the allowed keys."""
    if not isinstance(obj, dict):
        raise RegistryError(f"{where}: expected a mapping, got {type(obj).__name__}")
    missing = required - obj.keys()
    unknown = obj.keys() - required - optional
    if missing:
        raise RegistryError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise RegistryError(f"{where}: unknown keys {sorted(unknown)}")
    return obj


def _check_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{where}: expected non-empty text")
    return value


def _check_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise RegistryError(f"{where}: expected a list, got {type(value).__name__}")
    return value


def _check_text_list(value: Any, where: str) -> list[str]:
    """Require a non-empty list of non-empty strings (safe to hash)."""
    if not isinstance(value, list) or not value:
        raise RegistryError(f"{where}: must be a non-empty list")
    for item in value:
        _check_text(item, where)
    return value


def _check_positive_int(value: Any, where: str) -> int:
    # bool is an int subclass, so True would otherwise pass as 1.
    if type(value) is not int or value < 1:
        raise RegistryError(f"{where}: expected a positive integer, got {value!r}")
    return value


def _check_source_refs(refs: Any, pages: dict[str, int], where: str) -> None:
    if not isinstance(refs, list) or not refs:
        raise RegistryError(f"{where}: source_refs must be a non-empty list")
    for ref in refs:
        match = _SOURCE_REF.match(ref) if isinstance(ref, str) else None
        if match is None:
            raise RegistryError(f"{where}: malformed source ref {ref!r}")
        source, page = match["source"], int(match["page"])
        if source not in pages:
            raise RegistryError(f"{where}: unknown source {source!r}")
        if page > pages[source]:
            raise RegistryError(f"{where}: {ref} exceeds {pages[source]} pages")


def _check_criterion(criterion: Any, gate: dict[str, Any], where: str) -> None:
    _check_keys(
        criterion, {"quantity", "comparator", "source_value", "value", "unit"}, where
    )
    _check_text(criterion["quantity"], f"{where}.quantity")
    if criterion["comparator"] not in _COMPARATORS:
        raise RegistryError(f"{where}: comparator must be one of {_COMPARATORS}")
    if criterion["unit"] not in _REGISTRY_UNITS:
        raise RegistryError(f"{where}: unit must be one of {_REGISTRY_UNITS}")

    source_value, value = criterion["source_value"], criterion["value"]
    if gate["source_status"] == "unspecified_source":
        if source_value is not None or value is not None:
            raise RegistryError(
                f"{where}: an unspecified_source gate must not carry a threshold"
            )
        return
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RegistryError(f"{where}: value must be a number")
    if not math.isfinite(value):
        raise RegistryError(f"{where}: value must be finite")

    # Traceability: the threshold must be quoted from the source text and
    # must convert exactly to the registered SI value.
    match = _SOURCE_VALUE.match(source_value) if isinstance(source_value, str) else None
    if match is None:
        raise RegistryError(f"{where}: source_value {source_value!r} is not parseable")
    if source_value not in gate["source_text"]:
        raise RegistryError(f"{where}: {source_value!r} does not appear in source_text")
    si_unit, multiplier = _SOURCE_UNITS[match["unit"]]
    if si_unit != criterion["unit"]:
        raise RegistryError(
            f"{where}: {source_value!r} converts to {si_unit}, "
            f"registered unit is {criterion['unit']}"
        )
    expected = float(match["number"]) * multiplier
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RegistryError(
            f"{where}: value {value} does not equal {source_value!r} = {expected}"
        )


def _check_gate(
    gate: Any, requirement_ids: set[str], pages: dict[str, int], where: str
) -> None:
    _check_keys(
        gate,
        {
            "id",
            "requirement_ids",
            "category",
            "tiers",
            "source_refs",
            "source_text",
            "source_status",
            "combine",
            "criteria",
            "interpretation",
        },
        where,
    )
    _check_text(gate["id"], f"{where}.id")
    where = f"gate {gate['id']}"
    refs = _check_text_list(gate["requirement_ids"], f"{where}.requirement_ids")
    unknown = set(refs) - requirement_ids
    if unknown:
        raise RegistryError(f"{where}: unknown requirement ids {sorted(unknown)}")
    if gate["category"] not in _GATE_CATEGORIES:
        raise RegistryError(f"{where}: category must be one of {_GATE_CATEGORIES}")
    tiers = _check_text_list(gate["tiers"], f"{where}.tiers")
    if not set(tiers) <= set(TIERS):
        raise RegistryError(f"{where}: tiers must be a non-empty subset of {TIERS}")
    _check_source_refs(gate["source_refs"], pages, where)
    _check_text(gate["source_text"], f"{where}.source_text")
    if gate["source_status"] not in _SOURCE_STATUSES:
        raise RegistryError(f"{where}: source_status must be one of {_SOURCE_STATUSES}")
    if gate["combine"] not in ("all", "any"):
        raise RegistryError(f"{where}: combine must be 'all' or 'any'")

    criteria = gate["criteria"]
    if not isinstance(criteria, list):
        raise RegistryError(f"{where}: criteria must be a list")
    if gate["source_status"] == "specified" and not criteria:
        raise RegistryError(f"{where}: a specified gate needs numeric criteria")
    if gate["source_status"] == "specified_qualitative" and criteria:
        raise RegistryError(f"{where}: a qualitative gate cannot carry thresholds")
    for index, criterion in enumerate(criteria):
        _check_criterion(criterion, gate, f"{where}.criteria[{index}]")

    interpretation = _check_keys(
        gate["interpretation"],
        {"status", "policy", "open_questions"},
        f"{where}.interpretation",
    )
    if interpretation["status"] not in _INTERPRETATION_STATUSES:
        raise RegistryError(
            f"{where}: interpretation status must be one of {_INTERPRETATION_STATUSES}"
        )
    unspecified = gate["source_status"] == "unspecified_source"
    if unspecified != (interpretation["status"] == "unspecified"):
        raise RegistryError(
            f"{where}: interpretation 'unspecified' must match source_status "
            "'unspecified_source'"
        )
    _check_text(interpretation["policy"], f"{where}.interpretation.policy")
    if not isinstance(interpretation["open_questions"], list):
        raise RegistryError(f"{where}: open_questions must be a list")


def validate_gate_registry(registry: Any) -> dict[str, Any]:
    """Validate a parsed gate registry and return it unchanged.

    Raises:
        RegistryError: if the registry is malformed, references unknown
            requirements or sources, or carries a threshold that cannot be
            traced verbatim to its quoted source text.
    """
    _check_keys(
        registry,
        {
            "registry_version",
            "schema_authority",
            "sources",
            "unavailable_sources",
            "requirements",
            "gates",
        },
        "registry",
    )
    # Exact type: True and 1.0 both compare equal to 1.
    version = registry["registry_version"]
    if type(version) is not int or version != 1:
        raise RegistryError(f"registry: unsupported registry_version {version!r}")
    if registry["schema_authority"] != "project_owned":
        raise RegistryError("registry: schema_authority must be 'project_owned'")

    pages: dict[str, int] = {}
    for index, source in enumerate(_check_list(registry["sources"], "sources")):
        where = f"sources[{index}]"
        _check_keys(source, {"id", "file", "sha256", "pages", "authority"}, where)
        source_id = _check_text(source["id"], f"{where}.id")
        if source_id in pages:
            raise RegistryError(f"{where}: duplicate source {source_id!r}")
        _check_text(source["file"], f"{where}.file")
        if not re.fullmatch(r"[0-9a-f]{64}", str(source["sha256"])):
            raise RegistryError(f"{where}: sha256 must be 64 lowercase hex digits")
        if source["authority"] not in ("primary", "derived"):
            raise RegistryError(f"{where}: authority must be 'primary' or 'derived'")
        pages[source_id] = _check_positive_int(source["pages"], f"{where}.pages")
    if not pages:
        raise RegistryError("registry: at least one source document is required")

    unavailable = _check_list(registry["unavailable_sources"], "unavailable_sources")
    for index, missing in enumerate(unavailable):
        where = f"unavailable_sources[{index}]"
        _check_keys(missing, {"id", "description", "consequence"}, where)

    requirement_ids: set[str] = set()
    requirements = _check_list(registry["requirements"], "requirements")
    for index, requirement in enumerate(requirements):
        where = f"requirements[{index}]"
        _check_keys(requirement, {"id", "title", "source_refs"}, where)
        if not re.fullmatch(r"G[0-9]{2}", str(requirement["id"])):
            raise RegistryError(f"{where}: requirement id must look like G01")
        if requirement["id"] in requirement_ids:
            raise RegistryError(f"{where}: duplicate requirement {requirement['id']}")
        _check_text(requirement["title"], f"{where}.title")
        _check_source_refs(requirement["source_refs"], pages, where)
        requirement_ids.add(requirement["id"])

    gate_ids: set[str] = set()
    for index, gate in enumerate(_check_list(registry["gates"], "gates")):
        _check_gate(gate, requirement_ids, pages, f"gates[{index}]")
        if gate["id"] in gate_ids:
            raise RegistryError(f"gates[{index}]: duplicate gate {gate['id']}")
        gate_ids.add(gate["id"])
    if not gate_ids:
        raise RegistryError("registry: no gates defined")
    return registry


@dataclass(frozen=True)
class GateRegistry:
    """A validated gate registry plus the identity of the file it came from."""

    path: Path
    sha256: str
    data: dict[str, Any]

    @property
    def gates(self) -> list[dict[str, Any]]:
        return self.data["gates"]


def load_gate_registry(path: Path = DEFAULT_GATES_PATH) -> GateRegistry:
    """Read, parse and validate the gate registry at ``path``."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RegistryError(f"cannot read gate registry {path}: {error}") from error
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise RegistryError(
            f"gate registry {path} is not valid YAML: {error}"
        ) from error
    validate_gate_registry(data)
    return GateRegistry(path=path, sha256=hashlib.sha256(raw).hexdigest(), data=data)


def initial_gate_status(gate: dict[str, Any]) -> dict[str, str]:
    """Return the only verdicts a non-benchmark run may report for ``gate``.

    Pass/fail verdicts come exclusively from a scored benchmark run (P04+).
    """
    if gate["source_status"] == "unspecified_source":
        return {
            "id": gate["id"],
            "status": "unspecified_source",
            "reason": "No threshold exists in the supplied sources; the quantity "
            "is reported without a pass/fail verdict.",
        }
    return {
        "id": gate["id"],
        "status": "unverified",
        "reason": "Not scored: no benchmark run has evaluated this gate and this "
        "build produces no measurements.",
    }


def summarize_gate_statuses(items: list[dict[str, Any]]) -> dict[str, int]:
    """Count gate items per status, including zero counts."""
    summary: dict[str, int] = {name: 0 for name in GATE_STATUSES}
    for item in items:
        summary[item["status"]] += 1
    return summary


# --------------------------------------------------------------------------
# Diagnostic run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRequest:
    """One CLI invocation of ``floscan run``."""

    input_dir: Path
    output_dir: Path
    tier: str
    mode: str
    gates_path: Path = DEFAULT_GATES_PATH
    argv: tuple[str, ...] = field(default_factory=tuple)
    run_id: str = field(
        default_factory=lambda: (
            "run-"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )


def check_output_location(input_dir: Path, output_dir: Path) -> str | None:
    """Return why ``output_dir`` is unusable, or None if it is acceptable.

    Raw inputs are immutable, so results may never be written inside the input
    tree. Run directories are append-only, so an existing result is never
    overwritten.

    Raises:
        RunIOError: if the filesystem cannot answer these checks.
    """
    try:
        input_resolved = input_dir.expanduser().resolve()
        output_resolved = output_dir.expanduser().resolve()
        if output_resolved == input_resolved or output_resolved.is_relative_to(
            input_resolved
        ):
            return (
                f"output directory {str(output_resolved)!r} is inside the input "
                f"{str(input_resolved)!r}; raw inputs are immutable"
            )
        if output_resolved.exists() and not output_resolved.is_dir():
            return f"output path {str(output_resolved)!r} exists and is not a directory"
        if (output_resolved / RESULT_FILENAME).exists():
            return (
                f"{str(output_resolved / RESULT_FILENAME)!r} already exists; run "
                "directories are append-only, choose a new output directory"
            )
    except OSError as error:
        raise RunIOError(
            f"cannot check output location {output_dir}: {error}"
        ) from error
    return None


def _inventory(input_dir: Path) -> tuple[dict[str, Any], str | None]:
    """Count files and bytes under ``input_dir`` without reading contents.

    Returns the inventory record and, if the input is unusable, the reason.

    Raises:
        RunIOError: if any part of the input tree cannot be read. An unreadable
            subdirectory is a failure, never a silently smaller count.
    """

    def _raise(error: OSError) -> None:
        raise error

    resolved = input_dir.expanduser().resolve()
    try:
        record: dict[str, Any] = {
            "path": str(resolved),
            "exists": resolved.exists(),
            "is_directory": resolved.is_dir(),
            "file_count": None,
            "total_bytes": None,
        }
        if not resolved.exists():
            return record, f"input directory {str(resolved)!r} does not exist"
        if not resolved.is_dir():
            return record, f"input path {str(resolved)!r} is not a directory"
        file_count = 0
        total_bytes = 0
        # os.walk skips unreadable directories unless onerror re-raises.
        for root, _dirs, files in os.walk(resolved, onerror=_raise):
            for name in files:
                file_count += 1
                total_bytes += (Path(root) / name).lstat().st_size
    except OSError as error:
        raise RunIOError(
            f"cannot inventory input {str(resolved)!r}: {error}"
        ) from error
    record["file_count"] = file_count
    record["total_bytes"] = total_bytes
    if file_count == 0:
        return record, f"input directory {str(resolved)!r} contains no files"
    return record, None


def _stage(name: str, status: str, reason: str) -> dict[str, str]:
    return {"name": name, "status": status, "reason": reason}


def _sections(reason: str) -> dict[str, dict[str, str]]:
    sections = {
        name: {"status": "unavailable", "reason": reason} for name in CONTRACT_SECTIONS
    }
    sections["public_schema_export"] = {
        "status": "blocked_external",
        "reason": EXTERNAL_SCHEMA_REASON,
    }
    return sections


def _run_implemented(
    request: RunRequest, diagnostics: list[dict[str, str]]
) -> tuple[str, str, list[dict[str, str]], dict[str, str] | None, dict[str, Any]]:
    """Run this tier's implemented stages.

    Returns (status, reason, stages, records pointer, section coverage).

    A failure inside a stage is reported as status ``failed`` with the error,
    never hidden; the envelope is still written.
    """
    # Imported here so other tiers never load Open3D, PyAV or Pillow.
    from floscan.reconstruction import lidar

    try:
        outcome = lidar.run_live(request.input_dir, request.output_dir)
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        reason = f"{type(error).__name__}: {error}"
        diagnostics.append({"code": "stage_failed", "message": reason})
        return (
            "failed",
            f"A LiDAR stage failed: {reason}",
            [
                _stage("capture.normalize", "failed", f"LiDAR stages failed: {reason}"),
            ],
            None,
            {},
        )
    diagnostics.extend(outcome.diagnostics)
    if outcome.status != "unsupported":  # capture or reconstruction stopped
        return outcome.status, outcome.status_reason, outcome.stages, None, {}
    from floscan.io import exporter

    try:
        local = exporter.finish_lidar(request.output_dir, request.run_id, request.mode)
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        reason = f"{type(error).__name__}: {error}"
        diagnostics.append({"code": "stage_failed", "message": reason})
        failed = _stage("rooms.build", "failed", f"local result failed: {reason}")
        return "failed", reason, [*outcome.stages, failed], None, {}
    diagnostics.extend(local.diagnostics)
    return (
        local.status,
        local.status_reason,
        [*outcome.stages, *local.stages],
        local.records,
        local.sections,
    )


def execute(request: RunRequest, registry: GateRegistry) -> dict[str, Any]:
    """Run the pipeline and return a validated result envelope.

    Only the stages in ``IMPLEMENTED_STAGES`` for the requested tier run; the
    output contract stays incomplete, so a successful live run still ends
    with status ``unsupported``. Replay never falls back to live inference.
    """
    if request.tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {request.tier!r}")
    if request.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {request.mode!r}")

    inventory, input_problem = _inventory(request.input_dir)
    stages: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []
    records: dict[str, str] | None = None

    if input_problem is not None:
        stages.append(_stage(INVENTORY_STAGE, "invalid_input", input_problem))
        skipped = "Not attempted: the input is invalid."
        stages += [_stage(name, "skipped", skipped) for name in PIPELINE_STAGES]
        status, exit_code = "invalid_input", EXIT_INVALID_INPUT
        status_reason = input_problem
        sections = _sections(skipped)
        diagnostics.append({"code": "input_invalid", "message": input_problem})
    elif request.mode == "replay":
        stages.append(_stage(INVENTORY_STAGE, "ok", "Input directory inventoried."))
        miss = (
            "No replay cache exists in this build. Replay requires a cache entry "
            "matching the input, config, model, code, tier and schema hashes and "
            "never falls back to live inference."
        )
        stages.append(_stage(REPLAY_STAGE, "unsupported", miss))
        skipped = "Not attempted: replay cache unavailable."
        stages += [_stage(name, "skipped", skipped) for name in PIPELINE_STAGES]
        status, exit_code = "unsupported", EXIT_REPLAY_UNAVAILABLE
        status_reason = miss
        sections = _sections(skipped)
        diagnostics.append({"code": "replay_cache_unavailable", "message": miss})
    elif IMPLEMENTED_STAGES.get(request.tier):
        stages.append(_stage(INVENTORY_STAGE, "ok", "Input directory inventoried."))
        status, status_reason, done, records, produced = _run_implemented(
            request, diagnostics
        )
        stages += done
        names = {stage["name"] for stage in done}
        not_built = "Not implemented in this build."
        stages += [
            _stage(name, "not_implemented", not_built)
            for name in PIPELINE_STAGES
            if name not in names
        ]
        exit_code = {
            "invalid_input": EXIT_INVALID_INPUT,
            "failed": EXIT_FAILED,
        }.get(status, EXIT_INCOMPLETE)
        sections = _sections(
            "Unavailable: no stage producing this section is implemented in this "
            "build. Stage outputs (capture records, point clouds, plane "
            "candidates) are files in the run directory listed under diagnostics."
        )
        sections.update(produced)
        diagnostics.append({"code": "contract_incomplete", "message": status_reason})
    else:
        stages.append(_stage(INVENTORY_STAGE, "ok", "Input directory inventoried."))
        not_built = "Not implemented in this build (P01 diagnostic skeleton)."
        stages += [
            _stage(name, "not_implemented", not_built) for name in PIPELINE_STAGES
        ]
        status, exit_code = "unsupported", EXIT_INCOMPLETE
        status_reason = (
            "The output contract is incomplete: no reconstruction, measurement, "
            "damage, rule, export or rendering stage exists in this build. No "
            "rooms, geometry or measurements were produced."
        )
        sections = _sections(
            "Unavailable: no stage producing this section is implemented in this "
            "build (P01 diagnostic skeleton)."
        )
        diagnostics.append({"code": "contract_incomplete", "message": status_reason})

    gate_items = [initial_gate_status(gate) for gate in registry.gates]
    envelope: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "schema_authority": {
            "owner": "project",
            "external_schema_status": "unavailable",
            "external_conformance_claimed": False,
            "note": EXTERNAL_SCHEMA_REASON,
        },
        "result_kind": RESULT_KIND,
        "run": {
            "run_id": request.run_id,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "floscan_version": __version__,
            "mode": request.mode,
            "tier": request.tier,
            "argv": list(request.argv),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "input": inventory,
            "output_dir": str(request.output_dir.expanduser().resolve()),
        },
        "status": status,
        "status_reason": status_reason,
        "exit_code": exit_code,
        "coverage": {"contract_complete": False, "sections": sections},
        "capture": None,
        "coordinate_frames": [],
        "scale": None,
        "rooms": [],
        "surfaces": [],
        "walls": [],
        "openings": [],
        "property_graph": None,
        "measurements": [],
        "damage_regions": [],
        "concealed_damage_flags": [],
        "scope_items": [],
        "artifacts": [],
        "records": records,
        "stages": stages,
        "gates": {
            "registry": {
                "path": str(registry.path),
                "sha256": registry.sha256,
                "registry_version": registry.data["registry_version"],
            },
            "summary": summarize_gate_statuses(gate_items),
            "items": gate_items,
        },
        "diagnostics": diagnostics,
    }
    validate_envelope(envelope)
    return envelope


def write_envelope(envelope: dict[str, Any], output_dir: Path) -> Path:
    """Validate and write ``envelope`` to ``output_dir/result.json``.

    The file is created exclusively, so an existing result is never
    overwritten. The written bytes are read back and validated again.

    Raises:
        RunIOError: if the directory or file cannot be created or written. A
            partially written file created by this call is removed.
    """
    validate_envelope(envelope)
    path = output_dir / RESULT_FILENAME
    text = json.dumps(envelope, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        handle = path.open("x", encoding="utf-8")
    except OSError as error:
        raise RunIOError(f"cannot create result {path}: {error}") from error
    try:
        with handle:
            handle.write(text)
    except OSError as error:
        # Only ever the file this call created; the write failure is reported
        # either way, so a failed cleanup must not mask it.
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise RunIOError(f"cannot write result {path}: {error}") from error
    written = read_envelope(path)
    validate_envelope(written)
    validate_records(written, output_dir)
    return path


def validate_records(envelope: dict[str, Any], run_dir: Path) -> None:
    """Check the records file the envelope points to, if any.

    The file must exist with the recorded SHA-256, validate as an
    ``internal-v0`` PropertyResult (records, references and coverage), and
    agree with every envelope section status.

    Raises:
        EnvelopeError: describing the first disagreement.
    """
    from floscan.contracts.result import PropertyResult

    records = envelope["records"]
    if records is None:
        return
    path = run_dir / records["path"]
    try:
        data = path.read_bytes()
    except OSError as error:
        raise EnvelopeError(f"records file {path} cannot be read: {error}") from error
    if hashlib.sha256(data).hexdigest() != records["sha256"]:
        raise EnvelopeError(f"records file {path} does not match its recorded hash")
    try:
        result = PropertyResult.model_validate_json(data)
    except ValueError as error:
        raise EnvelopeError(
            f"records file {path} is not a valid result: {error}"
        ) from error
    for name, section in envelope["coverage"]["sections"].items():
        recorded = result.coverage[name].status
        if section["status"] != recorded:
            raise EnvelopeError(
                f"coverage.sections.{name} is {section['status']!r} but the records "
                f"say {recorded!r}"
            )


def _reject_constant(name: str) -> None:
    raise EnvelopeError(f"non-finite JSON number {name} is not allowed")


def read_envelope(path: Path) -> dict[str, Any]:
    """Parse a result envelope, rejecting NaN and Infinity literals."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise EnvelopeError(f"cannot read {path}: {error}") from error
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise EnvelopeError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise EnvelopeError(f"{path}: top level must be a JSON object")
    return data


# --------------------------------------------------------------------------
# Envelope validation
# --------------------------------------------------------------------------


def _expect_exact_keys(obj: Any, keys: frozenset[str] | set[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise EnvelopeError(f"{where}: expected an object")
    missing = set(keys) - obj.keys()
    unknown = obj.keys() - set(keys)
    if missing:
        raise EnvelopeError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise EnvelopeError(f"{where}: unknown keys {sorted(unknown)}")


def _expect_reason(obj: dict[str, Any], key: str, where: str) -> None:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EnvelopeError(f"{where}: a non-empty {key!r} is required")


def _check_finite(value: Any, where: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EnvelopeError(f"{where}: non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{where}[{index}]")


def _expected_exit_code(status: str, mode: str) -> set[int]:
    # Status "ok" is rejected before this is consulted: the diagnostic schema
    # can never certify a complete result.
    if status == "failed":
        return {EXIT_FAILED}
    if status == "invalid_input":
        return {EXIT_INVALID_INPUT}
    if mode == "replay":
        return {EXIT_REPLAY_UNAVAILABLE, EXIT_INCOMPLETE}
    return {EXIT_INCOMPLETE}


def validate_envelope(envelope: Any) -> None:
    """Check ``envelope`` against the project-owned diagnostic contract.

    Beyond shape checks this enforces the honesty rules of P01: no NaN or
    Infinity, every section unavailable with a reason and an empty payload,
    no entity records (rooms, measurements, ...) because no validator for them
    exists before P03, public-schema export blocked while the external schema
    is unavailable, and no gate verdict other than ``unverified`` or
    ``unspecified_source``.

    This diagnostic schema has no producer for any contract section, so it
    rejects status ``ok``, ``contract_complete: true`` and any ``available``
    or ``partial`` section. A stage after the input inventory may report
    ``ok`` only if it is implemented for the run's tier in live mode
    (``IMPLEMENTED_STAGES``); its outputs are files in the run directory,
    never records embedded in the envelope.

    Raises:
        EnvelopeError: describing the first violation found.
    """
    _expect_exact_keys(envelope, ENVELOPE_KEYS, "envelope")
    _check_finite(envelope, "envelope")

    if envelope["schema_version"] != RESULT_SCHEMA_VERSION:
        raise EnvelopeError(f"schema_version must be {RESULT_SCHEMA_VERSION!r}")
    authority = envelope["schema_authority"]
    _expect_exact_keys(
        authority,
        {"owner", "external_schema_status", "external_conformance_claimed", "note"},
        "schema_authority",
    )
    if authority["owner"] != "project":
        raise EnvelopeError("schema_authority.owner must be 'project'")
    if authority["external_schema_status"] != "unavailable":
        raise EnvelopeError(
            "schema_authority.external_schema_status must be 'unavailable' until an "
            "evaluator schema is supplied and an exporter maps to it"
        )
    if authority["external_conformance_claimed"] is not False:
        raise EnvelopeError("conformance to an unavailable external schema is claimed")
    if envelope["result_kind"] != RESULT_KIND:
        raise EnvelopeError(f"result_kind must be {RESULT_KIND!r}")

    run = envelope["run"]
    _expect_exact_keys(run, RUN_KEYS, "run")
    if run["mode"] not in MODES:
        raise EnvelopeError(f"run.mode must be one of {MODES}")
    if run["tier"] not in TIERS:
        raise EnvelopeError(f"run.tier must be one of {TIERS}")

    status = envelope["status"]
    if status not in RESULT_STATUSES:
        raise EnvelopeError(f"status must be one of {RESULT_STATUSES}")
    if status == "ok":
        raise EnvelopeError(
            f"status 'ok' is not allowed in {RESULT_SCHEMA_VERSION}: this "
            "diagnostic schema has no producer for any contract section and "
            "cannot certify a complete result"
        )
    _expect_reason(envelope, "status_reason", "envelope")
    exit_code = envelope["exit_code"]
    allowed_exits = _expected_exit_code(status, run["mode"])
    if type(exit_code) is not int or exit_code not in allowed_exits:
        raise EnvelopeError(
            f"exit_code {exit_code!r} does not match status {status!r} "
            f"(expected one of {sorted(allowed_exits)})"
        )

    coverage = envelope["coverage"]
    _expect_exact_keys(coverage, {"contract_complete", "sections"}, "coverage")
    if coverage["contract_complete"] is not False:
        raise EnvelopeError(
            f"coverage.contract_complete must be false in {RESULT_SCHEMA_VERSION}: "
            "no contract section can be produced or validated"
        )
    sections = coverage["sections"]
    records = envelope["records"]
    if records is not None:
        _expect_exact_keys(records, {"path", "sha256", "schema_version"}, "records")
        if records["schema_version"] != RECORDS_SCHEMA_VERSION:
            raise EnvelopeError(
                f"records.schema_version must be {RECORDS_SCHEMA_VERSION}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", str(records["sha256"])):
            raise EnvelopeError("records.sha256 must be a SHA-256 hex digest")
        if PurePosixPath(str(records["path"])).is_absolute() or ".." in str(
            records["path"]
        ):
            raise EnvelopeError("records.path must be relative to the run directory")
    _expect_exact_keys(sections, frozenset(CONTRACT_SECTIONS), "coverage.sections")
    export = sections["public_schema_export"]
    if authority["external_schema_status"] == "unavailable" and not (
        isinstance(export, dict) and export.get("status") in _NO_EXPORT
    ):
        raise EnvelopeError(
            "coverage.sections.public_schema_export must be 'blocked_external' or "
            "'unavailable' while schema_authority.external_schema_status is "
            "'unavailable'"
        )
    for name, fields in CONTRACT_SECTIONS.items():
        section = sections[name]
        where = f"coverage.sections.{name}"
        _expect_exact_keys(section, {"status", "reason"}, where)
        if section["status"] not in SECTION_STATUSES:
            raise EnvelopeError(f"{where}: status must be one of {SECTION_STATUSES}")
        if section["status"] in ("available", "partial") and records is None:
            raise EnvelopeError(
                f"{where}: status {section['status']!r} needs a records file that "
                "substantiates it; this envelope has none"
            )
        _expect_reason(section, "reason", where)
        for field_name in fields:
            payload = envelope[field_name]
            empty = [] if field_name not in NULLABLE_FIELDS else None
            if payload != empty:
                # Records live in the validated records file, never inline.
                raise EnvelopeError(
                    f"{field_name}: the envelope carries no entity records; they "
                    "belong in the validated records file"
                )

    stages = envelope["stages"]
    if not isinstance(stages, list) or not stages:
        raise EnvelopeError("stages must be a non-empty list")
    # A tuple, not a set: membership must not hash a malformed (unhashable) name.
    known_stages = (INVENTORY_STAGE, REPLAY_STAGE, *PIPELINE_STAGES)
    for index, stage in enumerate(stages):
        where = f"stages[{index}]"
        _expect_exact_keys(stage, {"name", "status", "reason"}, where)
        if stage["name"] not in known_stages:
            raise EnvelopeError(f"{where}: unknown stage {stage['name']!r}")
        if stage["status"] not in STAGE_STATUSES:
            raise EnvelopeError(f"{where}: status must be one of {STAGE_STATUSES}")
        _expect_reason(stage, "reason", where)
        implemented = (
            IMPLEMENTED_STAGES.get(run["tier"], ()) if run["mode"] == "live" else ()
        )
        if (
            stage["status"] == "ok"
            and stage["name"] != INVENTORY_STAGE
            and stage["name"] not in implemented
        ):
            raise EnvelopeError(
                f"{where}: stage {stage['name']!r} reports 'ok', but no such stage "
                f"is implemented for a {run['mode']} {run['tier']} run"
            )

    gates = envelope["gates"]
    _expect_exact_keys(gates, {"registry", "summary", "items"}, "gates")
    _expect_exact_keys(
        gates["registry"], {"path", "sha256", "registry_version"}, "gates.registry"
    )
    items = gates["items"]
    if not isinstance(items, list) or not items:
        raise EnvelopeError("gates.items must be a non-empty list")
    for index, item in enumerate(items):
        where = f"gates.items[{index}]"
        _expect_exact_keys(item, {"id", "status", "reason"}, where)
        if item["status"] not in GATE_STATUSES:
            raise EnvelopeError(f"{where}: status must be one of {GATE_STATUSES}")
        if item["status"] in ("measured_pass", "measured_fail"):
            raise EnvelopeError(
                f"{where}: gate {item['id']!r} reports {item['status']!r}; gate "
                "verdicts come only from a scored benchmark run, never from "
                "`floscan run`"
            )
        _expect_reason(item, "reason", where)
    if gates["summary"] != summarize_gate_statuses(items):
        raise EnvelopeError("gates.summary disagrees with gates.items")

    diagnostics = envelope["diagnostics"]
    if not isinstance(diagnostics, list):
        raise EnvelopeError("diagnostics must be a list")
    for index, entry in enumerate(diagnostics):
        _expect_exact_keys(entry, {"code", "message"}, f"diagnostics[{index}]")
