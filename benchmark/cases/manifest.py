"""Benchmark case manifests: what was collected, per split, and what is missing.

Manifests reference evidence by relative path and content hash; they never hold
ground-truth values. Composition (the assignment's minimum benchmark) is
computed from the recorded evidence, so a case cannot be declared complete
while anything is missing.

    uv run python -m benchmark.cases.manifest status      # missing-evidence checklist
    uv run python -m benchmark.cases.manifest hash DIR    # raw hash of a capture
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from floscan.contracts.base import Contract, Id, Sha256, Text, Tier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_ROOT / "benchmark" / "manifests"

Alias = Literal[
    "dev_property", "calibration_set", "heldout_property", "walkin_property"
]
Split = Literal["development", "calibration", "heldout", "walkin"]
ALIAS_SPLIT: dict[str, str] = {
    "dev_property": "development",
    "calibration_set": "calibration",
    "heldout_property": "heldout",
    "walkin_property": "walkin",
}
MANIFEST_FILES = {
    "development": "development.json",
    "calibration": "calibration.json",
    "heldout": "heldout.json",
}
TIERS: tuple[str, ...] = ("photo", "video", "lidar")
PHOTOS_PER_ROOM = (2, 8)
DAMAGE_CLASSES = ("stain", "crack")
_IGNORED_FILES = {".DS_Store"}


def raw_manifest_hash(directory: Path) -> str:
    """SHA-256 over every file's relative POSIX path and content hash.

    Two recordings with this hash equal are byte-identical, whatever their
    folder names, so a copy can never pass as an independent capture.
    """
    lines = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if path.name in _IGNORED_FILES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        lines.append(
            f"{path.relative_to(directory).as_posix()}\t{digest.hexdigest()}\n"
        )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


class AppInfo(Contract):
    name: Text
    version: Text | None = None


class RoomPlan(Contract):
    """A room from the operator's sketch (IDs as in the measurement sheet)."""

    id: Id = Field(pattern=r"^R\d{2}$")
    label: Text
    kind: Literal["room", "connector"]
    furnished: bool = False
    staged_damage_classes: list[Literal["stain", "crack"]] = Field(default_factory=list)


class CaptureEntry(Contract):
    """One recording. ``missing`` entries are placeholders for planned captures."""

    capture_id: Id
    tier: Tier
    scope: Literal["whole_property", "room"]
    room_ids: list[Id] = Field(default_factory=list)
    purpose: Literal["primary", "repeat"]
    repeat_of: Id | None = None
    status: Literal["missing", "received", "rejected"]
    path: Text | None = None
    raw_manifest_hash: Sha256 | None = None
    device_model: Text | None = None
    app: AppInfo | None = None
    recorded_on: Text | None = None
    photo_counts: dict[str, int] = Field(default_factory=dict)
    reason: Text | None = None

    @model_validator(mode="after")
    def _evidence(self) -> CaptureEntry:
        # A planned (missing) room capture may not know its room until the sketch.
        if (
            self.scope == "room"
            and self.status != "missing"
            and len(self.room_ids) != 1
        ):
            raise ValueError(
                f"{self.capture_id}: a room capture names exactly one room"
            )
        if self.purpose == "repeat" and self.repeat_of is None:
            raise ValueError(
                f"{self.capture_id}: a repeat names the capture it repeats"
            )
        if self.status == "rejected" and self.reason is None:
            raise ValueError(f"{self.capture_id}: a rejected capture needs a reason")
        if self.status == "received":
            missing = [
                name
                for name in (
                    "path",
                    "raw_manifest_hash",
                    "device_model",
                    "app",
                    "recorded_on",
                )
                if getattr(self, name) is None
            ]
            if self.app is not None and self.app.version is None:
                missing.append("app.version")
            if missing:
                raise ValueError(f"{self.capture_id}: received capture lacks {missing}")
            if self.tier == "photo":
                low, high = PHOTOS_PER_ROOM
                if not self.photo_counts:
                    raise ValueError(
                        f"{self.capture_id}: photo capture needs per-room counts"
                    )
                for room, count in self.photo_counts.items():
                    if not low <= count <= high:
                        raise ValueError(
                            f"{self.capture_id}: {room} has {count} photos; the photo "
                            f"tier takes {low} to {high} per room"
                        )
        return self


class GroundTruthStatus(Contract):
    status: Literal["missing", "partial", "complete"]
    measurements_file: Text | None = None
    measurements_sha256: Sha256 | None = None
    ground_truth_file: Text | None = None
    ground_truth_sha256: Sha256 | None = None
    instrument: Text | None = None
    operator: Text | None = None
    missing_items: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def _evidence(self) -> GroundTruthStatus:
        if self.status in ("partial", "complete") and (
            self.measurements_file is None or self.measurements_sha256 is None
        ):
            raise ValueError("ground truth marked present needs its measurements file")
        if self.status == "complete":
            missing = [
                n
                for n in (
                    "ground_truth_file",
                    "ground_truth_sha256",
                    "instrument",
                    "operator",
                )
                if getattr(self, n) is None
            ]
            if missing or self.missing_items:
                raise ValueError(
                    "ground truth marked complete but lacks "
                    f"{missing + self.missing_items}"
                )
        return self


class IncumbentStatus(Contract):
    app: AppInfo
    status: Literal["missing", "received"]
    room_ids: list[Id] = Field(default_factory=list)
    export_files: list[Text] = Field(default_factory=list)
    export_sha256: list[Sha256] = Field(default_factory=list)
    export_kind: Literal["app_export", "transcription"] | None = None

    @model_validator(mode="after")
    def _evidence(self) -> IncumbentStatus:
        if self.status == "received":
            if len(self.room_ids) != 2 or self.room_ids[0] == self.room_ids[1]:
                raise ValueError("an incumbent comparison covers two distinct rooms")
            if not self.export_files or len(self.export_files) != len(
                self.export_sha256
            ):
                raise ValueError(
                    "received incumbent output needs files and their hashes"
                )
            if self.app.version is None or self.export_kind is None:
                raise ValueError("received incumbent output needs app version and kind")
        return self


class Case(Contract):
    alias: Alias
    split: Split
    property_id: Id
    description: Text
    rooms: list[RoomPlan] = Field(default_factory=list)
    captures: list[CaptureEntry] = Field(default_factory=list)
    ground_truth: GroundTruthStatus
    incumbent: IncumbentStatus | None = None
    declared_complete: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Case:
        if ALIAS_SPLIT[self.alias] != self.split:
            raise ValueError(
                f"case {self.alias} belongs to split {ALIAS_SPLIT[self.alias]}"
            )
        ids = [c.capture_id for c in self.captures]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.alias}: duplicate capture ids")
        rooms = {r.id for r in self.rooms}
        by_id = {c.capture_id: c for c in self.captures}
        for capture in self.captures:
            unknown = set(capture.room_ids) - rooms
            if unknown:
                raise ValueError(
                    f"{capture.capture_id}: unknown rooms {sorted(unknown)}"
                )
            if capture.repeat_of is not None:
                original = by_id.get(capture.repeat_of)
                if original is None or original.tier != capture.tier:
                    raise ValueError(
                        f"{capture.capture_id}: repeats unknown or other-tier capture "
                        f"{capture.repeat_of}"
                    )
        _reject_duplicate_recordings(self.captures)
        if self.declared_complete and not all(i["met"] for i in self.composition()):
            unmet = [i["item"] for i in self.composition() if not i["met"]]
            raise ValueError(f"{self.alias} is declared complete but misses {unmet}")
        return self

    def composition(self) -> list[dict[str, Any]]:
        """Required evidence, item by item, computed from what was recorded.

        The benchmark property (``dev_property``) carries the assignment's full
        minimum composition. Calibration, held-out and walk-in spaces need
        their rooms, all three tiers and tape ground truth.
        """
        benchmark = self.alias == "dev_property"
        received = [c for c in self.captures if c.status == "received"]
        rooms = [r for r in self.rooms if r.kind == "room"]
        connectors = [r for r in self.rooms if r.kind == "connector"]

        def item(name: str, met: bool, detail: str) -> dict[str, Any]:
            return {"item": name, "met": met, "detail": detail}

        counts = f"{len(rooms)} rooms, {len(connectors)} connectors"
        if benchmark:
            items = [
                item(
                    "at least 3 rooms plus a connector sketched",
                    len(rooms) >= 3 and bool(connectors),
                    counts,
                )
            ]
            staged = {
                r.id: sorted(set(r.staged_damage_classes))
                for r in self.rooms
                if r.staged_damage_classes
            }
            items.append(
                item(
                    "furnished room with staged damage of two classes",
                    any(
                        r.furnished and len(set(r.staged_damage_classes)) >= 2
                        for r in self.rooms
                    ),
                    ", ".join(f"{k}: {v}" for k, v in staged.items()) or "none staged",
                )
            )
        else:
            items = [item("rooms sketched", bool(self.rooms), counts)]
        for tier in TIERS:
            have = [
                c.capture_id
                for c in received
                if c.tier == tier and c.scope == "whole_property"
            ]
            items.append(
                item(
                    f"whole-property {tier} capture received",
                    bool(have),
                    ", ".join(have) or "not received",
                )
            )
        if benchmark:
            repeats = [
                f"{c.capture_id} repeats {c.repeat_of}"
                for c in received
                if c.purpose == "repeat"
                and any(
                    o.capture_id == c.repeat_of and o.status == "received"
                    for o in self.captures
                )
            ]
            items.append(
                item(
                    "one room captured twice at the same tier",
                    bool(repeats),
                    ", ".join(repeats) or "not received",
                )
            )
        gt = self.ground_truth
        items.append(
            item(
                "tape ground truth complete",
                gt.status == "complete",
                "; ".join([gt.status, *gt.missing_items]),
            )
        )
        if benchmark:
            incumbent = self.incumbent
            items.append(
                item(
                    "incumbent export for two rooms received",
                    incumbent is not None and incumbent.status == "received",
                    "not planned"
                    if incumbent is None
                    else f"{incumbent.app.name}: {incumbent.status}",
                )
            )
        return items


def _reject_duplicate_recordings(captures: Sequence[CaptureEntry]) -> None:
    """Identical content or the same folder is one recording, not two."""
    seen: dict[str, str] = {}
    for capture in captures:
        if capture.status != "received":
            continue
        for key in (capture.raw_manifest_hash, capture.path):
            if key is None:
                continue
            if key in seen:
                raise ValueError(
                    f"{capture.capture_id} is the same recording as {seen[key]}; "
                    "a copy "
                    "cannot count as a separate capture or a repeat"
                )
            seen[key] = capture.capture_id


class Fixture(Contract):
    """Development-only input: never benchmark evidence."""

    id: Id
    path: Text
    raw_manifest_hash: Sha256
    file_count: int = Field(ge=1)
    kind: Literal["stray_lidar_session"]
    usable_as: list[Literal["lidar_development", "video_development"]]
    duplicate_of: Id | None = None
    notes: Text


class CaseManifest(Contract):
    manifest_version: Literal[1]
    split: Literal["development", "calibration", "heldout"]
    cases: list[Case] = Field(default_factory=list)
    fixtures: list[Fixture] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> CaseManifest:
        allowed = {self.split} | ({"walkin"} if self.split == "heldout" else set())
        for case in self.cases:
            if case.split not in allowed:
                raise ValueError(f"case {case.alias} does not belong in {self.split}")
        hashes: dict[str, Fixture] = {}
        for fixture in self.fixtures:
            first = hashes.get(fixture.raw_manifest_hash)
            if first is not None and fixture.duplicate_of != first.id:
                raise ValueError(
                    f"fixture {fixture.id} is byte-identical to {first.id} and must "
                    "say so"
                )
            hashes.setdefault(fixture.raw_manifest_hash, fixture)
        return self


def load_manifests(directory: Path = MANIFEST_DIR) -> dict[str, CaseManifest]:
    """Load every split and check cross-split rules."""
    manifests = {
        split: CaseManifest.model_validate_json((directory / name).read_text("utf-8"))
        for split, name in MANIFEST_FILES.items()
    }
    check_splits(manifests)
    return manifests


def check_splits(manifests: dict[str, CaseManifest]) -> None:
    """A property, alias or recording appears in one split only."""
    owner: dict[str, str] = {}
    aliases: set[str] = set()
    recordings: dict[str, str] = {}
    for split, manifest in manifests.items():
        for case in manifest.cases:
            if case.alias in aliases:
                raise ValueError(f"case alias {case.alias} appears twice")
            aliases.add(case.alias)
            if case.property_id in owner and owner[case.property_id] != case.alias:
                raise ValueError(
                    f"property {case.property_id} is in both {owner[case.property_id]} "
                    f"and {case.alias}; keep each property in one split"
                )
            owner[case.property_id] = case.alias
            for capture in case.captures:
                key = capture.raw_manifest_hash
                if capture.status == "received" and key is not None:
                    if key in recordings:
                        raise ValueError(
                            f"{capture.capture_id} duplicates {recordings[key]} "
                            "across cases"
                        )
                    recordings[key] = f"{split}/{case.alias}/{capture.capture_id}"


def status_report(manifests: dict[str, CaseManifest]) -> str:
    """Human-readable collection status and missing-evidence checklist."""
    lines = ["# Benchmark collection status", ""]
    for split, manifest in manifests.items():
        for case in manifest.cases:
            items = case.composition()
            done = sum(i["met"] for i in items)
            lines.append(
                f"## {case.alias} ({split}, property {case.property_id}): "
                f"{done}/{len(items)} met"
            )
            for item in items:
                lines.append(
                    f"- [{'x' if item['met'] else ' '}] {item['item']} "
                    f"({item['detail']})"
                )
            lines.append("")
        for fixture in manifest.fixtures:
            lines.append(
                f"- fixture {fixture.id}: {fixture.kind}, development only"
                + (
                    f", duplicate of {fixture.duplicate_of}"
                    if fixture.duplicate_of
                    else ""
                )
            )
        if manifest.fixtures:
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.cases.manifest")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="print the missing-evidence checklist")
    status.add_argument("--dir", type=Path, default=MANIFEST_DIR)
    digest = commands.add_parser("hash", help="raw manifest hash of a capture folder")
    digest.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "hash":
        if not args.directory.is_dir():
            print(f"error: {args.directory} is not a directory", file=sys.stderr)
            return 1
        print(raw_manifest_hash(args.directory))
        return 0
    try:
        print(status_report(load_manifests(args.dir)), end="")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
