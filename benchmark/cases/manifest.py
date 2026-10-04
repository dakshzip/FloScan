"""Benchmark case manifests: what was collected, per split, and what is missing.

Manifests reference evidence by relative path and content hash; they never hold
ground-truth values. Composition (the assignment's minimum benchmark) is
computed from the recorded evidence, so a case cannot be declared complete
while anything is missing.

    uv run python -m benchmark.cases.manifest status      # missing-evidence checklist
    uv run python -m benchmark.cases.manifest hash DIR    # hashes of a capture

Two kinds of hash are recorded. ``raw_manifest_hash`` is provenance: it pins
the exact folder tree, names included. ``media_sha256`` is recording identity:
the content digests of the photo and video files, names ignored, so a renamed
or re-foldered copy is still recognised as the same recording. Distinct media
does not prove a fresh sensor session; that rests on the operator's record
(device, app, date), which stays visible in the status report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
# Photo and video files identify a recording. Depth and confidence PNGs do not:
# the two supplied sessions share byte-identical PNG frames although they are
# different recordings (runs/p05a-evidence/png-identity.log).
MEDIA_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".heic", ".heif", ".dng", ".mov", ".mp4", ".m4v"}
)


def _file_digests(directory: Path) -> list[tuple[str, str]]:
    """(relative POSIX path, SHA-256) of every file, sorted by path."""
    digests = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if path.name in _IGNORED_FILES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        digests.append((path.relative_to(directory).as_posix(), digest.hexdigest()))
    return digests


def raw_manifest_hash(directory: Path) -> str:
    """Provenance hash: SHA-256 over every file's relative path and content hash.

    It pins the exact tree, so renaming a file inside the folder changes it.
    It is not recording identity; see ``media_digests``.
    """
    return _raw_hash(_file_digests(directory))


def media_digests(directory: Path) -> list[str]:
    """Recording identity: sorted content digests of every photo and video file.

    File and folder names are ignored and repeated content is kept, so a copy
    stays recognisable after renaming, and a padded photo set shows its repeats.
    """
    return _media(_file_digests(directory))


def _raw_hash(digests: Sequence[tuple[str, str]]) -> str:
    lines = "".join(f"{rel}\t{digest}\n" for rel, digest in digests)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _media(digests: Sequence[tuple[str, str]]) -> list[str]:
    return sorted(
        digest for rel, digest in digests if Path(rel).suffix.lower() in MEDIA_SUFFIXES
    )


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


class PhotoGroup(Contract):
    """One photo folder (album) and the sketch room it shows."""

    folder: Text
    room_id: Id = Field(pattern=r"^R\d{2}$")
    count: int


class CaptureEntry(Contract):
    """One recording. ``missing`` entries are placeholders for planned captures.

    ``room_ids`` lists the sketch rooms (connectors included) the recording
    covers; for photos it must match ``photo_groups``. A received whole-property
    capture with no ``room_ids`` is kept but counts as unrecorded coverage.
    """

    capture_id: Id
    tier: Tier
    scope: Literal["whole_property", "room"]
    room_ids: list[Id] = Field(default_factory=list)
    purpose: Literal["primary", "repeat"]
    repeat_of: Id | None = None
    status: Literal["missing", "received", "rejected"]
    path: Text | None = None
    raw_manifest_hash: Sha256 | None = None
    media_sha256: list[Sha256] = Field(default_factory=list)
    device_model: Text | None = None
    app: AppInfo | None = None
    recorded_on: Text | None = None
    photo_groups: list[PhotoGroup] = Field(default_factory=list)
    reason: Text | None = None

    @model_validator(mode="after")
    def _evidence(self) -> CaptureEntry:
        name = self.capture_id
        if len(self.room_ids) != len(set(self.room_ids)):
            raise ValueError(f"{name}: room_ids lists a room twice")
        # A planned (missing) room capture may not know its room until the sketch.
        if (
            self.scope == "room"
            and self.status != "missing"
            and len(self.room_ids) != 1
        ):
            raise ValueError(f"{name}: a room capture names exactly one room")
        if self.purpose == "repeat" and self.repeat_of is None:
            raise ValueError(f"{name}: a repeat names the capture it repeats")
        if self.purpose == "primary" and self.repeat_of is not None:
            raise ValueError(f"{name}: a primary capture does not repeat another")
        if self.repeat_of == name:
            raise ValueError(f"{name}: a recording cannot repeat itself")
        if self.status == "rejected" and self.reason is None:
            raise ValueError(f"{name}: a rejected capture needs a reason")
        if self.status == "received":
            missing = [
                field
                for field in (
                    "path",
                    "raw_manifest_hash",
                    "device_model",
                    "app",
                    "recorded_on",
                )
                if getattr(self, field) is None
            ]
            if not self.media_sha256:
                missing.append("media_sha256")
            if self.app is not None and self.app.version is None:
                missing.append("app.version")
            if missing:
                raise ValueError(f"{name}: received capture lacks {missing}")
            if len(self.media_sha256) != len(set(self.media_sha256)):
                raise ValueError(
                    f"{name}: the same media file appears twice; a copy is not "
                    "another photo or recording"
                )
            if self.tier == "photo":
                self._check_photo_groups()
        return self

    def _check_photo_groups(self) -> None:
        name = self.capture_id
        low, high = PHOTOS_PER_ROOM
        if not self.photo_groups:
            raise ValueError(f"{name}: photo capture needs per-room photo groups")
        folders = [g.folder for g in self.photo_groups]
        rooms = [g.room_id for g in self.photo_groups]
        if len(folders) != len(set(folders)) or len(rooms) != len(set(rooms)):
            raise ValueError(f"{name}: each photo folder maps to its own room")
        for group in self.photo_groups:
            if not low <= group.count <= high:
                raise ValueError(
                    f"{name}: {group.folder} ({group.room_id}) has {group.count} "
                    f"photos; the photo tier takes {low} to {high} per room"
                )
        if set(rooms) != set(self.room_ids):
            raise ValueError(
                f"{name}: room_ids {sorted(self.room_ids)} must match the photo "
                f"groups {sorted(rooms)}"
            )
        total = sum(g.count for g in self.photo_groups)
        if total != len(self.media_sha256):
            raise ValueError(
                f"{name}: photo groups count {total} photos but media_sha256 "
                f"lists {len(self.media_sha256)}"
            )


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
        sketch = [r.id for r in self.rooms]
        if len(sketch) != len(set(sketch)):
            raise ValueError(f"{self.alias}: a sketch room id appears twice")
        rooms = set(sketch)
        by_id = {c.capture_id: c for c in self.captures}
        for capture in self.captures:
            unknown = set(capture.room_ids) - rooms
            if unknown:
                raise ValueError(
                    f"{capture.capture_id}: unknown rooms {sorted(unknown)}"
                )
            if capture.repeat_of is not None:
                _check_repeat_link(capture, by_id.get(capture.repeat_of))
        if self.incumbent is not None:
            unknown = set(self.incumbent.room_ids) - rooms
            if unknown:
                raise ValueError(
                    f"incumbent names rooms {sorted(unknown)} that are not in the "
                    "sketch"
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
        their rooms, all three tiers and tape ground truth. A whole-property
        tier counts only when one received primary recording covers every
        sketched room and connector.
        """
        benchmark = self.alias == "dev_property"
        received = [c for c in self.captures if c.status == "received"]
        by_id = {c.capture_id: c for c in self.captures}
        rooms = [r for r in self.rooms if r.kind == "room"]
        connectors = [r for r in self.rooms if r.kind == "connector"]
        sketch = {r.id for r in self.rooms}

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
            candidates = [
                c
                for c in received
                if c.tier == tier
                and c.scope == "whole_property"
                and c.purpose == "primary"
            ]
            covering = [
                c.capture_id for c in candidates if sketch and sketch <= set(c.room_ids)
            ]
            details = [_coverage_detail(c, sketch) for c in candidates]
            items.append(
                item(
                    f"whole-property {tier} capture received",
                    bool(covering),
                    "; ".join(details) or "not received",
                )
            )
        if benchmark:
            repeats = []
            for capture in received:
                original = by_id.get(capture.repeat_of or "")
                if original is None or original.status != "received":
                    continue
                shared = sorted(set(capture.room_ids) & set(original.room_ids))
                if shared:
                    repeats.append(
                        f"{capture.capture_id} ({capture.recorded_on}) repeats "
                        f"{original.capture_id} ({original.recorded_on}) in "
                        f"{', '.join(shared)}; distinct media, session per "
                        "operator record"
                    )
            items.append(
                item(
                    "one room captured twice at the same tier",
                    bool(repeats),
                    "; ".join(repeats) or "not received",
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
            items.append(
                item(
                    "incumbent export for two rooms received",
                    self.incumbent is not None
                    and self.incumbent.status == "received"
                    and self.incumbent.export_kind == "app_export",
                    _incumbent_detail(self.incumbent),
                )
            )
        return items


def _check_repeat_link(capture: CaptureEntry, original: CaptureEntry | None) -> None:
    """A repeat names a distinct primary recording of the same tier and room."""
    name = capture.capture_id
    if original is None or original.tier != capture.tier:
        raise ValueError(
            f"{name}: repeats unknown or other-tier capture {capture.repeat_of}"
        )
    # Repeats of repeats are refused, so links cannot form chains or cycles.
    if original.purpose != "primary":
        raise ValueError(
            f"{name}: repeats {original.capture_id}, which is itself a repeat; "
            "name the primary recording"
        )
    if (
        capture.status == "received"
        and original.status == "received"
        and capture.room_ids
        and original.room_ids
        and not set(capture.room_ids) & set(original.room_ids)
    ):
        raise ValueError(
            f"{name}: covers {sorted(capture.room_ids)}, which {original.capture_id} "
            f"does not ({sorted(original.room_ids)}); a repeat records the same room"
        )


def _coverage_detail(capture: CaptureEntry, sketch: set[str]) -> str:
    if not capture.room_ids:
        return f"{capture.capture_id}: room coverage not recorded"
    if not sketch:
        return f"{capture.capture_id}: no rooms sketched"
    missing = sorted(sketch - set(capture.room_ids))
    if missing:
        return f"{capture.capture_id}: missing {', '.join(missing)}"
    return f"{capture.capture_id}: covers all {len(sketch)} sketched rooms"


def _incumbent_detail(incumbent: IncumbentStatus | None) -> str:
    if incumbent is None:
        return "not planned"
    detail = f"{incumbent.app.name}: {incumbent.status}"
    if incumbent.status == "received":
        detail += f" for {', '.join(incumbent.room_ids)}"
        if incumbent.export_kind == "transcription":
            detail += " as a transcription (supporting evidence, not the app's export)"
    return detail


def _reject_duplicate_recordings(captures: Sequence[CaptureEntry]) -> None:
    """The same folder, tree or media content is one recording, not two."""
    seen: dict[str, str] = {}
    for capture in captures:
        if capture.status != "received":
            continue
        keys = [capture.path, capture.raw_manifest_hash, *capture.media_sha256]
        for key in keys:
            if key is None:
                continue
            if key in seen and seen[key] != capture.capture_id:
                raise ValueError(
                    f"{capture.capture_id} is the same recording as {seen[key]}; "
                    "a copy, even renamed, cannot count as a separate capture or a "
                    "repeat"
                )
            seen[key] = capture.capture_id


class Fixture(Contract):
    """Development-only input: never benchmark evidence."""

    id: Id
    path: Text
    raw_manifest_hash: Sha256
    media_sha256: list[Sha256] = Field(min_length=1)
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
        seen: dict[str, Fixture] = {}
        for fixture in self.fixtures:
            keys = [fixture.raw_manifest_hash, *fixture.media_sha256]
            first = next((seen[k] for k in keys if k in seen), None)
            if first is not None and fixture.duplicate_of != first.id:
                raise ValueError(
                    f"fixture {fixture.id} shares content with {first.id} and must "
                    "say it is a duplicate"
                )
            for key in keys:
                seen.setdefault(key, fixture)
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
    """A property, alias or recording appears in one split only.

    A development fixture is never a capture of any case.
    """
    owner: dict[str, str] = {}
    aliases: set[str] = set()
    fixtures = {
        key: fixture.id
        for manifest in manifests.values()
        for fixture in manifest.fixtures
        for key in (fixture.raw_manifest_hash, *fixture.media_sha256)
    }
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
                if capture.status != "received":
                    continue
                where = f"{split}/{case.alias}/{capture.capture_id}"
                for key in (capture.raw_manifest_hash, *capture.media_sha256):
                    if key is None:
                        continue
                    if key in fixtures:
                        raise ValueError(
                            f"{where} is development fixture {fixtures[key]}; "
                            "fixtures are never benchmark captures"
                        )
                    if recordings.get(key, where) != where:
                        raise ValueError(
                            f"{capture.capture_id} duplicates {recordings[key]} "
                            "across cases"
                        )
                    recordings[key] = where


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
    digest = commands.add_parser(
        "hash", help="provenance hash and media digests of a capture folder"
    )
    digest.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "hash":
        if not args.directory.is_dir():
            print(f"error: {args.directory} is not a directory", file=sys.stderr)
            return 1
        digests = _file_digests(args.directory)
        record = {
            "raw_manifest_hash": _raw_hash(digests),
            "media_sha256": _media(digests),
            "file_count": len(digests),
        }
        print(json.dumps(record, indent=2))
        return 0
    try:
        print(status_report(load_manifests(args.dir)), end="")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
