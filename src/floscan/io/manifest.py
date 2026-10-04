"""Raw file manifests of capture directories.

``raw_manifest_hash`` is provenance: SHA-256 over every file's relative POSIX
path and content hash, so it pins the exact tree. ``media_digests`` is
recording identity: the content hashes of photo and video files with names
ignored. Both use the same algorithm as ``benchmark/cases/manifest.py``
(tests compare them on the supplied samples); inference code never imports
the benchmark package.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from floscan.contracts.base import Contract
from floscan.io.assets import sha256_file

IGNORED_FILES = frozenset({".DS_Store"})
MEDIA_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".heic", ".heif", ".dng", ".mov", ".mp4", ".m4v"}
)


@dataclass(frozen=True)
class FileEntry:
    """One file of a capture directory."""

    relative: str
    sha256: str
    byte_count: int


def file_manifest(directory: Path) -> list[FileEntry]:
    """Every file under ``directory`` (Finder metadata ignored), sorted by path."""
    entries = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if path.name in IGNORED_FILES:
            continue
        entries.append(
            FileEntry(
                path.relative_to(directory).as_posix(),
                sha256_file(path),
                path.stat().st_size,
            )
        )
    return entries


def raw_manifest_hash(entries: Sequence[FileEntry]) -> str:
    lines = "".join(f"{e.relative}\t{e.sha256}\n" for e in entries)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def media_digests(entries: Sequence[FileEntry]) -> list[str]:
    return sorted(
        e.sha256
        for e in entries
        if PurePosixPath(e.relative).suffix.lower() in MEDIA_SUFFIXES
    )


def check_new_output_dir(input_dir: Path, output_dir: Path) -> str | None:
    """Why ``output_dir`` cannot receive new outputs, or None if it can.

    Outputs never go inside the input (raw captures are immutable), and run
    directories are append-only, so an existing non-empty directory is refused.
    """
    source = input_dir.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    if target == source or target.is_relative_to(source):
        return f"output {str(target)!r} is inside the input {str(source)!r}"
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        return f"output {str(target)!r} already exists and is not an empty directory"
    return None


def write_capture_outputs(
    output_dir: Path,
    capture: Contract,
    report: dict[str, Any],
    frames: Iterable[dict[str, Any]],
) -> list[Path]:
    """Write ``capture.json``, ``inspection.json`` and ``frames.jsonl``.

    Files are created exclusively, so nothing existing is ever overwritten.
    JSON is strict: a NaN or infinity raises ValueError instead of being
    written as a non-standard token.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (
        ("capture.json", capture.to_json() + "\n"),
        ("inspection.json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
    ):
        path = output_dir / name
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        written.append(path)
    path = output_dir / "frames.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for row in frames:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    written.append(path)
    return written
