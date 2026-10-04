"""Content-addressed references to files inside a capture or run root.

Inputs are only ever opened for reading here; nothing in this module writes
to a capture directory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from floscan.contracts.base import AssetRef

CHUNK_BYTES = 8 * 1024 * 1024
MIME_TYPES = {
    ".csv": "text/csv",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".jsonl": "application/jsonl",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".npy": "application/x-npy",
    ".png": "image/png",
}


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def mime_type(path: PurePosixPath | Path) -> str:
    return MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")


def asset_ref(
    root: Path,
    relative: str,
    sha256: str | None = None,
    byte_count: int | None = None,
    **extra: object,
) -> AssetRef:
    """AssetRef for ``root / relative``; hashes the file unless given its hash."""
    path = root / relative
    return AssetRef(
        uri=relative,
        sha256=sha256 if sha256 is not None else sha256_file(path),
        byte_count=byte_count if byte_count is not None else path.stat().st_size,
        mime_type=mime_type(PurePosixPath(relative)),
        **extra,
    )
