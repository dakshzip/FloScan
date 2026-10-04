"""Pinned model checkpoints: lock, fetch, verify, offline loading and smoke runs.

Weights live outside Git under ``$FLOSCAN_MODELS_DIR`` (default
``<project>/models``), one directory per model. ``scripts/fetch_models.sh``
downloads each pinned file by exact revision and checks its size and SHA-256.
Inference only reads those files: before a model loads, its files are verified
again and outbound network connections are refused, so a missing or altered
checkpoint fails loudly instead of triggering a download.

Process isolation: on macOS the torch and pycolmap wheels each bundle their
own ``libomp``, and loading both into one process aborts with "OMP: Error
#15". The suggested ``KMP_DUPLICATE_LIB_OK`` workaround can silently produce
incorrect results, so it is not used. Model inference runs in a worker process
(``python -m floscan.runtime.models smoke ...``) that refuses to start if
pycolmap is already loaded. This module imports torch only inside functions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from floscan import __version__
from floscan.runtime import limits
from floscan.runtime.timing import StageTimer, accelerator_memory, peak_rss_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = PROJECT_ROOT / "configs" / "models.lock.json"
MODELS_DIR_ENV = "FLOSCAN_MODELS_DIR"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
_CHUNK_BYTES = 8 * 1024 * 1024


class ModelLockError(ValueError):
    """The model lock file is malformed."""


class ModelUnavailable(RuntimeError):
    """A pinned checkpoint is missing, incomplete or does not match the lock."""


class ModelRuntimeError(RuntimeError):
    """The process is not in a state where a model may be loaded."""


class NetworkForbidden(ConnectionError):
    """An outbound connection was attempted while network access is forbidden."""


# --------------------------------------------------------------------------
# Lock
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFile:
    """One pinned checkpoint file."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    """A selected model as pinned in the lock."""

    name: str
    repo: str
    revision: str
    files: tuple[ModelFile, ...]
    model_class: str
    processor_class: str


@dataclass(frozen=True)
class ModelLock:
    """The parsed model lock and the identity of its file."""

    path: Path
    sha256: str
    endpoint: str
    models: dict[str, ModelSpec]
    not_installed: tuple[dict[str, str], ...]


_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelLockError(message)


def _text(value: Any, where: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{where}: expected text")
    return value


def _keys(obj: Any, required: set[str], where: str) -> dict[str, Any]:
    _require(isinstance(obj, dict), f"{where}: expected an object")
    missing = required - obj.keys()
    unknown = obj.keys() - required
    _require(not missing, f"{where}: missing keys {sorted(missing)}")
    _require(not unknown, f"{where}: unknown keys {sorted(unknown)}")
    return obj


def _parse_file(entry: Any, where: str) -> ModelFile:
    _keys(entry, {"path", "size", "sha256"}, where)
    path = _text(entry["path"], f"{where}.path")
    pure = PurePosixPath(path)
    _require(
        not pure.is_absolute() and ".." not in pure.parts and "\\" not in path,
        f"{where}: path {path!r} must be relative and stay inside the model directory",
    )
    size = entry["size"]
    _require(type(size) is int and size > 0, f"{where}: size must be a positive int")
    sha = entry["sha256"]
    _require(
        isinstance(sha, str) and bool(_HEX64.fullmatch(sha)), f"{where}: bad sha256"
    )
    return ModelFile(path=path, size=size, sha256=sha)


def _parse_model(entry: Any, where: str) -> ModelSpec:
    _keys(
        entry,
        {
            "name",
            "status",
            "role",
            "loader",
            "checkpoint",
            "code_license",
            "weights_license",
        },
        where,
    )
    name = _text(entry["name"], f"{where}.name")
    where = f"model {name}"
    _require(entry["status"] == "selected", f"{where}: status must be 'selected'")
    _text(entry["role"], f"{where}.role")
    loader = _keys(
        entry["loader"],
        {"library", "model_class", "processor_class"},
        f"{where}.loader",
    )
    _require(
        loader["library"] == "transformers", f"{where}: loader must be transformers"
    )
    checkpoint = _keys(
        entry["checkpoint"], {"repo", "revision", "files"}, f"{where}.checkpoint"
    )
    revision = checkpoint["revision"]
    _require(
        isinstance(revision, str) and bool(_HEX40.fullmatch(revision)),
        f"{where}: revision must be a 40-character commit hash, not a branch",
    )
    files = checkpoint["files"]
    _require(isinstance(files, list) and bool(files), f"{where}: files must be a list")
    parsed = tuple(
        _parse_file(item, f"{where}.files[{index}]") for index, item in enumerate(files)
    )
    _require(
        len({item.path for item in parsed}) == len(parsed),
        f"{where}: duplicate file paths",
    )
    code = _keys(
        entry["code_license"],
        {"name", "source", "commit", "license_url", "license_sha256"},
        f"{where}.code_license",
    )
    _text(code["name"], f"{where}.code_license.name")
    _require(bool(_HEX40.fullmatch(str(code["commit"]))), f"{where}: bad code commit")
    _require(
        bool(_HEX64.fullmatch(str(code["license_sha256"]))),
        f"{where}: bad code licence sha256",
    )
    weights = entry["weights_license"]
    _require(isinstance(weights, dict), f"{where}.weights_license: expected an object")
    _text(weights.get("name"), f"{where}.weights_license.name")
    _require(
        weights.get("status") in ("declared", "conflicting_metadata"),
        f"{where}: weights licence status must be declared or conflicting_metadata",
    )
    evidence = weights.get("evidence")
    _require(
        isinstance(evidence, list) and bool(evidence),
        f"{where}: weights licence needs evidence",
    )
    return ModelSpec(
        name=name,
        repo=_text(checkpoint["repo"], f"{where}.checkpoint.repo"),
        revision=revision,
        files=parsed,
        model_class=_text(loader["model_class"], f"{where}.loader.model_class"),
        processor_class=_text(
            loader["processor_class"], f"{where}.loader.processor_class"
        ),
    )


def load_lock(path: Path = LOCK_PATH) -> ModelLock:
    """Read and validate the model lock.

    Raises:
        ModelLockError: if the lock is unreadable or malformed, including any
            revision that is a branch name rather than a commit hash.
    """
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ModelLockError(f"cannot read model lock {path}: {error}") from error
    _keys(
        data,
        {"lock_version", "description", "endpoint", "models", "not_installed"},
        "lock",
    )
    version = data["lock_version"]
    _require(type(version) is int and version == 1, "lock: unsupported lock_version")
    endpoint = _text(data["endpoint"], "lock.endpoint")
    _require(endpoint.startswith("https://"), "lock.endpoint must be https")
    entries = data["models"]
    _require(isinstance(entries, list) and bool(entries), "lock.models must be a list")
    models: dict[str, ModelSpec] = {}
    for index, entry in enumerate(entries):
        spec = _parse_model(entry, f"models[{index}]")
        _require(spec.name not in models, f"duplicate model {spec.name!r}")
        models[spec.name] = spec
    deferred = data["not_installed"]
    _require(isinstance(deferred, list), "lock.not_installed must be a list")
    for index, item in enumerate(deferred):
        _keys(item, {"name", "status", "reason"}, f"not_installed[{index}]")
        _require(
            item["status"] in ("conditional", "deferred"),
            f"not_installed[{index}]: status must be conditional or deferred",
        )
        _text(item["reason"], f"not_installed[{index}].reason")
    return ModelLock(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        endpoint=endpoint.rstrip("/"),
        models=models,
        not_installed=tuple(deferred),
    )


def models_dir(override: Path | None = None) -> Path:
    """Return the checkpoint root: override, then $FLOSCAN_MODELS_DIR, then default."""
    if override is not None:
        return override
    env = os.environ.get(MODELS_DIR_ENV)
    return Path(env) if env else DEFAULT_MODELS_DIR


# --------------------------------------------------------------------------
# Verification and fetching
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Stream ``path`` through SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(spec: ModelSpec, root: Path) -> dict[str, Any]:
    """Check every pinned file of ``spec`` under ``root`` by size and SHA-256.

    Never downloads anything. The overall status is ``ok``, ``missing``,
    ``size_mismatch`` or ``hash_mismatch`` (the worst file decides).
    """
    start = time.perf_counter()
    directory = root / spec.name
    files = []
    for item in spec.files:
        path = directory / item.path
        if not path.is_file():
            status = "missing"
        elif path.stat().st_size != item.size:
            status = "size_mismatch"
        elif sha256_file(path) != item.sha256:
            status = "hash_mismatch"
        else:
            status = "ok"
        files.append({"path": item.path, "status": status})
    statuses = {entry["status"] for entry in files}
    overall = next(
        (s for s in ("missing", "size_mismatch", "hash_mismatch") if s in statuses),
        "ok",
    )
    return {
        "name": spec.name,
        "status": overall,
        "directory": str(directory),
        "files": files,
        "seconds": time.perf_counter() - start,
    }


def require_verified(spec: ModelSpec, root: Path) -> dict[str, Any]:
    """Verify ``spec`` and raise ``ModelUnavailable`` unless every file matches."""
    report = verify_model(spec, root)
    if report["status"] != "ok":
        bad = [
            f"{f['path']} ({f['status']})"
            for f in report["files"]
            if f["status"] != "ok"
        ]
        raise ModelUnavailable(
            f"{spec.name}: checkpoint {report['status']} in {report['directory']}: "
            f"{', '.join(bad)}. Run scripts/fetch_models.sh; inference never downloads."
        )
    return report


def _file_url(endpoint: str, spec: ModelSpec, item: ModelFile) -> str:
    return (
        f"{endpoint}/{spec.repo}/resolve/{spec.revision}/"
        f"{urllib.parse.quote(item.path)}"
    )


def fetch_model(
    spec: ModelSpec,
    root: Path,
    endpoint: str,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Download missing or invalid files of ``spec`` and verify each one.

    Each file streams into ``<name>.partial`` and is renamed into place only
    after its size and SHA-256 match the lock; a mismatch deletes the partial
    file and raises ``ModelUnavailable``. Valid existing files are kept.
    """
    start = time.perf_counter()
    directory = root / spec.name
    downloaded_bytes = 0
    downloaded_files = 0
    for item in spec.files:
        target = directory / item.path
        if (
            target.is_file()
            and target.stat().st_size == item.size
            and sha256_file(target) == item.sha256
        ):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        url = _file_url(endpoint, spec, item)
        digest = hashlib.sha256()
        received = 0
        file_start = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with partial.open("wb") as out:
                    while chunk := response.read(_CHUNK_BYTES):
                        digest.update(chunk)
                        received += len(chunk)
                        out.write(chunk)
        except (OSError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            raise ModelUnavailable(
                f"{spec.name}: download of {url} failed: {error}"
            ) from error
        if received != item.size or digest.hexdigest() != item.sha256:
            partial.unlink(missing_ok=True)
            raise ModelUnavailable(
                f"{spec.name}: {item.path} from {url} does not match the lock "
                f"({received} bytes, sha256 {digest.hexdigest()}; expected "
                f"{item.size} bytes, sha256 {item.sha256})"
            )
        os.replace(partial, target)
        seconds = time.perf_counter() - file_start
        downloaded_bytes += received
        downloaded_files += 1
        log(
            f"  {spec.name}/{item.path}: {received / 1e6:.1f} MB in {seconds:.1f} s, "
            "sha256 verified"
        )
    return {
        "name": spec.name,
        "files_downloaded": downloaded_files,
        "bytes_downloaded": downloaded_bytes,
        "seconds": time.perf_counter() - start,
    }


# --------------------------------------------------------------------------
# Offline, isolated, sequential loading
# --------------------------------------------------------------------------


@contextmanager
def network_forbidden() -> Iterator[None]:
    """Refuse outbound connections and DNS lookups inside the block.

    Local (AF_UNIX) sockets stay usable. Restores the socket module on exit.
    """
    unix = getattr(socket, "AF_UNIX", None)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    original_create_connection = socket.create_connection

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise NetworkForbidden(
            "network access is forbidden during inference; models load only "
            "from verified local files"
        )

    def connect(self: socket.socket, address: Any) -> None:
        if unix is not None and self.family == unix:
            return original_connect(self, address)
        refuse()

    def connect_ex(self: socket.socket, address: Any) -> int:
        if unix is not None and self.family == unix:
            return original_connect_ex(self, address)
        refuse()
        return 1  # Unreachable: refuse() always raises.

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = refuse  # type: ignore[assignment]
    socket.create_connection = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]


def ensure_model_process() -> None:
    """Refuse to load models in a process where pycolmap is already loaded."""
    if "pycolmap" in sys.modules:
        raise ModelRuntimeError(
            "pycolmap is loaded in this process. Its bundled libomp conflicts with "
            "torch's (OMP Error #15 aborts the process on macOS), so model "
            "inference must run in a separate worker process."
        )


def _set_offline_environment() -> None:
    # Must be set before huggingface_hub/transformers are first imported.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _empty_cache(torch: Any, device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


_ACTIVE_MODEL: str | None = None


@contextmanager
def loaded_model(
    spec: ModelSpec, root: Path, device: str, timer: StageTimer | None = None
) -> Iterator[tuple[Any, Any]]:
    """Load ``spec`` from verified local files onto ``device``; free it on exit.

    Only one model may be loaded per process at a time, which bounds memory
    on 16 GB machines and 10 GB GPUs. Files are re-verified before loading and
    the network is refused while transformers reads them.

    Raises:
        ModelRuntimeError: if another model is loaded or pycolmap is present.
        ModelUnavailable: if any checkpoint file is missing or altered.
    """
    global _ACTIVE_MODEL
    if _ACTIVE_MODEL is not None:
        raise ModelRuntimeError(
            f"cannot load {spec.name} while {_ACTIVE_MODEL} is loaded; models load "
            "one at a time"
        )
    ensure_model_process()
    stages = timer if timer is not None else StageTimer()
    with stages.stage("verify_checkpoint"):
        require_verified(spec, root)
    _set_offline_environment()
    import torch
    import transformers

    directory = root / spec.name
    model_class = getattr(transformers, spec.model_class)
    processor_class = getattr(transformers, spec.processor_class)
    _ACTIVE_MODEL = spec.name
    model: Any = None
    processor: Any = None
    try:
        with stages.stage("load"), network_forbidden():
            processor = processor_class.from_pretrained(
                directory, local_files_only=True
            )
            model = model_class.from_pretrained(
                directory, local_files_only=True, dtype=torch.float32
            )
            model = model.to(device).eval()
        yield model, processor
    finally:
        model = None
        processor = None
        gc.collect()
        _empty_cache(torch, device)
        _ACTIVE_MODEL = None


# --------------------------------------------------------------------------
# Smoke inference
# --------------------------------------------------------------------------


def synthetic_room_image() -> tuple[Any, tuple[int, int, int, int]]:
    """Draw a deterministic 640x480 room scene owned by this project.

    Returns the PIL image and the pixel box (x0, y0, x1, y1) of its door. The
    image exercises model code paths only; it carries no accuracy meaning.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    width, height = 640, 480
    image = Image.new("RGB", (width, height), (200, 196, 186))
    draw = ImageDraw.Draw(image)
    back = (180, 120, 460, 330)
    draw.polygon(
        [(0, 0), (width, 0), (back[2], back[1]), (back[0], back[1])],
        fill=(236, 234, 228),
    )
    draw.polygon(
        [(0, height), (width, height), (back[2], back[3]), (back[0], back[3])],
        fill=(150, 116, 84),
    )
    draw.polygon(
        [(0, 0), (back[0], back[1]), (back[0], back[3]), (0, height)],
        fill=(214, 205, 188),
    )
    draw.polygon(
        [(width, 0), (back[2], back[1]), (back[2], back[3]), (width, height)],
        fill=(206, 198, 182),
    )
    draw.rectangle(back, fill=(222, 214, 198))
    door = (280, 180, 350, 330)
    draw.rectangle(door, fill=(110, 72, 42))
    draw.ellipse((338, 252, 344, 258), fill=(210, 180, 90))
    draw.polygon([(510, 150), (600, 120), (600, 250), (510, 250)], fill=(170, 205, 230))
    draw.ellipse((60, 230, 130, 285), fill=(120, 108, 86))
    noise = np.random.default_rng(0).normal(0.0, 3.0, (height, width, 3))
    pixels = np.clip(np.asarray(image, dtype=np.float64) + noise, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8)), door


def _smoke_depth(
    torch: Any, model: Any, processor: Any, image: Any, box: Any, device: str
) -> dict[str, Any]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    post = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]
    depth = post["predicted_depth"].float().cpu()
    fov = post.get("field_of_view")
    checks = {
        "depth_shape_matches_image": tuple(depth.shape) == (image.height, image.width),
        "depth_finite": bool(torch.isfinite(depth).all()),
        "depth_positive": bool((depth > 0).all()),
        "field_of_view_finite": fov is not None
        and bool(torch.isfinite(fov.float().cpu()).all()),
    }
    return {"checks": checks, "output": {"depth_shape": list(depth.shape)}}


# Smoke-test prompt and thresholds only; P19 freezes the real ones.
_DINO_PROMPT = "a stain. a crack. a door. a window."
_DINO_BOX_THRESHOLD = 0.3
_DINO_TEXT_THRESHOLD = 0.25


def _smoke_grounding_dino(
    torch: Any, model: Any, processor: Any, image: Any, box: Any, device: str
) -> dict[str, Any]:
    inputs = processor(images=image, text=_DINO_PROMPT, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=_DINO_BOX_THRESHOLD,
        text_threshold=_DINO_TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]
    boxes = result["boxes"].float().cpu()
    scores = result["scores"].float().cpu()
    labels = result.get("text_labels", result.get("labels"))
    checks = {
        "boxes_shape_valid": boxes.ndim == 2 and boxes.shape[-1] == 4,
        "boxes_finite": bool(torch.isfinite(boxes).all()),
        "scores_in_unit_interval": bool(((scores >= 0) & (scores <= 1)).all()),
        "one_score_per_box": int(scores.numel()) == int(boxes.shape[0]),
    }
    return {
        "checks": checks,
        "output": {
            "prompt": _DINO_PROMPT,
            "detections": int(scores.numel()),
            "labels": [str(label) for label in labels],
        },
    }


def _smoke_sam2(
    torch: Any, model: Any, processor: Any, image: Any, box: Any, device: str
) -> dict[str, Any]:
    inputs = processor(images=image, input_boxes=[[list(box)]], return_tensors="pt").to(
        device
    )
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )[0]
    iou = outputs.iou_scores.float().cpu()
    mask = masks.reshape(-1, image.height, image.width)[0].bool()
    x0, y0, x1, y1 = box
    inside = int(mask[y0:y1, x0:x1].sum())
    total = int(mask.sum())
    checks = {
        "mask_shape_matches_image": tuple(masks.shape[-2:])
        == (image.height, image.width),
        "mask_nonempty": total > 0,
        "iou_score_finite": bool(torch.isfinite(iou).all()),
    }
    return {
        "checks": checks,
        "output": {
            "mask_pixels": total,
            "mask_fraction_inside_prompt_box": inside / total if total else None,
        },
    }


_SMOKES: dict[str, Callable[..., dict[str, Any]]] = {
    "DepthProForDepthEstimation": _smoke_depth,
    "GroundingDinoForObjectDetection": _smoke_grounding_dino,
    "Sam2Model": _smoke_sam2,
}


def run_smoke(
    name: str, device_request: str, root: Path, lock: ModelLock | None = None
) -> dict[str, Any]:
    """Load one model and run two forward passes on the synthetic image.

    Runs in the calling process; use ``smoke_in_subprocess`` for isolation and
    per-model memory figures. Returns timings, memory, output summaries and
    sanity checks. Raises on any failure.
    """
    lock = lock or load_lock()
    if name not in lock.models:
        raise ModelLockError(f"unknown model {name!r}; lock has {sorted(lock.models)}")
    spec = lock.models[name]
    smoke = _SMOKES.get(spec.model_class)
    if smoke is None:
        raise ModelLockError(f"no smoke test for {spec.model_class}")
    ensure_model_process()
    _set_offline_environment()
    timer = StageTimer()
    with timer.stage("import_torch"):
        import torch
        import transformers
    device = limits.resolve_device(device_request)
    torch.manual_seed(0)
    image, box = synthetic_room_image()
    memory_samples: list[dict[str, Any]] = []
    with loaded_model(spec, root, device, timer) as (model, processor):
        memory_samples.append({"after": "load", **accelerator_memory(torch, device)})
        with timer.stage("inference_cold"):
            first = smoke(torch, model, processor, image, box, device)
        with timer.stage("inference_warm"):
            second = smoke(torch, model, processor, image, box, device)
        memory_samples.append(
            {"after": "inference", **accelerator_memory(torch, device)}
        )
    checks = {
        **first["checks"],
        **{f"warm_{k}": v for k, v in second["checks"].items()},
    }
    return {
        "name": name,
        "device": device,
        "status": "ok" if all(checks.values()) else "failed_checks",
        "checks": checks,
        "output": first["output"],
        "timings_s": timer.durations_s,
        "peak_rss_bytes": peak_rss_bytes(),
        "accelerator_memory": memory_samples,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "image": {
            "kind": "synthetic_room_image",
            "width": image.width,
            "height": image.height,
        },
    }


def smoke_in_subprocess(
    name: str, device: str, root: Path, timeout: float = 1800.0
) -> dict[str, Any]:
    """Run ``run_smoke`` in a fresh worker process and return its report.

    A crash, timeout or failure inside the worker is returned as a failed
    report with the worker's stderr tail; it never raises.
    """
    command = [
        sys.executable,
        "-m",
        "floscan.runtime.models",
        "smoke",
        name,
        "--device",
        device,
        "--models-dir",
        str(root),
    ]
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "device": device,
            "status": "timeout",
            "detail": f"no result after {timeout:.0f} s",
            "process_wall_s": time.perf_counter() - start,
        }
    wall = time.perf_counter() - start
    lines = completed.stdout.strip().splitlines()
    report: dict[str, Any] | None = None
    if lines:
        try:
            report = json.loads(lines[-1])
        except json.JSONDecodeError:
            report = None
    if report is None:
        report = {
            "name": name,
            "device": device,
            "status": "crashed",
            "returncode": completed.returncode,
            "detail": " | ".join(completed.stderr.strip().splitlines()[-4:]),
        }
    report["process_wall_s"] = wall
    return report


# --------------------------------------------------------------------------
# Doctor
# --------------------------------------------------------------------------


def run_doctor(
    *,
    live_models: bool,
    devices: Sequence[str] | None,
    root: Path,
    lock_path: Path = LOCK_PATH,
) -> dict[str, Any]:
    """Collect hardware, library, checkpoint and optional live-model evidence.

    ``devices=None`` tests every device torch reports. An explicitly requested
    device that is unavailable is recorded as ``not_available``, never
    skipped. The doctor process imports torch but never pycolmap; libraries
    and models are exercised in child processes.
    """
    lock = load_lock(lock_path)
    report: dict[str, Any] = {
        "report_kind": "floscan.doctor",
        "report_version": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "floscan_version": __version__,
        "hardware": limits.detect_hardware(),
        "lock": {"path": str(lock.path), "sha256": lock.sha256},
        "models_dir": str(root),
        "libraries": limits.check_libraries(),
        "coexistence": limits.check_coexistence(),
        "torch": limits.torch_capabilities(),
        "checkpoints": [verify_model(spec, root) for spec in lock.models.values()],
        "not_installed": list(lock.not_installed),
        "live": [],
        "profiles": [],
    }
    libraries_ok = all(item["status"] == "ok" for item in report["libraries"])
    checkpoints_ok = all(item["status"] == "ok" for item in report["checkpoints"])
    available = limits.available_devices()
    tested = list(devices) if devices else available

    if live_models:
        for device in tested:
            if device not in available:
                report["profiles"].append(
                    {
                        "device": device,
                        "status": "not_available",
                        "detail": f"torch reports {available}",
                    }
                )
                continue
            results = [smoke_in_subprocess(name, device, root) for name in lock.models]
            report["live"].extend(results)
            failed = [r["name"] for r in results if r["status"] != "ok"]
            report["profiles"].append(
                {
                    "device": device,
                    "status": "usable" if not failed else "failed",
                    "failed_models": failed,
                }
            )
    usable = [p["device"] for p in report["profiles"] if p["status"] == "usable"]
    ok = libraries_ok and checkpoints_ok and (bool(usable) if live_models else True)
    report["summary"] = {
        "libraries_ok": libraries_ok,
        "checkpoints_ok": checkpoints_ok,
        "live_models_tested": live_models,
        "usable_devices": usable,
        "ok": ok,
    }
    report["exit_code"] = 0 if ok else 1
    return report


# --------------------------------------------------------------------------
# Worker entry point
# --------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m floscan.runtime.models",
        description="Fetch, verify or smoke-test pinned model checkpoints.",
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--models", help="comma-separated model names (default: all)")
        sub.add_argument("--models-dir", type=Path, default=None)
    smoke = commands.add_parser("smoke", help="worker: one model, one device, JSON out")
    smoke.add_argument("name")
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--models-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _selected(lock: ModelLock, names: str | None) -> list[ModelSpec]:
    if not names:
        return list(lock.models.values())
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in lock.models]
    if unknown:
        raise ModelLockError(
            f"unknown models {unknown}; lock has {sorted(lock.models)}"
        )
    return [lock.models[name] for name in wanted]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m floscan.runtime.models``."""
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    root = models_dir(args.models_dir)
    if args.command == "smoke":
        try:
            lock = load_lock(args.lock)
            report = run_smoke(args.name, args.device, root, lock)
        except Exception as error:  # Worker boundary: report every failure as data.
            report = {
                "name": args.name,
                "device": args.device,
                "status": "failed",
                "error_type": type(error).__name__,
                "detail": str(error),
                "peak_rss_bytes": peak_rss_bytes(),
            }
        print(json.dumps(report))
        return 0 if report["status"] == "ok" else 1

    try:
        lock = load_lock(args.lock)
        specs = _selected(lock, args.models)
        if args.command == "verify":
            reports = [verify_model(spec, root) for spec in specs]
            for item in reports:
                print(f"{item['name']}: {item['status']} ({item['seconds']:.1f} s)")
            return 0 if all(item["status"] == "ok" for item in reports) else 1
        total = sum(item.size for spec in specs for item in spec.files)
        print(f"fetching {len(specs)} models ({total / 1e9:.2f} GB pinned) into {root}")
        start = time.perf_counter()
        downloaded = 0
        for spec in specs:
            downloaded += fetch_model(spec, root, lock.endpoint)["bytes_downloaded"]
        print(
            f"done: {downloaded / 1e9:.2f} GB downloaded in "
            f"{time.perf_counter() - start:.0f} s; all files match the lock"
        )
        return 0
    except (ModelLockError, ModelUnavailable) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
