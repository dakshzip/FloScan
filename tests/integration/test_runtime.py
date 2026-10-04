"""Integration tests for the P02 runtime: pinned checkpoints and live models.

Tests marked by the ``real_weights`` fixture need ``scripts/fetch_models.sh``
to have completed; they fail (not skip) otherwise, because P02's acceptance
requires real inference evidence.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import weakref
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from floscan.runtime import limits, models
from floscan.runtime.models import (
    LOCK_PATH,
    ModelLockError,
    ModelRuntimeError,
    ModelUnavailable,
    NetworkForbidden,
)

SELECTED = ("depth_pro", "grounding_dino_tiny", "sam2_1_hiera_small")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _lock_data() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _write_lock(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "models.lock.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _tiny_lock(tmp_path: Path, payload: bytes) -> models.ModelLock:
    """A one-model lock pinning a single file whose content is ``payload``."""
    data = _lock_data()
    entry = copy.deepcopy(data["models"][0])
    entry["name"] = "tiny"
    entry["checkpoint"]["files"] = [
        {
            "path": "weights.bin",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    data["models"] = [entry]
    return models.load_lock(_write_lock(tmp_path, data))


@pytest.fixture
def file_server() -> Iterator[tuple[str, dict[str, bytes]]]:
    """Serve ``routes[path] = bytes`` over local HTTP; yields (base_url, routes)."""
    routes: dict[str, bytes] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            body = routes.get(self.path)
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", routes
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="session")
def real_weights() -> Path:
    """The real checkpoint root, verified; fails with instructions if absent."""
    root = models.models_dir()
    lock = models.load_lock()
    bad = {
        name: report["status"]
        for name, spec in lock.models.items()
        if (report := models.verify_model(spec, root))["status"] != "ok"
    }
    if bad:
        pytest.fail(
            f"checkpoints not ready in {root}: {bad}; run scripts/fetch_models.sh"
        )
    return root


# --------------------------------------------------------------------------
# Lock integrity
# --------------------------------------------------------------------------


def test_lock_pins_selected_models_with_licence_evidence() -> None:
    lock = models.load_lock()
    assert tuple(lock.models) == SELECTED
    raw = {entry["name"]: entry for entry in _lock_data()["models"]}
    for name in SELECTED:
        entry = raw[name]
        assert entry["code_license"]["license_sha256"]
        assert entry["weights_license"]["evidence"]
    # The Depth Pro weight-licence conflict must stay visible, not be resolved
    # silently in our favour.
    assert raw["depth_pro"]["weights_license"]["status"] == "conflicting_metadata"
    assert {item["name"] for item in lock.not_installed} == {"disk_lightglue", "vggt"}
    # CPU always runs fp32; only measured accelerator paths may use fp16.
    for spec in lock.models.values():
        assert spec.dtypes["cpu"] == "float32"
    assert lock.models["depth_pro"].dtypes == {
        "cpu": "float32",
        "mps": "float16",
        "cuda": "float16",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d["models"][0]["checkpoint"].__setitem__("revision", "main"),
            "commit hash",
        ),
        (
            lambda d: d["models"][0]["checkpoint"]["files"][0].__setitem__(
                "path", "../escape.json"
            ),
            "relative",
        ),
        (
            lambda d: d["models"][0]["checkpoint"]["files"][0].__setitem__(
                "sha256", "abc"
            ),
            "sha256",
        ),
        (
            lambda d: d["models"][0]["checkpoint"]["files"][0].__setitem__(
                "size", True
            ),
            "positive int",
        ),
        (
            lambda d: d["models"][0]["weights_license"].__setitem__("evidence", []),
            "evidence",
        ),
        (lambda d: d.__setitem__("endpoint", "http://huggingface.co"), "https"),
        (
            lambda d: d["models"][0]["loader"]["dtype"].__setitem__("mps", "int8"),
            "dtype",
        ),
        (lambda d: d["models"][0]["loader"]["dtype"].pop("cuda"), "missing keys"),
    ],
    ids=[
        "branch-revision",
        "path-traversal",
        "bad-sha",
        "bool-size",
        "no-evidence",
        "http",
        "bad-dtype",
        "missing-device-dtype",
    ],
)
def test_lock_rejects_unpinned_or_unsafe_entries(
    tmp_path: Path, mutate, message
) -> None:
    data = _lock_data()
    mutate(data)
    with pytest.raises(ModelLockError, match=message):
        models.load_lock(_write_lock(tmp_path, data))


# --------------------------------------------------------------------------
# Missing and bad checkpoints
# --------------------------------------------------------------------------


def test_missing_checkpoint_is_reported_and_refused(tmp_path: Path) -> None:
    lock = _tiny_lock(tmp_path, b"abc")
    spec = lock.models["tiny"]
    report = models.verify_model(spec, tmp_path / "empty")
    assert report["status"] == "missing"
    with pytest.raises(ModelUnavailable, match="fetch_models.sh"):
        models.require_verified(spec, tmp_path / "empty")


@pytest.mark.parametrize(
    ("content", "status"),
    [(b"abd", "hash_mismatch"), (b"abcd", "size_mismatch")],
)
def test_bad_checkpoint_hash_or_size_is_refused(
    tmp_path: Path, content: bytes, status: str
) -> None:
    lock = _tiny_lock(tmp_path, b"abc")
    spec = lock.models["tiny"]
    root = tmp_path / "models"
    (root / "tiny").mkdir(parents=True)
    (root / "tiny" / "weights.bin").write_bytes(content)
    assert models.verify_model(spec, root)["status"] == status
    with pytest.raises(ModelUnavailable, match=status):
        models.require_verified(spec, root)


def test_load_refuses_altered_checkpoint_before_reading_it(tmp_path: Path) -> None:
    lock = _tiny_lock(tmp_path, b"abc")
    root = tmp_path / "models"
    (root / "tiny").mkdir(parents=True)
    (root / "tiny" / "weights.bin").write_bytes(b"abd")
    with pytest.raises(ModelUnavailable, match="hash_mismatch"):
        with models.loaded_model(lock.models["tiny"], root, "cpu"):
            pytest.fail("an altered checkpoint must never load")


def test_fetch_verifies_download_and_keeps_good_file(
    tmp_path: Path, file_server
) -> None:
    base, routes = file_server
    lock = _tiny_lock(tmp_path, b"abc")
    spec = lock.models["tiny"]
    routes[f"/{spec.repo}/resolve/{spec.revision}/weights.bin"] = b"abc"
    root = tmp_path / "models"
    result = models.fetch_model(spec, root, base, log=lambda _line: None)
    assert result["files_downloaded"] == 1
    assert models.verify_model(spec, root)["status"] == "ok"
    # A second fetch keeps the verified file and downloads nothing.
    again = models.fetch_model(spec, root, base, log=lambda _line: None)
    assert again["files_downloaded"] == 0


def test_fetch_rejects_tampered_download_and_leaves_nothing(
    tmp_path: Path, file_server
) -> None:
    base, routes = file_server
    lock = _tiny_lock(tmp_path, b"abc")
    spec = lock.models["tiny"]
    routes[f"/{spec.repo}/resolve/{spec.revision}/weights.bin"] = b"xyz"
    root = tmp_path / "models"
    with pytest.raises(ModelUnavailable, match="does not match the lock"):
        models.fetch_model(spec, root, base, log=lambda _line: None)
    assert not (root / "tiny" / "weights.bin").exists()
    assert not (root / "tiny" / "weights.bin.partial").exists()


# --------------------------------------------------------------------------
# Offline guard, isolation and sequential loading
# --------------------------------------------------------------------------


def test_network_is_refused_inside_guard_and_restored_after() -> None:
    original = socket.getaddrinfo
    with models.network_forbidden():
        with pytest.raises(NetworkForbidden):
            socket.getaddrinfo("huggingface.co", 443)
        with pytest.raises((NetworkForbidden, urllib.error.URLError)):
            urllib.request.urlopen("https://huggingface.co", timeout=5)
    assert socket.getaddrinfo is original


def test_model_process_refuses_when_pycolmap_is_loaded(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pycolmap", object())
    with pytest.raises(ModelRuntimeError, match="separate worker process"):
        models.ensure_model_process()


def test_real_pycolmap_process_fails_cleanly_instead_of_aborting(
    real_weights: Path,
) -> None:
    code = (
        "import pycolmap\n"
        "from pathlib import Path\n"
        "from floscan.runtime.models import run_smoke\n"
        f"run_smoke('sam2_1_hiera_small', 'cpu', Path({str(real_weights)!r}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 1, completed.stderr
    assert "ModelRuntimeError" in completed.stderr
    assert "OMP: Error" not in completed.stderr


def test_runtime_modules_do_not_import_torch() -> None:
    code = (
        "import sys, floscan.cli, floscan.runtime.models, floscan.runtime.limits\n"
        "assert 'torch' not in sys.modules, 'torch imported eagerly'\n"
        "assert 'transformers' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_models_load_one_at_a_time_and_are_freed(real_weights: Path) -> None:
    lock = models.load_lock()
    spec = lock.models["sam2_1_hiera_small"]
    with models.loaded_model(spec, real_weights, "cpu") as (model, _processor):
        ref = weakref.ref(model)
        with pytest.raises(ModelRuntimeError, match="one at a time"):
            with models.loaded_model(lock.models["depth_pro"], real_weights, "cpu"):
                pytest.fail("a second model must not load")
        del model, _processor
    gc.collect()
    assert ref() is None


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------


def test_device_resolution_never_substitutes() -> None:
    with pytest.raises(ValueError):
        limits.resolve_device("tpu")
    available = limits.available_devices()
    assert limits.resolve_device("auto") in available
    for device in limits.DEVICES:
        if device not in available:
            with pytest.raises(limits.DeviceUnavailable):
                limits.resolve_device(device)


# --------------------------------------------------------------------------
# Real inference smoke (offline)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", SELECTED)
def test_real_inference_smoke_in_isolated_offline_worker(
    real_weights: Path, name: str
) -> None:
    report = models.smoke_in_subprocess(name, "auto", real_weights)
    assert report["status"] == "ok", report
    assert all(report["checks"].values()), report["checks"]
    assert report["device"] in limits.available_devices()
    assert {"verify_checkpoint", "load", "inference_cold", "inference_warm"} <= set(
        report["timings_s"]
    )


def test_whole_smoke_path_runs_with_network_blocked(real_weights: Path) -> None:
    with models.network_forbidden():
        report = models.run_smoke("sam2_1_hiera_small", "cpu", real_weights)
    assert report["status"] == "ok", report


def test_doctor_refuses_to_overwrite_report(tmp_path: Path) -> None:
    from floscan.cli import main

    existing = tmp_path / "report.json"
    existing.write_text("{}", encoding="utf-8")
    assert main(["doctor", "--output", str(existing)]) == 2
    assert existing.read_text(encoding="utf-8") == "{}"


# Depth Pro runs fp16 on accelerators (8.5 GiB, 6 s on the M2 Pro) instead of
# fp32 (15.9 GiB, 53-168 s while swapping). This guards that choice: fp16 depth
# must stay within 1% per pixel and 0.1% global scale of fp32 CPU depth, an
# order of magnitude inside the tightest RGB wall gate (video, 3%).
_DEPTH_DUMP = """
import sys
import numpy as np
from pathlib import Path
from floscan.runtime import models
device, out = sys.argv[1], sys.argv[2]
spec = models.load_lock().models["depth_pro"]
image, _ = models.synthetic_room_image()
with models.loaded_model(spec, Path(sys.argv[3]), device) as (model, processor):
    import torch
    inputs = processor(images=image, return_tensors="pt").to(device, dtype=model.dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    depth = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]["predicted_depth"].float().cpu().numpy()
np.save(out, depth)
"""


def test_depth_pro_fp16_matches_fp32_reference(
    real_weights: Path, tmp_path: Path
) -> None:
    import numpy as np

    accelerators = [d for d in limits.available_devices() if d != "cpu"]
    if not accelerators:
        pytest.skip("no accelerator: Depth Pro runs fp32 everywhere here")
    device = accelerators[-1]
    depths = {}
    for run_device in (device, "cpu"):
        out = tmp_path / f"{run_device}.npy"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _DEPTH_DUMP,
                run_device,
                str(out),
                str(real_weights),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        depths[run_device] = np.load(out)
    reference, half = depths["cpu"], depths[device]
    relative = np.abs(half - reference) / reference
    assert float(relative.max()) <= 0.01, f"max per-pixel drift {relative.max():.4%}"
    ratio = float(np.median(half / reference))
    assert abs(ratio - 1.0) <= 0.001, f"global scale drift {ratio - 1.0:.4%}"
