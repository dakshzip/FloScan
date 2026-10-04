"""Hardware profile detection, explicit device selection and library checks.

Nothing here imports torch at module import time: geometry processes that load
pycolmap must be able to import this module (see ``models.py`` on the macOS
OpenMP conflict). Every library check runs in its own interpreter so a native
crash is recorded as a failed check instead of killing the caller.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

DEVICES = ("cpu", "mps", "cuda")


class DeviceUnavailable(RuntimeError):
    """The requested compute device does not exist on this machine."""


def _run(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _cpu_model() -> str | None:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            return None
        return None
    return platform.processor() or None


def _total_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        value = _run(["sysctl", "-n", "hw.memsize"])
        return int(value) if value and value.isdigit() else None
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return None


def _os_version() -> str:
    if sys.platform == "darwin":
        build = _run(["sw_vers", "-buildVersion"])
        version = platform.mac_ver()[0]
        return f"macOS {version}" + (f" ({build})" if build else "")
    if sys.platform == "win32":
        return f"Windows {platform.release()} ({platform.version()})"
    return platform.platform()


def _nvidia_smi() -> list[dict[str, str]]:
    """Query NVIDIA GPUs and the driver version, if nvidia-smi is present."""
    if shutil.which("nvidia-smi") is None:
        return []
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    gpus = []
    for line in (output or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            gpus.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total": parts[2],
                    "compute_capability": parts[3],
                }
            )
    return gpus


def detect_hardware() -> dict[str, Any]:
    """Describe this machine from the operating system, without torch.

    Every field is measured or reported by the OS; unknown values are None.
    """
    return {
        "os": _os_version(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "nvidia_smi": _nvidia_smi(),
    }


def torch_capabilities() -> dict[str, Any]:
    """Report torch's view of the accelerators. Imports torch."""
    import torch

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }


def available_devices() -> list[str]:
    """Devices torch can use here, always including ``cpu``. Imports torch."""
    import torch

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def resolve_device(requested: str) -> str:
    """Map a requested device to a usable one, never silently substituting.

    ``auto`` picks CUDA, then MPS, then CPU. An explicit device that is not
    available raises instead of falling back.
    """
    if requested not in ("auto", *DEVICES):
        raise ValueError(
            f"device must be 'auto' or one of {DEVICES}, got {requested!r}"
        )
    devices = available_devices()
    if requested == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if candidate in devices:
                return candidate
    if requested not in devices:
        raise DeviceUnavailable(
            f"device {requested!r} is not available here (available: {devices})"
        )
    return requested


# Each check imports one library and exercises one small native code path, so
# a broken binary wheel fails here rather than mid-pipeline.
LIBRARY_CHECKS: dict[str, str] = {
    "numpy": (
        "import numpy as np\n"
        "a = np.random.default_rng(0).normal(size=(64, 64))\n"
        "np.linalg.svd(a)\n"
        "print(np.__version__)"
    ),
    "scipy": (
        "import scipy\n"
        "from scipy.optimize import least_squares\n"
        "r = least_squares(lambda x: x - 3.0, [0.0])\n"
        "assert abs(r.x[0] - 3.0) < 1e-6\n"
        "print(scipy.__version__)"
    ),
    "shapely": (
        "import shapely\n"
        "from shapely.geometry import Polygon\n"
        "assert Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]).area == 2.0\n"
        "print(shapely.__version__)"
    ),
    "pydantic": (
        "import pydantic\n"
        "class M(pydantic.BaseModel):\n"
        "    x: float\n"
        "assert M.model_validate({'x': 1}).x == 1.0\n"
        "print(pydantic.__version__)"
    ),
    "opencv": (
        "import cv2, numpy as np\n"
        "img = (np.random.default_rng(0).random((240, 320)) * 255).astype('uint8')\n"
        "kp, desc = cv2.SIFT_create().detectAndCompute(img, None)\n"
        "assert desc is not None and len(kp) > 0\n"
        "print(cv2.__version__)"
    ),
    "av": (
        "import av, io, numpy as np\n"
        "buf = io.BytesIO()\n"
        "with av.open(buf, 'w', format='mp4') as out:\n"
        "    s = out.add_stream('mpeg4', rate=30)\n"
        "    s.width, s.height, s.pix_fmt = 64, 48, 'yuv420p'\n"
        "    for i in range(8):\n"
        "        f = av.VideoFrame.from_ndarray(\n"
        "            np.full((48, 64, 3), i * 20, 'uint8'), format='rgb24')\n"
        "        for p in s.encode(f):\n"
        "            out.mux(p)\n"
        "    for p in s.encode():\n"
        "        out.mux(p)\n"
        "buf.seek(0)\n"
        "with av.open(buf) as inp:\n"
        "    pts = [f.pts for f in inp.decode(video=0)]\n"
        "assert len(pts) == 8 and pts == sorted(pts)\n"
        "print(av.__version__)"
    ),
    "pycolmap": (
        "import pycolmap\n"
        "r = pycolmap.Reconstruction()\n"
        "assert r.num_images() == 0\n"
        "print(pycolmap.__version__)"
    ),
    "open3d": (
        "import open3d as o3d, numpy as np\n"
        "rng = np.random.default_rng(0)\n"
        "pts = np.c_[rng.random((500, 2)) * 4, np.full(500, 2.5)]\n"
        "pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))\n"
        "plane, inliers = pc.segment_plane(0.01, 3, 200)\n"
        "assert len(inliers) == 500 and abs(abs(plane[2]) - 1) < 1e-6\n"
        "print(o3d.__version__)"
    ),
    "torch": (
        "import torch\n"
        "x = torch.ones(64, 64)\n"
        "assert float((x @ x)[0, 0]) == 64.0\n"
        "print(torch.__version__)"
    ),
    "transformers": "import transformers\nprint(transformers.__version__)",
}

# Pairs that must share a process in the planned architecture, or that are
# known to conflict. A failure here is recorded as an isolation requirement.
COEXISTENCE_CHECKS: dict[str, str] = {
    "torch+pycolmap": "import torch, pycolmap\nprint('ok')",
    "torch+open3d": "import torch, open3d\nprint('ok')",
    "opencv+av": "import cv2, av\nprint('ok')",
}


def _check_snippet(name: str, code: str, timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "failed",
            "detail": f"timed out after {timeout:.0f} s",
            "seconds": time.perf_counter() - start,
        }
    stdout = completed.stdout.strip().splitlines()
    stderr = completed.stderr.strip().splitlines()
    result: dict[str, Any] = {
        "name": name,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - start,
        "version": stdout[-1] if completed.returncode == 0 and stdout else None,
    }
    if completed.returncode != 0:
        result["detail"] = " | ".join(stderr[-3:]) or "no stderr"
    warnings = [line for line in stderr if "is implemented in both" in line]
    if warnings:
        result["warnings"] = sorted(
            {w.split("Class ")[-1].split(" ")[0] for w in warnings}
        )
    return result


def check_libraries(timeout: float = 120.0) -> list[dict[str, Any]]:
    """Import and exercise each library in a fresh interpreter."""
    return [
        _check_snippet(name, code, timeout) for name, code in LIBRARY_CHECKS.items()
    ]


def check_coexistence(timeout: float = 120.0) -> list[dict[str, Any]]:
    """Import known-sensitive library pairs together in fresh interpreters."""
    results = []
    for name, code in COEXISTENCE_CHECKS.items():
        result = _check_snippet(name, code, timeout)
        result["status"] = "compatible" if result["status"] == "ok" else "conflict"
        result.pop("version", None)
        results.append(result)
    return results
