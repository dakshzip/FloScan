"""Stage timing and memory probes used as runtime evidence.

Durations are wall-clock seconds from ``time.perf_counter``. Peak resident
memory is the process high-water mark, so per-model numbers are only
meaningful when each model runs in its own process (see ``models.py``).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTimer:
    """Collects named wall-clock durations in insertion order."""

    durations_s: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block and record it under ``name``.

        The duration is recorded even if the block raises, so a failed stage
        still shows how long it ran.
        """
        if name in self.durations_s:
            raise ValueError(f"stage {name!r} was already timed")
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations_s[name] = time.perf_counter() - start


def peak_rss_bytes() -> int | None:
    """Return this process's peak resident set size in bytes, if known."""
    if sys.platform == "win32":
        return _windows_peak_working_set()
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kibibytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _windows_peak_working_set() -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    ):
        return None
    return int(counters.PeakWorkingSetSize)


def accelerator_memory(torch: Any, device: str) -> dict[str, Any]:
    """Return accelerator memory figures for ``device`` after a run.

    CUDA reports true peaks since the last reset. MPS exposes no peak API, so
    its figures are point samples taken by the caller and labelled as such.
    """
    if device == "cuda":
        return {
            "kind": "cuda_peak",
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    if device == "mps":
        return {
            "kind": "mps_sample",
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    return {"kind": "none"}
