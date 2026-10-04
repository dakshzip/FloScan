"""Diagnose Depth Pro latency and memory for one device and dtype.

Reproduces runs/p02-evidence/depth_pro_precision.log (macOS only, because it
samples swap with sysctl). Run each configuration in its own process:

    uv run python runs/p02-evidence/diagnostics/depth_pro_precision.py mps float32
    uv run python runs/p02-evidence/diagnostics/depth_pro_precision.py mps float16 \
        --save mps_fp16.npy
    uv run python runs/p02-evidence/diagnostics/depth_pro_precision.py cpu float32 \
        --save cpu_fp32.npy
    uv run python runs/p02-evidence/diagnostics/depth_pro_precision.py \
        --compare cpu_fp32.npy mps_fp16.npy

The log was produced by two earlier ad-hoc scripts with the same logic; this
file consolidates them in lint-clean form.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from floscan.runtime import models
from floscan.runtime.timing import peak_rss_bytes


def swap_used() -> str:
    """Return macOS swap usage as reported by sysctl."""
    output = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
    ).stdout
    return output.split("used = ")[1].split()[0]


def profile(device: str, dtype_name: str, save: Path | None) -> None:
    """Time preprocess, forward and post-process for two Depth Pro passes."""
    models.ensure_model_process()
    models._set_offline_environment()
    import torch
    import transformers

    spec = models.load_lock().models["depth_pro"]
    root = models.models_dir()
    models.require_verified(spec, root)
    directory = root / spec.name
    dtype = getattr(torch, dtype_name)
    image, _ = models.synthetic_room_image()
    with models.network_forbidden():
        processor = transformers.DepthProImageProcessor.from_pretrained(
            directory, local_files_only=True
        )
        model = transformers.DepthProForDepthEstimation.from_pretrained(
            directory, local_files_only=True, dtype=dtype
        )
    model = model.to(device).eval()
    tag = f"[{device} {dtype_name}]"
    print(f"{tag} loaded; swap {swap_used()}", flush=True)
    depth = None
    for run in range(2):
        start = time.perf_counter()
        inputs = processor(images=image, return_tensors="pt").to(device, dtype=dtype)
        if device == "mps":
            torch.mps.synchronize()
        prepared = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        if device == "mps":
            torch.mps.synchronize()
        forwarded = time.perf_counter()
        post = processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )
        depth = post[0]["predicted_depth"].float().cpu().numpy()
        done = time.perf_counter()
        memory = ""
        if device == "mps":
            driver = torch.mps.driver_allocated_memory() / 2**30
            memory = f" mps_driver {driver:.2f} GiB"
        print(
            f"{tag} run {run}: preprocess {prepared - start:.2f}s "
            f"forward {forwarded - prepared:.2f}s post {done - forwarded:.2f}s "
            f"finite {bool(np.isfinite(depth).all())} swap {swap_used()} "
            f"peak_rss {peak_rss_bytes() / 2**30:.2f} GiB{memory}",
            flush=True,
        )
    if save is not None and depth is not None:
        np.save(save, depth)


def compare(reference_path: Path, candidate_path: Path) -> None:
    """Print per-pixel relative differences of a candidate against a reference."""
    reference = np.load(reference_path)
    candidate = np.load(candidate_path)
    relative = np.abs(candidate - reference) / reference
    print(
        f"relative difference: median {np.median(relative):.3%} "
        f"p95 {np.percentile(relative, 95):.3%} p99 {np.percentile(relative, 99):.3%} "
        f"max {relative.max():.3%}; global scale ratio "
        f"{np.median(candidate / reference):.5f}"
    )


def main() -> None:
    """Parse arguments and run one profile or one comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("device", nargs="?", choices=("cpu", "mps", "cuda"))
    parser.add_argument("dtype", nargs="?", choices=("float32", "float16"))
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("REF", "CANDIDATE"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.device and args.dtype:
        profile(args.device, args.dtype, args.save)
    else:
        parser.error("give DEVICE DTYPE, or --compare REF CANDIDATE")


if __name__ == "__main__":
    main()
