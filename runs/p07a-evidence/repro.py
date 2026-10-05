"""P07A: the P07 review probes, each run on its own.

Same inputs as docs/reviews/p07-evidence/repro.py, but a refused setting is
reported instead of stopping the remaining probes. Only temporary synthetic
inputs are created.
"""

import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np

from floscan.capture import stray
from floscan.geometry.planes import robust_refit
from floscan.reconstruction import lidar


def main() -> None:
    rng = np.random.default_rng(22)
    xy = rng.uniform(-1, 1, (1000, 2))
    clean = np.column_stack([xy, np.zeros(1000)])
    band = np.column_stack([xy[:200], np.full(200, 0.029)])
    fit = robust_refit(np.concatenate([clean, band]), [0, 0, 1], 0, cutoff=0.03)
    print("1. exact plane z=0 plus 200/1200 points at z=0.029 m, cutoff 0.03 m")
    print("   returned offset_m", fit.offset, "support", int(fit.inliers.sum()))

    sys.path.insert(0, str(Path.cwd()))
    helpers = runpy.run_path("tests/synthetic/test_lidar_planes.py")
    with tempfile.TemporaryDirectory(prefix="floscan-p07a-") as temporary:
        root = helpers["make_room"](Path(temporary) / "session", frames=60)
        inspection = stray.inspect_session(stray.open_session(root))
        print("   capture status", inspection.status)
        print("2. submap_max_keyframes=2, submap_overlap_keyframes=60")
        for overlap in (60, 3, 1):
            try:
                config = lidar.LidarConfig(
                    submap_max_keyframes=2,
                    submap_overlap_keyframes=overlap,
                    submap_max_travel_m=100,
                )
            except ValueError as error:
                print(f"   overlap {overlap}: refused:", str(error).splitlines()[1])
                continue
            keyframes, _ = lidar.select_keyframes(
                inspection.session, inspection, config
            )
            plan = lidar.plan_submaps(inspection.session, keyframes, config)
            print(
                f"   overlap {overlap}: max keyframes per submap", max(map(len, plan))
            )
        print("3. LidarConfig(max_depth_m=0.16): every point filtered")
        try:
            manifest = lidar.reconstruct(
                inspection.session,
                inspection,
                Path(temporary) / "reconstruction",
                lidar.LidarConfig(max_depth_m=0.16),
            )
            print("   evidence_status", manifest["evidence_status"])
            print(
                "   kept",
                manifest["filter_counts"]["kept"],
                "preview",
                manifest["preview"],
            )
        except Exception as error:  # noqa: BLE001 - the probe reports anything
            print("   exception", type(error).__name__, str(error))


if __name__ == "__main__":
    main()
