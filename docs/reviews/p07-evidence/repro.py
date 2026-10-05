"""Independent P07 probes; only temporary synthetic inputs are created."""

import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np

from floscan.capture import stray
from floscan.geometry.planes import robust_refit
from floscan.reconstruction import lidar


def main():
    rng = np.random.default_rng(22)
    xy = rng.uniform(-1, 1, (1000, 2))
    clean = np.column_stack([xy, np.zeros(1000)])
    band = np.column_stack([xy[:200], np.full(200, 0.029)])
    fit = robust_refit(np.concatenate([clean, band]), [0, 0, 1], 0, cutoff=0.03)
    print("Exact main plane z=0; contamination: 200/1200 points at z=0.029 m")
    print("returned offset_m", fit.offset, "inliers", int(fit.inliers.sum()))

    sys.path.insert(0, str(Path.cwd()))
    helpers = runpy.run_path("tests/synthetic/test_lidar_planes.py")
    with tempfile.TemporaryDirectory(prefix="floscan-p07-review-") as temporary:
        root = helpers["make_room"](Path(temporary) / "session", frames=60)
        inspection = stray.inspect_session(stray.open_session(root))
        print("capture status", inspection.status)
        config = lidar.LidarConfig(
            submap_max_keyframes=2,
            submap_overlap_keyframes=60,
            submap_max_travel_m=100,
        )
        keyframes, _ = lidar.select_keyframes(inspection.session, inspection, config)
        plan = lidar.plan_submaps(inspection.session, keyframes, config)
        print("configured max keyframes", config.submap_max_keyframes)
        print("actual max keyframes", max(map(len, plan)))
        try:
            manifest = lidar.reconstruct(
                inspection.session,
                inspection,
                Path(temporary) / "reconstruction",
                lidar.LidarConfig(max_depth_m=0.16),
            )
            print("empty-filter result", manifest["filter_counts"], manifest["summary"])
        except Exception as error:
            print("empty-filter exception", type(error).__name__, str(error))


if __name__ == "__main__":
    main()
