"""P06 probe: which odometry row each decoded video frame belongs to.

Compares the decoded PTS gap pattern with the odometry timestamp gaps at row
offsets 0 and 1, with and without the MP4 edit list.

Usage: uv run python runs/p06-evidence/edit_list_probe.py SESSION_DIR
"""

import sys
from pathlib import Path

import av
import numpy as np

root = Path(sys.argv[1])
t = np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1, usecols=[0])
odometry_gaps = np.round(np.diff(t) * 60).astype(int)
print("odometry first gaps", odometry_gaps[:6].tolist())
for options in ({}, {"ignore_editlist": "1"}):
    with av.open(str(root / "rgb.mp4"), options=options) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        pts = np.array([frame.pts for frame in container.decode(stream)])
    video_gaps = np.diff(pts)
    print(options, "decoded", len(pts), "first pts", pts[:5].tolist())
    for offset in (0, 1):
        n = min(len(video_gaps), len(odometry_gaps) - offset)
        agree = int((video_gaps[:n] == odometry_gaps[offset : offset + n]).sum())
        print(f"   row offset {offset}: gap agreement {agree}/{n}")
    offset = len(t) - len(pts)
    sensor = t[offset:]
    residual = (sensor - sensor[0]) - (pts - pts[0]) / 60.0
    design = np.column_stack([np.ones_like(sensor), sensor - sensor[0]])
    coef, *_ = np.linalg.lstsq(design, residual, rcond=None)
    post_fit = np.abs(residual - design @ coef).max()
    print(
        f"   offset {offset}: raw residual {1e3 * residual.min():.2f}.."
        f"{1e3 * residual.max():.2f} ms, clock-rate {coef[1] * 1e6:.1f} ppm, "
        f"post-fit max {1e3 * post_fit:.2f} ms"
    )
