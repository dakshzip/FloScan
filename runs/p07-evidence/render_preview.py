"""P07 evidence: top-down and side renders of a reconstruction preview.

Usage: uv run python runs/p07-evidence/render_preview.py PREVIEW_PLY OUT_PREFIX
"""

import sys

import numpy as np
import open3d as o3d
from PIL import Image

RESOLUTION_M = 0.01

cloud = o3d.io.read_point_cloud(sys.argv[1])
xyz, rgb = np.asarray(cloud.points), np.asarray(cloud.colors)
order = np.argsort(xyz[:, 2])  # higher points drawn last (top view)
xyz, rgb = xyz[order], (rgb[order] * 255).astype(np.uint8)
low, high = xyz.min(axis=0), xyz.max(axis=0)
size = ((high - low) / RESOLUTION_M).astype(int) + 1
cells = ((xyz - low) / RESOLUTION_M).astype(int)
top = np.full((size[1], size[0], 3), 255, np.uint8)
top[size[1] - 1 - cells[:, 1], cells[:, 0]] = rgb
side = np.full((size[2], size[0], 3), 255, np.uint8)
side[size[2] - 1 - cells[:, 2], cells[:, 0]] = rgb
Image.fromarray(top).save(f"{sys.argv[2]}-top.png")
Image.fromarray(side).save(f"{sys.argv[2]}-side.png")
print(f"{len(xyz)} points; extent {np.round(high - low, 2).tolist()} m")
