"""P08 evidence: room hypotheses from a P07 bundle, with a top-down render.

Usage: uv run python runs/p08-evidence/rooms_from_bundle.py BUNDLE_DIR OUT_DIR
Edges: green = observed wall, red = unknown, orange = unobserved span.
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from floscan.geometry.rooms import build_rooms, load_bundle, write_rooms

PIXELS_PER_M = 60

bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
start = time.perf_counter()
evidence = load_bundle(bundle)
model = build_rooms(evidence)
elapsed = time.perf_counter() - start
path = write_rooms(out, model)
print(f"{path}: {model.status} in {elapsed:.1f} s; {len(model.rooms)} rooms")
for reason in model.reasons:
    print("  -", reason)
floor = model.diagnostics["floor"]
print("  floor", round(floor.get("height_m", float("nan")), 3), floor["reason"])
print("  ceiling", model.diagnostics["ceiling"]["reason"])
print("  walls", model.diagnostics["walls"])
print("  segmentation", model.diagnostics["segmentation"])
for h in model.hypotheses:
    gaps = sum(len(e.gaps) for e in h.edges)
    unknown = sum(e.status == "unknown" for e in h.edges)
    print(
        f"  {h.room_id}: {h.outline.area:.2f} m2, {len(h.edges)} edges "
        f"({unknown} unknown, {gaps} spans), floor coverage {h.floor_coverage:.0%}, "
        f"{h.status}"
    )

points = np.concatenate([p.points for p in evidence.planes])[:, :2]
low = points.min(axis=0) - 0.5
size = ((points.max(axis=0) + 0.5 - low) * PIXELS_PER_M).astype(int)
image = Image.new("RGB", tuple(size), "white")
draw = ImageDraw.Draw(image)


def pixel(xy):
    return (
        float((xy[0] - low[0]) * PIXELS_PER_M),
        float(size[1] - (xy[1] - low[1]) * PIXELS_PER_M),
    )


for x, y in points[:: max(1, len(points) // 200_000)]:
    draw.point(pixel((x, y)), fill=(200, 200, 200))
for h in model.hypotheses:
    for edge in h.edges:
        colour = (0, 160, 0) if edge.status == "observed_wall" else (220, 0, 0)
        draw.line([pixel(edge.start), pixel(edge.end)], fill=colour, width=3)
        direction = (np.asarray(edge.end) - edge.start) / edge.length_m
        for a, b in edge.gaps:
            draw.line(
                [pixel(edge.start + a * direction), pixel(edge.start + b * direction)],
                fill=(255, 140, 0),
                width=5,
            )
    centre = h.outline.representative_point()
    draw.text(pixel((centre.x, centre.y)), h.room_id.replace("room:", ""), fill="black")
image.save(out / "rooms-top.png")
