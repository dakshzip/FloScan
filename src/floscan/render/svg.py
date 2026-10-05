"""Plan rendering. The renderer draws record geometry and never alters it.

``build_drawing`` maps world (x, y) metres to canvas pixels with a single
recorded similarity (scale and translation, y flipped). SVG and PNG are both
drawn from that drawing, so they show the same geometry; the SVG stores the
transform in ``data-*`` attributes so coordinates can be mapped back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

from floscan.contracts.geometry import Room, Wall
from floscan.contracts.inspection import Measurement
from floscan.render.labels import LEGEND, length_label, room_lines

PIXELS_PER_M = 60.0
MARGIN_PX = 60.0
LEGEND_PX = 120.0
# Wall labels are drawn only for walls at least this long; every length is
# still in the measurement records. Text width is estimated per character.
MIN_LABELLED_WALL_M = 0.5
CHAR_PX = 7.0
COLOURS = {
    "fill": "#eef3fb",
    "wall": "#1f2933",
    "unknown": "#d64545",
    "text": "#1f2933",
    "note": "#52606d",
}


@dataclass(frozen=True)
class Transform:
    """canvas = ((x - x0) * scale + margin, (y1 - y) * scale + margin)."""

    x0: float
    y1: float
    scale: float
    margin: float

    def __call__(self, xy) -> tuple[float, float]:
        x, y = float(xy[0]), float(xy[1])
        return (
            (x - self.x0) * self.scale + self.margin,
            (self.y1 - y) * self.scale + self.margin,
        )

    def inverse(self, px: float, py: float) -> tuple[float, float]:
        return (
            (px - self.margin) / self.scale + self.x0,
            self.y1 - (py - self.margin) / self.scale,
        )


@dataclass
class Drawing:
    width: float
    height: float
    transform: Transform
    title: str
    rooms: list[tuple[str, list[list[tuple[float, float]]]]] = field(
        default_factory=list
    )  # room id, rings (canvas)
    edges: list[tuple[str, tuple[float, float], tuple[float, float]]] = field(
        default_factory=list
    )  # kind, start, end (canvas)
    texts: list[tuple[float, float, str, int, str]] = field(
        default_factory=list
    )  # x, y, text, size, colour key
    scale_bar_m: float = 1.0


def _scale_bar_length(extent_m: float) -> float:
    for candidate in (0.5, 1.0, 2.0, 5.0, 10.0):
        if candidate >= extent_m / 8:
            return candidate
    return 10.0


def build_drawing(
    rooms: list[Room],
    walls: list[Wall],
    measurements: list[Measurement],
    title: str,
) -> Drawing:
    """Canvas geometry and labels for every room (records unchanged)."""
    by_subject: dict[tuple[str, str], Measurement] = {
        (m.subject_id, m.quantity): m for m in measurements
    }
    rings_world = [
        [r.boundary.outer, *r.boundary.holes] for r in rooms if r.boundary is not None
    ]
    points = np.array([p for rings in rings_world for ring in rings for p in ring])
    if len(points) == 0:
        points = np.zeros((1, 2))
    low, high = points.min(axis=0), points.max(axis=0)
    transform = Transform(float(low[0]), float(high[1]), PIXELS_PER_M, MARGIN_PX)
    longest_note = max(len(text) for _, text in LEGEND)
    width = max(
        (high[0] - low[0]) * PIXELS_PER_M + 2 * MARGIN_PX,
        len(title) * 7.5 + 2 * MARGIN_PX,
        MARGIN_PX + 180 + longest_note * CHAR_PX + MARGIN_PX,
    )
    height = (high[1] - low[1]) * PIXELS_PER_M + 2 * MARGIN_PX + LEGEND_PX
    drawing = Drawing(width, height, transform, title)
    drawing.scale_bar_m = _scale_bar_length(float(max(high - low)))
    by_wall = {w.id: w for w in walls}
    placed: list[tuple[float, float, float, float]] = []
    for room in rooms:
        if room.boundary is None:
            continue
        rings = [room.boundary.outer, *room.boundary.holes]
        drawing.rooms.append(
            (room.id, [[transform(p) for p in ring] for ring in rings])
        )
        wall_edges = {
            (tuple(np.round(by_wall[w].baseline[0][:2], 9)),
             tuple(np.round(by_wall[w].baseline[1][:2], 9))): w
            for w in room.wall_ids
        }  # fmt: skip
        outline = Polygon(room.boundary.outer, room.boundary.holes)
        for ring in rings:
            for k in range(len(ring)):
                a, b = ring[k], ring[(k + 1) % len(ring)]
                key = (tuple(np.round(a, 9)), tuple(np.round(b, 9)))
                wall_id = wall_edges.get(key)
                kind = "wall" if wall_id else "unknown"
                drawing.edges.append((kind, transform(a), transform(b)))
                if wall_id and np.linalg.norm(np.subtract(b, a)) >= MIN_LABELLED_WALL_M:
                    middle = (np.asarray(a) + np.asarray(b)) / 2
                    direction = np.asarray(b) - np.asarray(a)
                    inward = np.array([-direction[1], direction[0]])
                    inward /= np.linalg.norm(inward)
                    text = length_label(by_subject.get((wall_id, "wall_length")))
                    # Clear the wall by the label's own extent along the normal
                    # (canvas y is flipped).
                    half_w = len(text) * CHAR_PX * 11 / 24
                    offset_px = 6 + abs(inward[0]) * half_w + abs(inward[1]) * 7
                    cx, cy = transform(middle)
                    x = cx + inward[0] * offset_px
                    y = cy - inward[1] * offset_px
                    _place(drawing, placed, (x, y, text, 11, "text"))
        anchor = outline.representative_point()
        x, y = transform((anchor.x, anchor.y))
        lines = room_lines(
            room.label,
            room.status,
            by_subject.get((room.id, "floor_area")),
            by_subject.get((room.id, "ceiling_height")),
        )
        for k, line in enumerate(lines):
            _place(
                drawing,
                placed,
                (x, y + 14 * (k - len(lines) / 2), line, 12, "text"),
                force=True,
            )
    return drawing


def _place(
    drawing: Drawing,
    placed: list[tuple[float, float, float, float]],
    text: tuple[float, float, str, int, str],
    force: bool = False,
) -> None:
    """Add a label unless it would overlap one already placed."""
    x, y, content, size, _ = text
    half_w, half_h = len(content) * CHAR_PX * size / 24, size * 0.6
    box = (x - half_w, y - half_h, x + half_w, y + half_h)
    overlaps = any(
        box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
        for b in placed
    )
    if overlaps and not force:
        return
    placed.append(box)
    drawing.texts.append(text)


def to_svg(drawing: Drawing) -> str:
    t = drawing.transform
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{drawing.width:.1f}" '
        f'height="{drawing.height:.1f}" viewBox="0 0 {drawing.width:.6f} '
        f'{drawing.height:.6f}" data-x0="{t.x0!r}" data-y1="{t.y1!r}" '
        f'data-scale="{t.scale!r}" data-margin="{t.margin!r}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{t.margin:.1f}" y="24" font-family="sans-serif" font-size="14" '
        f'fill="{COLOURS["text"]}">{escape(drawing.title)}</text>',
    ]
    for room_id, rings in drawing.rooms:
        path = " ".join(
            "M " + " L ".join(f"{x!r},{y!r}" for x, y in ring) + " Z" for ring in rings
        )
        parts.append(
            f'<path data-room="{escape(room_id)}" d="{path}" fill="{COLOURS["fill"]}" '
            'fill-rule="evenodd" stroke="none"/>'
        )
    for kind, (x1, y1), (x2, y2) in drawing.edges:
        dash = ' stroke-dasharray="6,4"' if kind == "unknown" else ""
        width = 3 if kind == "wall" else 2
        parts.append(
            f'<line data-edge="{kind}" x1="{x1!r}" y1="{y1!r}" x2="{x2!r}" y2="{y2!r}" '
            f'stroke="{COLOURS[kind]}" stroke-width="{width}"{dash}/>'
        )
    for x, y, text, size, colour in drawing.texts:
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
            f'font-size="{size}" text-anchor="middle" fill="{COLOURS[colour]}">'
            f"{escape(text)}</text>"
        )
    parts.extend(_scale_bar_svg(drawing))
    parts.extend(_legend_svg(drawing))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _scale_bar_svg(drawing: Drawing) -> list[str]:
    t = drawing.transform
    y = drawing.height - LEGEND_PX + 20
    x0, x1 = t.margin, t.margin + drawing.scale_bar_m * t.scale
    return [
        f'<line data-scale-bar="{drawing.scale_bar_m!r}" x1="{x0:.1f}" y1="{y:.1f}" '
        f'x2="{x1:.1f}" y2="{y:.1f}" stroke="black" stroke-width="3"/>',
        f'<text x="{x0:.1f}" y="{y + 16:.1f}" font-family="sans-serif" font-size="12">'
        f"{drawing.scale_bar_m:g} m</text>",
    ]


def _legend_svg(drawing: Drawing) -> list[str]:
    x = drawing.transform.margin + 140
    y = drawing.height - LEGEND_PX + 20
    parts = []
    for k, (kind, text) in enumerate(LEGEND):
        yy = y + 18 * k
        if kind in ("wall", "unknown"):
            dash = ' stroke-dasharray="6,4"' if kind == "unknown" else ""
            parts.append(
                f'<line x1="{x:.1f}" y1="{yy:.1f}" x2="{x + 30:.1f}" y2="{yy:.1f}" '
                f'stroke="{COLOURS[kind]}" stroke-width="3"{dash}/>'
            )
        parts.append(
            f'<text x="{x + 40:.1f}" y="{yy + 4:.1f}" font-family="sans-serif" '
            f'font-size="12" fill="{COLOURS["note"]}">{escape(text)}</text>'
        )
    return parts


def to_png(drawing: Drawing) -> Image.Image:
    """Raster of the same drawing (same canvas coordinates as the SVG)."""
    image = Image.new(
        "RGB", (int(np.ceil(drawing.width)), int(np.ceil(drawing.height))), "white"
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=12)
    title_font = ImageFont.load_default(size=14)
    for _, rings in drawing.rooms:
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(rings[0], fill=255)
        for hole in rings[1:]:
            mask_draw.polygon(hole, fill=0)
        image.paste(COLOURS["fill"], mask=mask)
    for kind, start, end in drawing.edges:
        if kind == "wall":
            draw.line([start, end], fill=COLOURS["wall"], width=3)
        else:
            _dashed(draw, start, end, COLOURS["unknown"])
    fonts = {}
    for x, y, text, size, colour in drawing.texts:
        fonts.setdefault(size, ImageFont.load_default(size=size))
        draw.text((x, y), text, fill=COLOURS[colour], font=fonts[size], anchor="mm")
    draw.text(
        (drawing.transform.margin, 16),
        drawing.title,
        fill=COLOURS["text"],
        font=title_font,
    )
    t = drawing.transform
    y = drawing.height - LEGEND_PX + 20
    draw.line(
        [(t.margin, y), (t.margin + drawing.scale_bar_m * t.scale, y)],
        fill="black",
        width=3,
    )
    draw.text((t.margin, y + 6), f"{drawing.scale_bar_m:g} m", fill="black", font=font)
    x = t.margin + 140
    for k, (kind, text) in enumerate(LEGEND):
        yy = y + 18 * k
        if kind == "wall":
            draw.line([(x, yy), (x + 30, yy)], fill=COLOURS["wall"], width=3)
        elif kind == "unknown":
            _dashed(draw, (x, yy), (x + 30, yy), COLOURS["unknown"])
        draw.text((x + 40, yy - 6), text, fill=COLOURS["note"], font=font)
    return image


def _dashed(draw: ImageDraw.ImageDraw, start, end, colour: str) -> None:
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b - a))
    if length == 0:
        return
    direction = (b - a) / length
    for s in np.arange(0.0, length, 10.0):
        e = min(s + 6.0, length)
        draw.line(
            [tuple(a + s * direction), tuple(a + e * direction)], fill=colour, width=2
        )
