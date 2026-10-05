"""Label text for plans. Display rounding happens only here, never in data."""

from __future__ import annotations

from floscan.contracts.inspection import Measurement


def length_label(measurement: Measurement | None) -> str:
    if measurement is None or measurement.value is None:
        return "n/a"
    marker = "*" if measurement.quality_status == "degraded" else ""
    return f"{measurement.value:.2f} m{marker}"


def area_label(measurement: Measurement | None) -> str:
    if measurement is None or measurement.value is None:
        return "area n/a"
    marker = "*" if measurement.quality_status == "degraded" else ""
    return f"{measurement.value:.2f} m2{marker}"


def height_label(measurement: Measurement | None) -> str:
    if measurement is None or measurement.value is None:
        return "ceiling n/a"
    return f"ceiling {measurement.value:.2f} m"


def room_lines(
    label: str,
    status: str,
    area: Measurement | None,
    ceiling: Measurement | None,
) -> list[str]:
    lines = [label, area_label(area), height_label(ceiling)]
    if status != "ok":
        lines.append(f"({status})")
    return lines


LEGEND = [
    ("wall", "observed wall"),
    ("unknown", "no wall observed (opening or unseen)"),
    ("note", "* corner or edge not fully observed; n/a = not measured"),
    ("note", "values unrounded in data; no accuracy interval is available yet"),
]
