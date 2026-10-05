"""Measurement definitions: what each reported quantity means, exactly.

Every Measurement names one of these by ``definition_id``. Definitions are
versioned; changing a convention means a new ID, never an edited meaning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Definition:
    id: str
    quantity: str
    unit: str
    text: str


DEFINITIONS = {
    d.id: d
    for d in (
        Definition(
            "def:wall_length:v1",
            "wall_length",
            "m",
            "Horizontal (plan) length of a wall's interior finished face along "
            "its intersection with the floor plane, between the room-outline "
            "corners where it meets the adjoining walls.",
        ),
        Definition(
            "def:wall_height:v1",
            "wall_height",
            "m",
            "Vertical distance from the room's floor plane to its ceiling plane "
            "along the wall face, averaged over the wall's two ends. Requires an "
            "observed ceiling over the room.",
        ),
        Definition(
            "def:ceiling_height:v1",
            "ceiling_height",
            "m",
            "Vertical distance between the room's floor plane and ceiling plane "
            "above the centroid of the room outline. Requires an observed "
            "ceiling over the room.",
        ),
        Definition(
            "def:floor_area:v1",
            "floor_area",
            "m2",
            "Horizontal (plan) area enclosed by the room outline, holes "
            "(pillars, shafts) excluded.",
        ),
        Definition(
            "def:room_length:v1",
            "room_length",
            "m",
            "Longer side of the minimum-area rectangle enclosing the room "
            "outline in plan.",
        ),
        Definition(
            "def:room_width:v1",
            "room_width",
            "m",
            "Shorter side of the minimum-area rectangle enclosing the room "
            "outline in plan.",
        ),
    )
}


def definition_for(quantity: str) -> Definition:
    for definition in DEFINITIONS.values():
        if definition.quantity == quantity:
            return definition
    raise KeyError(f"no measurement definition for {quantity!r}")
