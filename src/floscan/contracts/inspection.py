"""Measurement, interval, damage, concealed-damage and scope records.

Every reported quantity has exactly one unit, fixed by its quantity, and its
interval must use the same unit. Missing values are ``None`` with a reason and
an ``unavailable`` interval; they are never zero, and missing evidence never
produces a zero-width "certain" interval.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from floscan.contracts.base import (
    AssetRef,
    Contract,
    Id,
    MeasurementUnit,
    Probability,
    Record,
    Text,
    Tier,
    Vec2,
)
from floscan.contracts.geometry import Polygon2D

Quantity = Literal[
    "wall_length",
    "wall_height",
    "ceiling_height",
    "room_length",
    "room_width",
    "floor_area",
    "opening_width",
    "opening_height",
    "sill_height",
    "wall_thickness",
    "footprint_area",
    "footprint_extent",
    "damage_area",
    "damage_length",
    "damage_width",
    "scope_area",
    "scope_length",
    "scope_count",
    "angle",
]

# The one unit each quantity may be reported in.
QUANTITY_UNITS: dict[str, str] = {
    "wall_length": "m",
    "wall_height": "m",
    "ceiling_height": "m",
    "room_length": "m",
    "room_width": "m",
    "floor_area": "m2",
    "opening_width": "m",
    "opening_height": "m",
    "sill_height": "m",
    "wall_thickness": "m",
    "footprint_area": "m2",
    "footprint_extent": "m",
    "damage_area": "m2",
    "damage_length": "m",
    "damage_width": "m",
    "scope_area": "m2",
    "scope_length": "m",
    "scope_count": "count",
    "angle": "rad",
}
NON_NEGATIVE_QUANTITIES = frozenset(QUANTITY_UNITS) - {"angle"}


class ConfidenceInterval(Contract):
    """Predictive error interval for the unknown true value of a measurement."""

    lower: float | None
    upper: float | None
    unit: MeasurementUnit
    nominal_level: float = Field(gt=0.0, lt=1.0)
    kind: Literal["predictive_error_interval"] = "predictive_error_interval"
    calibration_id: Id | None = None
    calibration_tier: Tier | None = None
    quantity_family: Text
    method: Text
    support_n: int = Field(ge=0)
    independent_unit: Literal["property", "room_cluster"]
    coverage_scope: Literal["marginal", "familywise"]
    status: Literal["calibrated", "provisional", "unavailable", "out_of_domain"]
    reason: Text | None = None

    @model_validator(mode="after")
    def _bounds(self) -> ConfidenceInterval:
        if self.status == "unavailable":
            if self.lower is not None or self.upper is not None:
                raise ValueError("an unavailable interval has no bounds")
            if self.reason is None:
                raise ValueError("an unavailable interval needs a reason")
            return self
        if (self.lower is None or self.upper is None) and self.reason is None:
            raise ValueError("an unbounded side needs a reason")
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper:
                raise ValueError("interval lower bound exceeds upper bound")
            if self.lower == self.upper:
                raise ValueError("a zero-width interval claims certainty")
        if self.status == "calibrated":
            if self.lower is None or self.upper is None:
                raise ValueError("a calibrated interval needs finite bounds")
            if self.calibration_id is None or self.support_n == 0:
                raise ValueError("a calibrated interval needs a calibrator and support")
        if self.status == "out_of_domain" and self.reason is None:
            raise ValueError("an out-of-domain interval needs a reason")
        return self


class Measurement(Record):
    """One scalar quantity of one subject, with one definition and one unit."""

    subject_id: Id
    subject_geometry_version: int = Field(ge=1)
    quantity: Quantity
    definition_id: Id
    value: float | None
    unit: MeasurementUnit
    interval: ConfidenceInterval
    method: Text
    evidence_ids: list[Id] = Field(default_factory=list)
    sample_group_id: Id | None = None
    bias_correction_id: Id | None = None
    quality_status: Literal["ok", "degraded", "unavailable"]
    unavailable_reason: Text | None = None

    @model_validator(mode="after")
    def _units_and_availability(self) -> Measurement:
        expected = QUANTITY_UNITS[self.quantity]
        if self.unit != expected:
            raise ValueError(
                f"{self.quantity} is reported in {expected}, not {self.unit} "
                "(mixed units)"
            )
        if self.interval.unit != self.unit:
            raise ValueError(
                f"interval unit {self.interval.unit} differs from measurement unit "
                f"{self.unit}"
            )
        if self.value is None:
            if self.unavailable_reason is None or self.quality_status != "unavailable":
                raise ValueError(
                    "a missing value needs quality_status 'unavailable' and a reason"
                )
            if self.interval.status != "unavailable":
                raise ValueError("a missing value cannot have an available interval")
        else:
            if self.quality_status == "unavailable":
                raise ValueError("an available value cannot be marked unavailable")
            if self.interval.status == "unavailable" and self.status == "ok":
                raise ValueError(
                    "every reported value needs an interval; mark the record partial "
                    "with a reason if the interval is unavailable"
                )
        if self.quantity in NON_NEGATIVE_QUANTITIES:
            if self.value is not None and self.value < 0:
                raise ValueError(f"{self.quantity} cannot be negative")
            if self.interval.lower is not None and self.interval.lower < 0:
                raise ValueError(f"{self.quantity} interval cannot extend below zero")
        return self


class ImageMask(Contract):
    frame_id: Id
    mask: AssetRef


class DamageRegion(Record):
    """One physical damage region, fused across views; image areas are not summed."""

    room_id: Id
    surface_id: Id | None = None
    surface_version: int | None = Field(default=None, ge=1)
    damage_class: Literal["stain", "crack", "other", "unknown"]
    class_probability: Probability | None = None
    classification_calibration_id: Id | None = None
    image_masks: list[ImageMask] = Field(default_factory=list)
    polygon_uv: Polygon2D | None = None
    centreline_uv: list[Vec2] | None = Field(default=None, min_length=2)
    extent_measurement_ids: list[Id] = Field(default_factory=list)
    association_score: Probability
    observed_fraction: Probability | None = None
    truncation_flags: list[Text] = Field(default_factory=list)
    observation_ids: list[Id] = Field(default_factory=list)

    @model_validator(mode="after")
    def _surface_geometry(self) -> DamageRegion:
        if (self.surface_id is None) != (self.surface_version is None):
            raise ValueError("surface_id and surface_version go together")
        if self.surface_id is None and (
            self.polygon_uv is not None or self.centreline_uv is not None
        ):
            raise ValueError("metric damage geometry needs an associated surface")
        return self


Scalar = float | bool | str | None


class PredicateEvaluation(Contract):
    """One inspectable predicate of a rule, with what was observed."""

    name: Text
    observed: Scalar
    comparator: Literal["<", "<=", ">", ">=", "==", "!=", "in", "present"]
    threshold: Scalar
    satisfied: bool | None
    evidence_ids: list[Id] = Field(default_factory=list)


class ConcealedDamageFlag(Record):
    """A rule recommends inspection; it does not prove a hidden defect."""

    room_id: Id
    surface_id: Id | None = None
    flag_status: Literal["inspect", "not_triggered", "insufficient_evidence"]
    rule_id: Id
    rule_version: Text
    predicates: list[PredicateEvaluation] = Field(min_length=1)
    affected_location: Text | None = None
    uncertainty_dependency_ids: list[Id] = Field(default_factory=list)
    explanation: Text

    @model_validator(mode="after")
    def _consistent_with_predicates(self) -> ConcealedDamageFlag:
        outcomes = [p.satisfied for p in self.predicates]
        if self.flag_status == "inspect" and not all(o is True for o in outcomes):
            raise ValueError("an 'inspect' flag needs every predicate satisfied")
        if self.flag_status == "insufficient_evidence" and None not in outcomes:
            raise ValueError("'insufficient_evidence' needs an unevaluated predicate")
        if self.flag_status == "not_triggered" and (
            None in outcomes or all(o is True for o in outcomes)
        ):
            raise ValueError("'not_triggered' needs a predicate evaluated as false")
        return self


class ScopeItem(Record):
    """A line of work keyed to one surface version; never a cost."""

    room_id: Id
    surface_id: Id
    surface_version: int = Field(ge=1)
    action_code: Id
    description: Text
    observed_region_ids: list[Id] = Field(default_factory=list)
    concealed_flag_ids: list[Id] = Field(default_factory=list)
    quantity_measurement_id: Id | None = None
    rule_id: Id
    rule_version: Text
    assumptions: list[Text] = Field(default_factory=list)
    deduplication_key: Text
    scope_status: Literal["observed_work", "inspection_required", "unavailable"]

    @model_validator(mode="after")
    def _traceable(self) -> ScopeItem:
        if self.scope_status == "observed_work":
            if not self.observed_region_ids or self.quantity_measurement_id is None:
                raise ValueError("observed work needs observed regions and a quantity")
        if self.scope_status == "inspection_required" and not self.concealed_flag_ids:
            raise ValueError("an inspection item needs the concealed flag behind it")
        return self
