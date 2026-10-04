"""Capture and frame records (schema ``capture-v0``).

A Capture is the immutable manifest of what a declared tier is allowed to use.
Strict tier isolation is enforced here: photo captures cannot declare depth,
pose or IMU modalities, and RGB video cannot ingest sensor sidecars. Room
folder names are labels only; they never imply adjacency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from floscan.contracts.base import (
    CAPTURE_SCHEMA_VERSION,
    AssetRef,
    Contract,
    Id,
    Positive,
    Record,
    Sha256,
    Text,
    Tier,
)
from floscan.contracts.geometry import PixelTransform

Modality = Literal[
    "rgb_image",
    "rgb_video",
    "image_metadata_intrinsics",
    "depth",
    "depth_confidence",
    "pose",
    "intrinsics",
    "imu",
]

# Modalities each strict tier may consume. Metadata intrinsics (EXIF focal
# length) are allowed for RGB tiers; depth, pose and IMU never are.
ALLOWED_MODALITIES: dict[str, frozenset[str]] = {
    "photo": frozenset({"rgb_image", "image_metadata_intrinsics"}),
    "video": frozenset({"rgb_video", "image_metadata_intrinsics"}),
    "lidar": frozenset(
        {
            "rgb_video",
            "rgb_image",
            "depth",
            "depth_confidence",
            "pose",
            "intrinsics",
            "imu",
        }
    ),
}
PHOTO_FRAMES_PER_ROOM = (2, 8)


class SourceApp(Contract):
    name: Text
    version: Text | None = None


class Device(Contract):
    model: Text | None = None
    os: Text | None = None
    lens: Text | None = None


class Clock(Contract):
    id: Id
    description: Text
    unit: Literal["s"]


class RoomGroup(Contract):
    """Frames the operator grouped under one room folder; a label, not geometry."""

    id: Id
    label: Text
    frame_ids: list[Id] = Field(min_length=1)


class Capture(Record):
    schema_version: Literal["capture-v0"] = CAPTURE_SCHEMA_VERSION
    property_session_id: Id
    tier: Tier
    source_app: SourceApp | None = None
    device: Device | None = None
    assets: list[AssetRef] = Field(min_length=1)
    frames_uri: Text
    room_groups: list[RoomGroup] = Field(default_factory=list)
    clock_domains: list[Clock] = Field(default_factory=list)
    allowed_modalities: list[Modality] = Field(min_length=1)
    scale_evidence_ids: list[Id] = Field(default_factory=list)
    capture_profile_id: Id
    raw_manifest_hash: Sha256
    quality_report_id: Id | None = None

    @model_validator(mode="after")
    def _tier_isolation(self) -> Capture:
        forbidden = set(self.allowed_modalities) - ALLOWED_MODALITIES[self.tier]
        if forbidden:
            raise ValueError(
                f"strict {self.tier} tier cannot use {sorted(forbidden)}; a "
                "sensor-assisted route must be a separately labelled capture"
            )
        if self.tier == "photo":
            low, high = PHOTO_FRAMES_PER_ROOM
            if not self.room_groups:
                raise ValueError("a photo capture needs one room group per room folder")
            for group in self.room_groups:
                if not low <= len(group.frame_ids) <= high:
                    raise ValueError(
                        f"room {group.label!r} has {len(group.frame_ids)} photos; "
                        f"the photo tier takes {low} to {high} per room"
                    )
        frame_ids = [fid for group in self.room_groups for fid in group.frame_ids]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("a frame belongs to at most one room group")
        return self


class FrameQuality(Contract):
    blur: float | None = None
    exposure: float | None = None
    tracking: Text | None = None
    reason_codes: list[Text] = Field(default_factory=list)


class Frame(Record):
    """One image (still or decoded video frame) with its own timing and camera."""

    capture_id: Id
    source_index: int = Field(ge=0)
    timestamp_s: float | None = None
    clock_id: Id | None = None
    presentation_time_s: float | None = None
    image: AssetRef
    camera_id: Id
    pose_id: Id | None = None
    depth_id: Id | None = None
    exposure_s: Positive | None = None
    orientation: PixelTransform
    quality: FrameQuality = Field(default_factory=FrameQuality)

    @model_validator(mode="after")
    def _timing(self) -> Frame:
        if self.timestamp_s is not None and self.clock_id is None:
            raise ValueError("a timestamp needs the clock it was measured on")
        return self
