"""Case manifest tests (P05): evidence bookkeeping, never ground-truth values."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmark.cases.manifest import (
    Case,
    CaseManifest,
    check_splits,
    load_manifests,
    raw_manifest_hash,
    status_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HASH_A, HASH_B, HASH_C, HASH_D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
APP = {"name": "native Camera", "version": "iOS 26.0"}


def _capture(capture_id: str, tier: str, hash_: str, **extra: Any) -> dict:
    data = {
        "capture_id": capture_id,
        "tier": tier,
        "scope": "whole_property",
        "purpose": "primary",
        "status": "received",
        "path": f"data/raw/P1/{capture_id}",
        "raw_manifest_hash": hash_,
        "device_model": "iPhone 16 Pro",
        "app": APP,
        "recorded_on": "2026-10-05",
    }
    if tier == "photo":
        data["photo_counts"] = {
            "room_001": 6,
            "room_002": 7,
            "room_003": 8,
            "room_004": 5,
        }
    data.update(extra)
    return data


def _complete_case() -> dict:
    """Every required item present, by recorded evidence only."""
    return {
        "alias": "dev_property",
        "split": "development",
        "property_id": "P1",
        "description": "test property",
        "rooms": [
            {"id": "R01", "label": "living", "kind": "room"},
            {
                "id": "R02",
                "label": "bedroom",
                "kind": "room",
                "furnished": True,
                "staged_damage_classes": ["stain", "crack"],
            },
            {"id": "R03", "label": "kitchen", "kind": "room"},
            {"id": "R04", "label": "hall", "kind": "connector"},
        ],
        "captures": [
            _capture("photo_01", "photo", HASH_A),
            _capture("video_01", "video", HASH_B),
            _capture("lidar_01", "lidar", HASH_C),
            _capture(
                "lidar_02",
                "lidar",
                HASH_D,
                scope="room",
                room_ids=["R02"],
                purpose="repeat",
                repeat_of="lidar_01",
            ),
        ],
        "ground_truth": {
            "status": "complete",
            "measurements_file": "benchmark/ground_truth/records/P1/measurements.json",
            "measurements_sha256": HASH_A,
            "ground_truth_file": "benchmark/ground_truth/records/P1/ground_truth.json",
            "ground_truth_sha256": HASH_B,
            "instrument": "tape-1",
            "operator": "operator",
        },
        "incumbent": {
            "app": {"name": "magicplan", "version": "9.9"},
            "status": "received",
            "room_ids": ["R01", "R03"],
            "export_files": ["data/raw/P1/magicplan/export.pdf"],
            "export_sha256": [HASH_C],
            "export_kind": "app_export",
        },
        "declared_complete": True,
    }


# --------------------------------------------------------------------------
# Committed manifests
# --------------------------------------------------------------------------


def test_committed_manifests_load_and_show_status() -> None:
    manifests = load_manifests()
    assert set(manifests) == {"development", "calibration", "heldout"}
    report = status_report(manifests)
    assert "## dev_property (development, property P1): 0/8 met" in report
    assert "duplicate of sample-1a8384c3f6" in report
    dev = manifests["development"].cases[0]
    assert not dev.declared_complete
    assert all(not item["met"] for item in dev.composition())


def test_status_command_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "benchmark.cases.manifest", "status"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "- [ ] whole-property lidar capture received" in completed.stdout


def test_complete_case_can_be_declared_complete() -> None:
    case = Case.model_validate(_complete_case())
    assert all(item["met"] for item in case.composition())


# --------------------------------------------------------------------------
# Duplicate recordings are never repeats
# --------------------------------------------------------------------------


def test_identical_recording_rejected_as_repeat() -> None:
    data = _complete_case()
    data["captures"][3]["raw_manifest_hash"] = HASH_C  # same content as lidar_01
    with pytest.raises(ValueError, match="same recording as lidar_01"):
        Case.model_validate(data)


def test_same_folder_rejected_as_repeat() -> None:
    data = _complete_case()
    data["captures"][3]["path"] = data["captures"][2]["path"]
    with pytest.raises(ValueError, match="same recording as lidar_01"):
        Case.model_validate(data)


def test_real_sample_copy_cannot_be_a_repeat() -> None:
    fixtures = {f.id: f for f in load_manifests()["development"].fixtures}
    original, copy_ = fixtures["sample-1a8384c3f6"], fixtures["sample-1a8384c3f6-copy"]
    assert original.raw_manifest_hash == copy_.raw_manifest_hash
    data = _complete_case()
    data["captures"][2].update(
        path=original.path, raw_manifest_hash=original.raw_manifest_hash
    )
    data["captures"][3].update(
        path=copy_.path, raw_manifest_hash=copy_.raw_manifest_hash
    )
    with pytest.raises(ValueError, match="a copy"):
        Case.model_validate(data)


def test_undeclared_fixture_duplicate_is_rejected() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "benchmark/manifests/development.json").read_text()
    )
    del manifest["fixtures"][1]["duplicate_of"]
    with pytest.raises(ValueError, match="byte-identical"):
        CaseManifest.model_validate(manifest)


# --------------------------------------------------------------------------
# Missing fields are never complete evidence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["captures"][1].pop("raw_manifest_hash"), "lacks"),
        (
            lambda d: d["captures"][1].update(app={"name": "native Camera"}),
            "app.version",
        ),
        (lambda d: d["captures"][0].pop("photo_counts"), "per-room counts"),
        (
            lambda d: d["ground_truth"].pop("ground_truth_file"),
            "marked complete but lacks",
        ),
        (
            lambda d: d["ground_truth"].update(missing_items=["ceiling R03"]),
            "marked complete but lacks",
        ),
        (lambda d: d["ground_truth"].pop("measurements_file"), "measurements file"),
        (lambda d: d["incumbent"].update(room_ids=["R01"]), "two distinct rooms"),
        (lambda d: d["incumbent"].update(export_sha256=[]), "files and their hashes"),
    ],
    ids=[
        "no-hash",
        "no-app-version",
        "no-photo-counts",
        "gt-no-derived-file",
        "gt-open-items",
        "gt-no-sheet",
        "incumbent-one-room",
        "incumbent-no-hash",
    ],
)
def test_incomplete_evidence_is_rejected(mutate, message: str) -> None:
    data = _complete_case()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        Case.model_validate(data)


@pytest.mark.parametrize(
    ("mutate", "unmet"),
    [
        (
            lambda d: d["captures"][1].update(status="missing"),
            "whole-property video capture received",
        ),
        (lambda d: d["captures"].pop(3), "one room captured twice"),
        (
            lambda d: d["rooms"][1].update(staged_damage_classes=["stain"]),
            "staged damage of two classes",
        ),
        (lambda d: d["rooms"].pop(3), "plus a connector"),
        (lambda d: d["ground_truth"].update(status="partial"), "tape ground truth"),
        (lambda d: d.update(incumbent=None), "incumbent export"),
    ],
    ids=[
        "missing-tier",
        "no-repeat",
        "one-damage-class",
        "no-connector",
        "partial-gt",
        "no-incumbent",
    ],
)
def test_case_cannot_be_declared_complete_with_gaps(mutate, unmet: str) -> None:
    data = _complete_case()
    mutate(data)
    with pytest.raises(ValueError, match=unmet):
        Case.model_validate(data)
    data["declared_complete"] = False
    items = Case.model_validate(data).composition()
    assert not next(i for i in items if unmet in i["item"])["met"]


@pytest.mark.parametrize(
    ("count", "ok"), [(1, False), (2, True), (8, True), (9, False)]
)
def test_photo_tier_takes_two_to_eight_per_room(count: int, ok: bool) -> None:
    data = _complete_case()
    data["captures"][0]["photo_counts"]["room_002"] = count
    if ok:
        Case.model_validate(data)
    else:
        with pytest.raises(ValueError, match="2 to 8"):
            Case.model_validate(data)


def test_manifest_cannot_carry_ground_truth_values() -> None:
    data = _complete_case()
    data["rooms"][0]["ceiling_height_m"] = 2.41
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Case.model_validate(data)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def _manifest(split: str, cases: list[dict]) -> CaseManifest:
    return CaseManifest.model_validate(
        {"manifest_version": 1, "split": split, "cases": cases}
    )


def _other_case(alias: str, split: str, property_id: str, captures: list) -> dict:
    return {
        "alias": alias,
        "split": split,
        "property_id": property_id,
        "description": "x",
        "captures": captures,
        "ground_truth": {"status": "missing"},
    }


def test_property_cannot_sit_in_two_splits() -> None:
    manifests = {
        "development": _manifest("development", [_complete_case()]),
        "calibration": _manifest(
            "calibration", [_other_case("calibration_set", "calibration", "P1", [])]
        ),
    }
    with pytest.raises(ValueError, match="keep each property in one split"):
        check_splits(manifests)


def test_recording_cannot_count_in_two_cases() -> None:
    reused = _capture("photo_01", "photo", HASH_A)
    manifests = {
        "development": _manifest("development", [_complete_case()]),
        "heldout": _manifest(
            "heldout", [_other_case("heldout_property", "heldout", "P3", [reused])]
        ),
    }
    with pytest.raises(ValueError, match="across cases"):
        check_splits(manifests)


def test_alias_must_match_split() -> None:
    with pytest.raises(ValueError, match="belongs to split development"):
        Case.model_validate({**_complete_case(), "split": "heldout"})


# --------------------------------------------------------------------------
# Raw manifest hash
# --------------------------------------------------------------------------


def test_raw_manifest_hash_identifies_content_not_names(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b copy"
    for folder in (a, b):
        (folder / "depth").mkdir(parents=True)
        (folder / "rgb.mp4").write_bytes(b"video")
        (folder / "depth" / "000000.png").write_bytes(b"depth")
    (b / ".DS_Store").write_bytes(b"finder")
    assert raw_manifest_hash(a) == raw_manifest_hash(b)
    (b / "depth" / "000000.png").write_bytes(b"depth!")
    assert raw_manifest_hash(a) != raw_manifest_hash(b)
    before = raw_manifest_hash(a)
    (a / "depth" / "000000.png").rename(a / "depth" / "000001.png")
    assert raw_manifest_hash(a) != before  # a renamed frame is a different recording
