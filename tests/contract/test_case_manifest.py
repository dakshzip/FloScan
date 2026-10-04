"""Case manifest tests (P05): evidence bookkeeping, never ground-truth values."""

from __future__ import annotations

import hashlib
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
    media_digests,
    raw_manifest_hash,
    status_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HASH_A, HASH_B, HASH_C, HASH_D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
APP = {"name": "native Camera", "version": "iOS 26.0"}
SKETCH = ["R01", "R02", "R03", "R04"]  # R04 is the connector


def _digests(label: str, count: int) -> list[str]:
    """Distinct synthetic media digests (test fixtures, not real files)."""
    return sorted(
        hashlib.sha256(f"{label}-{i}".encode()).hexdigest() for i in range(count)
    )


def _set_photo_groups(capture: dict, counts: dict[str, int]) -> None:
    """Photo groups room_00N -> sketch room, with matching media digests."""
    capture["photo_groups"] = [
        {"folder": f"room_{room[1:].zfill(3)}", "room_id": room, "count": count}
        for room, count in counts.items()
    ]
    capture["room_ids"] = list(counts)
    capture["media_sha256"] = _digests(capture["capture_id"], sum(counts.values()))


def _capture(capture_id: str, tier: str, hash_: str, **extra: Any) -> dict:
    data = {
        "capture_id": capture_id,
        "tier": tier,
        "scope": "whole_property",
        "room_ids": list(SKETCH),
        "purpose": "primary",
        "status": "received",
        "path": f"data/raw/P1/{capture_id}",
        "raw_manifest_hash": hash_,
        "media_sha256": _digests(capture_id, 1),
        "device_model": "iPhone 16 Pro",
        "app": APP,
        "recorded_on": "2026-10-05",
    }
    if tier == "photo":
        _set_photo_groups(data, {"R01": 6, "R02": 7, "R03": 8, "R04": 5})
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
    for capture, fixture in (
        (data["captures"][2], original),
        (data["captures"][3], copy_),
    ):
        capture.update(
            path=fixture.path,
            raw_manifest_hash=fixture.raw_manifest_hash,
            media_sha256=list(fixture.media_sha256),
        )
    with pytest.raises(ValueError, match="a copy"):
        Case.model_validate(data)


def test_undeclared_fixture_duplicate_is_rejected() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "benchmark/manifests/development.json").read_text()
    )
    del manifest["fixtures"][1]["duplicate_of"]
    with pytest.raises(ValueError, match="must say it is a duplicate"):
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
        (lambda d: d["captures"][0].pop("photo_groups"), "per-room photo groups"),
        (lambda d: d["captures"][1].update(media_sha256=[]), "media_sha256"),
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
        "no-photo-groups",
        "no-media-identity",
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
        (lambda d: d["rooms"][3].update(kind="room"), "plus a connector"),
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
    _set_photo_groups(data["captures"][0], {"R01": 6, "R02": count, "R03": 8, "R04": 5})
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
        "rooms": _complete_case()["rooms"],
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


def test_raw_manifest_hash_is_provenance_of_the_tree(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b copy"
    for folder in (a, b):
        (folder / "depth").mkdir(parents=True)
        (folder / "rgb.mp4").write_bytes(b"video")
        (folder / "depth" / "000000.png").write_bytes(b"depth")
    (b / ".DS_Store").write_bytes(b"finder")
    assert raw_manifest_hash(a) == raw_manifest_hash(b)  # root folder name ignored
    (b / "depth" / "000000.png").write_bytes(b"depth!")
    assert raw_manifest_hash(a) != raw_manifest_hash(b)
    before = raw_manifest_hash(a)
    (a / "depth" / "000000.png").rename(a / "depth" / "000001.png")
    assert raw_manifest_hash(a) != before  # provenance pins names, identity does not


def test_media_digests_survive_renaming(tmp_path: Path) -> None:
    a, b = tmp_path / "lidar_01", tmp_path / "elsewhere"
    (a / "depth").mkdir(parents=True)
    (b / "nested").mkdir(parents=True)
    (a / "rgb.mp4").write_bytes(b"same video")
    (a / "depth" / "000000.png").write_bytes(b"depth")
    (b / "nested" / "renamed.MP4").write_bytes(b"same video")
    (b / "odometry.csv").write_bytes(b"edited metadata")
    assert raw_manifest_hash(a) != raw_manifest_hash(b)
    assert media_digests(a) == media_digests(b)  # PNG and CSV are not media
    assert media_digests(a) == [hashlib.sha256(b"same video").hexdigest()]


def test_media_digests_keep_repeated_photos(tmp_path: Path) -> None:
    (tmp_path / "IMG_1.JPG").write_bytes(b"photo")
    (tmp_path / "IMG_2.jpg").write_bytes(b"photo")
    assert len(media_digests(tmp_path)) == 2


# --------------------------------------------------------------------------
# P05A: coverage is bound to unique sketch rooms, connector included
# --------------------------------------------------------------------------


def test_sketch_room_ids_are_unique() -> None:
    data = _complete_case()
    data["rooms"][2] = dict(data["rooms"][1])
    with pytest.raises(ValueError, match="sketch room id appears twice"):
        Case.model_validate(data)


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (
            lambda d: _set_photo_groups(d["captures"][0], {"R01": 2}),
            "photo_01: missing R02, R03, R04",
        ),
        (
            lambda d: d["captures"][1].update(room_ids=["R01"]),
            "video_01: missing R02, R03, R04",
        ),
        (
            lambda d: d["captures"][2].update(room_ids=["R01", "R02", "R03"]),
            "lidar_01: missing R04",
        ),
        (
            lambda d: d["captures"][1].update(room_ids=[]),
            "video_01: room coverage not recorded",
        ),
    ],
    ids=["photos-one-room", "video-one-room", "lidar-no-connector", "no-coverage"],
)
def test_partial_whole_property_coverage_stays_unmet(mutate, detail: str) -> None:
    data = _complete_case()
    mutate(data)
    with pytest.raises(ValueError, match="whole-property"):
        Case.model_validate(data)
    data["declared_complete"] = False
    case = Case.model_validate(data)  # partial data is kept, not rejected
    tier = next(i for i in case.composition() if detail.split("_")[0] in i["item"])
    assert not tier["met"]
    assert tier["detail"] == detail


def test_full_coverage_lists_every_sketched_room() -> None:
    items = Case.model_validate(_complete_case()).composition()
    photo = next(i for i in items if "photo" in i["item"])
    assert photo == {
        "item": "whole-property photo capture received",
        "met": True,
        "detail": "photo_01: covers all 4 sketched rooms",
    }


def test_coverage_needs_a_sketch() -> None:
    data = _complete_case()
    data.update(rooms=[], declared_complete=False, incumbent=None)
    for capture in data["captures"]:
        capture["room_ids"] = []
    data["captures"] = data["captures"][1:3]  # photo groups need sketch rooms
    items = Case.model_validate(data).composition()
    assert not any(i["met"] for i in items if "whole-property" in i["item"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d["captures"][0].update(room_ids=["R01", "R02", "R03"]),
            "must match the photo groups",
        ),
        (
            lambda d: d["captures"][0]["photo_groups"][1].update(room_id="R01"),
            "maps to its own room",
        ),
        (
            lambda d: d["captures"][0]["photo_groups"][1].update(folder="room_001"),
            "maps to its own room",
        ),
        (
            lambda d: d["captures"][0]["media_sha256"].pop(),
            "count 26 photos but media_sha256 lists 25",
        ),
        (
            lambda d: d["captures"][0]["media_sha256"].__setitem__(
                1, d["captures"][0]["media_sha256"][0]
            ),
            "same media file appears twice",
        ),
        (
            lambda d: d["captures"][0]["photo_groups"][0].update(room_id="R09"),
            "must match the photo groups",
        ),
        (
            lambda d: d["captures"][1].update(room_ids=["R01", "R01"]),
            "lists a room twice",
        ),
        (lambda d: d["captures"][1].update(room_ids=["R09"]), "unknown rooms"),
    ],
    ids=[
        "ids-differ-from-groups",
        "two-folders-one-room",
        "one-folder-twice",
        "media-count-mismatch",
        "padded-with-a-copy",
        "group-room-not-listed",
        "room-listed-twice",
        "unsketched-room",
    ],
)
def test_photo_groups_bind_folders_to_sketch_rooms(mutate, message: str) -> None:
    data = _complete_case()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        Case.model_validate(data)


# --------------------------------------------------------------------------
# P05A: repeat links name a distinct primary recording of the same room
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d["captures"][3].update(repeat_of="lidar_02"),
            "cannot repeat itself",
        ),
        (
            lambda d: d["captures"][2].update(purpose="repeat", repeat_of="lidar_02"),
            "itself a repeat",
        ),
        (
            lambda d: d["captures"][2].update(repeat_of="lidar_02"),
            "primary capture does not repeat",
        ),
        (
            lambda d: d["captures"][3].update(repeat_of="video_01"),
            "other-tier capture",
        ),
        (
            lambda d: d["captures"][2].update(room_ids=["R01", "R03", "R04"]),
            "a repeat records the same room",
        ),
    ],
    ids=["self", "cycle", "primary-with-link", "other-tier", "other-room"],
)
def test_invalid_repeat_links_are_rejected(mutate, message: str) -> None:
    data = _complete_case()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        Case.model_validate(data)


def test_whole_property_repeat_shares_rooms() -> None:
    data = _complete_case()
    data["captures"][3].update(scope="whole_property", room_ids=list(SKETCH))
    items = Case.model_validate(data).composition()
    repeat = next(i for i in items if "captured twice" in i["item"])
    assert repeat["met"]
    assert "in R01, R02, R03, R04" in repeat["detail"]


def test_repeat_detail_keeps_session_provenance_visible() -> None:
    data = _complete_case()
    data["captures"][3]["recorded_on"] = "2026-10-06"
    items = Case.model_validate(data).composition()
    repeat = next(i for i in items if "captured twice" in i["item"])["detail"]
    assert repeat == (
        "lidar_02 (2026-10-06) repeats lidar_01 (2026-10-05) in R02; "
        "distinct media, session per operator record"
    )


def test_renamed_copy_cannot_be_a_repeat(tmp_path: Path) -> None:
    first, second = tmp_path / "lidar_01", tmp_path / "lidar_02"
    first.mkdir()
    second.mkdir()
    (first / "rgb.mp4").write_bytes(b"one recording")
    (second / "renamed.mp4").write_bytes(b"one recording")
    data = _complete_case()
    for capture, folder in (
        (data["captures"][2], first),
        (data["captures"][3], second),
    ):
        capture.update(
            path=str(folder),
            raw_manifest_hash=raw_manifest_hash(folder),
            media_sha256=media_digests(folder),
        )
    assert (
        data["captures"][2]["raw_manifest_hash"]
        != (data["captures"][3]["raw_manifest_hash"])
    )
    with pytest.raises(ValueError, match="same recording as lidar_01"):
        Case.model_validate(data)


def test_partly_copied_photo_set_is_the_same_recording() -> None:
    data = _complete_case()
    data["captures"][3].update(
        capture_id="photo_02",
        tier="photo",
        repeat_of="photo_01",
        path="data/raw/P1/photo_02",
    )
    _set_photo_groups(data["captures"][3], {"R02": 2})
    data["captures"][3]["media_sha256"] = [
        data["captures"][0]["media_sha256"][0],
        _digests("fresh", 1)[0],
    ]
    with pytest.raises(ValueError, match="photo_02 is the same recording as photo_01"):
        Case.model_validate(data)


def test_fixture_media_shared_without_duplicate_flag_is_rejected() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "benchmark/manifests/development.json").read_text()
    )
    copy_ = manifest["fixtures"][1]
    copy_["raw_manifest_hash"] = HASH_D  # e.g. a renamed sidecar file
    del copy_["duplicate_of"]
    with pytest.raises(ValueError, match="must say it is a duplicate"):
        CaseManifest.model_validate(manifest)


def test_fixture_is_never_a_benchmark_capture() -> None:
    fixture = load_manifests()["development"].fixtures[2]
    data = _complete_case()
    data["captures"][2]["media_sha256"] = list(fixture.media_sha256)
    manifests = {
        "development": CaseManifest.model_validate(
            {
                "manifest_version": 1,
                "split": "development",
                "cases": [data],
                "fixtures": [fixture.model_dump()],
            }
        )
    }
    with pytest.raises(ValueError, match="is development fixture sample-c00a170fe1"):
        check_splits(manifests)


def test_renamed_recording_cannot_count_in_two_cases() -> None:
    reused = _capture("photo_09", "photo", HASH_D, path="data/raw/P3/photos")
    reused["media_sha256"] = _complete_case()["captures"][0]["media_sha256"]
    manifests = {
        "development": _manifest("development", [_complete_case()]),
        "heldout": _manifest(
            "heldout",
            [_other_case("heldout_property", "heldout", "P3", [reused])],
        ),
    }
    with pytest.raises(ValueError, match="across cases"):
        check_splits(manifests)


# --------------------------------------------------------------------------
# P05A: only an original export of two sketched rooms satisfies the incumbent
# --------------------------------------------------------------------------


def test_transcription_is_kept_but_is_not_the_export() -> None:
    data = _complete_case()
    data["incumbent"]["export_kind"] = "transcription"
    with pytest.raises(ValueError, match="incumbent export for two rooms"):
        Case.model_validate(data)
    data["declared_complete"] = False
    items = Case.model_validate(data).composition()
    incumbent = next(i for i in items if "incumbent" in i["item"])
    assert not incumbent["met"]
    assert incumbent["detail"] == (
        "magicplan: received for R01, R03 as a transcription "
        "(supporting evidence, not the app's export)"
    )


def test_original_export_of_two_sketched_rooms_counts() -> None:
    items = Case.model_validate(_complete_case()).composition()
    incumbent = next(i for i in items if "incumbent" in i["item"])
    assert incumbent == {
        "item": "incumbent export for two rooms received",
        "met": True,
        "detail": "magicplan: received for R01, R03",
    }


def test_incumbent_rooms_must_be_in_the_sketch() -> None:
    data = _complete_case()
    data["incumbent"]["room_ids"] = ["R99", "R98"]
    with pytest.raises(ValueError, match=r"\['R98', 'R99'\] that are not in the"):
        Case.model_validate(data)


def test_repeat_without_recorded_original_coverage_is_unmet() -> None:
    data = _complete_case()
    data["captures"][2]["room_ids"] = []  # lidar_01 coverage not recorded
    data["declared_complete"] = False
    items = Case.model_validate(data).composition()
    repeat = next(i for i in items if "captured twice" in i["item"])
    assert repeat == {
        "item": "one room captured twice at the same tier",
        "met": False,
        "detail": "not received",
    }
