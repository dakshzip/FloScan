"""P05A: reproduce the P05 review findings against the case manifest.

Each probe starts from a fresh ``_complete_case()`` taken from the committed
test file and applies one review mutation. ACCEPTED means the malformed case
validated with declared_complete=True. Works on the P05 and P05A schemas.
"""

from __future__ import annotations

import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.cases import manifest as m  # noqa: E402

TESTS = runpy.run_path(str(ROOT / "tests/contract/test_case_manifest.py"))


def fresh() -> dict:
    return TESTS["_complete_case"]()


def only_one_photo_group(d: dict) -> None:
    photo = d["captures"][0]
    if "photo_groups" in photo:  # P05A schema
        photo["photo_groups"] = [{"folder": "room_001", "room_id": "R01", "count": 2}]
        photo["media_sha256"] = photo["media_sha256"][:2]
        photo["room_ids"] = ["R01"]
    else:
        photo["photo_counts"] = {"room_001": 2}


def renamed_copy(d: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp, "lidar_01"), Path(tmp, "lidar_02")
        a.mkdir()
        b.mkdir()
        (a / "rgb.mp4").write_bytes(b"dummy video bytes")
        (b / "renamed.mp4").write_bytes(b"dummy video bytes")
        for capture, folder in ((d["captures"][2], a), (d["captures"][3], b)):
            capture["raw_manifest_hash"] = m.raw_manifest_hash(folder)
            if hasattr(m, "media_digests"):  # P05A schema
                capture["media_sha256"] = m.media_digests(folder)
        print(
            "    raw hashes differ:",
            d["captures"][2]["raw_manifest_hash"]
            != d["captures"][3]["raw_manifest_hash"],
        )


PROBES = {
    "1a whole-property photos: one group only": only_one_photo_group,
    "1b whole-property video: room_ids=['R01']": (
        lambda d: d["captures"][1].update(room_ids=["R01"])
    ),
    "1c third room replaced by a copy of the second": (
        lambda d: d["rooms"].__setitem__(2, dict(d["rooms"][1]))
    ),
    "2a lidar_02.repeat_of='lidar_02'": (
        lambda d: d["captures"][3].update(repeat_of="lidar_02")
    ),
    "2b repeat is the same bytes under a renamed file": renamed_copy,
    "3a incumbent export_kind='transcription'": (
        lambda d: d["incumbent"].update(export_kind="transcription")
    ),
    "3b incumbent room_ids=['R99','R98']": (
        lambda d: d["incumbent"].update(room_ids=["R99", "R98"])
    ),
}


def main() -> int:
    baseline = m.Case.model_validate(fresh()).composition()
    print(f"control: unmodified case, {sum(i['met'] for i in baseline)}/8 met")
    accepted = 0
    for name, mutate in PROBES.items():
        data = fresh()
        mutate(data)
        try:
            case = m.Case.model_validate(data)
        except ValueError as error:
            first = str(error).splitlines()[1:3]
            print(f"REJECTED  {name}\n    {' '.join(s.strip() for s in first)}")
            continue
        met = sum(i["met"] for i in case.composition())
        accepted += 1
        print(f"ACCEPTED  {name}: declared_complete=True, {met}/8 met")
    print(f"{accepted}/{len(PROBES)} malformed cases accepted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
