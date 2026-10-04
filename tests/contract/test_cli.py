"""Contract tests for the P01 CLI, gate registry and diagnostic envelope."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from floscan.cli import main
from floscan.pipeline import (
    CONTRACT_SECTIONS,
    DEFAULT_GATES_PATH,
    EXIT_INCOMPLETE,
    EXIT_INVALID_INPUT,
    EXIT_REPLAY_UNAVAILABLE,
    EXIT_USAGE,
    PIPELINE_STAGES,
    RESULT_FILENAME,
    EnvelopeError,
    RegistryError,
    RunIOError,
    load_gate_registry,
    read_envelope,
    validate_envelope,
    validate_gate_registry,
    write_envelope,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def capture_dir(tmp_path: Path) -> Path:
    """A tiny non-empty capture directory whose name ends with a space."""
    directory = tmp_path / "capture "
    (directory / "room_001").mkdir(parents=True)
    (directory / "room_001" / "image_001.jpg").write_bytes(b"not a real jpeg")
    return directory


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run(*args: str) -> int:
    return main(["run", *args])


def _live_envelope(capture_dir: Path, output: Path, tier: str = "lidar") -> dict:
    code = _run(
        "--input", str(capture_dir), "--tier", tier,
        "--output", str(output), "--mode", "live",
    )  # fmt: skip
    assert code == EXIT_INCOMPLETE
    return read_envelope(output / RESULT_FILENAME)


# --------------------------------------------------------------------------
# Help
# --------------------------------------------------------------------------


def test_cli_help_lists_tiers_modes_and_exit_codes(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["run", "--help"])
    assert raised.value.code == 0
    text = capsys.readouterr().out
    for token in ("photo", "video", "lidar", "live", "replay", "exit codes"):
        assert token in text


def test_run_sh_help_via_shell() -> None:
    completed = subprocess.run(
        [str(PROJECT_ROOT / "run.sh"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--tier {photo,video,lidar}" in completed.stdout
    assert "--mode {live,replay}" in completed.stdout


# --------------------------------------------------------------------------
# Missing and invalid input
# --------------------------------------------------------------------------


def test_missing_input_writes_invalid_input_envelope(tmp_path: Path) -> None:
    output = tmp_path / "out"
    code = _run(
        "--input", str(tmp_path / "does-not-exist"), "--tier", "photo",
        "--output", str(output), "--mode", "live",
    )  # fmt: skip
    assert code == EXIT_INVALID_INPUT
    envelope = read_envelope(output / RESULT_FILENAME)
    validate_envelope(envelope)
    assert envelope["status"] == "invalid_input"
    assert envelope["exit_code"] == EXIT_INVALID_INPUT
    assert envelope["run"]["input"]["exists"] is False
    assert envelope["stages"][0]["status"] == "invalid_input"
    assert all(stage["status"] == "skipped" for stage in envelope["stages"][1:])


def test_empty_input_directory_is_invalid(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code = _run(
        "--input", str(empty), "--tier", "video",
        "--output", str(tmp_path / "out"), "--mode", "live",
    )  # fmt: skip
    assert code == EXIT_INVALID_INPUT


def test_missing_required_arguments_are_usage_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        _run("--input", str(tmp_path), "--output", str(tmp_path / "out"))
    assert raised.value.code == EXIT_USAGE
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------
# Forbidden tiers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["rgbd", "auto", "lidar_assisted", "LIDAR", ""])
def test_forbidden_tier_is_rejected_and_writes_nothing(
    tier: str, capture_dir: Path, tmp_path: Path, capsys
) -> None:
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as raised:
        _run(
            "--input", str(capture_dir), "--tier", tier,
            "--output", str(output), "--mode", "live",
        )  # fmt: skip
    assert raised.value.code == EXIT_USAGE
    assert "invalid choice" in capsys.readouterr().err
    assert not output.exists()


def test_forbidden_tier_through_run_sh(capture_dir: Path, tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            str(PROJECT_ROOT / "run.sh"), "--input", str(capture_dir),
            "--tier", "rgbd", "--output", str(tmp_path / "out"), "--mode", "live",
        ],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    assert completed.returncode == EXIT_USAGE
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------
# Partial-status output
# --------------------------------------------------------------------------


def test_live_run_reports_incomplete_contract_honestly(
    capture_dir: Path, tmp_path: Path
) -> None:
    envelope = _live_envelope(capture_dir, tmp_path / "out")
    validate_envelope(envelope)

    assert envelope["status"] == "unsupported"
    assert envelope["exit_code"] == EXIT_INCOMPLETE
    assert envelope["coverage"]["contract_complete"] is False
    sections = envelope["coverage"]["sections"]
    assert set(sections) == set(CONTRACT_SECTIONS)
    for name, section in sections.items():
        assert section["status"] != "available", name
        assert section["reason"].strip(), name
    assert sections["public_schema_export"]["status"] == "blocked_external"

    # No fabricated geometry or measurements.
    for name in ("rooms", "walls", "openings", "surfaces", "measurements"):
        assert envelope[name] == []
    assert envelope["property_graph"] is None
    assert envelope["scale"] is None

    stages = {stage["name"]: stage["status"] for stage in envelope["stages"]}
    assert stages["input.inventory"] == "ok"
    assert all(stages[name] == "not_implemented" for name in PIPELINE_STAGES)

    authority = envelope["schema_authority"]
    assert authority["owner"] == "project"
    assert authority["external_schema_status"] == "unavailable"
    assert authority["external_conformance_claimed"] is False


def test_no_gate_is_marked_pass(capture_dir: Path, tmp_path: Path) -> None:
    envelope = _live_envelope(capture_dir, tmp_path / "out")
    statuses = {item["status"] for item in envelope["gates"]["items"]}
    assert statuses <= {"unverified", "unspecified_source"}
    assert envelope["gates"]["summary"]["measured_pass"] == 0
    assert envelope["gates"]["summary"]["measured_fail"] == 0
    registry = load_gate_registry()
    assert len(envelope["gates"]["items"]) == len(registry.gates)


def test_trailing_space_input_path_is_preserved(
    capture_dir: Path, tmp_path: Path
) -> None:
    envelope = _live_envelope(capture_dir, tmp_path / "out", tier="photo")
    assert envelope["run"]["input"]["path"].endswith("capture ")
    assert envelope["run"]["input"]["file_count"] == 1


def test_result_json_has_no_nonfinite_numbers(
    capture_dir: Path, tmp_path: Path
) -> None:
    _live_envelope(capture_dir, tmp_path / "out")
    text = (tmp_path / "out" / RESULT_FILENAME).read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda name: pytest.fail(f"found {name}"))


def test_replay_never_falls_back_to_live(capture_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    code = _run(
        "--input", str(capture_dir), "--tier", "lidar",
        "--output", str(output), "--mode", "replay",
    )  # fmt: skip
    assert code == EXIT_REPLAY_UNAVAILABLE
    envelope = read_envelope(output / RESULT_FILENAME)
    validate_envelope(envelope)
    stages = {stage["name"]: stage["status"] for stage in envelope["stages"]}
    assert stages["replay.lookup"] == "unsupported"
    assert all(stages[name] == "skipped" for name in PIPELINE_STAGES)
    assert envelope["run"]["mode"] == "replay"


def test_run_sh_end_to_end_writes_validated_envelope(
    capture_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            str(PROJECT_ROOT / "run.sh"), "--input", str(capture_dir),
            "--tier", "video", "--output", str(output), "--mode", "live",
        ],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    assert completed.returncode == EXIT_INCOMPLETE, completed.stderr
    assert "status:   unsupported (exit 4)" in completed.stderr
    assert main(["validate", str(output / RESULT_FILENAME)]) == 0


# --------------------------------------------------------------------------
# Input preservation and append-only output
# --------------------------------------------------------------------------


def test_output_inside_input_is_refused(capture_dir: Path) -> None:
    before = _tree_digest(capture_dir)
    code = _run(
        "--input", str(capture_dir), "--tier", "photo",
        "--output", str(capture_dir / "results"), "--mode", "live",
    )  # fmt: skip
    assert code == EXIT_USAGE
    assert _tree_digest(capture_dir) == before
    assert not (capture_dir / "results").exists()


def test_existing_result_is_never_overwritten(
    capture_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "out"
    _live_envelope(capture_dir, output)
    first = (output / RESULT_FILENAME).read_bytes()
    code = _run(
        "--input", str(capture_dir), "--tier", "lidar",
        "--output", str(output), "--mode", "live",
    )  # fmt: skip
    assert code == EXIT_USAGE
    assert (output / RESULT_FILENAME).read_bytes() == first


def test_live_run_does_not_modify_input(capture_dir: Path, tmp_path: Path) -> None:
    before = _tree_digest(capture_dir)
    _live_envelope(capture_dir, tmp_path / "out")
    assert _tree_digest(capture_dir) == before


# --------------------------------------------------------------------------
# Envelope validator rejects dishonest output
# --------------------------------------------------------------------------


@pytest.fixture
def envelope(capture_dir: Path, tmp_path: Path) -> dict:
    return _live_envelope(capture_dir, tmp_path / "out")


def test_validator_rejects_gate_marked_pass(envelope: dict) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["gates"]["items"][0]["status"] = "measured_pass"
    tampered["gates"]["summary"]["measured_pass"] = 1
    tampered["gates"]["summary"]["unverified"] -= 1
    with pytest.raises(EnvelopeError, match="benchmark"):
        validate_envelope(tampered)


def test_validator_rejects_fake_room(envelope: dict) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["rooms"] = [{"id": "room_001", "label": "kitchen"}]
    with pytest.raises(EnvelopeError, match="rooms"):
        validate_envelope(tampered)


def test_validator_rejects_ok_status_with_unavailable_sections(envelope: dict) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["status"] = "ok"
    tampered["exit_code"] = 0
    with pytest.raises(EnvelopeError, match="ok"):
        validate_envelope(tampered)


def test_validator_rejects_external_conformance_claim(envelope: dict) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["schema_authority"]["external_conformance_claimed"] = True
    with pytest.raises(EnvelopeError, match="conformance"):
        validate_envelope(tampered)


def test_validator_rejects_unavailable_section_without_reason(envelope: dict) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["coverage"]["sections"]["measurements"]["reason"] = " "
    with pytest.raises(EnvelopeError, match="reason"):
        validate_envelope(tampered)


def test_validator_rejects_nonfinite_numbers(envelope: dict) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["run"]["input"]["total_bytes"] = math.nan
    with pytest.raises(EnvelopeError, match="non-finite"):
        validate_envelope(tampered)


def test_validator_rejects_mismatched_exit_code(envelope: dict) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["exit_code"] = 0
    with pytest.raises(EnvelopeError, match="exit_code"):
        validate_envelope(tampered)


def test_validate_command_fails_on_tampered_file(
    envelope: dict, tmp_path: Path
) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["gates"]["items"][0]["status"] = "measured_pass"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert main(["validate", str(path)]) == 1


# --------------------------------------------------------------------------
# Gate registry
# --------------------------------------------------------------------------

# Thresholds as printed in Applied_AI_Case_Study.pdf, converted to SI.
EXPECTED_THRESHOLDS = {
    "opening_width": [0.02, 0.85],
    "ceiling_height_error": [0.015],
    "ceiling_height_repeat_spread": [0.01],
    "wall_repeatability": [0.01, 0.005],
    "photo_footprint": [0.08],
    "photo_wall_length": [0.08],
    "video_wall_length": [0.03],
    "incumbent_head_to_head": [0.7],
    "fresh_machine_to_result": [900],
    "custom_app_install": [600],
    "technical_report_pages": [6],
}
UNSPECIFIED_GATES = {"lidar_wall_length", "floor_area", "round1_gates"}


def _registry_data() -> dict:
    return yaml.safe_load(DEFAULT_GATES_PATH.read_text(encoding="utf-8"))


def test_registry_thresholds_match_source_documents() -> None:
    registry = load_gate_registry()
    numeric = {
        gate["id"]: [criterion["value"] for criterion in gate["criteria"]]
        for gate in registry.gates
        if gate["source_status"] == "specified"
    }
    assert numeric == EXPECTED_THRESHOLDS


def test_registry_keeps_missing_round1_thresholds_unspecified() -> None:
    registry = load_gate_registry()
    unspecified = {
        gate["id"]
        for gate in registry.gates
        if gate["source_status"] == "unspecified_source"
    }
    assert unspecified == UNSPECIFIED_GATES
    for gate in registry.gates:
        if gate["id"] in UNSPECIFIED_GATES:
            assert all(c["value"] is None for c in gate["criteria"])


def test_registry_rejects_threshold_not_in_source_text() -> None:
    data = _registry_data()
    gate = next(g for g in data["gates"] if g["id"] == "opening_width")
    gate["criteria"][0]["source_value"] = "3 cm"
    gate["criteria"][0]["value"] = 0.03
    with pytest.raises(RegistryError, match="does not appear"):
        validate_gate_registry(data)


def test_registry_rejects_value_that_differs_from_quote() -> None:
    data = _registry_data()
    gate = next(g for g in data["gates"] if g["id"] == "ceiling_height_error")
    gate["criteria"][0]["value"] = 0.02
    with pytest.raises(RegistryError, match="does not equal"):
        validate_gate_registry(data)


def test_registry_rejects_invented_round1_threshold() -> None:
    data = _registry_data()
    gate = next(g for g in data["gates"] if g["id"] == "lidar_wall_length")
    gate["criteria"][0]["value"] = 0.02
    with pytest.raises(RegistryError, match="unspecified_source"):
        validate_gate_registry(data)


def test_registry_rejects_a_verdict_field() -> None:
    data = _registry_data()
    data["gates"][0]["status"] = "measured_pass"
    with pytest.raises(RegistryError, match="unknown keys"):
        validate_gate_registry(data)


def test_registry_source_hashes_match_pdfs() -> None:
    for source in _registry_data()["sources"]:
        pdf = PROJECT_ROOT / source["file"]
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == source["sha256"]


def test_adr_lists_every_requirement_id() -> None:
    adr = (PROJECT_ROOT / "docs" / "adr" / "001-requirements.md").read_text(
        encoding="utf-8"
    )
    for requirement in _registry_data()["requirements"]:
        assert f"| {requirement['id']} |" in adr, requirement["id"]


def test_invalid_registry_fails_run_without_output(
    capture_dir: Path, tmp_path: Path
) -> None:
    bad = tmp_path / "gates.yaml"
    bad.write_text("registry_version: 1\n", encoding="utf-8")
    output = tmp_path / "out"
    code = _run(
        "--input", str(capture_dir), "--tier", "lidar", "--output", str(output),
        "--mode", "live", "--gates", str(bad),
    )  # fmt: skip
    assert code == 1
    assert not (output / RESULT_FILENAME).exists()


# --------------------------------------------------------------------------
# Benchmark entry point
# --------------------------------------------------------------------------


def test_benchmark_reports_missing_scorer_and_cases(capsys) -> None:
    assert main(["benchmark", "dev_property", "lidar"]) == EXIT_INCOMPLETE
    err = capsys.readouterr().err
    assert "P04" in err and "P05" in err and "not run" in err


def test_benchmark_rejects_forbidden_tier() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["benchmark", "dev_property", "rgbd"])
    assert raised.value.code == EXIT_USAGE


def test_run_benchmark_sh_exits_nonzero() -> None:
    completed = subprocess.run(
        [str(PROJECT_ROOT / "run_benchmark.sh"), "dev_property", "photo"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == EXIT_INCOMPLETE
    assert "not run" in completed.stderr


# --------------------------------------------------------------------------
# P01A regressions (docs/reviews/P01-review.md)
# --------------------------------------------------------------------------


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in a fresh interpreter, as a user would."""
    return subprocess.run(
        [sys.executable, "-m", "floscan.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_concise_failure(completed: subprocess.CompletedProcess[str]) -> None:
    assert completed.returncode == 1, completed.stderr
    assert "Traceback" not in completed.stderr
    assert len(completed.stderr.strip().splitlines()) == 1, completed.stderr


def _complete_false_success(envelope: dict) -> dict:
    """The review's reproduction: consistent-looking success, empty payloads."""
    fake = copy.deepcopy(envelope)
    fake["status"] = "ok"
    fake["exit_code"] = 0
    fake["coverage"]["contract_complete"] = True
    for section in fake["coverage"]["sections"].values():
        section["status"] = "available"
    for stage in fake["stages"]:
        stage["status"] = "ok"
    return fake


def _mutate(data: Any, path: tuple[Any, ...], value: Any) -> None:
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


# Defect 1: diagnostic schema must never certify success.


def test_validator_rejects_complete_false_success(envelope: dict) -> None:
    with pytest.raises(EnvelopeError, match="status 'ok' is not allowed"):
        validate_envelope(_complete_false_success(envelope))


def test_validate_command_rejects_complete_false_success(
    envelope: dict, tmp_path: Path
) -> None:
    path = tmp_path / "fake-success.json"
    path.write_text(json.dumps(_complete_false_success(envelope)), encoding="utf-8")
    completed = _cli("validate", str(path))
    _assert_concise_failure(completed)
    assert "status 'ok' is not allowed" in completed.stderr


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("coverage", "contract_complete"), True, "contract_complete must be false"),
        (
            ("coverage", "sections", "measurements", "status"),
            "available",
            "not allowed",
        ),
        (("coverage", "sections", "per_room_plan", "status"), "partial", "not allowed"),
        (("stages", 8, "status"), "ok", "reports 'ok'"),
    ],
    ids=["contract-complete", "section-available", "section-partial", "stage-ok"],
)
def test_validator_rejects_each_false_success_component(
    envelope: dict, path: tuple[Any, ...], value: Any, message: str
) -> None:
    """Each false claim is rejected on its own, with the status left honest."""
    tampered = copy.deepcopy(envelope)
    _mutate(tampered, path, value)
    assert tampered["status"] == "unsupported"
    with pytest.raises(EnvelopeError, match=message):
        validate_envelope(tampered)


def test_validator_rejects_ok_replay_lookup(capture_dir: Path, tmp_path: Path) -> None:
    _run(
        "--input", str(capture_dir), "--tier", "lidar",
        "--output", str(tmp_path / "out"), "--mode", "replay",
    )  # fmt: skip
    tampered = read_envelope(tmp_path / "out" / RESULT_FILENAME)
    lookup = next(s for s in tampered["stages"] if s["name"] == "replay.lookup")
    lookup["status"] = "ok"
    with pytest.raises(EnvelopeError, match="reports 'ok'"):
        validate_envelope(tampered)


@pytest.mark.parametrize("status", ["available", "partial"])
def test_validator_rejects_public_export_claim(envelope: dict, status: str) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["coverage"]["sections"]["public_schema_export"]["status"] = status
    with pytest.raises(EnvelopeError, match="public_schema_export must be"):
        validate_envelope(tampered)


def test_validate_command_rejects_public_export_claim(
    envelope: dict, tmp_path: Path
) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["coverage"]["sections"]["public_schema_export"]["status"] = "available"
    path = tmp_path / "export-available.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    completed = _cli("validate", str(path))
    _assert_concise_failure(completed)
    assert "public_schema_export must be" in completed.stderr


def test_validator_still_accepts_diagnostic_failure_states(envelope: dict) -> None:
    unavailable_export = copy.deepcopy(envelope)
    unavailable_export["coverage"]["sections"]["public_schema_export"]["status"] = (
        "unavailable"
    )
    validate_envelope(unavailable_export)

    failed = copy.deepcopy(envelope)
    failed["status"] = "failed"
    failed["exit_code"] = 1
    validate_envelope(failed)


def test_validator_rejects_boolean_exit_code(envelope: dict) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["status"] = "failed"
    tampered["exit_code"] = True  # True == 1, but it is not an exit code.
    with pytest.raises(EnvelopeError, match="exit_code"):
        validate_envelope(tampered)


def test_validator_rejects_unhashable_stage_name(envelope: dict) -> None:
    tampered = copy.deepcopy(envelope)
    tampered["stages"][0]["name"] = ["input.inventory"]
    with pytest.raises(EnvelopeError, match="unknown stage"):
        validate_envelope(tampered)


# Defect 2: filesystem errors stay inside the CLI error boundary.


def test_output_parent_is_a_file_fails_without_traceback(
    capture_dir: Path, tmp_path: Path
) -> None:
    blocker = tmp_path / "file-not-directory"
    blocker.write_text("", encoding="utf-8")
    before = _tree_digest(capture_dir)
    completed = subprocess.run(
        [
            str(PROJECT_ROOT / "run.sh"), "--input", str(capture_dir),
            "--tier", "lidar", "--output", str(blocker / "output"), "--mode", "live",
        ],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    _assert_concise_failure(completed)
    assert completed.stderr.startswith("floscan run: error: cannot create result")
    assert blocker.is_file() and blocker.read_text(encoding="utf-8") == ""
    assert _tree_digest(capture_dir) == before


def test_write_envelope_keeps_exclusive_creation(
    envelope: dict, tmp_path: Path
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / RESULT_FILENAME).write_text("sentinel", encoding="utf-8")
    with pytest.raises(RunIOError, match="cannot create result"):
        write_envelope(envelope, output)
    assert (output / RESULT_FILENAME).read_text(encoding="utf-8") == "sentinel"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any directory")
def test_unreadable_input_subdirectory_fails_instead_of_undercounting(
    capture_dir: Path, tmp_path: Path
) -> None:
    locked = capture_dir / "locked"
    locked.mkdir()
    (locked / "hidden.bin").write_bytes(b"x")
    locked.chmod(0)
    try:
        code = _run(
            "--input", str(capture_dir), "--tier", "photo",
            "--output", str(tmp_path / "out"), "--mode", "live",
        )  # fmt: skip
    finally:
        locked.chmod(0o755)
    assert code == 1
    assert not (tmp_path / "out" / RESULT_FILENAME).exists()


# Defect 3: malformed registry types raise RegistryError.


@pytest.mark.parametrize(
    "pages", ["not-a-number", "6", True, False, 0, -1, 6.5, None, [6]]
)
def test_registry_rejects_malformed_source_page_count(pages: Any) -> None:
    data = _registry_data()
    data["sources"][0]["pages"] = pages
    with pytest.raises(RegistryError, match="positive integer"):
        validate_gate_registry(data)


def test_gates_command_reports_malformed_registry_without_traceback(
    tmp_path: Path,
) -> None:
    text = DEFAULT_GATES_PATH.read_text(encoding="utf-8")
    bad_text = text.replace("    pages: 6\n", "    pages: not-a-number\n", 1)
    assert bad_text != text
    bad = tmp_path / "gates.yaml"
    bad.write_text(bad_text, encoding="utf-8")
    completed = _cli("gates", "--gates", str(bad))
    _assert_concise_failure(completed)
    assert "sources[0].pages: expected a positive integer" in completed.stderr


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("registry_version",), True),
        (("registry_version",), 1.0),
        (("sources",), 5),
        (("requirements",), "G01"),
        (("gates",), {"id": "opening_width"}),
        (("unavailable_sources",), None),
        (("sources", 0, "id"), ["case_study"]),
        (("sources", 1, "id"), "case_study"),
        (("gates", 0, "requirement_ids"), [["G04"]]),
        (("gates", 0, "tiers"), [{"photo": 1}]),
    ],
    ids=[
        "version-bool",
        "version-float",
        "sources-int",
        "requirements-str",
        "gates-dict",
        "unavailable-null",
        "source-id-list",
        "source-id-duplicate",
        "requirement-ids-unhashable",
        "tiers-unhashable",
    ],
)
def test_registry_type_errors_raise_registry_error(
    path: tuple[Any, ...], value: Any
) -> None:
    data = _registry_data()
    _mutate(data, path, value)
    with pytest.raises(RegistryError):
        validate_gate_registry(data)
