"""Command-line entry point: ``floscan run|benchmark|inspect|gates|validate``."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from floscan import __version__
from floscan.pipeline import (
    CONTRACT_SECTIONS,
    DEFAULT_GATES_PATH,
    EXIT_FAILED,
    EXIT_INCOMPLETE,
    EXIT_INVALID_INPUT,
    EXIT_OK,
    EXIT_USAGE,
    MODES,
    RESULT_FILENAME,
    TIERS,
    EnvelopeError,
    RegistryError,
    RunIOError,
    RunRequest,
    check_output_location,
    execute,
    initial_gate_status,
    load_gate_registry,
    read_envelope,
    validate_envelope,
    write_envelope,
)

EXIT_CODES_HELP = f"""\
exit codes:
  {EXIT_OK}  ok: the full output contract was produced (not reachable in this build)
  {EXIT_FAILED}  failed: internal error, invalid gate registry, invalid envelope,
     or a filesystem error reading the input or writing the result; no result
  {EXIT_USAGE}  usage error: bad arguments, unknown tier, output inside input,
     or an existing result (run directories are append-only); nothing written
  3  invalid_input: input missing, not a directory or empty; envelope written
  {EXIT_INCOMPLETE}  incomplete: envelope written, output contract not produced
  5  replay unavailable: no matching replay cache; live inference not attempted
"""

INSPECT_EXIT_CODES_HELP = f"""\
exit codes:
  {EXIT_OK}  verified: timing and conventions confirmed on the data (per-frame
     issues, if any, are listed)
  {EXIT_FAILED}  failed: filesystem error writing the outputs
  {EXIT_USAGE}  usage error: output inside the input or not a new directory
  {EXIT_INVALID_INPUT}  invalid_input: not a readable capture of this tier
  {EXIT_INCOMPLETE}  unverified: parsed, but timing or a convention is not confirmed,
     or the tier is not supported by inspect yet
"""

# Case aliases reserved by the task packets; manifests arrive with P05.
RESERVED_CASE_ALIASES = (
    "dev_property",
    "calibration_set",
    "heldout_property",
    "walkin_property",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``floscan`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="floscan",
        description="FloScan property capture pipeline (P01 diagnostic skeleton).",
    )
    parser.add_argument("--version", action="version", version=f"floscan {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser(
        "run",
        help="process one capture into a result envelope",
        description=(
            "Process one capture directory at an explicitly declared tier and "
            f"write OUTPUT/{RESULT_FILENAME}. This build implements no "
            "reconstruction: it inventories the input and writes a validated "
            "diagnostic envelope that reports every unavailable section."
        ),
        epilog=EXIT_CODES_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run.add_argument(
        "--input", required=True, type=Path, help="capture directory (never modified)"
    )
    run.add_argument(
        "--tier",
        required=True,
        choices=TIERS,
        help="declared input tier; never auto-detected or upgraded",
    )
    run.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new result directory, outside the input; existing results are kept",
    )
    run.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help="live processes the raw input; replay only reuses a matching cache",
    )
    run.add_argument(
        "--gates",
        type=Path,
        default=DEFAULT_GATES_PATH,
        help="gate registry (default: configs/gates.yaml)",
    )

    benchmark = commands.add_parser(
        "benchmark",
        help="benchmark a case live (inference and cases not built yet)",
        description=(
            "Run live inference on a benchmark case and score it against sealed "
            "ground truth. Live inference and case manifests (P05) do not exist "
            "yet, so this command reports that and exits nonzero. Scoring of "
            "existing predictions is available as `./run_benchmark.sh score`."
        ),
        epilog=EXIT_CODES_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    benchmark.add_argument("case_id", help="case alias, e.g. dev_property")
    benchmark.add_argument("tier", choices=TIERS, help="tier to benchmark")
    benchmark.add_argument("--mode", choices=MODES, default="live")

    inspect = commands.add_parser(
        "inspect",
        help="validate and audit one capture folder (no inference)",
        description=(
            "Parse a capture folder read-only, check its timing and conventions on "
            "the data, and report what is verified and what is not. Only LiDAR "
            "sessions in the Stray Scanner export format are supported so far."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=INSPECT_EXIT_CODES_HELP,
    )
    inspect.add_argument("--input", type=Path, required=True, help="capture folder")
    inspect.add_argument("--tier", choices=TIERS, required=True)
    inspect.add_argument(
        "--output",
        type=Path,
        help="new directory for capture.json, inspection.json and frames.jsonl",
    )

    gates = commands.add_parser(
        "gates", help="list registered gates and their current verdict status"
    )
    gates.add_argument("--json", action="store_true", help="print JSON")
    gates.add_argument("--gates", type=Path, default=DEFAULT_GATES_PATH)

    validate = commands.add_parser(
        "validate", help="validate a result envelope against the project schema"
    )
    validate.add_argument("result", type=Path, help=f"path to a {RESULT_FILENAME}")

    doctor = commands.add_parser(
        "doctor",
        help="check hardware, libraries, pinned checkpoints and live models",
        description=(
            "Report the hardware, exercise every library in its own process, "
            "verify pinned checkpoints by SHA-256 and, with --live-models, run "
            "each model on each device in an isolated, network-blocked worker."
        ),
    )
    doctor.add_argument(
        "--live-models", action="store_true", help="run a forward pass per model"
    )
    doctor.add_argument(
        "--devices",
        default="all",
        help="comma-separated cpu,mps,cuda, or 'all' available devices (default)",
    )
    doctor.add_argument(
        "--models-dir", type=Path, default=None, help="checkpoint root override"
    )
    doctor.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the JSON report here (new file)",
    )
    return parser


def _command_run(args: argparse.Namespace, argv: Sequence[str]) -> int:
    problem = check_output_location(args.input, args.output)
    if problem is not None:
        print(f"floscan run: error: {problem}", file=sys.stderr)
        return EXIT_USAGE
    registry = load_gate_registry(args.gates)
    request = RunRequest(
        input_dir=args.input,
        output_dir=args.output,
        tier=args.tier,
        mode=args.mode,
        gates_path=args.gates,
        argv=tuple(argv),
    )
    envelope = execute(request, registry)
    path = write_envelope(envelope, args.output)

    sections = envelope["coverage"]["sections"]
    available = sum(1 for s in sections.values() if s["status"] == "available")
    summary = envelope["gates"]["summary"]
    stream = sys.stdout if envelope["exit_code"] == EXIT_OK else sys.stderr
    print(f"FloScan run {envelope['run']['run_id']}", file=stream)
    print(
        f"  status:   {envelope['status']} (exit {envelope['exit_code']})", file=stream
    )
    print(f"  reason:   {envelope['status_reason']}", file=stream)
    print(
        f"  result:   {path} (validated, {envelope['schema_version']}, "
        "project-owned schema)",
        file=stream,
    )
    ran = [
        stage
        for stage in envelope["stages"]
        if stage["status"] not in ("not_implemented", "skipped")
        and stage["name"] != "input.inventory"
    ]
    for stage in ran:
        print(f"  stage:    {stage['name']} {stage['status']}", file=stream)
    print(f"  sections: {available}/{len(CONTRACT_SECTIONS)} available", file=stream)
    print(
        "  gates:    "
        + ", ".join(f"{count} {name}" for name, count in summary.items()),
        file=stream,
    )
    return envelope["exit_code"]


def _command_benchmark(args: argparse.Namespace) -> int:
    registered = "reserved" if args.case_id in RESERVED_CASE_ALIASES else "unknown"
    print(
        f"floscan benchmark: case {args.case_id!r} ({registered} alias), tier "
        f"{args.tier}, mode {args.mode}: not run.\n"
        "  Live inference is not implemented yet and no case manifests or ground "
        "truth are registered (packet P05).\n"
        "  No inference was run and no gate was scored. To score existing "
        "predictions use: ./run_benchmark.sh score --ground-truth GT.json "
        "--predictions P.json --output DIR",
        file=sys.stderr,
    )
    return EXIT_INCOMPLETE


def _command_inspect(args: argparse.Namespace) -> int:
    # Imported here so other commands never pay for video and image decoding.
    from floscan.capture import stray
    from floscan.io.manifest import check_new_output_dir, write_capture_outputs

    if args.tier != "lidar":
        print(
            f"floscan inspect: --tier {args.tier} is not supported yet; only LiDAR "
            "sessions (Stray Scanner export) can be inspected in this build",
            file=sys.stderr,
        )
        return EXIT_INCOMPLETE
    if args.output is not None:
        problem = check_new_output_dir(args.input, args.output)
        if problem:
            print(f"floscan inspect: error: {problem}", file=sys.stderr)
            return EXIT_USAGE
    try:
        session = stray.open_session(args.input)
        inspection = stray.inspect_session(session)
    except stray.StrayInputError as error:
        print(f"floscan inspect: invalid input: {error}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    print("\n".join(stray.summary_lines(inspection)))
    if args.output is not None:
        try:
            written = write_capture_outputs(
                args.output,
                stray.capture_record(inspection),
                stray.report(inspection),
                stray.frame_rows(inspection),
            )
        except OSError as error:
            raise RunIOError(f"cannot write inspection outputs: {error}") from error
        except ValueError as error:  # a non-finite number reached strict JSON
            raise RunIOError(
                f"inspection outputs are not strict JSON: {error}"
            ) from error
        print("wrote: " + ", ".join(str(path) for path in written))
    return EXIT_INCOMPLETE if inspection.status == "unverified" else EXIT_OK


def _format_criteria(gate: dict[str, Any]) -> str:
    if not gate["criteria"]:
        if gate["source_status"] == "unspecified_source":
            return "unspecified (defining source unavailable)"
        return "qualitative (no numeric threshold in sources)"
    joiner = " AND " if gate["combine"] == "all" else " OR "
    parts = []
    for criterion in gate["criteria"]:
        if criterion["value"] is None:
            parts.append(f"{criterion['quantity']}: threshold unspecified")
        else:
            parts.append(
                f"{criterion['quantity']} {criterion['comparator']} "
                f"{criterion['value']} {criterion['unit']}"
            )
    return joiner.join(parts)


def _command_gates(args: argparse.Namespace) -> int:
    registry = load_gate_registry(args.gates)
    rows = []
    for gate in registry.gates:
        verdict = initial_gate_status(gate)
        rows.append(
            {
                "id": gate["id"],
                "requirement_ids": gate["requirement_ids"],
                "tiers": gate["tiers"],
                "source_status": gate["source_status"],
                "interpretation": gate["interpretation"]["status"],
                "criteria": _format_criteria(gate),
                "status": verdict["status"],
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return EXIT_OK
    print(f"gate registry {registry.path} (sha256 {registry.sha256[:12]})")
    for row in rows:
        print(
            f"- {row['id']} [{', '.join(row['tiers'])}] "
            f"{row['status']}; source {row['source_status']}, "
            f"interpretation {row['interpretation']}\n    {row['criteria']}"
        )
    return EXIT_OK


def _command_validate(args: argparse.Namespace) -> int:
    envelope = read_envelope(args.result)
    validate_envelope(envelope)
    print(
        f"{args.result}: valid {envelope['schema_version']} envelope "
        f"(project-owned schema); status {envelope['status']}, "
        f"contract_complete {str(envelope['coverage']['contract_complete']).lower()}"
    )
    return EXIT_OK


def _gib(value: int | None) -> str:
    return "?" if value is None else f"{value / 2**30:.2f} GiB"


def _accelerator_peak(item: dict[str, Any]) -> str:
    """Largest accelerator memory figure in a smoke report.

    Process RSS misses Metal and CUDA allocations, so it is shown separately.
    """
    figures = [
        value
        for sample in item.get("accelerator_memory", [])
        for key, value in sample.items()
        if key.endswith("_bytes")
    ]
    return _gib(max(figures)) if figures else "n/a"


def _prepare_report_path(path: Path) -> bool:
    """Create the report's parent directory before the (slow) doctor run.

    Returns False if the report already exists, which is a usage error.

    Raises:
        RunIOError: if the location cannot be checked or created.
    """
    try:
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RunIOError(f"cannot prepare report path {path}: {error}") from error
    return True


def _write_new_report(path: Path, report: dict[str, Any]) -> None:
    """Write ``report`` to a file that must not exist yet.

    Exclusive creation never overwrites a file that appeared during the run.
    A partially written file created by this call is removed.

    Raises:
        RunIOError: if the file cannot be created or written.
    """
    text = json.dumps(report, indent=2) + "\n"
    try:
        handle = path.open("x", encoding="utf-8")
    except OSError as error:
        raise RunIOError(f"cannot create report {path}: {error}") from error
    try:
        with handle:
            handle.write(text)
    except OSError as error:
        with contextlib.suppress(OSError):
            path.unlink()
        raise RunIOError(f"cannot write report {path}: {error}") from error


def _command_doctor(args: argparse.Namespace) -> int:
    # Imported here so other commands never pay for the runtime modules.
    from floscan.runtime import models

    devices = None if args.devices == "all" else args.devices.split(",")
    # Fail on an unusable report path before minutes of checks, not after.
    if args.output is not None and not _prepare_report_path(args.output):
        print(f"floscan doctor: error: {args.output} already exists", file=sys.stderr)
        return EXIT_USAGE
    try:
        report = models.run_doctor(
            live_models=args.live_models,
            devices=devices,
            root=models.models_dir(args.models_dir),
        )
    except models.ModelLockError as error:
        print(f"floscan doctor: error: {error}", file=sys.stderr)
        return EXIT_FAILED

    hardware = report["hardware"]
    print(
        f"hardware: {hardware['cpu_model']}, {hardware['logical_cpus']} CPUs, "
        f"{_gib(hardware['total_memory_bytes'])} RAM, {hardware['os']}, "
        f"Python {hardware['python']}"
    )
    for gpu in hardware["nvidia_smi"]:
        print(
            f"  nvidia: {gpu['name']}, driver {gpu['driver_version']}, "
            f"{gpu['memory_total']}"
        )
    torch_info = report["torch"]
    print(
        f"torch {torch_info['torch_version']}: cuda_build={torch_info['cuda_build']} "
        f"cuda={torch_info['cuda_available']} mps={torch_info['mps_available']}"
    )
    for item in report["libraries"]:
        detail = item.get("version") or item.get("detail")
        print(f"  library {item['name']:<12} {item['status']:<7} {detail}")
    for item in report["coexistence"]:
        print(f"  same-process {item['name']:<15} {item['status']}")
    for item in report["checkpoints"]:
        print(
            f"  checkpoint {item['name']:<20} {item['status']} "
            f"({item['seconds']:.1f} s to hash)"
        )
    for item in report["live"]:
        timings = item.get("timings_s", {})
        print(
            f"  live {item['name']:<20} {item['device']:<4} "
            f"{item.get('dtype', '?'):<8} {item['status']:<13} "
            f"load {timings.get('load', float('nan')):6.1f} s  "
            f"cold {timings.get('inference_cold', float('nan')):6.2f} s  "
            f"warm {timings.get('inference_warm', float('nan')):6.2f} s  "
            f"peak RSS {_gib(item.get('peak_rss_bytes'))}  "
            f"accelerator {_accelerator_peak(item)}"
            + ("" if item["status"] == "ok" else f"  {item.get('detail', '')}")
        )
    for profile in report["profiles"]:
        print(f"  profile {profile['device']:<4} {profile['status']}")
    summary = report["summary"]
    print(
        f"doctor: {'ok' if summary['ok'] else 'NOT OK'}; usable devices "
        f"{summary['usable_devices'] or 'none'}"
        + ("" if summary["live_models_tested"] else " (live models not tested)")
    )
    if args.output is not None:
        _write_new_report(args.output, report)
        print(f"report: {args.output}")
    return report["exit_code"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``floscan`` CLI and return its exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    try:
        if args.command == "run":
            return _command_run(args, arguments)
        if args.command == "benchmark":
            return _command_benchmark(args)
        if args.command == "inspect":
            return _command_inspect(args)
        if args.command == "gates":
            return _command_gates(args)
        if args.command == "doctor":
            return _command_doctor(args)
        return _command_validate(args)
    except (RegistryError, EnvelopeError, RunIOError) as error:
        print(f"floscan {args.command}: error: {error}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
