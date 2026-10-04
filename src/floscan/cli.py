"""Command-line entry point: ``floscan run|benchmark|gates|validate``."""

from __future__ import annotations

import argparse
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
        help="score a benchmark case (not implemented until P04/P05)",
        description=(
            "Run live inference on a benchmark case and score it against sealed "
            "ground truth. The scorer (P04) and case manifests (P05) do not "
            "exist yet, so this command always reports that and exits nonzero."
        ),
        epilog=EXIT_CODES_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    benchmark.add_argument("case_id", help="case alias, e.g. dev_property")
    benchmark.add_argument("tier", choices=TIERS, help="tier to benchmark")
    benchmark.add_argument("--mode", choices=MODES, default="live")

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
        "  The benchmark scorer is not implemented (packet P04) and no case "
        "manifests or ground truth are registered (packet P05).\n"
        "  No inference was run and no gate was scored; all gates remain "
        "unverified or unspecified_source.",
        file=sys.stderr,
    )
    return EXIT_INCOMPLETE


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


def _command_doctor(args: argparse.Namespace) -> int:
    # Imported here so other commands never pay for the runtime modules.
    from floscan.runtime import models

    devices = None if args.devices == "all" else args.devices.split(",")
    if args.output is not None and args.output.exists():
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
            f"  live {item['name']:<20} {item['device']:<4} {item['status']:<13} "
            f"load {timings.get('load', float('nan')):6.1f} s  "
            f"cold {timings.get('inference_cold', float('nan')):6.2f} s  "
            f"warm {timings.get('inference_warm', float('nan')):6.2f} s  "
            f"peak RSS {_gib(item.get('peak_rss_bytes'))}"
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
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
