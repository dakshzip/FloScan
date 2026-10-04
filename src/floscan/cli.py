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
  {EXIT_FAILED}  failed: internal error, invalid gate registry or invalid envelope
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
        return _command_validate(args)
    except (RegistryError, EnvelopeError) as error:
        print(f"floscan {args.command}: error: {error}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
