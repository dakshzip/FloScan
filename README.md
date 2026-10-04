# FloScan

FloScan turns a handheld iPhone capture (photos, video or LiDAR) into a measured whole-property floor plan with damage, concealed-damage flags and scope, every measurement carrying a confidence interval.

## Status: P01 diagnostic skeleton

This build produces **no geometry, no rooms and no measurements**.
It provides the command-line contract, the requirement and gate registry, and a validated diagnostic result envelope that reports every unavailable part of the output contract.
No gate has been measured; every gate is `unverified` or `unspecified_source`.
Reconstruction, measurement, damage, export and rendering arrive in later packets (see `docs/implementation-strategy/06-task-packets.md`).

The output JSON follows a **project-owned** schema (`floscan-result/0.1.0-diagnostic`).
The assignment's published JSON schema and the earlier Round 1 gates were not supplied and are unavailable, so no conformance to them is claimed.
See `docs/adr/001-requirements.md`.

## Requirements

- macOS or Linux with `bash`
- [uv](https://docs.astral.sh/uv/) 0.11 or newer (it fetches a suitable Python automatically)

## Quickstart

```sh
uv sync                 # install the locked environment
./run.sh --help         # command contract and exit codes
```

Process one capture (one command per capture):

```sh
./run.sh --input "example input /1a8384c3f6" --tier lidar --output runs/demo-lidar --mode live
```

Note that the sample directory name ends with a space, so always quote it.
The input directory is only read; results may never be written inside it.

With this build the command exits `4` and writes `runs/demo-lidar/result.json` with `"status": "unsupported"`, every contract section `unavailable` with a reason, empty `rooms`/`measurements`, and all stages after the input inventory reported as `not_implemented`.

Other commands:

```sh
uv run floscan gates                          # requirement-traced gate registry
uv run floscan validate runs/demo-lidar/result.json
./run_benchmark.sh dev_property lidar         # exits 4: scorer (P04) and cases (P05) not built yet
```

## Command contract

`./run.sh --input <capture-dir> --tier photo|video|lidar --output <result-dir> --mode live|replay`

- `--tier` is mandatory and never auto-detected or upgraded.
- `--mode live` processes the raw input; `--mode replay` only reuses a matching cache and never falls back to live inference.
- `--output` must be a new directory outside the input; an existing `result.json` is never overwritten.

| Exit | Meaning |
|---|---|
| 0 | Full output contract produced (not reachable in this build) |
| 1 | Internal failure, invalid gate registry or invalid envelope |
| 2 | Usage error (bad arguments, unknown tier, output inside input, existing result); nothing written |
| 3 | `invalid_input`: input missing, not a directory or empty; envelope written |
| 4 | Incomplete: envelope written, output contract not produced |
| 5 | Replay unavailable: no matching cache; live inference not attempted |

## Development

```sh
uv run pytest tests/contract/test_cli.py -q
uv run ruff check . && uv run ruff format --check .
```

## Repository map

| Path | Purpose |
|---|---|
| `configs/gates.yaml` | Requirement IDs G01-G19 and gates, each threshold quoted verbatim from the source PDFs |
| `docs/adr/001-requirements.md` | Frozen requirements, unavailable sources, gate interpretations, diagnostic contract |
| `docs/implementation-strategy/` | Approved architecture, contracts, benchmark design and task packets |
| `src/floscan/cli.py`, `src/floscan/pipeline.py` | CLI and diagnostic pipeline skeleton |
| `AGENTS.md` | Rules for coding agents working in this repository |
| `example input /` | Raw sample RGB-D recordings, preserved unchanged and not tracked in Git |
