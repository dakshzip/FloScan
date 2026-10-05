# FloScan

FloScan turns a handheld iPhone capture (photos, video or LiDAR) into a measured whole-property floor plan with damage, concealed-damage flags and scope, every measurement carrying a confidence interval.

## Status: local LiDAR plan (P09)

For LiDAR captures in the Stray Scanner export format, one command produces a partial local result: verified capture timing and conventions, metric points and planes, room outlines, walls and surfaces, measurements, and an SVG/PNG plan.
Every edge without an observed wall, every room without an observed ceiling and every corner not observed is marked as such; nothing is filled in.
No measurement has an accuracy interval yet (no calibration exists), and no gate has been measured: every gate is `unverified` or `unspecified_source`.
Photo and video tiers, openings, the stitched whole-property plan, damage, concealed-damage flags and scope are not implemented; their sections are reported unavailable.

The run envelope `result.json` follows a **project-owned** schema (`floscan-result/0.2.0`); the records are in `property_result.json` (project-owned `internal-v0`), and `floscan validate` checks both.
The assignment's published JSON schema and the earlier Round 1 gates were not supplied and are unavailable, so no conformance to them is claimed.
See `docs/adr/001-requirements.md`.

## Requirements

- macOS or Linux with `bash`, or Windows with PowerShell
- [uv](https://docs.astral.sh/uv/) 0.11 or newer (it fetches a suitable Python automatically)
- About 7 GB of disk: 1.4 GB environment, 2.8 GB model weights, plus captures

## Quickstart

```sh
scripts/bootstrap.sh       # one-time: install the locked environment
scripts/fetch_models.sh    # one-time: download and verify 2.8 GB of weights
./run.sh --help            # command contract and exit codes
uv run floscan doctor      # check hardware, libraries and weights
```

On Windows, run `uv sync --locked` instead of `scripts/bootstrap.sh`, `uv run python -m floscan.runtime.models fetch` instead of `scripts/fetch_models.sh`, and `uv run floscan ...` in place of the `.sh` launchers.

On macOS, `scripts/bootstrap.sh` keeps the environment in `.venv.nosync` with `.venv` as a symlink to it.
When the project sits in a folder synced by iCloud Drive (such as a Desktop under "Desktop & Documents"), iCloud sets the hidden flag on dot-named paths like `.venv`, and Python skips hidden `.pth` files, so a plain `uv sync` environment eventually fails with `No module named 'floscan'`.
iCloud leaves `.nosync` names alone.
The launchers detect the broken state and say to rerun `scripts/bootstrap.sh`.

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
uv run pytest -q                       # contract and runtime tests (needs fetched weights)
uv run ruff check . && uv run ruff format --check .
```

## Repository map

| Path | Purpose |
|---|---|
| `configs/gates.yaml` | Requirement IDs G01-G19 and gates, each threshold quoted verbatim from the source PDFs |
| `docs/adr/001-requirements.md` | Frozen requirements, unavailable sources, gate interpretations, diagnostic contract |
| `docs/implementation-strategy/` | Planning handoff (architecture, contracts, benchmarks, task packets); kept local and not distributed, like the assignment PDFs |
| `src/floscan/cli.py`, `src/floscan/pipeline.py` | CLI and diagnostic pipeline skeleton |
| `AGENTS.md` | Rules for coding agents working in this repository |
| `example input /` | Raw sample RGB-D recordings, preserved unchanged and not tracked in Git |
