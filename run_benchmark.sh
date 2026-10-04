#!/usr/bin/env bash
# Benchmark one case at one tier:
#   ./run_benchmark.sh <case-id> photo|video|lidar [--mode live|replay]
# The scorer (P04) and case manifests (P05) do not exist yet; this command
# reports that and exits nonzero without scoring anything.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "run_benchmark.sh: error: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 127
fi

# Fail fast with the fix if the macOS environment cannot import floscan.
source "$ROOT/scripts/preflight.sh"
floscan_preflight || exit 1

exec uv run --project "$ROOT" --locked --quiet floscan benchmark "$@"
