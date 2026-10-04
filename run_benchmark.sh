#!/usr/bin/env bash
# Benchmark one case at one tier:
#   ./run_benchmark.sh <case-id> photo|video|lidar [--mode live|replay]
#     Live inference and case manifests (P05) do not exist yet; this reports
#     that and exits nonzero without scoring anything.
#   ./run_benchmark.sh score --ground-truth GT.json --predictions P.json ... \
#       --output DIR [--correspondence C.json] [--incumbent I.json]
#     Scores existing predictions with the evaluator in benchmark/evaluator.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "run_benchmark.sh: error: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 127
fi

# Fail fast with the fix if the macOS environment cannot import floscan.
source "$ROOT/scripts/preflight.sh"
floscan_preflight || exit 1

if [[ "${1:-}" == "score" ]]; then
    # The evaluator sits outside the installed package, so put the project
    # root on the import path for this process only.
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec uv run --project "$ROOT" \
        --locked --quiet python -m benchmark.evaluator.report "$@"
fi
exec uv run --project "$ROOT" --locked --quiet floscan benchmark "$@"
