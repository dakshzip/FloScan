#!/usr/bin/env bash
# One command per capture:
#   ./run.sh --input <capture-dir> --tier photo|video|lidar --output <result-dir> --mode live|replay
# Quote paths with spaces, e.g. --input "example input /1a8384c3f6".
# See ./run.sh --help for exit codes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "run.sh: error: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 127
fi

# --locked fails loudly if pyproject.toml and uv.lock disagree.
exec uv run --project "$ROOT" --locked --quiet floscan run "$@"
