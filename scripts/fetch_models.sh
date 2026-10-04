#!/usr/bin/env bash
# Download every pinned model checkpoint and verify it against
# configs/models.lock.json (exact revision, size and SHA-256).
#
#   scripts/fetch_models.sh                      # all models into ./models
#   scripts/fetch_models.sh --models depth_pro   # a subset
#   FLOSCAN_MODELS_DIR=/Volumes/w scripts/fetch_models.sh   # a preloaded volume
#
# Valid files are kept, so rerunning only fetches what is missing or altered.
# Inference never downloads; it fails if this script has not completed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "fetch_models.sh: error: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 127
fi

# Fail fast with the fix if the macOS environment cannot import floscan.
source "$ROOT/scripts/preflight.sh"
floscan_preflight || exit 1

exec uv run --project "$ROOT" --locked --quiet python -m floscan.runtime.models fetch "$@"
