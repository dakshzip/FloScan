#!/usr/bin/env bash
# One-time setup: install the locked Python environment. Rerunnable.
# Then fetch model weights with scripts/fetch_models.sh.
#
# macOS keeps the environment in .venv.nosync, with .venv a symlink to it.
# When the project lives in an iCloud Drive-synced folder (for example a
# Desktop synced by "Desktop & Documents"), iCloud sets the hidden flag on
# dot-named paths such as .venv and everything inside. CPython skips hidden
# .pth files, so the editable install's floscan.pth stops loading and every
# command fails with "No module named 'floscan'". iCloud leaves names ending
# in .nosync alone, so the real files never get the flag.
#
# Windows: run `uv sync --locked` instead of this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "bootstrap.sh: error: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 127
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ -e .venv && ! -L .venv ]]; then
        echo "bootstrap.sh: replacing the plain .venv directory (regenerable) with" \
            "a symlink to .venv.nosync"
        rm -rf .venv
    fi
    UV_PROJECT_ENVIRONMENT="$ROOT/.venv.nosync" uv sync --locked
    ln -sfn .venv.nosync .venv
else
    uv sync --locked
fi

uv run --locked floscan --version
echo "bootstrap.sh: environment ready; next run scripts/fetch_models.sh"
