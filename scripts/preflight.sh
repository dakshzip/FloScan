# Sourced by the launcher scripts. Fails fast, with the fix, when the macOS
# environment would break with "No module named 'floscan'"; see bootstrap.sh.
# Expects ROOT to be the project root. Does nothing outside macOS.

floscan_preflight() {
    [[ "$(uname -s)" == "Darwin" ]] || return 0
    local venv="$ROOT/.venv"
    if [[ ! -e "$venv" ]]; then
        echo "error: no environment at $venv; run scripts/bootstrap.sh first" >&2
        return 1
    fi
    local pth
    for pth in "$venv"/lib/python3.*/site-packages/floscan.pth; do
        [[ -e "$pth" ]] || continue
        # UF_HIDDEN (0x8000): CPython skips hidden .pth files.
        if (( $(stat -f %f "$pth") & 0x8000 )); then
            echo "error: $pth has the macOS hidden flag (set by iCloud Drive on" \
                "dot-named paths), so Python will not load it; run" \
                "scripts/bootstrap.sh to move the environment to .venv.nosync" >&2
            return 1
        fi
    done
    return 0
}
