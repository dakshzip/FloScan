"""P06 probe: convention-check residual on the synthetic box room vs resolution.

Usage: uv run python runs/p06-evidence/residual_vs_resolution.py SCRATCH_DIR
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "integration"))

import test_stray as t  # noqa: E402

from floscan.capture import stray  # noqa: E402

for scale in (1, 2, 4):
    t.RGB = (128 * scale, 96 * scale)
    t.DEPTH = (64 * scale, 48 * scale)
    t.FX = 100.0 * scale
    root = t.make_session(Path(sys.argv[1]) / f"s{scale}", frames=40)
    inspection = stray.inspect_session(stray.open_session(root))
    error = inspection.conventions["hypotheses"][0]["median_error_m"]
    print(scale, t.DEPTH, round(error * 1e3, 2), "mm")
