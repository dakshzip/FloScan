# P02 review

Reviewed commits `70ae0d1` and `421fc42` against accepted P01A commit `21d1979`. Disposition: **changes required**, limited to installation reproducibility and doctor output error handling. No reconstruction work is requested.

## Findings

### 1. Normal installed commands fail on the available Mac

`uv run floscan doctor --live-models --devices mps` fails before any checks with `ModuleNotFoundError: No module named 'floscan'`. The first independent `uv run pytest -q` returned **97 passed, 6 failed in 15.80 s**; all six failures involve subprocess imports. A second run after attempting an environment repair returned **89 passed, 14 failed in 10.36 s**, including the P01 shell entry points.

The generated `.venv/lib/python3.11/site-packages/floscan.pth` contains the correct absolute source path, but has the macOS `hidden` file flag. This environment's CPython 3.11.15 `site.addpackage` explicitly skips files with `UF_HIDDEN`. Clearing the flag makes a direct `.venv/bin/python` import succeed. The next `uv run --no-sync floscan --help` restores the flag and fails again. This is an observed installation/environment interaction, not evidence that the model implementation cannot run. A one-off `chflags` repair is insufficient.

With `PYTHONPATH="$PWD/src"` explicitly supplied, the CLI and workers import correctly and real MPS inference succeeds. That workaround is diagnostic evidence only: it is absent from the published operator commands and does not establish their reproducibility. Resolve the installation interaction or provide a durable documented launcher strategy, then prove the normal entry points work repeatedly without a manually exported source path. Preserve Python's safeguards rather than modifying its installed standard library.

### 2. Doctor report filesystem failures escape the CLI boundary

`src/floscan/cli.py` creates the output parent and opens the JSON file without wrapping expected filesystem failures. An independently reproduced call with an output parent that is a regular file raises an unhandled `FileExistsError`. The reproduction substitutes a minimal doctor report so the output boundary is isolated; it preserves the existing parent file. Permission errors and an output creation race reach the same unprotected boundary.

Use a narrow typed error boundary, return a controlled nonzero exit with a concise message, preserve existing files, and test the parent-is-a-file case and exclusive-creation behavior. The P01A error handling already offers a suitable pattern. Avoid catching every exception in `main`.

## Verified implementation and limitations

- Ruff lint and format checks pass; 24 Python files formatted. `git diff --check 21d1979..HEAD` passes.
- Full suite with the explicitly recorded import workaround, `PYTHONPATH="$PWD/src" uv run pytest -q`: **103 passed in 160.16 s**. This includes real model inference and the synthetic-image fp16/fp32 comparison; it does not erase the normal-command failures above.
- Independent doctor execution with the explicit source-path workaround verifies all three checkpoint hashes and all ten library checks.
- Depth Pro, Grounding DINO tiny and SAM 2.1 small complete real cold and warm MPS inference and all output sanity checks. The new report is `docs/reviews/p02-evidence/doctor-mps-with-pythonpath.json`.
- MPS memory samples are approximately 8.52 GiB, 2.39 GiB and 3.08 GiB respectively. These are sampled allocator figures, not measured peak whole-machine memory. The review ran tests concurrently with the doctor, so its timings are functional evidence under contention, not a replacement for the author's isolated timing baseline.
- Checkpoint pins, separate code/weight license evidence, the disclosed Depth Pro metadata conflict, offline loading, corrupted-file rejection and isolated workers are substantively implemented. Inference fixtures are synthetic and prove runtime behavior only.
- Torch/PyCOLMAP same-process conflict reproduces; the separate worker design is required on this Mac. Other tested library pairs are compatible.
- RTX 3080/CUDA and evaluator hardware remain untested. Windows is asserted for the RTX host in the dependency documentation and matrix, while this conversation only confirms the GPU. Cite operator confirmation if available; otherwise label the OS unknown and retain the Windows commands as a conditional profile.
- The one-image fp16/fp32 comparison cannot establish real-room metric accuracy or guarantee drift on every future capture. No geometry or accuracy gate is passed by P02.

## Next packet: P02A

**TASK:** Fix the two findings and make the existing runtime commands reproducible. Do not start P03.

**FILES TO TOUCH:** P02 runtime/package configuration and installation documentation only as required; `src/floscan/cli.py` for the doctor error boundary; `tests/integration/test_runtime.py` and relevant existing CLI regressions; new P02A evidence. Any launcher adjustment must remain installation-related and preserve its public arguments and behavior. Do not modify raw captures, gates, model checkpoints, old evidence or algorithm modules.

**ACCEPTANCE:** From a new shell without `PYTHONPATH`, run `uv sync --locked`, repeated `uv run floscan --help`, `./run.sh --help`, full tests and a real `floscan doctor --live-models --devices mps` using the documented launch method. Record any required durable setup step and verify it survives another normal invocation. Doctor output failures must return controlled errors without tracebacks or overwriting existing files. Keep CUDA claims untested unless that hardware is actually exercised. Confirm or correct the RTX host OS provenance.

Prompt for Opus:

> Read docs/reviews/P02-review.md. Implement P02A only: resolve the observed macOS editable-install hidden .pth interaction so documented commands work repeatedly without an ad hoc PYTHONPATH, and handle expected doctor-report filesystem failures through a narrow typed error boundary. Reproduce both findings, preserve all existing model/runtime evidence, run the acceptance commands from a clean shell, and return exact commands and outputs. Confirm the RTX host OS from operator evidence or label it unknown. Do not add models, reconstruction, contracts or P03 work. Stop for review after P02A.
