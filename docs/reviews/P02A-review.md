# P02A review

Reviewed commit `2f28ed5` against P02 commit `421fc42`. Disposition: **accept**. This supersedes the two changes-required findings in `P02-review.md`.

## Verified repairs

- Independent full suite, `env -u PYTHONPATH uv run pytest -q`: **107 passed in 182.27 seconds**.
- Independent `env -u PYTHONPATH -u UV_PROJECT_ENVIRONMENT -u VIRTUAL_ENV uv run floscan doctor --live-models --devices mps`: exit 0, all ten library checks and all three checkpoint hashes pass; all three models complete real cold/warm MPS inference. Evidence: `docs/reviews/p02-evidence/doctor-mps-after-p02a.json`. The doctor ran alongside the suite, so its timing figures reflect contention and should not replace isolated performance measurements.
- `uv sync --locked`, repeated `uv run floscan --help`, and `./run.sh --help` succeed without `PYTHONPATH`.
- `scripts/bootstrap.sh` is rerunnable. The Mac environment resides in `.venv.nosync`, reached through the `.venv` symlink. The editable-install `.pth` file remains unhidden through sync, bootstrap and repeated launches. Python's path safeguards are untouched.
- `uv run floscan --version` succeeds with `PYTHONPATH`, `UV_PROJECT_ENVIRONMENT` and `VIRTUAL_ENV` explicitly removed.
- Independent parent-is-a-file reproduction returns controlled exit 1 without a traceback and preserves the parent file. Exclusive report creation raises `RunIOError` and preserves the existing report.
- Report path preparation happens before the slow doctor checks; expected create/write errors use the existing typed CLI boundary. Partial report cleanup is restricted to a report created by the current write call.
- Ruff lint and formatting pass, with 25 Python files formatted. Shell syntax checks and `git diff --check 421fc42..HEAD` pass.

## Limits

This acceptance is for the runtime packet on the available M2 Pro. CUDA and evaluator hardware remain untested. The Torch/PyCOLMAP process-isolation requirement remains. Synthetic runtime and precision checks establish executable model paths, not floor-plan accuracy or assignment gate compliance. The RTX host OS now cites operator confirmation in the implementation session; this review does not independently exercise that workstation.

The macOS setup migration removes and regenerates a plain project `.venv` when necessary. Its dependencies are regenerable from the lock; captures and model checkpoints are untouched. The reviewer reran bootstrap against the already migrated environment and verified it remains usable.

## Next packet: P03

Prompt for Opus after acceptance:

> P01/P01A and P02/P02A are accepted. Read docs/reviews/P02A-review.md and implement P03 only from docs/implementation-strategy/06-task-packets.md. Read the coordinate and record specifications in docs/implementation-strategy/02-data-contracts.md and the frozen requirements in docs/adr/001-requirements.md before coding. Follow the P03 file allowlist. Implement validated typed records, explicit rigid and similarity transforms, and centralized vendor-frame/resize conversions. Reject mixed units, nonfinite values, invalid rotations, broken references and metric operations on up-to-scale geometry. Keep unknown covariance unknown. Use asymmetric synthetic fixtures to prove inverse/composition, unproject/reproject, plane inverse transpose, handedness, quaternion ordering and applying scale exactly once. Run the P03 tests and existing regression checks; return the change, evidence, remaining limitations and next packet P04. Do not start reconstruction, ingestion, scoring or P04 implementation. Stop for review after P03.
