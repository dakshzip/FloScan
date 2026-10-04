# P01 review

Reviewed commit `52453cc` against baseline `7865cb5`. Review disposition: **changes required**, limited to diagnostic validation and error handling. No application source was changed during this review.

## What is correct

- Required check: `uv run pytest tests/contract/test_cli.py -q`: **40 passed in 0.97 seconds**.
- `./run.sh --help`: exit 0, explicit tiers/modes and documented exit codes.
- `uv run ruff check .`: passed. `uv run ruff format --check .`: 15 files already formatted.
- Normal generated results honestly report unsupported/incomplete behavior, empty measurement/geometry arrays, unavailable external schema and unverified gates.
- Live/replay are explicit; replay does not run live inference. Raw-input/output separation and exclusive result creation are implemented.
- Thresholds, provisional scoring interpretations and unavailable Round 1/schema sources are recorded. No measured accuracy claim is made.
- Package marker, lock and packet evidence are justified small additions to the P01 scope. No reconstruction implementation was introduced.

## What is wrong

### 1. Diagnostic validator accepts a fabricated complete success

Location: `src/floscan/pipeline.py`, lines 782-805.

Independent reproduction: copy a generated P01 result; set `status=ok`, `exit_code=0`, `coverage.contract_complete=true`, every section status to `available`, and every stage status to `ok`. Leave rooms, measurements, artifacts and other payloads empty/null. `validate_envelope` accepts it.

This also allows `public_schema_export` to claim availability while `external_schema_status` remains unavailable. The current tests reject changing only the top-level status; they do not reject consistent-looking false metadata. The P01 diagnostic schema must not certify a full result with no pipeline implementation.

### 2. Output filesystem errors escape the CLI error boundary

Locations: `src/floscan/pipeline.py`, lines 652-655; `src/floscan/cli.py`, line 256.

Independent reproduction: create a regular file `file-not-directory`, then request output at `file-not-directory/output`. The CLI raises uncaught `NotADirectoryError`, rather than returning the documented failure with a concise diagnostic. Actual subprocess invocation produces a traceback.

### 3. Malformed registry field types escape as ordinary exceptions

Locations: `src/floscan/pipeline.py`, source-page conversion immediately after line 355; `src/floscan/cli.py`, line 256.

Independent reproduction: replace `pages: 6` in a temporary gate registry with `pages: not-a-number`, then invoke `gates --gates <temporary-file>`. An uncaught `ValueError` is raised by `int(source["pages"])`. A malformed registry should raise `RegistryError` and produce documented exit 1.

## What is risky

- The test suite covers expected output and simple tampering well, but stronger contradictory-state and invalid-type cases are missing.
- The diagnostic validator is specific to P01. P03 must deliberately replace its empty-payload restrictions with real record validation, using a distinct schema version; do not weaken it now to anticipate future success.
- The ADR derives an approximate deadline from a PDF file timestamp. That is only a scheduling assumption, not the timestamp of the user's 40-hour statement. Keep the deadline labeled approximate and confirm remaining time before scheduling major work.
- No reconstruction/measurement accuracy was evaluated here, appropriately for P01.

## What should change

Make only the narrow fixes below. Keep the passing normal flows and existing evidence. No broad refactor, reconstruction, dependency expansion, benchmark scoring or P02 work is needed to resolve this review.

## Next exact packet: P01A

**TASK:** Close the three reproduced P01 validation/error-boundary defects.

**CONTEXT:** This review found false success acceptance and uncaught output/registry exceptions despite 40 passing tests.

**FILES TO TOUCH:** `src/floscan/pipeline.py`, `src/floscan/cli.py`, `tests/contract/test_cli.py`. Permit a new `runs/p01a-evidence/` directory for commands/test results only.

**FILES NOT TO TOUCH:** Gate values/policies, original capture assets, schema identifiers, dependency files, architecture documents, frozen P01 evidence and all reconstruction/model code.

**INPUTS:** Current P01 commit and the three reproductions above.

**OUTPUTS:** Minimal correction diff and passing regression checks.

**IMPLEMENTATION REQUIREMENTS:**

1. For `floscan-result/0.1.0-diagnostic`, reject `status=ok` and `contract_complete=true`; enforce unavailable/blocked external export consistently with schema authority. Keep support for diagnostic failure states and empty payloads.
2. Convert anticipated filesystem errors at the run/inventory/write boundary into a concise documented failure, exit 1, without traceback or a fabricated successful result. Preserve exclusive creation and raw-input protection.
3. Validate source page count as a positive integer (excluding booleans) before using it; malformed registry types must raise `RegistryError`. Audit neighboring type conversions for the same issue. Do not indiscriminately catch all `ValueError`/`Exception` at the top level and hide programmer bugs.

**TESTS:** Add the complete false-success mutation, unavailable external export marked available, output parent is a file, and nonnumeric/boolean/nonpositive source page counts. Exercise user-facing error behavior through a CLI subprocess where relevant.

**ACCEPTANCE CRITERIA:** All three reproductions are rejected with the intended diagnostic; existing normal live/replay/missing-input behavior passes; raw inputs and frozen results remain unchanged; no new dependencies or scope expansion.

**COMMANDS TO RUN:** `uv run pytest tests/contract/test_cli.py -q`; `./run.sh --help`; `uv run ruff check .`; `uv run ruff format --check .`; `git diff --check`.

**EXPECTED RESULT:** Every current test plus meaningful new regressions passes, with a small correction commit ready for review. Stop before P02.

**KNOWN RISKS:** Overbroad exception catching hides real faults; changing the diagnostic schema prematurely creates P03 migration problems.

## Prompt to send Opus

Read `docs/reviews/P01-review.md`. Implement P01A only, following its file allowlist, requirements and acceptance checks. Reproduce each defect before fixing it, retain the existing P01 evidence, run the listed commands, and return the diff and test output. Do not start P02 or add reconstruction/models. Stop after this correction for review.
