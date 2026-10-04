# P06 review

Reviewed commits `daa22e4` and `37bbda0` against accepted P05A commit `82b6d3e`. Disposition: **changes required**. Resolve P06A before starting P07.

## Findings

### 1. [P2] Corrupt confidence can abort the entire inspection

Location: `src/floscan/capture/stray.py:733`, with the depth loader at lines 274-280.

The image scan catches a corrupt confidence PNG and records `confidence_corrupt`, but leaves its depth marked usable. If that frame is an endpoint of a convention-check pair, `session.depth()` reads the corrupt confidence again and raises `StrayInputError`. The CLI then reports the whole session as invalid input and produces no inspection or frame diagnostics. The same corruption outside sampled pairs does not abort the scan, so the outcome depends on sampling.

Independent reproduction: replace synthetic `confidence/000000.png` with invalid PNG bytes. Baseline is `verified_with_frame_issues`; the damaged variant raises `StrayInputError` from the convention audit. Handle unavailable confidence consistently, exclude corrupt/invalid confidence from convention evidence, and preserve the recorded frame issue. If too few valid pairs remain, return an explicit unverified convention result rather than aborting or claiming verification. Keep corrupt depth handling intact.

### 2. [P2] Unused malformed IMU data can crash inspection or produce invalid JSON

Location: `src/floscan/capture/stray.py:652-668`.

The IMU summary assumes nonempty finite rows after `loadtxt`. An otherwise valid synthetic capture with a header-only `imu.csv` raises `ValueError` at `times.min()`; the CLI does not catch that exception. A seven-column all-NaN row instead reaches a verified inspection whose report fails `json.dumps(..., allow_nan=False)`. The normal output writer permits NaN, producing nonstandard JSON. This optional sensor stream is deliberately excluded from inference and must not prevent usable RGB-D evidence from being audited.

Validate row count, seven-column shape, finite values and timestamp ordering before statistics. Report unusable IMU with a reason and keep it excluded. Ensure every report serializes with strict JSON, without inventing replacement measurements. Add header-only, non-finite and malformed-column cases through both the API and CLI/output path.

### 3. [P2] A zero start-offset configuration crashes synchronization

Location: `src/floscan/capture/sync.py:151-158` and `SyncProfile`.

`max_start_offset=0` is accepted by the profile and is a natural configuration when neither stream may lose leading samples. It creates one hypothesis, then indexes `results[1]` and raises `IndexError`. Negative values can leave no hypotheses. Neither configuration yields an auditable association result or a deliberate configuration error.

Define the supported domain and validate it at profile/API boundaries. Either handle a single hypothesis with explicitly stated evidence limits or reject this setting with a clear validation error. Do not treat the absence of alternatives as proof that ambiguous timestamps are synchronized. Add zero/negative offset regression coverage and validate fraction/threshold domains in the same bounded repair.

## Verification and working behavior

- P06 integration suite: **24 passed in 31.77 seconds**, including both supplied sessions and duplicate-folder hashing.
- Full regression suite: **367 passed in 175.24 seconds**, including the earlier packets and real model smoke tests. The independent probes expose cases not covered by this passing suite.
- Independent CLI inspection of `example input /1a8384c3f6` succeeds: 5,251 sensor rows, 5,250 presented video frames, first presented frame associated to sensor row 1, all presented frames matched, maximum clock-fit residual 0.122 ms. Reprojection convention error is 8.4 mm versus 43.4 mm for the runner-up. Row 0 lacks RGB; rows 5,199 and 5,200 carry possible tracking-reset diagnostics.
- Real-sample integration tests independently inspect the 1,715-row session and confirm 1,714 presented frames, the same leading offset, sub-millisecond residual and verified conventions. They confirm the copied folder's manifest equals the first recording's manifest.
- Ruff check and format checks pass. `git diff --check 82b6d3e..HEAD` passes. The review reproduction script also passes Ruff.
- Independent probes and their output are in `docs/reviews/p06-evidence/repro.py` and `repro.log`. Run from the project root with `env -u PYTHONPATH .venv/bin/python docs/reviews/p06-evidence/repro.py`. They mutate only temporary synthetic captures, not supplied inputs.

The two distinct floor/ceiling walkthroughs remain separate capture worlds. Duplicate-folder evidence does not make these two recordings identical. Recorder version, device, lens distortion, world-up sign, IMU conventions and unavailable tracking state remain explicit gaps. Ingestion verification and reprojection consistency establish neither benchmark accuracy nor independent physical scale certification. The small camera-axis option added to the shared geometry helper is justified by the source's already-optical poses; its existing Apple behavior remains the default and has regression coverage.

## Bounded P06A prompt for Opus

> Read docs/reviews/P06-review.md and implement P06A only. Reproduce all three findings using docs/reviews/p06-evidence/repro.py before changing code. Repair corrupt-confidence handling across scanning, normalized loading and convention auditing; unusable confidence must become explicit frame evidence and must not be used to verify conventions. Repair optional IMU validation so empty, malformed or non-finite rows produce an honest unavailable diagnostic and strict JSON rather than an exception or NaN. Define and validate the synchronization/profile configuration domain, including zero and negative start offsets, without turning the absence of alternate hypotheses into verification. Stay within src/floscan/capture/{stray,sync}.py, configs/capture_profiles/stray.yaml and tests/integration/test_stray.py; touch the CLI/output writer only if needed for these diagnostics or strict JSON. Preserve originals and packet definitions. Add meaningful regression cases, rerun the independent probes, P06 tests, full regressions and Ruff checks, and return exact evidence. Do not implement P07 point clouds or planes. Stop for review after P06A.
