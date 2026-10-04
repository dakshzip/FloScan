# P05 review

Reviewed commit `966cd5d` against `470a00f`. Disposition: **changes required**. The work order and visible collection checklist are useful; manifest completion and repeat identity checks need the bounded P05A below.

## Verified checks and scope

- P05 manifest tests: **30 passed in 0.19 seconds**.
- Full regression suite: **308 passed in 125.12 seconds**, without an import workaround.
- Ruff lint and formatting pass; 45 files formatted. `git diff --check HEAD^..HEAD` passes.
- Committed development status is honestly **0/8 met**. Calibration, held-out and walk-in cases are also incomplete. Missing human captures, tape readings and incumbent files are not fabricated.
- Existing samples are labeled development-only. The duplicated folder is distinguished from the separate complementary floor/ceiling session. No inference or geometry was added by P05.
- The capture card gives stock-app steps, 2-8 photos per group, connector coverage and no required metric aids. The work order requests fresh repeats, removable staged defects, instrument metadata and original incumbent export.
- This packet validates recorded metadata, not the truth of physical collection or file contents merely because a path/hash is present. Those remain evidence verification dependencies.

## Findings

### 1. [P1] Whole-property coverage and unique room composition are not enforced

Starting from `_complete_case()` in `tests/contract/test_case_manifest.py`, each mutation below is accepted with `declared_complete=True` and **8/8 composition checks met**:

- Replace the whole-property photo counts with only `{'room_001': 2}`. Three rooms plus the connector remain in the sketch, but no photos are recorded for them.
- Set the whole-property video's explicit `room_ids` to only `['R01']`.
- Replace the third room with a copy of the second, producing duplicate physical room IDs. The list length still satisfies the three-room count.

Validate unique sketch IDs and bind capture coverage to those physical IDs, including the connector. Photo folder labels need an explicit mapping to the sketch, or a canonical room-key convention; counting arbitrary keys is insufficient. Compute whole-property completeness from received room coverage, not the `scope` label alone. Keep partial coverage representable with an unmet checklist item; do not invent photos or reject all partial data.

### 2. [P1] Invalid repeat links and renamed copies satisfy independence

Setting `lidar_02.repeat_of='lidar_02'` is accepted as a complete case. The repeat list checks only whether its target ID exists, is received and has the same tier, so a recording can establish its own repeatability. Validate a distinct original recording, a sound non-cyclic relationship and overlapping physical room coverage.

The duplicate check also uses a path-sensitive raw manifest hash as proof of recording identity. Independent scratch reproduction: two directories contain the exact same video bytes, one under `rgb.mp4` and the other under `renamed.mp4`. Their raw manifest hashes differ; registering them as the primary and repeat is accepted with `declared_complete=True`. The existing hash unit test even describes a renamed frame as a different recording, contrary to the collection rule.

Retain path-sensitive raw manifest hashes for provenance, but add appropriate content identity for deduplicating the underlying recording independently of filenames/folder names. For still sets, preserve the media-content multiplicity needed to identify a copied set. Recording identity must not depend on changing metadata or renaming a file. Distinct media content alone still does not prove a fresh sensor session; keep operator recording/session provenance visible. Reject copied media as an independent repeat without comparing real-world geometry or inventing capture dates.

### 3. [P1] Transcriptions and nonexistent incumbent rooms count as the required export

Two independent `_complete_case()` mutations pass all eight checks:

- `incumbent.export_kind='transcription'`.
- `incumbent.room_ids=['R99', 'R98']`, although those rooms do not exist in the sketch.

The work order explicitly says a transcription is not an export, but `composition()` treats any received incumbent entry as satisfying the export requirement. Check provenance kind and membership of both distinct incumbent rooms in the case. A transcription can remain recorded as partial supporting evidence; it must not satisfy the original-export checklist item. Do not demand a particular free-tier export format or silently assume app capabilities.

### 4. [P2] Ground-truth protocol supplies unsupported fixed uncertainty

`benchmark/ground_truth/schema.md` assigns `uncertainty_m` a default of **0.003 m for a tape**, regardless of instrument accuracy, distance, access or placement. The planning protocol requires instrument/placement uncertainty to be recorded and explicitly cautions against unverified millimetre accuracy. Repeated readings do not alone justify this fixed number.

Remove the numeric default. Record an uncertainty supported by the instrument and procedure, or mark it unknown with a reason. If a worked example uses 0.003 m, label it a synthetic illustration, not a default to populate actual readings. Continue using central GT values for gates and show uncertainty only as the documented sensitivity convention.

## Independent reproduction

Probes ran with `env -u PYTHONPATH uv run python`, using a fresh `_complete_case()` from `runpy.run_path('tests/contract/test_case_manifest.py')` for each mutation. All six malformed metadata cases above were accepted as declared complete; the renamed-media probe used newly created scratch directories and dummy bytes, not supplied captures. Product source, tests and raw inputs were untouched.

## Next packet: P05A

**TASK:** Fix composition/coverage, repeat identity/association, incumbent export eligibility and uncertainty instructions. Do not begin P06 or fabricate missing collection evidence.

**FILES TO TOUCH:** P05 manifest code, manifest schema/example metadata as needed, P05 tests, capture/GT work-order documentation for the new room mapping/identity and uncertainty conventions, and new P05A evidence. Preserve raw captures, numeric gates, scorer behavior and earlier evidence. Keep existing received/missing/rejected states and named case aliases.

**TESTS:** Reject or mark incomplete the independent examples above. Positive cases must support genuine whole-property coverage across all declared rooms/connector; partial captures with unmet checklist entries; distinct same-room same-tier primary/repeat recordings; copies detected despite renamed media; original exports for two actual rooms; and transcription recorded without an export pass. Update the misleading renamed-frame test. Run P05 tests, full regressions, Ruff and status reporting. Keep real manifests incomplete until actual evidence arrives.

Prompt for Opus:

> Read docs/reviews/P05-review.md and implement P05A only. Bind whole-property and photo coverage to unique physical room IDs including connectors, preserving incomplete collection states. Reject self/cyclic/incorrect-room repeat links and copied recordings even when media files are renamed; keep raw manifest provenance distinct from media recording identity. Count only an original app export for two actual case rooms as satisfying the incumbent requirement, while allowing transcription as explicitly incomplete supporting evidence. Remove unsupported fixed GT uncertainty defaults and preserve unknown uncertainty honestly. Reproduce the review failures, add focused positive and negative tests, run the P05 suite, full regressions and status command, and return evidence and compatibility notes. Do not start P06 or invent human captures, GT readings or incumbent exports. Stop for review after P05A.
