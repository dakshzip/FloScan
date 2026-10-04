# P04 review

Reviewed commit `a899614` against `880a14f`. Disposition: **changes required**. Revision hashes changed before publication; `docs/history-rewrite.md` records the mapping for earlier reviews. This review does not alter history or product code.

## Verified checks

- P04 scorer tests: **30 passed in 1.91 seconds**.
- Full regression suite: **259 passed in 126.97 seconds**.
- Ruff lint and formatting pass; 36 files formatted.
- Numeric gate registry is unchanged by P04.
- `git diff --check HEAD^..HEAD` finds only a trailing blank line in `runs/p04-evidence/sample-score/report.md:40`. This is a minor generated-report formatting issue, not an accuracy finding.

The hand-calculated tests cover the 8/11 opening denominator, missing walls/rooms/values, duplicates, biased repeatable ceilings, OR/AND repeatability, rigid-only alignment, wrong footprint shape, overlaps and incumbent missing output. Live inference remains an explicit nonzero placeholder. These checks establish scorer behavior on synthetic examples; they are not real capture compliance evidence.

## Findings

### 1. [P1] Unplaced local geometry can pass whole-property gates

`plan_from_result` keeps a room's local coordinates when `T_property_from_room` is absent, then emits those coordinates in a scoring view that claims one common frame. It drops placement status and graph registration status, retaining only the number of graph components. The subsequent rigid matching can make the local geometry look like a placed property.

Independent reproduction: start with `_complete_result()` from `tests/contract/test_records.py`, set the tier to photo, derive GT from the valid placed plan, then change the prediction room to `placement_status='unplaced'`, remove its placement transform, and set graph registration to `not_attempted`. Mark result/stitched coverage partial with reasons so it remains a valid diagnostic result. The scorer returns **measured_pass for both `photo_stitch_single_plan` and `photo_footprint`**, despite placement never being established.

Preserve placement and registration evidence in the scoring boundary. Unplaced room coordinates must not enter a common-property union or receive a whole-property gate pass. Keep valid local scalar measurements scoreable; do not discard them merely because stitching failed. Treat placement failures explicitly rather than relying on accidental mismatch of room-local coordinates. Validate equivalent direct scoring-view inputs, not only internal-result conversion.

### 2. [P1] Incumbent gate does not enforce the two-room shared set or uniqueness

`IncumbentExport.room_ids` requires two list entries, but does not ensure distinct valid rooms or that the reported dimensions actually cover them. `incumbent_comparison` never consults those room IDs when scoring. Independently reproduced with the usual three-room GT and a LiDAR prediction: export declares `['living', 'kitchen']`, but contains only `living-S`. The gate returns **measured_pass with `n_shared=1`**, although only one room was compared.

Repeating that identical dimension five times is also accepted and produces **`n_shared=5`**. Duplication can arbitrarily weight an easy win in the nominally frozen shared set.

Validate unique physical dimension keys `(kind, gt_id)`, distinct GT room IDs, and ownership of every compared dimension in the declared benchmark rooms. A measured pass requires eligible shared measurements from both required rooms. Preserve explicit incumbent-unavailable exclusions, but do not let an excluded/missing room satisfy the two-room requirement. Our missing prediction for an eligible incumbent dimension must continue to count as a loss. Do not change the frozen 70% threshold or select dimensions using our errors.

### 3. [P2] Polygon holes are silently filled by the scoring adapter

The internal `Polygon2D` supports holes. `plan_from_result` copies only `room.boundary.outer`, while `RoomView` represents only a single outer ring. Footprint union, overlap and room-IoU matching therefore see filled geometry instead of the actual polygon.

Independent reproduction: add a clockwise 1x1 m hole to the asymmetric 3.1x4.7 m room fixture. The validated internal room area is **13.57 m2**, but its emitted scoring polygon has area **14.57 m2**. Matching/footprint geometry is already wrong before any error formula runs.

Preserve holes through transformation and all polygon consumers, including union, IoU, overlap and boundary sampling. Add known-area fixtures and JSON round trips. Retain backward compatibility for existing hole-free scoring views where feasible; if an unsupported polygon cannot be represented, reject it explicitly rather than silently filling it.

## Independent probes

Probes ran with `env -u PYTHONPATH uv run python`, using builders loaded through `runpy.run_path('tests/unit/test_benchmark.py')` and `runpy.run_path('tests/contract/test_records.py')`. They used fresh fixtures and the real validated gate registry. The internal unplaced and holed results passed `PropertyResult.model_validate` before conversion. Product source, tests, GT fixtures and gate values were untouched.

## Next packet: P04A

**TASK:** Correct these three scorer-boundary/eligibility issues. Do not build the estimator or begin P05.

**FILES TO TOUCH:** P04 evaluator modules, `tests/unit/test_benchmark.py`, additional tiny scoring fixtures and new P04A evidence. No gate-value changes, capture edits, model/runtime changes or unrelated contract refactors. Any newly required scoring-view fields must have documented semantics and compatibility.

**TESTS:** Unplaced/not-attempted and disconnected internal results must fail dependent whole-property gates while retaining valid local dimension errors; valid placed plans still pass. Incumbent exports with duplicate dimensions, duplicate/unknown rooms, out-of-set dimensions or comparison from only one declared room cannot earn the two-room pass; valid two-room comparisons retain exact win/tie/loss counts and our missing-output penalty. Holed polygons preserve exact area under rigid transforms and produce correct union/IoU/overlap/boundary metrics. Run P04 tests, full regressions, Ruff and diff checks. Preserve earlier evidence.

Prompt for Opus:

> Read docs/reviews/P04-review.md and implement P04A only. Preserve placement/registration status in scoring views so unplaced local geometry cannot pass whole-property gates; retain valid local measurements. Enforce unique incumbent dimensions and a genuine comparison covering both distinct declared GT rooms without changing thresholds or missing-output losses. Preserve polygon holes throughout conversion and spatial metrics, with explicit scoring-view compatibility. Reproduce the independent failures, add positive and negative tests, run the P04 suite and full regressions, and return exact evidence and remaining limitations. Do not add inference, datasets, P05 or new gate definitions. Stop for review after P04A.
