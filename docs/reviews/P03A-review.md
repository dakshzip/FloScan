# P03A review

Reviewed commit `a81643d` against P03 commit `0f5dc07`. Disposition: **changes required**, limited to remaining association and ID-uniqueness gaps in finding 2 of `P03-review.md`. The original six malformed-construction reproductions are now rejected, and the original nested-mutation paths are blocked.

## Verified repairs

- Focused contract/frame suite: **118 passed in 0.25 seconds**.
- Full regression suite: **225 passed in 115.58 seconds**, without an import workaround.
- Independent execution of `runs/p03a-evidence/repro.py`: empty false success, dangling opening height, dangling wall thickness, wrong opening-width binding, duplicate coordinate frame and indefinite covariance are rejected. Nested list and dictionary assignment raise `TypeError`; ordinary serialization preserves the valid coordinate.
- Coverage now checks actual records and distinguishes a recorded negative inspection from inspection never performed. External export remains explicitly unavailable.
- Known covariance/information is checked for positive semidefiniteness, with valid nonzero singular matrices preserved.
- Validated containers block normal in-place mutation. The supported `to_json()` boundary revalidates the record tree; the regression suite verifies deliberate low-level bypass cannot silently serialize NaN as null.
- Ruff lint and formatting pass; 34 files formatted. `git diff --check 0f5dc07..HEAD` passes.
- The added optional coverage evidence field and stricter semantics are documented in `runs/p03a-evidence/compatibility.md`. Generated schema snapshots match models in the focused tests. New documents include the extra evidence field and therefore are not readable by the earlier strict model; compatibility is explicitly recorded rather than assumed.

## Remaining findings

### 1. [P1] Bundle depth association is checked in only one direction

`ReconstructionBundle._one_unit_and_valid_references` verifies that a depth record's owning frame points back to it. A different frame can still point to that same depth because the frame loop checks only that `depth_id` exists. Independently reproduced with two frames (`frame-1` and `frame-2`) both declaring `depth_id='depth-1'`, while the depth declares `frame_id='frame-1'`: validation accepts both. The second image can therefore consume another image's explicitly associated depth, corrupting unprojection and geometry without a contract error.

For every frame with depth, require the referenced depth's `frame_id` to equal that frame's ID. If future synchronization intentionally shares a raw depth observation, represent its derived association explicitly rather than silently disagreeing with the current record ownership.

### 2. [P2] Duplicate graph-node IDs still bypass uniqueness checks

The new result index checks coordinate frames and entity IDs before indexing, but omits graph nodes and edges. `PropertyGraph._consistency` still creates a set of node IDs without rejecting duplicate nodes. Appending a copy of `node-1` to the otherwise valid `_complete_result()` graph is independently accepted, including with top-level `status='ok'`.

Reject duplicate node and edge IDs on their raw lists at the graph's validation boundary, and define their uniqueness scope explicitly. Duplicate node identities make factor endpoints and the anchor ambiguous even when room geometry is valid. Keep legitimate graph nodes with distinct IDs valid.

## Independent probe method

Probes used `env -u PYTHONPATH uv run python` and fixture builders from `runpy.run_path('tests/contract/test_records.py')`. The depth fixture used valid canonical depth/mask metadata, `alignment='aligned_rgb'`, and zero synchronization residual; a second frame was linked to depth expressly owned by the first. The duplicate-node probe used `_complete_result()` and appended a fresh copy of its first graph node. Both probes were accepted. A separate differing-camera-ID probe is not a finding: aligned depth may legitimately have a different resolution and camera calibration record. Product source and tests were untouched.

## Next packet: P03B

**TASK:** Complete finding 2's association/uniqueness checks. The other three original findings are closed; do not rework them.

**FILES TO TOUCH:** `src/floscan/contracts/result.py`, `src/floscan/contracts/geometry.py`, `tests/contract/test_records.py`, and new P03B evidence. Generated schemas only if an explicitly necessary contract representation change requires them. Do not alter frame mathematics, gates, captures, runtime, earlier evidence or unrelated coverage semantics.

**TESTS:** Reject a frame referencing another frame's depth; reject duplicate graph nodes and edges. Positive tests must preserve correct one-to-one depth associations, distinct valid graph nodes/edges and existing separate-camera behavior. Run focused P03 tests, full regressions, Ruff and schema snapshots.

Prompt for Opus:

> Read docs/reviews/P03A-review.md and implement P03B only. The original false-success, covariance and mutation fixes are accepted; finish the remaining reference checks. Validate frame-to-depth ownership in both directions. Reject duplicate graph-node and edge IDs before lookup/set construction and document their uniqueness scope. Reproduce the two independently accepted cases, add focused negative and positive regressions, run the P03 tests and full suite, and return exact evidence. Keep existing coverage semantics, camera-alignment modes and frame mathematics unchanged. Do not start P04; stop for review after P03B.
