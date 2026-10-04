# P03 review

Reviewed commit `0f5dc07` against `6b66af4`. Disposition: **changes required**. The coordinate mathematics has useful coverage, but record validation still accepts invalid and misleading results. Complete the bounded P03A below before P04.

## Verified checks

- Required packet tests: **85 passed in 0.19 seconds**.
- Full regression suite: **192 passed in 125.05 seconds**, without a source-path workaround.
- Ruff lint and format checks pass; 36 Python files formatted.
- `git diff --check 6b66af4..HEAD` passes.
- Committed schema/model equality is covered by the passing suite. The published schemas remain project-owned; semantic validation is required in addition to JSON Schema.

The tests exercise asymmetric geometry, camera projection, rigid inverse/composition, plane inverse transpose, Apple/COLMAP conventions, pixel resizing, handedness and applying scale once. No reconstruction, real-room accuracy or benchmark gate has been established by this packet.

## Findings

### 1. [P1] Empty results can claim complete success

At `src/floscan/contracts/result.py:240`, result validation compares the top-level status with the coverage labels but never verifies that those labels match the actual contents. Independently reproduced from `_result()` in `tests/contract/test_records.py`: clear rooms, surfaces, walls, openings, measurements and coordinate frames; set `capture_id=None`; set every coverage entry to `available`; set `status='ok'`. `PropertyResult.model_validate` accepts it, even with no property graph, scale, rendered artifacts or external export.

This repeats the false-success failure closed in P01A, now in the new internal schema. Validate coverage against the records/evidence it describes. At minimum, absent capture identity, coordinates, room geometry, measurements and stitched graph cannot be certified as available. Distinguish a valid negative inspection finding from inspection never performed; do not require fabricated damage to make an inspection complete. Keep the unavailable external schema/export explicit. Partial and unavailable results must remain representable with reasons.

### 2. [P1] Referential integrity is incomplete and measurement bindings can be wrong

Independent mutations of the valid fixture are accepted:

- `openings[0].height_measurement_id='nonexistent'`.
- `walls[0].thickness_measurement_id='nonexistent'`.
- `openings[0].width_measurement_id='m-wall'`, where `m-wall` is a wall-length measurement of another subject. Mere ID existence is insufficient: a consumer would report wall length as opening width.
- Appending a duplicate coordinate-frame record with ID `R1`. `_check_references` builds the frame dictionary before checking uniqueness, silently collapsing duplicates.

Check uniqueness on raw record lists before indexing. Audit references that can be resolved within each containing result/bundle, including optional measurement and connector references; verify the expected subject, quantity, room/surface ownership and relevant version. Do not demand unresolved external observation/asset records be invented; explicitly distinguish external references from references to the containing record tables. Extend the same approach to bundle associations between frames, depths and cameras where those records are present. Preserve unavailable optional values as null.

### 3. [P2] Indefinite covariance is accepted as known uncertainty

`check_symmetric_covariance` checks symmetry and nonnegative diagonal entries, which do not ensure positive semidefiniteness. A 6x6 identity matrix with entries `[0,1]` and `[1,0]` changed to `2.0` passes `Pose.model_validate` with `covariance_status='known'`, despite its smallest eigenvalue being **-1.0**. It describes negative variance in a direction and cannot support valid uncertainty propagation.

Reject covariance/information matrices with materially negative eigenvalues using a documented numerical tolerance. Preserve legitimate nonzero singular matrices for partial observability. Keep unknown covariance unknown and the existing rejection of all-zero covariance claiming certainty. Apply validation consistently where these fields are declared; do not silently project an invalid matrix into a valid one.

### 4. [P2] Nested mutation bypasses validation and silently changes geometry on export

`Contract` is declared immutable with `frozen=True`, but its list/dictionary fields remain mutable. Independently reproduced:

```python
r = PropertyResult.model_validate(valid_fixture)
r.surfaces[0].origin[0] = float('nan')
r.to_json()  # origin becomes [null, 0.0, 0.0], without an error
```

The same mutation path can alter a validated rotation, reference list or coverage dictionary without running any validator. This breaks the promised immutable records and allows invalid geometry to reach output. Provide immutable nested containers or an equally explicit safe contract boundary that prevents ordinary nested mutation and revalidates serialization. Preserve normal JSON array/object representation and strict input validation. Test nested list and dictionary changes, including nonfinite geometry; an invalid number must never silently become a missing value without a reason.

## Reproduction method

Independent probes ran with `env -u PYTHONPATH uv run python`, loading fixture builders through `runpy.run_path('tests/contract/test_records.py')`. Each mutation started from a fresh `_result()`; the covariance case used `_pose('p1', 'm')`. The probes did not alter product source, captured inputs or the committed tests. All six malformed construction cases listed above were accepted; nested NaN mutation serialized to null.

## Next packet: P03A

**TASK:** Close the four findings with focused semantic validation and immutable record handling. Do not add estimator, ingestion or scoring functionality.

**FILES TO TOUCH:** P03 contract modules, including the shared `base.py`; generated `schemas/{internal-v0,capture-v0}.schema.json` if required by the representation; `tests/contract/test_records.py`; new P03A evidence. Coordinate helpers/tests may change only if a necessary contract representation adaptation requires it. Keep numeric gates, raw captures, previous evidence, model runtime and public CLI behavior unchanged.

**TESTS:** Reproduce every accepted malformed construction above, nested list/dictionary mutations and nonfinite serialization. Include positive cases for partial coverage, complete no-damage inspection, unavailable optional measurements, valid measurement bindings, nonzero singular PSD covariance, and ordinary JSON round trips. Keep existing frame tests and full regressions passing. Regenerate schemas and verify snapshots; explain representation changes and whether serialized v0 shapes remain compatible.

**ACCEPTANCE:** No empty false-success result; internally resolvable references are unique, valid and bound to the correct entity/quantity; invalid covariance is rejected without fabricated certainty; validated records cannot silently become invalid through normal nested mutation or serialization. Existing working geometry and explicit partial/unavailable behavior remain intact.

Prompt for Opus:

> Read docs/reviews/P03-review.md and implement P03A only. Reproduce and fix its four findings: truthful coverage/completeness, complete internally resolvable reference validation with subject/quantity/ownership checks and duplicate rejection before indexing, positive-semidefinite covariance validation, and immutable nested records with safe serialization. Keep partial results and complete negative inspection findings valid. Use meaningful regression and positive tests, regenerate schema snapshots as needed, run P03 tests and the full suite, and document any v0 representation compatibility effects. Preserve gates, captures, model/runtime evidence and frame mathematics. Return the diff, exact check results and remaining limits. Do not start P04 or reconstruction; stop for review after P03A.
