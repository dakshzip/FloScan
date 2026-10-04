# P03A representation and compatibility notes

## Serialized shape

Validated records now hold `FrozenList` and `FrozenDict` containers instead of plain lists and dicts.
Both are `list`/`dict` subclasses, so `model_dump`, `model_dump_json` and the JSON Schemas still describe and emit ordinary arrays and objects.
A JSON document produced before P03A and one produced after are identical for every pre-existing field.

## Schema changes

`schemas/internal-v0.schema.json` gains one optional field, `SectionCoverage.evidence_ids` (array of IDs, default empty); this is the only diff (10 insertions, 1 deletion).
`schemas/capture-v0.schema.json` is byte-identical.
Old v0 documents remain loadable because the field has a default.
Documents written now always include `evidence_ids`, so a reader built before P03A (strict, `extra="forbid"`) would reject them; no such reader exists outside this repository.

## Semantic tightening (intended)

Documents that P03 accepted may now be rejected when they make false or unbacked claims:

- A coverage section marked `available` must be backed by records (capture id, frames, resolved scale, per-room walls with wall lengths plus ceiling height and floor area, one connected graph with every room placed, measured values with intervals, a plan artifact).
- A section marked `unavailable` or `blocked_external` must be empty; content with gaps is `partial`.
- `public_schema_export` can never be `available` (no evaluator schema, no exporter); `status: ok` needs every internal section available and carries the export as `blocked_external` or `unavailable` with a reason.
- A no-damage result is complete only with the inspected surfaces listed in `coverage.damage_regions.evidence_ids`; concealed-damage coverage needs recorded rule evaluations (`not_triggered` flags are negative findings).
- Record IDs must be unique across all record kinds, checked on the raw lists before indexing.
- Measurement references must exist and measure the right subject and quantity (for example an opening's width must be that opening's `opening_width`), and owned records must belong to the referencing room, wall or surface.
- Covariance and information matrices must be positive semidefinite within a relative eigenvalue tolerance of 1e-9 (singular PSD matrices stay valid; all-zero stays rejected); `Plane.parameter_covariance` is now validated too.

## Serialization safety

`Contract.to_json()` re-validates the whole record tree before writing.
`ser_json_inf_nan="constants"` makes `model_dump_json` emit `NaN` rather than `null` if a non-finite value ever reaches output through a validation bypass, so readers fail loudly instead of losing the value.

## External references

`EXTERNAL_REFERENCE_FIELDS` in `src/floscan/contracts/result.py` lists the ID fields that point outside a `PropertyResult` (observations, evidence, planes and masks in bundles, rules, calibrators, hypotheses, levels).
They are provenance links and are not resolved inside a result; every other reference is.
