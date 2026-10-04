# Ground-truth records

Ground truth is evaluator-only data.
It lives under `benchmark/ground_truth/records/<property_id>/`, is read only by `benchmark/evaluator`, and is never imported or read by inference code (`tests/unit/test_benchmark.py` checks that `src/` never imports `benchmark`).
Nothing here is ever estimated or filled in: every value is a tape reading transcribed from the signed measurement sheet.

## Layer 1: `measurements.json` (raw readings, transcribed from the sheet)

The example below is a synthetic illustration of the format; none of its numbers is a default or a value to copy into a real record.

```json
{
  "property_id": "P1",
  "instrument": {"id": "tape-1", "description": "brand and model", "length_m": 5.0,
                 "accuracy_class": "EC class II", "resolution_m": 0.001},
  "operator": "name",
  "sheet_files": ["data/raw/P1/sheet/sheet.jpg"],
  "readings": [
    {"id": "R01-W1", "quantity": "wall_length", "values_m": [3.412, 3.414],
     "endpoint_definition": "inside corner to inside corner, finished surface, 1.0 m above floor",
     "measured_at_height_m": 1.0, "instrument_id": "tape-1",
     "timestamp": "2026-10-05T10:12:00+05:30",
     "uncertainty_m": null,
     "uncertainty_basis": "unknown: tape tolerance at this length not yet looked up",
     "notes": ""}
  ]
}
```

- `id` is the stable physical identifier from the sketch: rooms `R01`, walls `R01-W1` (clockwise from the entrance-door wall), doors `R01-D1`, windows `R01-N1`, damage `R02-X1`.
- `quantity` is one of `wall_length`, `ceiling_height`, `diagonal`, `opening_width`, `opening_height`, `sill_height`, `opening_offset`, `wall_thickness`, `damage_length`, `damage_width`, `damage_position_along`, `damage_position_height`.
- `values_m` holds every reading in metres; the central value is their mean. Two readings are required; a spread above 5 mm requires a third.
- `uncertainty_m` has no default.
It is filled only when it can be supported: the tolerance of the instrument's stated accuracy class at the measured length, plus a placement allowance the operator states for that reading (for example a jamb face that is hard to reach).
`uncertainty_basis` says how the number was obtained.
When it cannot be supported, `uncertainty_m` is `null` and `uncertainty_basis` starts with `unknown:` and gives the reason.
The spread of repeated readings shows repeatability only; it does not establish accuracy and is not copied into `uncertainty_m`.
- Uncertainty is reported beside results as a sensitivity band only; gates use the central value, and an error is never reduced by the instrument uncertainty.
- Items that could not be measured are listed with `"values_m": []` and a reason in `notes`; they are never filled in.

## Layer 2: `ground_truth.json` (scoring view, derived)

The scorer consumes a `PlanView` with `"kind": "ground_truth"` (see `benchmark/evaluator/matching.py`): rooms with polygons (and holes), walls with endpoints and lengths, openings with centres and widths, ceiling heights, floor areas and the traversable adjacency between rooms, all in one surveyed frame in metres.

Scalars (wall lengths, opening widths, ceiling heights) are copied from Layer 1 means without change.
Coordinates are derived from Layer 1 by a documented construction: each room polygon from its wall lengths and diagonals, rooms placed relative to each other through shared openings (offsets along both walls and the wall thickness).
That derivation is not implemented yet; until it exists, a property's `ground_truth.json` is reported as missing in the manifest status, and no coordinates are drawn by hand or taken from any reconstruction.

## Damage and concealed damage

Each staged damage item has an ID, a class (`stain` or `crack`), a host surface, its centre position and its extent.
The damage room's undamaged walls are listed as negative surfaces.
Ground truth cannot say whether hidden damage exists; concealed-damage flags are scored only for traceability, not for truth.

## Incumbent export

`benchmark/ground_truth/records/<property_id>/incumbent.json` follows `IncumbentExport` in `benchmark/evaluator/metrics.py`: app name and version, the original export file, the two declared room IDs, and one entry per dimension the app reports (`null` where it reports none).
Its dimension list is written from the app's export before any FloScan error is looked at.
