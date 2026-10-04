# P04A scoring-view compatibility

## New fields (all optional, defaults keep pre-P04A views loadable)

| Field | Default | Meaning |
|---|---|---|
| `RoomView.holes` | `[]` | Interior rings; kept through rigid alignment, union, IoU, overlap and boundary sampling. A hole that makes the shape invalid is rejected, never filled. |
| `RoomView.placement` | `"placed"` | `placed`, `unplaced` or `ambiguous`. Non-placed rooms carry room-local coordinates: their walls and openings are matched in their own frame for local dimensions, and they never enter the property footprint, overlap or adjacency. |
| `PlanView.registration_status` | `null` | `connected`, `disconnected`, `ambiguous`, `not_attempted` or `failed`. |

## Behaviour changes (intended tightening)

- Whole-property gates (`photo_stitch_single_plan`, `photo_footprint`, `photo_stitch_adjacency`, `photo_stitch_no_overlap`) pass only for a stitched plan: `registration_status` explicitly `connected`, one component, and every ground-truth room matched to a placed room. A pre-P04A view without `registration_status` loads but cannot pass them.
- A view claiming `connected` with any non-placed room is rejected; ground truth must have every room placed.
- `plan_from_result` now carries holes, each room's `placement_status` and the graph's `registration_status` (`not_attempted` when there is no graph).
- Incumbent exports must name two distinct ground-truth rooms and list each `(kind, gt_id)` once; every dimension must belong to a declared room. A comparison without a shared dimension in both rooms is `unverified`; our missing outputs still count as losses and the 70 % threshold is unchanged.

## Unchanged

Gate thresholds in `configs/gates.yaml`, the matching distances, and every formula for placed, hole-free plans.
Earlier P04 evidence is preserved as written, including its report's trailing blank line, which the renderer no longer emits.
