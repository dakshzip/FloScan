# Benchmark score: case-1 (photo)

Scorer scorer-v1, matching matching-v1, gate registry sha256 a30fc29d28cd.
Thresholds come from configs/gates.yaml; provisional interpretations are listed there.

## Gates

| Gate | Status | Reason |
|---|---|---|
| opening_width | measured_pass | G / (N + FP) in every capture (provisional denominator) |
| ceiling_height_error | measured_fail | every room in every capture; missing heights fail |
| ceiling_height_repeat_spread | unverified | fewer than two captures of the same tier |
| wall_repeatability | unverified | fewer than two captures of the same tier |
| drift_accountability | unverified | needs the drift-correction ON/OFF ablation (P13) |
| photo_stitch_single_plan | measured_fail | a disconnected or single-room output fails |
| photo_stitch_adjacency | measured_fail | missed, extra or unplaced edges fail |
| photo_stitch_no_overlap | measured_pass | no pairwise interior overlap beyond numerical tolerance |
| photo_footprint | measured_fail | area of the union of rooms (provisional definition); extents and IoU reported alongside |
| photo_wall_length | measured_fail | every expected wall in every capture; missing walls fail |
| floor_area | unspecified_source | no threshold in the supplied sources; reported without a verdict |
| round1_gates | unspecified_source | no threshold in the supplied sources; reported without a verdict |
| confidence_calibration | unverified | no numeric calibration band is supplied; empirical coverage is reported |
| fresh_machine_to_result | unverified | not a benchmark measurement (deliverable gate) |
| custom_app_install | unverified | not a benchmark measurement (deliverable gate) |
| technical_report_pages | unverified | not a benchmark measurement (deliverable gate) |

Not applicable to the photo tier: video_wall_length, lidar_wall_length, stitched_plan_adjacency, incumbent_head_to_head.

## Capture cap-a

| Quantity | Expected | Matched | Missing | Unmeasured | Phantom | MAE | Max rel |
|---|---|---|---|---|---|---|---|
| wall_length | 12 | 8 | 4 | 0 | 0 | 0 | 0 |
| opening_width | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| ceiling_height | 3 | 2 | 1 | 0 | 0 | 0 | 0 |
| floor_area | 3 | 2 | 1 | 0 | 0 | 0 | 0 |

Openings: N=2 TP=2 FN=0 FP=0 G=2 score=1.
Footprint: area error 0.1765, IoU 0.8235, Hausdorff 1.5 m.

