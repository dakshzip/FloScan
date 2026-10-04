# ADR 001: Frozen requirements, project-owned schema and diagnostic contract

- Status: accepted for P01
- Date: 2026-10-04
- Packet: P01, "Freeze requirements and a runnable diagnostic contract"

## Context

FloScan answers the Applied AI Engineer case study.
The original assignment outranks the derived PRD.
The approved plan lives in `docs/implementation-strategy/`.

| Source | File | SHA-256 | Pages | Authority |
|---|---|---|---|---|
| `case_study` | `Applied_AI_Case_Study.pdf` | `26a6392f3171ff0303c0ba7bec3e19cc9ba1b7db982aac0f52622d76d396f769` | 6 | primary |
| `prd` | `applied_ai_case_study_engineering_prd.pdf` | `4b6bfa0aac073455406d152a0636769251392a7db74d785365be4ffcad95895d` | 10 | derived |

Page references (`case_study:p3`) are 1-based PDF page indexes, not printed footers.
On the case study, PDF page 3 carries printed footer "2".

### Unavailable sources

1. **Published JSON schema.** The assignment requires "JSON to the published schema", and the PRD says it is "supplied with the assignment". Neither PDF contains it, and the user confirms it is unavailable.
2. **Round 1 specification and gates.** The assignment says "Round 1 gates apply, with five additions". Only the five additions are in the PDFs, and the user confirms the Round 1 document is unavailable.

These gaps block a final external-schema compliance claim and any verdict on Round 1-only gates.
They do not block most engineering work, so the project does not wait for them.

### Deadline accounting

The user's update said approximately 40 hours remained.
Its exact timestamp is not recorded.
The earliest workspace artifact, `Applied_AI_Case_Study.pdf`, was written at 10:53 IST on 4 October 2026, so the conservative deadline is **about 02:53 IST on 6 October 2026**.
P01 started at 13:41 IST on 4 October, leaving about 37 hours.
The roadmap's stop-loss points keep their distance from the deadline: freeze the baseline about 14 hours before it (about 12:50 IST, 5 October), protect at least 4 hours for the shipped fix and at least 7 hours for packaging, rehearsal and the report.

### Hardware (as reported, not yet measured)

- RTX 3080 workstation: VRAM variant, OS, driver and CPU unconfirmed; P02 records them.
- 2023 MacBook Pro M2 Pro, base configuration: this P01 build ran here on macOS with a uv-managed CPython 3.11.15.
- Evaluator machine: unknown; no runtime promise applies to it.

## Decision

### 1. Requirement IDs

`configs/gates.yaml` is the machine-readable registry.
G01-G16 are the requirement IDs of the approved plan (`docs/implementation-strategy/README.md`).
G17-G19 are added here to close coverage gaps: the plan's table had no explicit row for the per-capture output contract, the capture route, or the hard-conditions and weights constraints.
"Status at P01" uses the compliance vocabulary of 04; nothing is measured yet.

| ID | Requirement | Sources | Gates | Status at P01 |
|---|---|---|---|---|
| G01 | Photo tier: 2 to 8 stills per room, no depth or poses, same whole-property product | case_study:p2; prd:p2-3 | photo_stitch_single_plan, photo_wall_length | not_started |
| G02 | Video tier: ordinary handheld iPhone 15+ clip | case_study:p2; prd:p3 | video_wall_length | not_started |
| G03 | LiDAR tier: depth, poses, intrinsics on Pro-class devices | case_study:p2; prd:p3 | lidar_wall_length | not_started |
| G04 | Opening width, missed and phantom openings count as misses | case_study:p3; prd:p4 | opening_width | not_started |
| G05 | Ceiling height per room and repeat spread | case_study:p3; prd:p4 | ceiling_height_error, ceiling_height_repeat_spread | not_started |
| G06 | Same-room same-tier wall repeatability | case_study:p3; prd:p4 | wall_repeatability | not_started |
| G07 | Drift accountability with ON/OFF ablation | case_study:p3; prd:p4 | drift_accountability | not_started |
| G08 | Photo whole-property stitch: one plan, adjacency, no overlap, footprint, walls | case_study:p3; prd:p4 | photo_stitch_*, photo_footprint, photo_wall_length | not_started |
| G09 | Interval on every measurement; calibration scored per tier | case_study:p2-3; prd:p2, p4 | confidence_calibration | not_started |
| G10 | Damage regions, concealed flags with fired rule, surface-keyed scope | case_study:p2; prd:p4-5 | none numeric | not_started |
| G11 | Benchmark composition with laser or tape ground truth and raw data | case_study:p3; prd:p5 | none numeric | not_started |
| G12 | LiDAR head-to-head against a named consumer app on two rooms | case_study:p4; prd:p5 | incumbent_head_to_head | not_started |
| G13 | Fix loop with declared prediction, shipped fix, regenerable before and after | case_study:p4; prd:p5-6 | none numeric | not_started |
| G14 | Fresh machine to result under 15 minutes; one command per capture; custom app install under 10 minutes | case_study:p2, p5; prd:p6 | fresh_machine_to_result, custom_app_install | not_started (CLI entry point exists, produces no result) |
| G15 | No own infrastructure; live cold walk-in at any tier; deterministic replay allowed | case_study:p5-6; prd:p7 | none numeric | not_started (live/replay distinction exists in CLI only) |
| G16 | Compliance matrix, device matrix, raw data, benchmark and technical report (max 6 pages), commit history | case_study:p4-5; prd:p6-7 | technical_report_pages | not_started |
| G17 | Full output contract per capture at every tier; Round 1 gates | case_study:p2-3; prd:p1, p3 | stitched_plan_adjacency, floor_area, lidar_wall_length, round1_gates | not_started |
| G18 | Capture route followed literally, plus device matrix | case_study:p2, p5; prd:p2-3 | custom_app_install | not_started |
| G19 | Mirrors, glass, wet-look, low light covered; weights by script or volume; disclosed models | case_study:p6; prd:p9 | none numeric | not_started |

### 2. Gate registry rules

- Every numeric threshold carries `source_value`, a verbatim fragment of its quoted `source_text`. The loader converts it to SI and rejects any value that differs, so a number cannot enter the registry without a source quotation.
- Thresholds are stored in SI units: metres, ratios, seconds, pages.
- Thresholds missing from the sources are `unspecified_source` with `value: null`: the LiDAR wall-length tolerance, the floor-area tolerance and the whole Round 1 gate list.
- Qualitative requirements (drift accountability, adjacency, no overlap, calibration) are `specified_qualitative` and carry no invented numbers.
- The registry has no field for a pass or fail verdict, and the loader rejects unknown keys. Verdicts come only from scored benchmark runs (P04 onward).

### 3. Provisional interpretations (frozen before any scoring)

These follow `docs/implementation-strategy/04-benchmarks.md` and stay labelled provisional until an evaluator clarifies them.

| Gate | Ambiguity | Provisional policy |
|---|---|---|
| opening_width | Denominator | `G / (N_gt + FP)`; misses stay in the denominator, phantoms add to it |
| opening_width, ceiling_height_error | Tier exemptions | None stated, so applied to every tier |
| ceiling_height_error | Per room or mean | Every room in every capture |
| ceiling_height_repeat_spread | Cross-tier captures | Same-tier independent captures only |
| wall_repeatability | "1 cm or 0.5%" | OR reading is primary; AND reading always published; relative to ground truth |
| photo_wall_length, video_wall_length | Tolerated failing fraction | Every expected wall must pass; fractions and quantiles also published |
| photo_footprint | Footprint definition | Area of the union of interior room and connector polygons; extents, IoU and boundary distance also published |
| photo_stitch_no_overlap | Numerical tolerance | Not in sources; P04 freezes a numerical-geometry tolerance before scoring, never a centimetre buffer |
| drift_accountability | Photo-tier applicability | Open question; registered for all tiers |
| custom_app_install | Route choice | Applies only if Route 1 is chosen; stock Route 2 is planned, P05 confirms |

### 4. Project-owned schema and exporter boundary

- Result JSON is `floscan-result/0.1.0-diagnostic`, owned by this project.
- Every envelope states `external_schema_status: unavailable` and `external_conformance_claimed: false`, and the `public_schema_export` section is `blocked_external`. The validator rejects any other value.
- If an evaluator schema is supplied later, it goes read-only under `schemas/public/` and an exporter maps the internal result to it. Unmappable required fields fail the export explicitly instead of being zero-filled.

### 5. Diagnostic result envelope

`floscan run` writes `<output>/result.json` containing:

- `schema_version`, `schema_authority`, `result_kind: diagnostic_envelope`
- `run`: run ID, UTC time, version, tier, mode, argv, Python, platform, input inventory (path, existence, file count, bytes), output directory
- `status`, `status_reason`, `exit_code`
- `coverage`: `contract_complete` plus one status and reason per contract section (`capture`, `coordinate_frames`, `scale`, `per_room_plan`, `stitched_plan`, `measurements`, `damage_regions`, `concealed_damage_flags`, `scope_items`, `rendered_plan`, `public_schema_export`)
- The 02 payload fields, empty or null in P01: `capture`, `coordinate_frames`, `scale`, `rooms`, `surfaces`, `walls`, `openings`, `property_graph`, `measurements`, `damage_regions`, `concealed_damage_flags`, `scope_items`, `artifacts`
- `stages`: `input.inventory`, optionally `replay.lookup`, then the 14 interface stages of 01
- `gates`: registry path, SHA-256 and version; per-gate status; status counts
- `diagnostics`

Stage statuses extend the failure vocabulary of 01 with `not_implemented` (no code exists in this build) and `skipped` (not attempted because an earlier stage stopped the run).
Neither is ever treated as `ok`.

The validator enforces these honesty rules:

- No NaN or Infinity anywhere.
- Every unavailable section has a reason and an empty or null payload.
- Entity payloads (rooms, measurements and the rest) are rejected outright until P03 supplies record validators, so no fake room can pass.
- `status: ok` requires every section available and every stage `ok`.
- `exit_code` must agree with `status`.
- Gate items may only be `unverified` or `unspecified_source`; `measured_pass` and `measured_fail` are rejected because verdicts come from benchmark scoring, never from `floscan run`.

### 6. Live and replay

`--mode` is mandatory.
Live processes the raw input and never reads a cache.
Replay needs a cache entry keyed by input, config, model, code, tier and schema hashes, and never falls back to live inference.
No cache exists in P01, so replay always exits `5` with `replay.lookup: unsupported`.

### 7. Exit codes

| Exit | Status | Envelope written |
|---|---|---|
| 0 | `ok` | yes (unreachable in P01) |
| 1 | `failed`, invalid registry or invalid envelope | no |
| 2 | usage error, output inside input, existing result | no |
| 3 | `invalid_input` | yes |
| 4 | `unsupported`, `insufficient_evidence` or `ambiguous` | yes |
| 5 | replay cache unavailable | yes |

### 8. Input preservation and append-only output

- The CLI refuses an output directory equal to or inside the input directory, so raw inputs cannot be modified.
- `result.json` is created exclusively and never overwritten.
- The sample directory `example input ` (trailing space) is read only, ignored by Git, and handled through quoted paths.

### 9. Interim dependency record (moves to DEPENDENCIES.md in P02)

`DEPENDENCIES.md` belongs to P02's allowlist, so P01 records its dependencies here.
Exact resolved versions are in `uv.lock`.

| Name | Use | License | Source | Offline behaviour |
|---|---|---|---|---|
| PyYAML (>=6.0.2,<7) | Read `configs/gates.yaml` (`safe_load` only) | MIT | PyPI | Fully offline after install |
| pytest (dev) | Contract tests | MIT | PyPI | Fully offline after install |
| ruff (dev) | Lint and format checks | MIT | PyPI | Fully offline after install |
| uv_build (build backend) | Builds the `floscan` package | MIT or Apache-2.0 | PyPI | Needed only at install time |

No models, weights, datasets or network APIs are used.

### 10. Deviations from the P01 file allowlist

These were needed to make the packet runnable and safe:

- `src/floscan/__init__.py`: package marker required by the build backend; holds only the version lookup.
- `.gitignore`: keeps `example input /` (about 670 MB of raw captures) and `tmp/` out of Git (rule 13).
- `uv.lock`: written automatically by `uv run`. It is committed so P01's test environment is reproducible. P02 owns auditing and freezing it.
- `runs/p01-evidence/`: the packet's own run artifacts, which the packet rules permit.

## Consequences

- One command reaches a validated diagnostic output today, and that output says honestly that the contract is incomplete.
- Later packets fill sections and replace the P01 entity-payload ban with P03 record validators. They must not loosen the honesty rules.
- Gate thresholds cannot drift silently: changing a number requires changing its quoted source text, which review will catch.
- Open questions stay visible in the registry and in `floscan gates` until an evaluator answers them.
