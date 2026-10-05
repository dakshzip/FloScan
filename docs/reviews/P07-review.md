# P07 review

Reviewed commit `89295a3` against P06A commit `029efa8`. Disposition: **changes required**. Resolve P07A before starting P08. No production code was changed during this review.

## Findings

### 1. [P2] The final unweighted fit undoes robust outlier suppression

Location: `src/floscan/geometry/planes.py:146-164`.

After Tukey IRLS converges, the function keeps every point whose absolute residual is below the cutoff, then replaces the robust result with an unweighted least-squares plane on that support. Points just inside the cutoff had almost no influence during IRLS but regain full weight here. Consequently the returned plane and covariance do not reflect the advertised robust estimate.

Independent reproduction: 1,000 exact points on z=0 plus 200 overlapping points at z=0.029 m, starting from the exact main plane with the production 0.03 m cutoff. The returned offset is -0.00481225 m, about 4.8 mm away from the main plane, and all 1,200 points become support. This is a biased fit to a clean dominant surface, not a benchmark accuracy claim. The existing uniformly distributed outlier test does not exercise this near-cutoff contamination.

Retain the robust estimate and its weighting when finalizing support, residual statistics and covariance, or explicitly define a robust final support rule that does not reintroduce suppressed returns. Keep support indices consistent with the returned plane. Add a near-cutoff contamination regression and retain the analytic noise-free and covariance tests.

### 2. [P2] An empty filtered cloud fails because the visualization is mandatory

Location: `src/floscan/reconstruction/lidar.py:601-610,628-638`.

A verified capture can legitimately yield zero retained points after configurable range/confidence filters. The stage still asks Open3D to write an empty PLY; Open3D returns false and `_write_preview` raises `OSError`. No `bundle.json` is written, so the filter counts and zero-evidence explanation are lost. Through the pipeline this becomes a stage failure rather than an explicit insufficient-evidence result.

Independent reproduction: the supplied synthetic room passes capture inspection, but `LidarConfig(max_depth_m=0.16)` filters all points and reconstruction raises `OSError: cannot write .../preview.ply`. This is a supported filter configuration with max depth greater than the minimum depth. Geometry can also be empty under default filters when a capture has only distant or rejected returns.

Preserve an auditable empty/partial bundle and filter diagnostics, mark reconstruction insufficient when no geometry survives, and omit or explicitly mark an unavailable preview. A visualization failure must not decide whether measured evidence exists. Add API and live-pipeline coverage; distinguish zero points from a nonempty cloud with zero discovered planes.

### 3. [P2] Accepted overlap settings defeat the submap memory bound

Location: `src/floscan/reconstruction/lidar.py:77-107,188-198`.

`LidarConfig` validates individual values but permits overlap greater than or equal to `submap_max_keyframes`. On reaching the configured limit, the planner retains the whole current submap when the overlap is larger, then appends more frames. Submaps exceed the declared cap, and a sufficiently large accepted overlap retains the growing capture rather than bounding it. Reconstruction materializes each planned submap's raw point arrays, so this also defeats the packet's streaming memory constraint.

The independent probe uses a verified synthetic capture, `submap_max_keyframes=2`, `submap_overlap_keyframes=60` and a large travel limit. It produces a submap with 23 keyframes despite the cap of two. Even an overlap of three already produces four-keyframe submaps despite the cap of two.

Require overlap strictly below the keyframe cap or implement an equivalent bounded planning rule. Validate other directly coupled filter bounds, such as min depth below max depth, with explicit configuration errors. Add tests that every planned submap obeys its keyframe cap across accepted overlap settings and tracking segments.

## Verification and working behavior

- Focused P07 and P06 suite: **80 passed in 86.52 seconds** with the existing uv cache accessible. The initial sandbox-only run failed three `run.sh` tests solely because the sandbox denied access to the uv cache; those failures are environmental and are not findings against P07.
- Full regression suite with uv cache access: **423 passed in 344.33 seconds**. The earlier sandbox-limited full run was stopped after cache-access errors; it is not counted as validation. These passing tests do not cover the three independently reproduced findings above.
- P06 review probes now behave correctly: corrupt confidence is retained as a frame issue; empty and non-finite IMU are diagnosed without crashing or emitting invalid JSON; zero start offset raises an explicit configuration error. The three original P06 findings are closed.
- Independent live run of the 1,715-row sample writes a validated incomplete envelope with exit 4 and both implemented stages marked ok. It produces 180 keyframes in six submaps, 1,950,084 retained observations and 60 plane candidates. About 96.2 percent of vertical-plane support agrees with a perpendicular direction family. No rooms or measurements are claimed; gates remain unverified or source-unspecified.
- Frame/pixel observation links, raw support indices, explicit metric units, unknown plane roles, optimistic covariance labels, missing-depth handling, pose-jump submap splitting and deterministic seeded RANSAC have meaningful regression coverage.
- Inspected the supplied top-view point preview: the geometry is plausible for development, with substantial occlusion and disconnected visible boundaries. This is visual plausibility evidence, not dimensional validation or a finished floor plan.
- Ruff lint and formatting pass; `git diff --check 029efa8..HEAD` passes. Review probes also pass Ruff. Reproduction code and output: `docs/reviews/p07-evidence/repro.py` and `repro.log`; all input mutations are confined to temporary synthetic sessions.

Independent live artifacts are under `/tmp/floscan-p07-review-live-89295a3/`. The command was `env -u PYTHONPATH uv run --locked floscan run --input 'example input /c00a170fe1' --tier lidar --output /tmp/floscan-p07-review-live-89295a3 --mode live`.

The pipeline/CLI wiring is justified by P07's required live command. The capture-convention audit changes broaden P07 beyond its importer exclusion and should remain explicitly documented, with their regression evidence; changing candidate selection is not proof of physical scale or gravity sign. The two distinct walkthroughs remain separate sessions. Pose-jump segments must not be treated as proven cross-segment registration during later room or property assembly.

## Bounded P07A prompt for Opus

> Read docs/reviews/P07-review.md and implement P07A only. First run docs/reviews/p07-evidence/repro.py to reproduce the biased near-cutoff plane refit, empty-cloud preview failure and overlap/keyframe cap violation. Preserve the robust final estimate and consistent support/statistics/covariance instead of replacing it with an unweighted fit that restores suppressed points. Preserve an explicit insufficient-evidence bundle with filter counts when no points survive, and make the preview optional in that case; ensure the live pipeline reports the actual evidence state. Validate coupled submap/filter settings so every accepted plan respects the keyframe bound. Limit changes to src/floscan/geometry/planes.py, src/floscan/reconstruction/{base,lidar}.py and tests/synthetic/test_lidar_planes.py, with pipeline changes only if required for the insufficient-evidence status. Add meaningful regression cases, rerun the independent probes, focused P06/P07 tests, full regressions and Ruff. Preserve raw captures, unknown plane semantics and incomplete output claims. Do not implement P08 rooms or surfaces. Return the repair evidence and stop for review.
