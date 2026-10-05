# P08 review

Reviewed commit `f5778b4` against P07A commit `80e4bd7`. Disposition: **changes required**. Resolve P08A before implementing P09 measurements. No production code was changed during review.

## Findings

### 1. [P1] Global ceiling evidence creates unsupported ceilings and wall heights in other rooms

Location: `src/floscan/geometry/rooms.py:667-695` and the global level selection in `build_rooms`.

The chosen ceiling is shared by every room. `_records` creates a ceiling over each entire room outline using the same global plane IDs and assigns floor-to-ceiling wall height profiles from that level, without checking whether the ceiling's support overlaps that room. This can manufacture an observed ceiling and height for a room whose ceiling was never captured. Different room ceiling levels also collapse to the single highest level.

Independent reproduction: two disconnected, fully walled 4x3 m synthetic rooms with floors at z=0, but ceiling support only over the first room. Both rooms receive 12 m2 ceiling surfaces at z=2.5 with `observation_ids=['ceiling-a']`, including the second room six metres away. Both rooms and the model report `ok`. The unsupported ceiling also supplies the second room's wall height profiles.

Assign horizontal evidence spatially per room and retain each room's observed coverage and level. A room without ceiling support must have no inferred ceiling or floor-to-ceiling height. Preserve observed wall extents without presenting them as full heights. Propagate missing/partial evidence consistently into room/surface statuses and reasons: the independent no-ceiling rectangle currently has model status `partial` but room status `ok` with no reason. Add negative cases for an unseen adjacent-room ceiling and different observed ceiling heights.

### 2. [P2] Accepted tilted planes are silently flattened into different surface geometry

Location: `src/floscan/geometry/rooms.py:179-207,258-265` and `src/floscan/geometry/surfaces.py:24-75`.

Horizontal candidates may tilt by up to ten degrees, but the selected level retains only a scalar median height. Floor and ceiling surfaces are then constructed with exact vertical normals, and wall intersections use z=floor.height. The accepted reconstructed plane's normal and offset are discarded. This is a geometric change, not a harmless chart conversion, and can bias intersections, heights and downstream areas.

Independent reproduction: shear the analytic rectangle scene so floor z=0.05*x and ceiling z=2.5+0.05*x, about 2.86 degrees from horizontal. The scene remains inside the accepted orientation threshold and reports `ok`. Its floor surface becomes z=0.1 with normal [0,0,1], reaching 0.1 m error at either side of the four-metre room.

Preserve the observed plane and construct metric charts/intersections on it. If tilted floors/ceilings are not supported yet, diagnose them and abstain from producing a falsely exact horizontal surface. Do not change the world axes or sensor poses to conceal the error. Add an analytic tilted-plane case checking the exported 3D surface and wall intersections, not just its xy footprint.

### 3. [P2] A tabletop becomes a floor when the actual floor is unseen

Location: `src/floscan/geometry/rooms.py:210-227` and floor surface construction in `_records`.

The first sufficiently large up-facing level below the median camera is always labeled floor. A table meets all those conditions when there is no observed lower floor. The existing table negative includes a real floor and only checks the easy ordering case; it does not establish that the sole horizontal surface is floor.

Independent reproduction: only a 2.5x1.5 m tabletop at z=0.75 and a camera at z=1.4. P08 chooses `plane_ids=['table']` as floor, forms a partial room from its footprint, and emits a floor surface with status `ok`. The model's unknown boundary warning does not retract the incorrect floor role.

Require independent structural support for the floor role, or preserve the level as an ambiguous candidate and report insufficient evidence. Lowest visible surface and camera height alone cannot distinguish an unseen floor from furniture. Do not solve this with assumed furniture heights. Add tabletop-only and ambiguous floor/furniture scenes alongside the existing real-floor-plus-table test.

### 4. [P2] Loading a bundle erases tracking-segment boundaries before room assembly

Location: `src/floscan/geometry/rooms.py:128-155` and cross-submap merging at lines 298-319.

P07 splits captures at possible tracking resets and lists continuous tracking segments in its bundle. `load_bundle` concatenates every submap's planes and cameras into a single `SceneEvidence`, discarding the segment membership. Level grouping, wall merging and floor polygonization then operate over the combined scene. `RoomModel` records everything in one W frame and has no diagnostic preserving the lost component boundary. `placement_status='unplaced'` concerns the property frame; it does not restore the tracking-component separation.

Confirmed on the supplied 1a8384c3f6 bundle: it contains three segments (keyframes 0-5190, 5199, and 5200-5220), but the resulting SceneEvidence retains only planes, cameras, source and hashes. The published P08 output uses that bundle without any tracking-component metadata. There is no registration evidence proving the reset segments can participate in the same local room.

Preserve tracking-component membership and process uncertain components independently, or explicitly refuse/omit their uncertain assembly with a diagnostic. Cross-component merging requires registration evidence; numerical coordinate agreement alone is insufficient. Add a reset case with incompatible segment geometry and verify it is not combined into one accepted room. Keep any intentional omissions visible.

## Verification and repairs closed

- Focused P08/P07 suite: **45 passed in 38.48 seconds**.
- Full regression suite: **447 passed in 210.18 seconds**. These tests do not cover the independently reproduced findings above.
- Independently reran `runs/p07a-evidence/repro.py`: the exact-plane contamination bias drops from 4.8 mm to 0.026 mm and support excludes the 200 near-cutoff returns; invalid overlaps are refused and an accepted two-keyframe cap is respected; zero-point reconstruction writes `no_points` with an unavailable preview. The three original P07 findings are closed.
- Ruff lint/format checks and `git diff --check 80e4bd7..HEAD` pass. The review reproduction script also passes Ruff.
- Rectangle, concave L-room, diagonal wall, pillar hole, absent wall, doorway gap, optional orthogonal snapping and surface orientation have meaningful analytic tests. P08 stays within its packet files and is an API stage; P09 is responsible for pipeline wiring.
- Independently built rooms from the corrected P07A sample bundle at `runs/p07a-evidence/run-c00a170fe1/reconstruction`: two partial rooms, observed floor coverage about 84 and 70 percent, no ceiling surfaces, and 14 surfaces total. Inspected the supplied top-view preview; red unknown edges remain visibly distinguished from observed walls. No benchmark or surveyed-layout accuracy claim is established by this preview or by the reported room count.
- Reproductions: `docs/reviews/p08-evidence/repro.py` and `repro.log`. Run from the project root with `env -u PYTHONPATH .venv/bin/python docs/reviews/p08-evidence/repro.py`. Synthetic dimensions are explicit fixtures; sample bundles are read only.

Real evidence generated in `runs/p08-evidence` used the earlier P07 bundles. The independent sample check above used P07A's corrected plane fits; regenerated evidence should identify the exact upstream bundle. Distinct walkthrough sessions remain separate worlds. Source world-up sign and physical accuracy remain unresolved rather than being certified by successful polygonization.

## Bounded P08A prompt for Opus

> Read docs/reviews/P08-review.md and implement P08A only. Reproduce the four findings with docs/reviews/p08-evidence/repro.py first. Associate ceiling support and height with each room spatially; do not copy another room's ceiling or manufacture floor-to-ceiling profiles, and keep partial statuses/reasons consistent. Preserve the selected floor/ceiling planes in metric surface charts and wall intersections, or explicitly abstain for unsupported tilt instead of flattening them. Treat a tabletop-only/ambiguous horizontal level as insufficient evidence for a floor role unless structural evidence resolves it; do not use standard heights. Preserve tracking-component membership from reconstruction and prevent unsupported assembly across reset segments, with visible diagnostics for excluded evidence. Stay within src/floscan/geometry/{rooms,surfaces,polygons}.py and tests/synthetic/test_rooms.py; change reconstruction metadata only if the necessary component identity cannot be recovered from its existing segment/keyframe metadata, and explain that dependency. Add meaningful negative/analytic cases, rerun probes, focused P07/P08 tests, full regressions and Ruff. Regenerate real evidence from the corrected P07A bundles. Do not implement P09 measurements or rendering yet. Return exact repair evidence and stop for review.
