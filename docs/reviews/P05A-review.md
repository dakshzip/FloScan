# P05A review

Reviewed commit `82b6d3e` against P05 commit `966cd5d`. Disposition: **accept**. This supersedes the four changes-required findings in `P05-review.md`.

## Verified repairs

- Focused P05 tests: **64 passed in 0.28 seconds**.
- Full regression suite: **342 passed in 124.94 seconds**, without an import workaround.
- Independent status command confirms development remains **0/8 met**, with calibration, held-out and walk-in cases also incomplete.
- Independent replay of `runs/p05a-evidence/repro.py`: **0/7 malformed cases accepted**. Incomplete photo/video room coverage, duplicate sketch IDs, self-repeat, renamed video copy, transcription-only incumbent output and unknown incumbent rooms are all rejected as declared-complete evidence.
- Unique physical room IDs include connectors. Explicit photo groups map folder names to those IDs; group counts agree with unique media digests. Partial received coverage remains representable with an unmet checklist entry.
- Repeat links target a distinct primary recording of the same tier, with shared room coverage. Raw tree provenance is separate from photo/video content identity, and duplicated content cannot become a repeat by renaming media files.
- Incumbent transcriptions remain supporting evidence and cannot satisfy the original-export requirement. Export room IDs must exist in the sketch.
- The GT instructions remove the fixed uncertainty default, preserve unknown uncertainty with a basis/reason, and label examples synthetic.
- Ruff lint and format checks pass; 46 files formatted. `git diff --check HEAD^..HEAD` passes.

## Limits and compatibility

`photo_counts` is replaced by explicit `photo_groups`; received captures/fixtures now need `media_sha256`. This is an intentional metadata migration: the supplied real manifests remain incomplete, and existing development fixtures received the new recording identity fields. Do not represent hashes, counts or operator statements as proof of surveyed geometry or a newly captured sensor session. Physical collection, content verification and GT derivation remain separate work.

The two distinct development walkthrough sessions are complementary floor/ceiling views, with separate session coordinates until registered. The byte-identical copied folder remains deduplicated development input. No real benchmark gate or collection-completeness claim is made.

## When the user's own inputs are needed

- P06-P09 can use the existing RGB-D samples for ingestion, points, planes and local geometry development.
- P10 should use the user's photos and short native RGB video as soon as available, to test the real sparse-view risks. RGB-only subsets of existing samples can support development smoke but do not replace a strict-tier benchmark capture.
- P11/P13 benefit from the user's LiDAR capture and surveyed openings/heights/property layout for real refinement and drift validation.
- P14-P16 implement the photo route and whole-property photo plan; use photo folders for at least three rooms plus a connector.
- P17-P18 implement the native video route and measured property output; use the independent ordinary RGB walkthrough, without LiDAR/pose sidecars.
- P19 needs the furnished room with the two staged damage classes visible in its RGB captures.
- P23 needs property-disjoint calibration captures and GT for defensible calibration. If unavailable, calibration remains provisional.
- P24's real all-tier baseline requires the user's independent photo, video and LiDAR captures, repeat evidence and measured GT. P25 additionally requires the incumbent's original two-room export. P30 needs a separate fresh walk-in capture set.

Start collection now using `docs/capture_protocol.md` and `benchmark/ground_truth/protocol.md`; coding P06 does not need to wait for it. The manifests should remain incomplete until the evidence is actually supplied and registered.

## Next packet: P06

Prompt for Opus after acceptance:

> P05/P05A are accepted. Read docs/reviews/P05A-review.md and implement P06 only from docs/implementation-strategy/06-task-packets.md. Use the existing sample sessions under the quoted trailing-space path "example input /". Read the capture/coordinate contracts and source convention requirements first. Implement Stray-style normalization, timestamps/synchronization auditing, content-addressed assets and explicit source profiles. Preserve every original file. Treat distinct floor/ceiling sessions as separate worlds and the copied folder as duplicate evidence; do not register or fuse sessions in P06. Do not assume undocumented IMU units, camera basis, depth kind or confidence semantics. Carry unknown source conventions and tracking resets as explicit diagnostics. Run meaningful synthetic and real-sample ingestion tests and existing regressions, then return source assumptions, data coverage, synchronization diagnostics, test evidence and next packet P07. Do not implement point-cloud reconstruction, planes, registration or benchmark inference. Stop for review after P06.
