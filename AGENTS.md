# AGENTS.md

Rules for any coding agent working in this repository.
Rules 1-25 are copied verbatim from `docs/implementation-strategy/07-agent-rules-review.md`, section I, which remains the source of truth.

## Project facts

- Requirements, unavailable sources and gate interpretations are frozen in `docs/adr/001-requirements.md` and `configs/gates.yaml`.
- The output schema is project-owned (`floscan-result`); the assignment's published schema and Round 1 gates are unavailable.
- Raw sample inputs live in `example input /`, a directory whose name ends with a space; always quote it and never modify, move or commit it.
- Commands: `./run.sh` (one capture), `./run_benchmark.sh` (one benchmark case), `uv run floscan --help`.
- Checks: `uv run pytest -q` and `uv run ruff check . && uv run ruff format --check .`.
- Work proceeds packet by packet (`docs/implementation-strategy/06-task-packets.md`); each packet returns the review envelope in 07 section J.
- Do not spawn other coding agents unless the user authorizes it.
- Never use the em dash character in project text; use a plain hyphen.

## Rules

1. Implement only an approved packet. Read the requirement matrix, contracts and relevant algorithm specification first. Do not treat this planning package as implementation approval.
2. The original assignment outranks the derived PRD. Keep source gaps visible: unavailable public JSON schema, missing earlier Round 1 gates, unspecified evaluator hardware and unresolved scoring interpretations. The user confirms the missing documents are unavailable; do not keep blocking routine work asking for them.
3. Freeze a clearly named project output schema and exporter boundary. Never change the supplied public schema, if later obtained, without approval. Never silently add breaking fields, change quantity definitions or weaken validation to make a result pass.
4. Use metres/radians/seconds internally. Every transform is named destination-from-source. No silent unit, axis, quaternion-order, depth-kind or timebase change. Up-to-scale geometry cannot enter metric measurement functions.
5. Keep modality-specific input permissions and reconstruction isolated. Strict photo cannot access depth/poses; strict video cannot access companion sensor logs. A sensor-assisted experiment is separately labeled and cannot count as a strict-tier result.
6. Never use ground truth, incumbent measurements, benchmark geometry, room identity lookup, or evaluator-only correspondence during inference. Learned calibration statistics must not become a property lookup table. Test this isolation.
7. Never hard-code room dimensions, standard ceiling height, standard door width, benchmark-specific thresholds or manual room placement. Synthetic dimensions belong only in explicit test fixtures.
8. Never hide missed openings, phantom openings, absent rooms, failed reconstructions, unavailable measurements, partial damage extents or disconnected layouts. A schema-valid partial result is not an accuracy or completeness pass.
9. A score is a claim about measured evidence. Never claim an accuracy improvement without the same-input benchmark comparison, exact command and run manifest. Do not move gates or change matching after seeing errors.
10. Keep predictions/GT matching independent of whether widths or lengths are accurate. Report denominators, missingness and reference uncertainty. A fresh sensor capture is different from a repeated deterministic run.
11. Prefer established implementations for SfM, BA, feature matching, registration primitives, segmentation and numerical optimization. Build project-specific residuals, contracts, evidence logic, rules and evaluation. Do not write SLAM or train foundation models in this schedule.
12. Introduce a dependency only with name, version/commit, source, code license, weight license, transitive concerns, download/hash, size, runtime hardware and offline behavior in DEPENDENCIES.md. Do not conflate a repository license with every checkpoint’s terms.
13. Pin versions and content hashes before the baseline. No implicit network/model download during inference. Large assets use documented scripts or volumes; never commit multi-GB data/weights into Git.
14. Every new geometric/ML integration algorithm needs a meaningful test, a failure case and observation-level diagnostics. Unit tests prove invariants; real held-out data establish performance. Avoid suites that merely assert the implementation's own output.
15. Never convert an uncalibrated detector score into a measurement CI. Preserve correlations from scale, poses, planes and masks. Report empirical coverage, width, independent sample count and OOD limitations for every tier.
16. Prefer deterministic geometry and pure rules when ML is unnecessary. Do not use an LLM to invent dimensions, adjacency, rule justifications, repair quantities or benchmark conclusions.
17. Concealed flags mean an inspectable rule recommended investigation. They do not prove a hidden condition. Scope quantities must reference measured surfaces and explicit assumptions; do not invent costs or hidden area.
18. Retain raw captures unchanged. Duplicate file hashes are deduplication evidence, not proof all recordings are identical. Separate floor and ceiling sessions have different coordinate worlds until registration proves otherwise.
19. Cache by input/config/model/code/tier/schema/calibrator hashes. Surface geometry changes invalidate dependent measurement/damage caches. Live mode must process new inputs; never present replay as a cold fresh run.
20. Keep run directories append-only. Preserve baseline before fixes. Commit the prediction and causal hypothesis before implementing the fix. If evidence refutes it, say so; do not rewrite history.
21. Make small genuine commits as work progresses. Never fabricate process history or manufacture a failing baseline. The user must be able to defend each major decision without tools.
22. One local process, files and a CLI are enough. No service fleet, hosted infrastructure, mandatory web UI or tracking server. Keep the stock capture route primary for the 40-hour deadline.
23. Benchmark the RTX 3080 and M2 Pro profiles separately. Check actual VRAM, OS and operator support. Do not promise arbitrary evaluator-machine performance or equate model GPU claims with whole-pipeline timing.
24. Respect the packet allowlist. A required cross-cutting change must state the dependency, migration and tests and receive review before broadening work; ordinary in-scope implementation decisions do not require repeated permission.
25. Finish each packet with its evidence envelope and the next precise packet. Stop adding algorithm features when the protected fix/reproduction/report window begins. If a gate remains impossible or unverified, report it plainly and preserve the best working system.
