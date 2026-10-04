# History rewrite before publication (2026-10-04)

Before the repository was made public, the assignment PDFs and the planning handoff (`docs/implementation-strategy/`, derived from them) were removed from every commit with `git filter-branch`.
They remain on the development machine and are ignored by Git.
Commit order, messages, authorship and dates are unchanged, except that the first commit's message was reworded because it no longer adds those files.

Reviews and evidence written before the rewrite cite the old hashes; this table maps them.

| Old | New | Subject |
|---|---|---|
| `864a667` | `85e366d` | Stop tracking the planning handoff |
| `f532dc6` | `b04904e` | Stop tracking the assignment PDFs |
| `9080bfb` | `0eb49ff` | P03B: two-way frame/depth ownership and unique graph node and edge IDs |
| `a81643d` | `0e75a91` | P03A: truthful coverage, bound references, PSD covariance, deep immutability |
| `0f5dc07` | `3040cac` | P03: typed internal records and the coordinate chain |
| `6b66af4` | `eb7cb35` | Add architecture reviews for P01, P01A and P02 |
| `2f28ed5` | `3d8a756` | P02A: durable macOS environment and doctor report error boundary |
| `421fc42` | `e72a73c` | P02 part 2: fp16 Depth Pro on accelerators, device matrix |
| `70ae0d1` | `d31990d` | P02 part 1: pinned runtime, model lock, offline loading and doctor |
| `21d1979` | `0a0ed73` | P01A: close P01 review defects in validation and error handling |
| `52453cc` | `2fed3c9` | P01: freeze requirements and a runnable diagnostic contract |
| `7865cb5` | `5bd6065` | Initial .gitignore |
