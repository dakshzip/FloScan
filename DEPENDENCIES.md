# Dependencies

Every external package, native library and model checkpoint FloScan uses, with the evidence behind each licence claim.
Recorded in P02 on 2026-10-04 and superseding the interim list in `docs/adr/001-requirements.md`.
Licences come from each installed distribution's own metadata or from licence files fetched at pinned commits; nothing here is from memory.

FloScan calls no infrastructure of its own.
Packages come from PyPI (plus PyTorch's CUDA index on Windows), checkpoints from Hugging Face; both are fetched once at setup and never during inference.

## Environment

| Item | Value |
|---|---|
| Python | 3.11 only (`requires-python = ">=3.11,<3.12"`); this Mac uses uv-managed CPython 3.11.15 |
| Resolver | uv 0.11.5; `uv.lock` resolves 116 entries for macOS, Linux and Windows |
| Install | `uv sync --locked` (measured: 66 s on this Mac and network, `.venv` 1.4 GB) |
| Wheels only | Every pinned binary ships cp311 wheels for macOS arm64, Linux x86_64 and Windows amd64; nothing compiles from source |
| Windows | torch and torchvision come from `https://download.pytorch.org/whl/cu126` (CUDA 12.6) because PyPI's Windows wheels are CPU-only |
| Linux | torch from PyPI targets CUDA 13.0 and needs an NVIDIA driver of 580 or later (not verified on any machine) |

Version policy: established releases, not the newest.
Each pin is a patch release or at least several weeks old on 2026-10-04 (for example torch 2.12.1 rather than 2.14.1 from 2026-09-30, Open3D 0.19.0 rather than 0.20.0 from 2026-09-16).

## Direct dependencies

Sizes are installed sizes on macOS arm64.

| Package | Version | Purpose | Licence (metadata) | Size | Offline behaviour |
|---|---|---|---|---|---|
| numpy | 2.4.6 | Arrays | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 26 MB | Fully offline |
| scipy | 1.17.1 | Robust least squares, optimisation | BSD (classifier) | 83 MB | Fully offline |
| shapely | 2.1.2 | Polygon area, union, overlap | BSD 3-Clause; bundles GEOS 3.13.1 (LGPL-2.1) | 7 MB | Fully offline |
| pydantic | 2.12.5 | Typed records (P03) | MIT | small | Fully offline |
| opencv-python-headless | 4.13.0.92 | Image processing, SIFT | Apache 2.0; bundles FFmpeg 7.1.1 built `--enable-gpl` (see below) | 122 MB | Fully offline |
| av (PyAV) | 18.1.0 | Video decode with presentation timestamps | BSD-3-Clause; bundles FFmpeg 8.1.2 plus libx264/libx265 (see below) | 46 MB | Fully offline |
| pycolmap | 4.1.1 | SfM, bundle adjustment | BSD-3-Clause; bundles libomp | 52 MB | Fully offline |
| open3d | 0.19.0 | Point clouds, RANSAC planes, ICP | MIT; bundles libomp | 282 MB | Fully offline |
| torch | 2.12.1 | Model inference (CPU, MPS, CUDA) | BSD-3-Clause; bundles libomp | 404 MB (Mac) | Fully offline |
| torchvision | 0.27.1 | Fast image processors used by transformers | BSD | small | Fully offline |
| transformers | 5.15.1 | Depth Pro, Grounding DINO, SAM 2 implementations | Apache 2.0 | 51 MB | Offline: loads only from verified local directories with `HF_HUB_OFFLINE=1` and the network guard |
| huggingface-hub | 1.31.0 | Required by transformers | Apache-2.0 | small | Offline mode forced during inference; FloScan fetches with its own verified downloader |
| safetensors | 0.8.0 | Checkpoint format | Apache Software License | small | Fully offline |
| pillow | 12.1.1 | Image I/O | MIT-CMU | small | Fully offline |
| pyyaml | 6.0.3 | Gate registry (`safe_load` only) | MIT | small | Fully offline |
| pytest (dev) | 9.1.1 | Tests | MIT | small | Fully offline |
| ruff (dev) | 0.16.10 | Lint and format | MIT | small | Fully offline |

## Bundled native libraries and their consequences

These ship inside the wheels above and are not visible in Python metadata.

| Library | Inside | Licence | Finding |
|---|---|---|---|
| FFmpeg 7.1.1 | opencv-python-headless | GPL (configured `--enable-gpl --enable-version3`) | The wheel's `LICENSE-3RD-PARTY.txt` describes FFmpeg under the LGPL, but the shipped `libavcodec` was configured with `--enable-gpl`, so this copy is GPL-3.0-or-later. |
| FFmpeg 8.1.2 | av | LGPL-3.0 build (`--enable-version3`, no `--enable-gpl` found) linked with libx264/libx265 | Same outcome in practice: it bundles GPL encoders. |
| libx264, libx265 | av and opencv (separate copies) | GPL-2.0-or-later | Encoders only; FloScan decodes with FFmpeg's native decoders and never encodes H.264/HEVC in the pipeline. |
| GnuTLS, OpenSSL | opencv | LGPL-2.1-or-later, Apache-2.0 | Network protocols FloScan never uses. |
| GEOS 3.13.1 | shapely | LGPL-2.1 | Dynamic library, unmodified. |
| libomp (four copies) | torch, open3d, pycolmap, scikit-learn | Apache-2.0 with LLVM exception | **Runtime conflict, see below.** |

GPL obligations attach to *distribution*.
FloScan installs these wheels from PyPI on the user's machine and does not redistribute them.
If FloScan were ever shipped as a bundled binary, the GPL builds above would govern that bundle, and LGPL-only FFmpeg builds would be needed.

### Measured runtime conflicts (macOS arm64)

Pairwise imports in fresh interpreters (`floscan doctor` repeats these checks):

| Pair | Result | Consequence |
|---|---|---|
| torch + pycolmap | **Aborts** with "OMP: Error #15" (two libomp runtimes), in either import order; reproduced with torch 2.12.1–2.14.1 and pycolmap 3.12.6–4.2.1 | Model inference runs in a separate worker process that refuses to start if pycolmap is loaded. `KMP_DUPLICATE_LIB_OK=TRUE` is **not** used: the runtime warns it can silently produce incorrect results. |
| torch + open3d | Compatible | Can share a process. |
| opencv + av | Compatible, with Objective-C duplicate-class warnings for FFmpeg's AVFoundation capture classes | Those classes serve camera/microphone capture, which FloScan never uses. |

Linux and Windows have not been tested for these conflicts.

## Model checkpoints

Pinned in `configs/models.lock.json` by Hugging Face commit, with every file's size and SHA-256.
`scripts/fetch_models.sh` downloads into `$FLOSCAN_MODELS_DIR` (default `./models`, ignored by Git), verifying each file before renaming it into place.
Inference re-verifies every file, sets `HF_HUB_OFFLINE=1` and refuses outbound connections while loading, so a missing checkpoint fails instead of downloading.

| Model | Checkpoint (revision) | Size | Code licence | Weight licence | Status |
|---|---|---|---|---|---|
| Depth Pro | `apple/DepthPro-hf` (`de816c8`) | 1.90 GB | Apple Sample Code License (ASCL), `apple-aiml-research/ml-depth-pro@9e65e4d` | ASCL per the official README; **conflicting metadata**, see below | Selected |
| Grounding DINO tiny | `IDEA-Research/grounding-dino-tiny` (`a2bb814`) | 0.69 GB | Apache-2.0, `IDEA-Research/GroundingDINO@856dde2` | Apache-2.0 (model card); pretraining datasets carry their own terms | Selected |
| SAM 2.1 Hiera small | `facebook/sam2.1-hiera-small` (`ee5bba1`) | 0.18 GB | Apache-2.0, `facebookresearch/sam2@2b90b9f` | Apache-2.0 (official README and model card) | Selected |
| DISK + LightGlue | none | none | none | none | Conditional: only if a matching ablation earns it; not fetched |
| VGGT | none | none | none | none | Deferred: weight licence must be audited first |

Total pinned download: 2.78 GB.

**Depth Pro weight-licence conflict.**
The official repository README (commit `9e65e4d`) says "The model weights are released under the [LICENSE](LICENSE) terms", the permissive Apple Sample Code License, and the `apple/DepthPro-hf` model card body says "License: Apple-ASCL".
That same model card's metadata tag says `apple-amlr`, Apple's research-use licence.
This case-study use is non-commercial and fits either reading.
Any commercial use needs Apple's terms confirmed first.

All three models run through transformers' pure-PyTorch implementations, so no CUDA or C++ extension needs compiling.
The original Grounding DINO and SAM 2 repositories build custom extensions, which this Mac cannot compile.

## Transitive dependencies (installed on macOS arm64)

Licences as declared in each distribution's metadata.
`addict` declares no licence in its metadata.
Most of the list comes from Open3D 0.19, which requires dash, flask, pandas, scikit-learn, matplotlib and nbformat even though FloScan uses none of them.

| Package | Version | Licence (metadata) |
|---|---|---|
| addict | 2.4.0 | UNKNOWN |
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| attrs | 26.1.0 | MIT |
| blinker | 1.9.0 | MIT License |
| certifi | 2026.7.22 | MPL-2.0 |
| charset-normalizer | 3.5.2 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| comm | 0.2.3 | BSD License |
| configargparse | 1.8.0 | MIT |
| contourpy | 1.3.3 | BSD License |
| cycler | 0.12.1 | BSD License |
| dash | 4.4.1 | MIT |
| fastjsonschema | 2.22.2 | BSD-3-Clause |
| filelock | 4.0.10 | MIT |
| flask | 3.1.3 | BSD-3-Clause |
| fonttools | 4.66.1 | MIT |
| fsspec | 2026.9.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.6.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| importlib_metadata | 9.0.1 | Apache-2.0 |
| iniconfig | 2.3.0 | MIT |
| itsdangerous | 2.2.0 | BSD License |
| janus | 2.0.0 | Apache 2 |
| jinja2 | 3.1.6 | BSD License |
| joblib | 1.6.0 | BSD-3-Clause |
| jsonschema | 4.26.0 | MIT |
| jsonschema-specifications | 2025.9.1 | MIT |
| jupyter_core | 5.9.1 | BSD-3-Clause |
| kiwisolver | 1.5.1 | BSD License |
| markdown-it-py | 4.2.0 | MIT License |
| markupsafe | 3.0.4 | BSD-3-Clause |
| matplotlib | 3.11.2 | Python Software Foundation License |
| mdurl | 0.1.2 | MIT License |
| mpmath | 1.3.0 | BSD |
| narwhals | 2.26.0 | MIT |
| nbformat | 5.11.1 | BSD License |
| nest-asyncio | 1.6.0 | BSD |
| networkx | 3.6.1 | BSD-3-Clause |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.6 | BSD License |
| platformdirs | 4.12.3 | MIT |
| plotly | 7.1.0 | MIT |
| pluggy | 1.6.0 | MIT |
| pydantic_core | 2.41.5 | MIT |
| pygments | 2.21.0 | BSD-2-Clause |
| pyparsing | 3.3.3 | MIT |
| pyquaternion | 0.9.9 | MIT |
| python-dateutil | 2.9.0.post0 | Dual License |
| referencing | 0.37.0 | MIT |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache-2.0 |
| retrying | 1.4.2 | Apache-2.0 |
| rich | 15.0.0 | MIT |
| rpds-py | 2026.6.3 | MIT |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| setuptools | 81.0.0 | MIT |
| shellingham | 1.5.4 | ISC License |
| six | 1.17.0 | MIT |
| sympy | 1.14.0 | BSD |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
| tokenizers | 0.22.2 | Apache Software License |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| traitlets | 5.16.1 | BSD License |
| typer | 0.27.2 | MIT |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |
| werkzeug | 3.1.9 | BSD-3-Clause |
| zipp | 4.1.1 | MIT |

Not installed on this Mac but present in `uv.lock` for other platforms, with licences as declared on PyPI (checked 2026-10-04, not inspected on a machine):

| Package | Platform | Licence (PyPI metadata) |
|---|---|---|
| nvidia-cublas, nvidia-cusparselt-cu13 and siblings | Linux | NVIDIA proprietary (`LicenseRef-NVIDIA-Proprietary` / "NVIDIA Proprietary Software") |
| cuda-toolkit, nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-nvshmem-cu13 | Linux | Undeclared in metadata; NVIDIA software licence terms apply |
| cuda-bindings, cuda-pathfinder | Linux | Apache-2.0 |
| triton | Linux | MIT License |
| torch, torchvision `+cu126` | Windows | BSD (as above); bundle NVIDIA CUDA 12.6 runtime libraries |
| colorama | Windows | BSD License |
| tzdata | Windows | Apache-2.0 |
