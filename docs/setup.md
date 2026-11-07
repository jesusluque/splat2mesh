# Setup

splat2mesh itself is plain Python and runs on CPU. The models around it (Idol, Iris-3B, ENEAS, Qwen3-VL) and the
renderer need GPUs; the pipeline reaches them on the machines where they are installed, over ssh. This page says
what each piece needs and how the code finds it.

## 1. The Python package (CPU)

```sh
git clone https://github.com/jesusluque/splat2mesh.git && cd splat2mesh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/synthetic.py          # mesh + UV + bake + USD on a procedural object, ~40 s
.venv/bin/python tests/glass_selftest.py     # the glass estimator recovers a hidden IOR and tint
```

`tests/synthetic.py` prints the mesh statistics and the errors against the known object; on a correct install:
0 open edges, 0 non-manifold edges, `chamfer_mean` ≈ 0.0023, `texture_psnr_db` ≈ 31.

This is enough for **direct mode on an existing splat**:

```sh
.venv/bin/python scripts/splat2usd.py object.ply --direct --out out/demo
```

## 2. GPU pieces

Each one is optional; a stage that needs a missing piece fails with the step's name.

| Piece | Used for | Developed on | Install |
|---|---|---|---|
| [Idol SDK](https://github.com/speridlabs/idol-sdk) | picture → 3D Gaussian splat | Windows + RTX 3070 (8 GB); also Linux + L4 | its README; it needs access to the gated DINOv3 and FLUX.2 repositories on Hugging Face |
| [ENEAS](https://github.com/speridlabs/eneas) | text → part masks tracked across views | WSL Ubuntu 24.04 on the same Windows host; Linux + L4 | `uv sync` in its repository (~9 GB of weights on first use) |
| [Iris-3B](https://github.com/speridlabs/iris-3b) | text → picture; restorer; its text encoder Qwen3-VL-4B describes materials | Apple Silicon Mac (MPS, bf16) | cloned to `third_party/iris-3b` with its own `.venv` |
| athenea | renders splat views (render path), orbit frames, path-traced stills and turntables | Apple Silicon Mac (Metal) | separate project, not included; any renderer that writes the same view layout works (below) |
| DeepSeek API (`deepseek-flash`, vision) | optional checks: multi-view validator, naming colour clusters, component lists | — | `ds()` in `scripts/gen_views.py` takes the API URL and key from a local helper module (`dsr`) that is not part of this repository; point it at your own client |

### The view layout expected by the render path

One folder per asset; `scripts/athenea_cams.py` writes the cameras and `scripts/render_athenea.sh` the views:

```
render/
├── cameras.json          OpenCV convention: shared intrinsics, per-view world-to-camera R|t
├── views/v000.png        colour
├── views/v000_mask.png   object mask
└── views/v000.exr        depth in channel Z   (or v000_depth.npy)
```

## 3. Environment variables

| Variable | Default | What it points at |
|---|---|---|
| `SPLAT2MESH_IDOL_HOST` | `ixwin-tb` | ssh host that runs Idol |
| `SPLAT2MESH_IDOL_LINUX` | unset (Windows host) | `1` when that host is plain Linux (paths `~/splat2mesh_eneas/idol-sdk`) |
| `SPLAT2MESH_ENEAS_HOST` | `ixwin-tb` | ssh host that runs ENEAS |
| `SPLAT2MESH_ENEAS_LINUX` | `0` on the default host, `1` otherwise | plain Linux instead of Windows + WSL |
| `ATHENEA_BUILD` | a local build folder | athenea build used by the render and view scripts |
| `SPLAT2MESH_GPU_LOCK` | `~/luc/athenea-gpu/shared.sh` | wrapper that runs one GPU job at a time on the local machine |
| `SPLAT2MESH_NOTIFY` | `~/bin/tg` | notifier called by `--notify` with a text and images |

On a Windows Idol host the scripts expect the SDK in `C:\splat2mesh\idol-sdk` with its Python at
`C:\splat2mesh\venv`; on Linux, in `~/splat2mesh_eneas/idol-sdk/.venv`. These paths are constants at the top of
`scripts/splat2usd.py`.

Not everything reads these variables yet. Hard-coded:

- `scripts/gen_views.py` (used by `--views` and `--multiview gen`) and the seed and probe scripts use the Windows
  host `ixwin-tb` and its paths, whatever `SPLAT2MESH_IDOL_HOST` says;
- `scripts/flow_batch.py` and `scripts/seed_views.sh` call the GPU lock at `~/luc/athenea-gpu/shared.sh` directly;
- the batch scripts (`night_*.sh`, `redo_*.sh`) name the development hosts.

Change those constants for another setup.

### Offline Idol host

If the Idol host has no internet, set `HF_HUB_OFFLINE=1` there; the weights load from its Hugging Face cache.
`scripts/idol_diffusers.py --blocks <local snapshot folder>` does the same for the Diffusers version of Idol, whose
pipeline code is otherwise fetched from the Hub.

## 4. Memory and time

Measured on the development machines (RTX 3070 8 GB for Idol and ENEAS, Apple M-series Mac for the rest):

| Stage | Measured |
|---|---|
| Idol SDK, 262 144 gaussians, eager mode | 2.5 min per sample (3.1 min with transfers) |
| Idol through Diffusers, bf16 | 4.55 GB peak VRAM, 3–6 min |
| Direct mode, 448³ grid, 80 000 faces, 4096² texture | about 1 min; the bake runs in chunks (it peaked at 15 GB RAM before chunking) |
| Render path, 87 views, 80 000 faces | 12 min end to end including Idol (6 min of rendering) |
| ENEAS | 28 min the first time (downloads 9 GB), then ~13 s per phrase |
| Iris restorer, bf16 | ~6 GB, 31 s per crop |
