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

## 2. Configuration

Everything machine-specific is in one configuration (`splat2mesh/config.py`), so the scripts carry no paths or
host names. Defaults run everything on this machine. To change them, copy the example and edit it:

```sh
cp splat2mesh.example.toml splat2mesh.toml        # git-ignored; or ~/.config/splat2mesh/config.toml
.venv/bin/python -m splat2mesh.config             # what is resolved (secrets hidden)
```

| Section | Keys | What for |
|---|---|---|
| `[athenea]` | `bin`, `shaders` | the renderer; default `third_party/athenea/build/*/bin/athenea`, then `$PATH` |
| `[gpu]` | `lock` | optional script that serialises GPU jobs (`bash LOCK COMMAND…`) |
| `[idol]` | `host`, `os`, `sdk`, `python`, `work` | picture → splat: `host = "local"` or an ssh host; `os = "linux"` or `"windows"` |
| `[eneas]` | `host`, `os`, `dir`, `work`, `distro` | part masks: `os = "linux"` or `"wsl"` (a Windows host running ENEAS in WSL) |
| `[iris]` | `dir`, `python` | Iris-3B (text → picture, restorer, local describer) |
| `[deepseek]` | `model`, `env_file` | vision checks and part lists; key from `$DEEPSEEK_API_KEY` or a file with that line |
| `[render]` | `hdri` | studio light for stills and turntables; default `assets/hdri/brown_photostudio_02.hdr` (CC0) |
| `[notify]` | `command` | optional notifier for `--notify` (`COMMAND [-f IMAGE] TEXT`) |

Every key can also be set by an environment variable: `ATHENEA_BIN`, `ATHENEA_SHADER_DIR`, `SPLAT2MESH_GPU_LOCK`,
`SPLAT2MESH_IDOL_HOST|OS|SDK|PYTHON|WORK`, `SPLAT2MESH_ENEAS_HOST|OS|DIR`, `SPLAT2MESH_IRIS_DIR|PYTHON`,
`DEEPSEEK_API_KEY`, `DEEPSEEK_ENV`, `SPLAT2MESH_HDRI`, `SPLAT2MESH_NOTIFY` (and `ATHENEA_BUILD`, a build folder).

## 3. athenea (renderer)

[athenea](https://github.com/jesusluque/athenea) (Apache-2.0) renders the splat views, the orbits for ENEAS, the
validator views, stills and turntables, and relights splats with materials. It is a git submodule:

```sh
git submodule update --init third_party/athenea
cd third_party/athenea
# its dependencies are built by its own scripts: OpenUSD with MaterialX and OpenVDB (scripts/build-usd.sh), OIDN
# (scripts/build-oidn.sh), OpenColorIO (scripts/build-ocio.sh), and slangc
cmake --preset macos-arm64-release && cmake --build --preset macos-arm64-release    # or linux-x86_64-release,
                                                                                    #    windows-x64-release
```

The binary is `build/<preset>/bin/athenea` and its shaders `build/<preset>/shaders` (compiled at run time); splat2mesh
finds both there. A build elsewhere is set with `athenea.bin` (shaders next to it, or `athenea.shaders`).

`splat2mesh/athenea.py` drives it from Python, with a CLI:

```sh
.venv/bin/python -m splat2mesh.athenea turntable ASSET.usda out.mp4 --frames 96 --technique raster   # one process
.venv/bin/python -m splat2mesh.athenea splat-turntable SPLAT.ply out.mp4 --y-down
.venv/bin/python -m splat2mesh.athenea view ASSET.usda out.png 1.3 0.9 3.3
.venv/bin/python -m splat2mesh.athenea_splat SPLAT_MATERIALS.ply splat.usdc --y-down   # relightable splat for athenea
```

`stage --turntable` renders all frames with one scene load: 24 raster frames at 1080² took 63 s in total, against
46 s per frame with one process each. Raster with the prefiltered HDRI looks like the path tracer on opaque assets;
use `--technique rt` for glass and translucency.

## 4. The other GPU pieces

Each one is optional; a stage that needs a missing piece fails with the step's name.

| Piece | Used for | Developed on | Install |
|---|---|---|---|
| [Idol SDK](https://github.com/speridlabs/idol-sdk) | picture → 3D Gaussian splat | Windows + RTX 3070 (8 GB); also Linux + L4 | its README; it needs access to the gated DINOv3 and FLUX.2 repositories on Hugging Face |
| [ENEAS](https://github.com/speridlabs/eneas) | text → part masks tracked across views | WSL Ubuntu 24.04 on the same Windows host; Linux + L4 | `uv sync` in its repository (~9 GB of weights on first use); `~/.local/bin/uv` is expected |
| [Iris-3B](https://github.com/speridlabs/iris-3b) | text → picture; restorer; its text encoder Qwen3-VL-4B is the local fallback describer | Apple Silicon Mac (MPS, bf16) | cloned to `third_party/iris-3b` with its own `.venv` |
| DeepSeek API (`deepseek-flash`, vision) | part lists, mask checks, cluster names, multi-view validator | — | a key (`[deepseek]`) |

### The view layout expected by the render path

One folder per asset; `scripts/athenea_cams.py` writes the cameras and `splat2mesh.athenea.render_splats` the views:

```
render/
├── cameras.json          OpenCV convention: shared intrinsics, per-view world-to-camera R|t
├── views/v000.png        colour
├── views/v000_mask.png   object mask
└── views/v000.exr        depth in channel Z   (or v000_depth.npy)
```

### Offline Idol host

If the Idol host has no internet, set `HF_HUB_OFFLINE=1` there; the weights load from its Hugging Face cache.
`scripts/idol_diffusers.py --blocks <local snapshot folder>` does the same for the Diffusers version of Idol, whose
pipeline code is otherwise fetched from the Hub.

## 5. Memory and time

Measured on the development machines (RTX 3070 8 GB for Idol and ENEAS, Apple M-series Mac for the rest):

| Stage | Measured |
|---|---|
| Idol SDK, 262 144 gaussians, eager mode | 2.5 min per sample (3.1 min with transfers) |
| Idol through Diffusers, bf16 | 4.55 GB peak VRAM, 3–6 min |
| Direct mode, 448³ grid, 80 000 faces, 4096² texture | about 1 min; the bake runs in chunks (it peaked at 15 GB RAM before chunking) |
| Render path, 87 views, 80 000 faces | 12 min end to end including Idol (6 min of rendering) |
| ENEAS | 28 min the first time (downloads 9 GB), then ~13 s per phrase |
| Iris restorer, bf16 | ~6 GB, 31 s per crop |
