# splat2mesh

3D Gaussian Splatting → UV-textured USD mesh, automated end to end. Apache-2.0.

**Tutorial: [docs/TUTORIAL.md](docs/TUTORIAL.md)** (image → Idol → splat → USD, with the UV map explained)

## Setup

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/synthetic.py        # end-to-end check on a procedural object, no GPU needed
.venv/bin/python tests/glass_selftest.py   # glass estimator check
```

The geometry, UV, bake, USD and glass stages run on CPU (`splat2mesh/`). Rendering the splat views uses the
athenea renderer (`render --splats`); any renderer that writes the same layout works (`cameras.json` in OpenCV
convention + `views/vNNN.png`, `vNNN_mask.png`, depth as `vNNN.exr` channel Z or `vNNN_depth.npy`).
Environment: `ATHENEA_BUILD` (renderer build dir), `SPLAT2MESH_GPU_LOCK` (wrapper that serialises GPU jobs),
`SPLAT2MESH_IDOL_HOST` (ssh host with CUDA + Idol), `SPLAT2MESH_NOTIFY` (notifier for `--notify`).

## One command

```sh
.venv/bin/python scripts/splat2usd.py INPUT [INPUT ...] [--out out/auto] [--notify]
```

`INPUT` is an image, a 3DGS `.ply`, or a folder of them (batch). For each input:

1. **Image → splat** (images only): Idol (speridlabs, Apache-2.0) on ixwin. Its output is Y-down; handled automatically.
2. **Cameras**: 87 views on rings at −30/0/25/50/75° + zenith, framing the splat (`scripts/athenea_cams.py`).
3. **Render**: athenea `render --splats` (unlit SH colour, depth Z), one `shared.sh` GPU job (`scripts/render_athenea.sh`).
4. **Geometry**: TSDF fusion + visual-hull clamp (fixes transparent parts) → clean → decimate (`--faces`) → cap unseen holes.
5. **UV + texture**: xatlas (scale-normalised, filled atlas) → robust multi-view bake (IRLS, occlusion test) → inpaint.
6. **USD**: `UsdGeomMesh` + `st` + `UsdPreviewSurface`, `metersPerUnit = 1`, relative texture path.
7. **QA**: mesh stats, re-render PSNR vs the splat, `compare.png`; `report.json` per input, `summary.json` per batch;
   `--notify` sends the summary and images to Telegram.

Flags: `--smooth X` (Taubin, 0 = off; 3 for glass), `--y-down` for COLMAP-style `.ply`, `--no-antialias` for plain 3DGS (non Mip-Splatting) `.ply`,
`--gaussians` for Idol, `--min-psnr` for the OK threshold.

## Glass (transparent objects)

```sh
.venv/bin/python scripts/splat2usd.py INPUT --glass background.json [--glass-views VIEWS_DIR]
.venv/bin/python scripts/glass_estimate.py ASSET_DIR VIEWS_DIR background.json --apply
```

Estimates IOR and absorption tint by tracing refracted rays through the mesh to a known background and fitting
the images (`splat2mesh/glass.py`); writes `glass.json` and turns the USD material into a dielectric
(`ior`, `opacity`, tint; full estimate in `customData`). Background JSON: a textured plane or a list of planes.
Self-test: `tests/glass_selftest.py`.

## Tests / benchmarks

- `tests/synthetic.py`: procedural object, Chamfer + texture PSNR.
- `scripts/run_pawn.py`: athenea's OpenChessSet pawn vs its original mesh (Chamfer, F-score, visible/hidden).
- `train/train_2dgs.py`: gsplat 2DGS (Apache-2.0) on ixwin, the retraining path.

Licences: all shipped code paths use permissive dependencies (Open3D, xatlas, fast-simplification, trimesh,
usd-core, gsplat). Idol's encoders: DINOv3 licence and FLUX.2 VAE; check before commercial use of generated splats.
