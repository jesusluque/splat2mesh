# splat2mesh

**3D Gaussian splat → production USD: a closed mesh, clean UVs, a baked texture and PBR materials, with no manual
step.** Give it a picture of an object (or a 3DGS `.ply`) and it returns a `UsdGeomMesh` with `UsdPreviewSurface`
and MaterialX materials, ready for usdview, Blender, Omniverse, Houdini or iPhone AR Quick Look.

<img src="docs/img/gallery.jpg" width="100%">

*33 test objects (porcelain, metals, wood, lacquer, stone, jade, wax, glass, leather and mixes), each made from one
generated picture. Geometry, UVs, texture and materials are all automatic. Path traced under a studio HDRI.*
Live gallery (USDZ in the browser and in AR): <https://splat2mesh.lucab.co.uk>

```
picture ──Idol──▶ 3D Gaussian splat ──splat2mesh──▶ closed mesh + UVs + texture ──materials──▶ USD (PBR)
                  (speridlabs)                      (this repository)              Qwen3-VL + ENEAS + library
```

## What it does

| Stage | How | Typical result |
|---|---|---|
| Picture → splat | [Idol](https://github.com/speridlabs/idol-sdk) (speridlabs) on a CUDA machine | 262 144–524 288 gaussians, ~3 min on an 8 GB GPU |
| Splat → mesh, **direct mode** | the splat's opacity field on a 448³ grid → solid → marching cubes → topology-preserving decimation | one watertight shell, 80 000 faces, ~1 min |
| Splat → mesh, **render path** | 87 rendered RGB-D views → TSDF fusion + visual hull | for splats that direct mode crumples; ~10 min |
| UVs + texture | xatlas, then colour baked from the gaussians (or the views) into a 4096² atlas | ~100–250 charts |
| Materials | parts named by Qwen3-VL, tracked on 16 views by [ENEAS](https://github.com/speridlabs/eneas), voted per texel, values from a 58-entry PBR library | roughness, metallic, transmission, subsurface maps |
| Output | USD with `UsdPreviewSurface` + MaterialX `standard_surface`; per-part and per-material variants | QA report: open edges, non-manifold edges, PSNR vs the splat |

## Quick start

```sh
git clone https://github.com/jesusluque/splat2mesh.git && cd splat2mesh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/synthetic.py      # end-to-end check on a procedural object, CPU only, ~40 s
```

From a splat you already have (CPU only):

```sh
.venv/bin/python scripts/splat2usd.py my_object.ply --direct --out out/demo   # --y-down for COLMAP/Idol .ply
```

From a picture (needs Idol on a CUDA host reachable by ssh, see [docs/setup.md](docs/setup.md)):

```sh
.venv/bin/python scripts/splat2usd.py lucky_cat.png --direct --out out/demo
```

Result: `out/demo/<name>/usd/asset.usda` with its textures, `compare.png` (splat vs mesh) and `report.json`.

## Documentation

| | |
|---|---|
| [Tutorial: basics](docs/TUTORIAL_basics.md) | one object step by step: picture → splat → USD, the UV map, direct mode vs render path |
| [Tutorial: materials](docs/TUTORIAL.md) | the full chain with automatic PBR materials, parts for compositing, web and video |
| [Pipeline reference](docs/pipeline.md) | every stage, its parameters and the reasons behind the defaults |
| [Materials reference](docs/materials.md) | how parts are found, the per-texel rules, the material library, glass |
| [Multi-view and seed selection](docs/multiview.md) | more than one view of the object; choosing the best Idol sample with a vision validator |
| [Setup](docs/setup.md) | dependencies, GPU hosts, environment variables |
| [Scripts](docs/scripts.md) | what every script does, grouped by purpose |
| [Findings](docs/findings.md) | what was tried, measured, kept and dropped, with numbers |
| [Licences](docs/licenses.md) | licences of the code, the dependencies and the models |

## Status and limits

splat2mesh is research code. It runs end to end on five test sets of eight generated pictures, and 33 of the
results are in the gallery. Known limits (details in [findings](docs/findings.md)):

- Idol invents the sides the picture does not show; those sides are only as good as Idol's guess.
- Direct mode makes solids: a vase has no inside, a bottle no liquid surface.
- Glass is hard for any splat. Glass that the splat leaves as an opening, and opaque parts inside glass (filaments,
  wicks), are recovered by dedicated stages, but refraction stays baked into the colour.
- Small parts (corks, eyelids, thin wires) are often missed by the segmenter.
- The GPU stages were developed on specific machines (an RTX 3070 Windows host for Idol and ENEAS, an Apple Silicon
  Mac for Iris and rendering); they are reached through ssh and environment variables, not packaged.

## Licence

splat2mesh is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Its Python dependencies are permissive.
The models it calls have their own licences; **one of Idol's image encoders, as Idol loads it by default, comes from
a non-commercial repository**. Read [docs/licenses.md](docs/licenses.md) before any commercial use.

## Credits

Idol, Iris-3B and ENEAS are by [speridlabs](https://github.com/speridlabs). Qwen3-VL-4B by the Qwen team. xatlas,
Open3D, trimesh, fast-simplification, manifold3d and OpenUSD do the geometry and file work. HDRIs from Poly Haven
(CC0).
