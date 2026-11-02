# Tutorial: from a text prompt to a UV-textured USD

Result of this tutorial: `qa.psnr_mean` 28.9 dB, 20 036 faces, 0 open edges, about 13 min end to end.

A fully open chain, every model and tool permissively licensed:

```
text ──Iris-3B Turbo──▶ image ──Idol──▶ 3D Gaussian splat ──splat2mesh──▶ USD mesh + UV + texture
     (speridlabs)               (speridlabs)                 (this repo)
```

You can also enter the chain with your own image or with an existing 3DGS `.ply`.

## 1. Install

```sh
git clone https://github.com/jesusluque/splat2mesh.git && cd splat2mesh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt gradio_client
```

## 2. Check it works (no GPU)

```sh
.venv/bin/python tests/synthetic.py
```

It builds a torus and a sphere with procedural colours, "photographs" them from 80 cameras and converts
them. It takes about 40 s and should print something like:

```json
"faces": 19998, "boundary_edges": 0, "non_manifold_edges": 0,
"chamfer_mean": 0.0023, "texture_psnr_db": 27.4
```

`chamfer_mean` is the shape error against the true geometry (here ≈ 0.25 % of the object size);
`texture_psnr_db` is how well the colour comes out.

## 3. Text → image (Iris-3B Turbo)

```sh
.venv/bin/python scripts/text2image.py \
  "A glossy white ceramic maneki-neko lucky cat figurine raising its left paw, red collar with a gold bell, \
holding a gold koban coin, red cherry blossom painting, whole object centred, studio product photo, \
plain black background" lucky_cat.png --seed 7
```

<img src="img/cat_input.jpg" width="320">

Prompt tips for 3D: **one object, whole, centred, plain light-grey or white background**, studio light. Avoid cropped objects,
scenes and strong reflections. The script uses the public Iris-3B Turbo Space; set `HF_TOKEN` for more
quota, or run Iris-3B locally (see [speridlabs/iris-3b](https://github.com/speridlabs/iris-3b)).

## 4. Image → splat → USD, in one command

```sh
.venv/bin/python scripts/splat2usd.py lucky_cat.png --gaussians 524288 --out out/tutorial
```

What happens:

1. **Idol** turns the image into a 3D Gaussian splat on a CUDA machine (`SPLAT2MESH_IDOL_HOST`), inventing
   the sides the picture never showed (~3 min on an 8 GB laptop GPU).
2. The splat is rendered from **87 cameras** around it, with colour, mask and depth (~7 min).
3. The depth maps are fused into a surface, cleaned, reduced to 20 000 faces and closed.
4. **xatlas** unwraps the UVs; the colour of every view is baked into one texture.
5. A **USD** is written with `UsdPreviewSurface`, and a QA pass compares the mesh with the splat.

The splat from two views the input image never had: three-quarter (left) and back (right):

<img src="img/cat_splat.jpg" width="640">

Idol invents the hidden side. Here the back came out dark and the tail black: the model took colour from
the **black background**. Prompting for a light grey or white background usually gives a cleaner back.

Left, the splat; right, the textured mesh, from three of the cameras:

<img src="img/cat_compare.jpg" width="480">

The UV atlas, with the chart outlines:

<img src="img/cat_uv.jpg" width="480">

Already have a splat? Start here instead:

```sh
.venv/bin/python scripts/splat2usd.py my_object.ply --out out/tutorial    # add --y-down for COLMAP-style .ply
```

A whole folder of images and `.ply` files is a batch: `splat2usd.py my_folder/ --out out/batch`.

## 5. What you get

```
out/tutorial/lucky_cat/
├── report.json          ← measurements and timings; "ok": true/false
├── render/              ← the 87 views of the splat (colour, mask, depth, cameras)
└── usd/
    ├── asset.usda       ← the mesh with UVs and material (usdview, Blender, Omniverse…)
    ├── basecolor.png    ← the texture
    ├── mesh.ply
    └── compare.png      ← splat vs textured mesh
```

Reading `report.json`:

| Field | Meaning | Good |
|---|---|---|
| `mesh.boundary_edges` | open edges (holes) | 0 |
| `mesh.non_manifold_edges` | edges shared by more than 2 faces | ≤ 20 |
| `mesh.uv_charts` | UV islands | fewer is better |
| `qa.psnr_mean` | how close the textured mesh looks to the splat, dB | ≥ 22 (26–30 is very good) |
| `t_total_s` | total time | |

Always look at `compare.png`: it is the quickest way to spot a problem.

## 6. Useful options

| Option | What for |
|---|---|
| `--faces 50000` | more geometric detail (default 20 000) |
| `--tex 4096` | bigger texture (default 2048) |
| `--smooth 3` | smoother surface; recommended for glass, slightly less accurate on opaque objects |
| `--min-psnr 25` | stricter threshold for "ok" |
| `--notify` | send the summary and images when a batch ends |

## 7. Glass

A splat does not know an object is transparent: it paints what shows through as colour. Given views of
the object in front of a known background (a checker, for instance), the tool estimates the material:

```sh
.venv/bin/python scripts/glass_estimate.py out/tutorial/vase/usd views_with_background/ background.json --apply
```

It finds the index of refraction and the tint by tracing refracted rays through the mesh to the
background, and turns the USD material into glass. `background.json` describes the background as one or
more textured planes (see `splat2mesh/glass.py`). Check it with `tests/glass_selftest.py`.

## 8. When something goes wrong

- **`"ok": false` with an `error`**: the message names the step (Idol, render, mesh).
- **Object upside down in `compare.png`**: rerun with `--y-down` (or `--y-up` for an Idol splat that is already upright).
- **Low PSNR on shiny objects**: expected; the texture is one fixed colour and cannot reproduce
  reflections that change with the viewing angle.
- **Many UV islands**: try `--smooth 1`; artist-clean UVs need quad remeshing.

Images in this tutorial were generated with Iris-3B Turbo and Idol; no third-party assets.
