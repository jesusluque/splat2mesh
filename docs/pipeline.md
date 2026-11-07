# Pipeline reference

What `scripts/splat2usd.py` does to each input, stage by stage, with the parameters in the code and why they are
set that way. The materials stage that follows is in [materials.md](materials.md).

```
INPUT (.png/.jpg/.webp or 3DGS .ply)
  │  image only
  ├─▶ 1. Idol on the CUDA host ─────────────▶ splat .ply (Y-down) + idol_input.png + idol_camera.json
  │
  ├─▶ 2a. direct mode (--direct) ──▶ folds ≤ 300 ? ──yes──▶ 4. UV + bake from the gaussians
  │                                     │ no
  └─▶ 2b. render path ◀──────────────────┘
          87 views ─▶ visual hull ─▶ TSDF ─▶ clean ─▶ decimate ─▶ cap holes ─▶ 4. UV + bake from the views
                                                                                   │
                                          5. USD ◀────────────────────────────────┘
                                          6. QA: mesh stats, re-render PSNR, compare.png, report.json
```

## 1. Picture → splat (Idol)

- The picture is copied to the Idol host and `idol.sample` runs there (`--num_gaussians`, default 262 144;
  `--gaussians 524288` was used for the gallery).
- **Crop mode** follows the file type: `.jpg/.jpeg` are treated as photos (`--mode real`, crop margin 1.7),
  everything else as renders or generated pictures (`synthetic`, margin 1.2).
- Idol removes the background itself (BiRefNet) and works on a 1024² crop of the object composited on black. Its
  output is **Y-down**; splat2usd turns it upright automatically. A `.ply` from elsewhere is assumed Y-up unless
  `--y-down` is given.
- Idol also predicts the camera of the input picture (`idol_camera.json`: yaw, pitch in radians, radius, fov). Later
  stages use it to find the view that matches the picture. (On a Linux Idol host, `SPLAT2MESH_IDOL_LINUX=1`, the
  camera file is not fetched yet.)
- **Use a white or light background in the picture.** Idol borrows colour from the background for the sides it has
  to invent: the same object on black came out with a dark back.

`--upscale` passes the object crop through the Iris-3B restorer first. It keeps the object (PSNR 38.6 dB at 256 px,
silhouette IoU 0.995 on the kettle) but did not improve results consistently (see [findings](findings.md)), so it is
off by default.

## 2a. Direct mode (`--direct`)

No rendering: the mesh comes straight from the gaussians (`splat2mesh/direct.py`).

1. **Opacity field.** Every gaussian with opacity above 0.02 adds its anisotropic kernel, evaluated to 3σ, to a grid of
   **448** cells across the object: `O(x) = 1 − Π(1 − αᵢ·Gᵢ(x))`, accumulated as `−Σ log(1 − αᵢGᵢ)`.
2. **Solid.** Cells above **0.15** are solid (0.3 left craters where Idol's back gaussians are faint); three closing
   iterations bridge small gaps, interior cavities are filled, pieces under 2 % of the largest are dropped, and the
   solid is blurred with
   σ = **1.2** voxels before marching cubes, which removes the small folds that would split the UVs into hundreds
   of charts.
3. **Surface.** Marching cubes, then quadric decimation to `--faces` (default **80 000**) with Open3D, which keeps
   one closed manifold shell; fast-simplification is the fallback.
4. **Fold check.** Edges with a dihedral angle above 120° are counted. Above `--max-folds` (default **300**) the
   surface is crumpled (flat black-lacquer walls, faceted glass) and splat2usd falls back to the render path.
5. **Colour.** Each texel takes the colour (SH degree 0) of its 12 nearest gaussians, weighted by α·G at the texel
   and by exp(−depth below the surface / h), so gaussians buried under the surface barely count. No views are needed.

Direct mode is meant for Idol-like splats, whose gaussians have volume. Thin, surface-aligned splats (converted from
meshes, some photogrammetry) leave holes in the opacity field: use the render path for them.

On the lucky cat: 1 component, watertight, 0 non-manifold edges, 108 UV charts, PSNR 27.1 dB against the splat,
39 s after Idol (render path on the same splat: 243 charts, 24.1 dB, 9 min).

## 2b. Render path (default)

1. **Cameras** (`scripts/athenea_cams.py`): **87** views on rings at −30° (12), 0° (24), 25° (24), 50° (18) and 75° (8)
   plus the zenith, framing the splat's bounding sphere.
2. **Render** (`scripts/render_athenea.sh`): athenea `render --splats`, unlit SH colour, mask and depth per view, as
   one job under the GPU lock. Any renderer that writes the same layout can replace it ([setup](setup.md)).
3. **Visual hull** (`geometry.visual_hull`, 320³): carving from the masks. Depth is clamped to the hull's front plus
   a margin, which fixes transparent parts whose depth goes through the object.
4. **TSDF fusion** (`geometry.tsdf_fuse`, Open3D `VoxelBlockGrid`, truncation 4 voxels).
5. **Clean**: components smaller than 2 % of the largest are dropped, non-manifold edges fixed.
6. **Decimate** to `--faces`, **cap** the remaining boundary loops (unseen undersides) with fans, optional Taubin
   smoothing `--smooth X` (X × 10 iterations; 3 is good for glass).

The texture is capped at 2048² on this path (`--render-tex`): each texel averages 87 views, and 4096² was four times
slower for little visible gain.

## 3. Orientation

Idol and COLMAP splats are Y-down; the USD is Y-up. Direct mode flips the gaussians when loading; the render path
renders with `--rotate-x 180`.

- **Picture input**: Idol's splat is taken as Y-down. `--y-up` overrides that if an Idol version ever writes Y-up.
- **`.ply` input**: taken as Y-up. Add `--y-down` for COLMAP-style and Idol splats.

If the result is upside down, the input was the other way round: switch the flag.

## 4. UVs and bake

- **Unwrap** (`texture.unwrap`): xatlas with default chart options, scaled so the charts fill one square atlas,
  4 texels of padding. Tuning xatlas changed little; the clean, smooth shell from direct mode is what cut the chart
  count (1 267 → 108 on the cat).
- **Bake from views** (`texture.bake`, render path): every texel is projected into each view; a sample counts if its
  depth agrees with the view's depth (occlusion test) and is weighted by cos⁴ of the viewing angle over the squared
  distance (grazing views, cos ≤ 0.05, are ignored). Two reweighted passes with a Cauchy weight (σ = 0.08) then
  down-weight samples far from the weighted mean (highlights, misregistered edges).
- **Bake from gaussians** (`direct.bake_direct`, direct mode): see 2a.5, in chunks of 400 000 texels.
- **Fill**: texels no view saw are filled by push-pull averaging of their neighbours, and colour is spread past the
  chart borders for mip-maps.

## 5. USD

`splat2mesh/usd_export.py` writes:

- `/Asset/Geom`: a `UsdGeomMesh` of triangles with per-vertex `st` and normals (vertices are split along UV seams);
  `metersPerUnit = 1`, `upAxis = Y`. The size is the splat's (an Idol object is about one unit across), not
  real-world: scale it to the real object's size.
- `UsdPreviewSurface` with relative texture paths (`basecolor.png` next to the file).
- The materials stage adds roughness, metallic, transmission, subsurface and opacity maps and a MaterialX
  `standard_surface` ([materials.md](materials.md)).

## 6. QA and the report

- Mesh statistics: faces, components, open (boundary) edges, non-manifold edges, watertight, UV charts, atlas use,
  covered texels, and in direct mode the fold count.
- **Re-render PSNR** (render path only): the textured mesh is ray cast at every fifth view and compared with the
  splat's renders; `compare.png` shows splat and mesh side by side. Direct mode has no views to compare with, so it
  writes neither; look at the mesh in a viewer, or render it (`scripts/athenea_view.sh`).
- `report.json` per input and `summary.json` per batch. `"ok"` means: render path — 0 open edges, ≤ 20
  non-manifold edges, > 99 % texels covered and PSNR ≥ `--min-psnr` (22); direct mode — 0 open edges and ≤ 200
  non-manifold edges.

PSNR is lower on glossy objects (porcelain highlights move with the view; a texture holds one colour): 24 dB on the
porcelain cat is a good result, 26–30 dB is typical of matte objects.

## Extra stages for glass

- **Glass estimate** (`--glass BACKGROUND.json`, `splat2mesh/glass.py`): given views of the object in front of a
  known background (a textured plane or several), refracted rays are traced through the mesh to the background and
  IOR and absorption tint are fitted to the images. The USD material becomes a dielectric. Self-test:
  `tests/glass_selftest.py`.
- **Openings** (`splat2mesh/openings.py`): glass the splat left as a hole (car windows, lantern panes). The object's
  envelope is closed with a large radius; envelope faces far from the real surface are cap candidates, and the
  ones that the segmenter's glass masks see first become a thin-walled `/Asset/Glass` sheet. Runs only when a
  described glass part got almost no texels.
- **Interior** (`splat2mesh/interior.py`): opaque gaussians more than 8 % of the object size under a transmissive
  surface (a filament, a wick, a dip tube) are meshed, unwrapped and baked on their own as `/Asset/Interior`.

## Other entry points

| Command | What |
|---|---|
| `python -m splat2mesh.cli VIEWS_DIR OUT_DIR` | render path from a folder of views you rendered yourself |
| `python -m splat2mesh.evaluate REF.usda RECON.ply` | Chamfer, F-score against a reference mesh |
| `python -m splat2mesh.interior ASSET_DIR SPLAT.ply [--y-down]` (Idol splats need `--y-down`) | the interior stage on its own |
| `scripts/curio_post.py ASSET.usda OUT` | optional quad retopology with CurioMesh, run as a separate program (GPL-3.0); UVs remade and every texture carried over texel by texel |
