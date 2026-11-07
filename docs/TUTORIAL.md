# Tutorial: from a picture to a production USD with PBR materials

splat2mesh turns a 3D Gaussian splat into a **UV-textured USD mesh with physically based materials**, with no
manual step. This tutorial runs the whole chain; the step-by-step basics (direct mode, the UV map, the render path)
are in [TUTORIAL_basics.md](TUTORIAL_basics.md).

```
text ──Iris-3B──▶ picture ──Idol──▶ 3D Gaussian splat ──splat2mesh──▶ HD mesh + UVs + texture
                                                                    │
       Qwen3-VL: "gold: the coin; porcelain: the body…" ◀───────────┤  which parts, which material
       ENEAS: tracks each part on a 16-view orbit        ◀───────────┤  where each part is, in 3D
       material library: roughness, metallic, IOR, F0, SSS ─────────▶ USD (UsdPreviewSurface + MaterialX)
                                                                       + one Mesh per part (Cryptomatte)
```

<img src="img/gallery.jpg" width="100%">

*33 test objects (porcelain, metals, wood, lacquer, stone, jade, wax, glass, leather and mixes); every material
assigned by the pipeline. Path traced with athenea under a studio HDRI (polished metals need something to
reflect), composited on grey.*

## 1. Install

```sh
git clone https://github.com/jesusluque/splat2mesh.git && cd splat2mesh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/synthetic.py            # end-to-end check, no GPU
```

The GPU pieces and how the code reaches them are in [setup.md](setup.md):

| Piece | What for | Where it ran here |
|---|---|---|
| [Idol](https://github.com/speridlabs/idol-sdk) (speridlabs) | picture → 3D Gaussian splat | CUDA, 8 GB (RTX 3070) or 24 GB (L4) |
| [Iris-3B](https://github.com/speridlabs/iris-3b) (speridlabs) | text → picture; its text encoder **Qwen3-VL-4B** describes materials | Apple Silicon (MPS, bf16) |
| [ENEAS](https://github.com/speridlabs/eneas) (speridlabs) | text → part masks tracked across views | CUDA in WSL or Linux |
| athenea | orbit frames, path-traced stills and turntables; renders for the render path | Metal |

## 2. One object, one command

```sh
mkdir -p pictures && cp lucky_cat.png pictures/00.png        # the materials flow (step 3) wants numbered names
.venv/bin/python scripts/splat2usd.py pictures/ --direct --gaussians 524288 --out out/demo
```

1. **Idol** turns the picture into a splat. It invents the hidden sides: use a white background.
2. **Direct mode** builds the mesh straight from the gaussians: opacity field on a 448³ grid → solid (gaps closed,
   cavities filled) → marching cubes → topology-preserving decimation to **80 000 faces**, one closed shell.
3. If the surface comes out crumpled (more than **300** folded edges: black lacquer walls, faceted glass), it
   **falls back to the render path** by itself: 87 rendered views, TSDF fusion and visual hull.
4. **xatlas** UVs; colour baked from the gaussians into a **4096²** atlas.

Details and the reasons for each default: [pipeline.md](pipeline.md).

## 3. Materials, automatically

```sh
.venv/bin/python scripts/flow_batch.py out/demo out/demo_pbr prompts.txt     # a folder of objects
```

The source folder holds one subfolder per object named by number (`00`, `01`…, as `splat2usd.py` writes them for
pictures `00.png`, `01.png`…), and line N of `prompts.txt` is object N's prompt (the one the picture was made from,
or a short description). For each object:

1. **Qwen3-VL describes** the parts by material, with the library's names: `glass: the bulb; brass: the screw base`.
   The prompt has the last word on the specific material ("gold" becomes "brass" if the prompt says brass).
2. **ENEAS tracks every part** on a 16-view orbit of the textured mesh, starting from the view that matches the
   picture. Masks are projected back onto the texels and voted.
3. **Overlaps are resolved** by colour: masks that claim the same pixels are settled by reliable small masks first,
   then by colour clusters that a vision model names.
4. **Sanity rules and colour regions** catch the segmenter's mistakes: borders follow colour edges, a part that
   covers most of the object is dropped, and colour priors recover kintsugi lines, gilt rims, chrome and rubber.
5. **The library** (58 materials) gives roughness, metallic, IOR, metal F0, transmission and subsurface.
6. **A finish pass** makes glossy pieces look finished rather than scanned. Polished metals take their physical
   colour plus the fine detail of the texture, highlights are removed from glossy paint and porcelain, shading
   normals are smoothed with edges kept, the invented side is brightened to match the seen side, and glass gets
   one tint or keeps its stained colours.
7. **Glass extras**: glass the splat left as a hole becomes a thin glass sheet; opaque things inside glass
   (filaments, wicks) are meshed on their own.

All the rules and thresholds: [materials.md](materials.md).

What lands in the USD:

| Map | Source |
|---|---|
| base colour | the baked splat colour; polished metals use the physical F0 colour; glass a uniform tint |
| roughness, metallic | library, per texel, polish read from the texture |
| transmission, subsurface | library (glass, crystal; jade, wax, marble, porcelain) |
| opacity, IOR | for `UsdPreviewSurface` viewers |

Two surfaces per material: `UsdPreviewSurface` (every viewer) and a **MaterialX `standard_surface`** with
transmission and subsurface (athenea, Karma, Arnold…).

<img src="img/bulb.jpg" width="320"> <img src="img/jade.jpg" width="320">

## 4. Parts for compositing (Cryptomatte)

Next to `asset.usda` the flow writes:

- `asset_parts.usda`: one `Mesh` per part (`/Asset/Parts/gold_bell`, `/Asset/Parts/red_collar`…), each with its own
  material. Renderers that build CryptoObject from the prim path give one matte per part.
- `asset_materials.usda`: one `Mesh` per material (`/Asset/MaterialGroups/gold`…), one matte per material.

## 5. Viewing and sharing

- **Web**: `scripts/export_web.py` builds a gallery that uses USD only. Each card shows a path-traced still and loads
  the USDZ on click (three.js `USDZLoader`, four CC0 HDRIs). By default the viewer reads the per-texel roughness,
  metalness and transmission maps, so glass and metal borders are as fine as the texture. iPhone opens the same
  USDZ in AR. Live: <https://splat2mesh.lucab.co.uk>.
- **Video**: `scripts/athenea_turntable.sh asset.usda out.mp4` path traces a turntable;
  `scripts/blender_turntable.py` does the same in Blender.

## 6. Better hidden sides: pick the best seed

Idol's invented sides change from seed to seed. Rendering each sample from four sides and letting a vision model
score it (`scripts/seed_views.sh`, `scripts/mv_check.py`) picks the best one. Over 16 objects the best of 4 seeds
scored 3.67 against 3.21 for a fixed seed. See [multiview.md](multiview.md), which also covers what did not help.

## 7. What still goes wrong

- Idol paints what it cannot see: dark undersides, refraction baked as colour on glass.
- Direct mode fills solids: a vase has no inside, a bottle no liquid surface. What sits inside glass is recovered,
  but as Idol made it: flat ribbons rather than wires.
- Parts below a few pixels (filaments, eyelids, corks) are often missed; colour priors only cover metals and rubber.
- A missed part that is not a metal or rubber (a rosewood stand under jade) takes the main material.
- Dark glass the segmenter misses (car windows) is taken by the near-black rubber prior.
- Library values are catalogue values, not measurements.

What was tried and measured along the way: [findings.md](findings.md).
