# Materials reference

How an asset from [the pipeline](pipeline.md) gets physically based materials: which parts it has, where each part
is on the surface, which material each part is made of, and how the maps are finished so the object looks made
rather than scanned. Everything here is automatic; the only input is the object's picture (and, if there is one,
the text prompt it was generated from).

```
picture (+ prompt) ──Qwen3-VL──▶ "glass: the bulb; brass: the screw base"
                                          │
textured mesh ──16-view orbit──▶ ENEAS tracks each phrase ──▶ masks per view
                                          │
                       overlap resolution (colour clusters, DeepSeek names them)
                                          │
                       back-projection: every texel collects votes per (material, part)
                                          │
                       per-texel rules: colour regions, colour priors, sanity checks
                                          │
                       library values + finish pass ──▶ maps ──▶ USD (PreviewSurface + MaterialX)
```

Entry points:

| Command | What |
|---|---|
| `scripts/flow_batch.py SRC OUT PROMPTS.txt` | the whole flow on a folder made by `splat2usd.py` (direct HD mesh, render-path fallback, description, ENEAS, maps) |
| `scripts/materials_eneas3d.py ASSET_DIR PICTURE "gold: the coin; porcelain: the body"` | one asset with a given description |
| `scripts/describe_materials.py` | only the description step |
| `scripts/reapply_par.py` | rerun the per-texel rules on finished assets (saved masks, no ENEAS) |

## 1. Description: which parts, which material

`scripts/describe_materials.py` runs **Qwen3-VL-4B** (the text encoder of Iris-3B, already on disk) on the picture
and the prompt. It writes one line per material, using the library's names, with the parts made of it. It is asked
for 3–10 parts, starting with the ones that cover most of the object (body, clothing, skin, hair, large panels); on
a painted figurine it names the material the paint represents (steel, wood, leather, skin), not the paint. Its
answer is cleaned: a small model sometimes loops ("the shield rim, the shield edge, the shield rim…"), so repeated
parts are dropped, every part starts with "the" and at most 10 are kept:

```
glass: the bulb; brass: the screw base
porcelain (glazed): the cat's body; gold: the bell, the coin; urushi lacquer: the red collar
```

**The prompt is the authority** on the specific material inside a family (applied by `flow_batch.py`; running
`describe_materials.py` or `materials_eneas3d.py` alone does not apply it). It also names the material of single
parts: a part whose noun the prompt puts right after a material word ("a round grey **stone** base", "with a
**brass** rim") takes that material. The noun must head its phrase: in "gold belt buckle" the gold is the buckle's,
not the belt's. If the prompt says brass and the model
says gold, it becomes brass. If the model names a family the prompt never mentions, it is replaced by the prompt's
unused material (a bowl described as porcelain that the prompt calls urushi lacquer becomes lacquer).

Asking Qwen3-VL for numbers (roughness, colour) did not work: answers were stereotyped (`#d4af37` for any gold) and
changed between crops of the same part. The numbers therefore come from the library and the colour from the
baked texture.

## 2. Where each part is: ENEAS on an orbit

- The textured mesh is rendered unlit from **16 views** on an orbit at 15° elevation, 768 px, together with a map
  of which texel each pixel shows (`materials.render_orbit`).
- **ENEAS** (speridlabs, text → masks, with tracking) grounds every phrase on the orbit frame that looks most like
  the input picture, chosen by comparing object crops, and on the opposite frame. It then tracks the phrase
  through all 16 frames (`scripts/eneas_seq.py`, on the ENEAS host).
- A transparent material is also searched by its own name ("the glass"), because descriptions miss panes. That
  phrase only fills pixels no listed part claims, and it is dropped if it covers more than 30 % of the object.

### On the splat instead of the mesh (`--gs`, recommended)

`flow_batch.py --gs` (or `materials_eneas3d.py … --gs SPLAT.ply`) segments the **splat**, not the textured mesh
(`splat2mesh/gs_materials.py`):

1. The splat is rendered with athenea (`render --splats`) on a 16-view orbit at 15° elevation, 1024 px, on white,
   with depth and OpenCV cameras.
2. The annotation frame is the orbit view nearest to the camera Idol predicted for the picture
   (`idol_camera.json`): the side that was really seen. Colour similarity, used otherwise, picked the back of a
   figure whose silhouette looks the same.
3. ENEAS segments and tracks the phrases on those frames; masks are checked and resolved as below.
4. **Votes per gaussian**: every gaussian centre is projected into every view and, where its depth matches the
   rendered depth (within 1.5 % of the object radius), it takes a vote weighted by its opacity for the phrase
   that covers that pixel. Gaussians no view sees take their neighbours' votes.
5. **Texels inherit** the votes of their 8 nearest gaussians (Gaussian kernel over distance × opacity), so every
   texel has votes; the per-texel rules below then run unchanged.
6. The splat is written back with per-gaussian material attributes (`mat/splat_materials.ply`: `mat_id`,
   `roughness`, `metallic`, `transmission`, `subsurface`, `ior`; names in `mat/mat_names.json`). Any other mesh of
   the same splat can inherit them without running ENEAS again.

On a dwarf figurine, 88 % of the gaussians got a direct vote; with the mesh orbit only 6–16 % of the texels did and
the rest came from colour regions, which spread one metal over the shield, the boots and the base.

Glass openings (`openings.py`) still read the mesh-orbit masks; with `--gs` they are not computed yet.

## 3. Checking and resolving the masks

**Check** (`overlap.verify`). DeepSeek sees the picture and one tile per phrase, with its mask in red on the
annotation frame, and answers yes / partly / no for each. Masks that show another part ("the face" grounded on
helmet and beard) are emptied; their class stays in the list. Answers are the majority of three calls.

**Overlaps**

ENEAS sometimes grounds a phrase on too much: "the dragon's body" on the whole figurine, "the chainmail" on the
whole dwarf. `splat2mesh/overlap.py` splits the object's pixels on the annotation frame into 8 colour clusters
(Lab k-means); DeepSeek names the listed part each cluster belongs to (majority of three calls; parts without a
mask can own a cluster). Then every pixel that several masks claim, or that only unreliable masks claim, is
decided in this order:

1. **A reliable specific mask wins.** A mask is reliable if it is not the largest and does not overlap a bigger one
   by an IoU between 0.5 and 0.9 (above 0.9 two phrases were grounded on the same part: duplicates, not swollen).
   Among reliable claimants, the one owning the pixel's colour cluster wins, otherwise the smallest.
2. **Otherwise colour decides.** The pixel goes to its cluster's part if that part claims it, or if only
   unreliable masks claim the pixel: a part ENEAS missed (a wooden shield inside "the chainmail") is recovered
   from its colour.
3. **Otherwise the smallest claimant** wins.

The cluster sheet is saved as `mat/overlap_clusters.jpg`. Without a DeepSeek key this step is skipped and the
sanity rules below handle overlaps.

## 4. Sanity rules (segmenter mistakes)

- A secondary part covering more than **75 %** of the object is a mislocated phrase and is dropped. A thin part
  (a phrase with the whole word rim, edge, wire, band, cap, cork, chain…; "chainmail" is not a chain) is dropped
  above **15 %**.
- Two phrases of different materials with nearly identical masks (IoU > 0.9) were not separated by ENEAS. The one
  listed first is kept.
- A dropped part keeps its material class, so the colour priors (6) can still claim texels for it.

## 5. From masks to texels

Masks are back-projected through the texel maps; every texel collects votes per (material, part) from every frame
that sees it.

- **Borders follow colour edges** (`refine_by_regions`). The texture is split into colour regions. A region only
  admits the classes that hold at least **25 %** of its votes, plus the classes for which it is a home region
  (the fewest regions that together hold **60 %** of the class's own votes; a steel helmet whose votes split over
  two greys keeps both). Votes then decide among the admitted classes. A mask that
  spills over white porcelain cannot make it gold, and a small part whose mask is mostly right is not voted away
  by a bigger neighbour.
- **Colour priors** recover thin parts the segmenter misses: kintsugi lines and gilt rims (hue of the metal's F0),
  chrome (bright, low-chroma grey), rubber (near black).
- A coloured metal never takes large white patches; they go to the unfound non-metal part.

## 6. The library

`splat2mesh/material_library.json` has **60 materials** with metallic, roughness and its plausible range, IOR, metal
F0 colour, transmission, subsurface, thin-walled and match keywords. Values are catalogue values, not measurements.

| Family | Entries |
|---|---|
| Metals | gold, brass, bronze, copper, silver, polished and brushed steel, chrome, aluminium, brushed aluminium, iron, cast iron, titanium, nickel, platinum, pewter / tin |
| Corroded metal (non-metallic surface) | rusted iron, patinated bronze |
| Ceramics | porcelain (glazed), glazed ceramic, crackle glaze, stoneware (matte glaze), unglazed clay / terracotta |
| Coatings | urushi lacquer, gloss lacquer, varnished wood, enamel, gloss / satin / matte paint, car paint |
| Wood and plant | raw wood, oiled hardwood, bamboo / rattan, cork |
| Plastics and rubber | glossy plastic, matte plastic, vinyl, rubber |
| Transparent | glass, frosted glass, crystal, quartz / gemstone, diamond, amber / resin |
| Stone and translucent | polished marble, honed stone, jade, concrete / plaster, mother-of-pearl, bone / ivory, wax / candle |
| Leather and textiles | leather, patent leather, cotton / linen, silk / satin, velvet / felt, paper / cardboard |
| Characters | skin (subsurface 0.5), hair / fur |

## 7. Finish pass

`materials.finish_maps` and the functions next to it, so glossy objects look finished:

- **Metals** take the library's F0 colour times only the *fine* luminance detail of the baked texture (a high-pass
  of log-luminance at 8 texels, darkening limited to ×0.75). Hammering and engraving stay; the reflections that the
  splat froze into its colour go. Dark strokes up to 4 texels wide are read as ink and stay non-metallic.
- **Polish from the texture.** Some materials can be satin or polished (leather, varnished wood, satin paint): a
  non-metal, non-transparent material whose default roughness is above 0.4 but whose range reaches 0.45 or less.
  If such a piece is one colour (chroma above 15, a\*b\* spread under 10), covers more than 2 000 texels and has
  more than 3 % pale-highlight texels, it is polished: roughness becomes the low end of its range minus 0.1 (at
  least 0.25), and the highlights are removed from the base colour.
- **Highlight removal** on glossy dielectrics (porcelain, lacquer, paint): the baked highlights are taken out of the
  base colour, since the renderer's lights put them back.
- **Shading normals** for glossy parts come from bilateral face-normal filtering (60 passes, edges above ~45° kept);
  the positions do not move. Reflections flow instead of showing the fused surface's small bumps.
- **The invented side.** Where the picture did not look, a material that came out much darker than on the seen
  side is lifted to the seen side's brightness, unless the seen side itself shows that lightness.
- **Glass, crystal, jade and wax** also get their vertex positions smoothed (120 Taubin iterations).

## 8. Glass colour

- One-colour glass (clear, blue, green): one uniform tint. A ray crosses two walls of a glass shell, so the colour
  per wall is the measured tint to the power 1/2.
- Multi-coloured glass (Tiffany shades, stained glass) keeps its baked colours with half transmission. A glass is
  multi-coloured when the spread of its texels in Lab a\*b\* is above 12 (single-colour glass measured 0.5–5.7,
  stained glass 15–20).
- Thin-walled glass parts get specular 0.5. Idol's double-surfaced walls otherwise act as four interfaces and look
  like chrome.
- Glass the splat left as a hole, and what sits inside the glass: see [pipeline.md](pipeline.md#extra-stages-for-glass).

## 9. Output files

In the asset's `usd/` folder:

| File | Content |
|---|---|
| `asset.usda` | one mesh; `UsdPreviewSurface` and MaterialX `standard_surface` (transmission, subsurface, thin-walled) |
| `asset_parts.usda` | one `Mesh` per part (`/Asset/Parts/gold_bell` …), each with its material. Renderers that build CryptoObject from prim paths give one matte per part |
| `asset_materials.usda` | one `Mesh` per material (`/Asset/MaterialGroups/gold` …) |
| `basecolor_delit.png` | the finished base colour the materials use (highlights removed, metals and glass finished) |
| `basecolor.png` | the raw bake from the splat, kept for reference |
| `roughness.png`, `metallic.png`, `orm.png` | maps; `orm.png` packs them glTF-style: R = 1, G = roughness, B = metallic |
| `transmission.png`, `transmission_color.png`, `subsurface.png`, `opacity.png` | maps for transparent and translucent parts |
| `parts.json`, `material_groups.json` | the part and material list, with glass tints |
| `mat/` | intermediate data: orbit frames, masks (`masks.npz`, `masks_used.npz`), texel classes, sheets |

## 10. Web gallery

`scripts/export_web.py OUTDIR NAME=ASSET_DIR[:TITLE[:SOURCE[:POSTER]]] …` builds a static gallery that uses USD
only:

- Each model is one USDZ (ASCII `.usda` inside), loaded on click with three.js' `USDZLoader`. iPhone opens the same
  file in AR Quick Look.
- `--materials texture` (default) keeps one mesh, and a small JSON gives the viewer the roughness, metalness and
  transmission maps. Borders are as fine as the texture.
- `--materials parts` uses one mesh per part. Each face takes the part of its centre texel, so borders are jagged
  on small features (it showed as broken glass triangles around a dragon's eye).
- Four CC0 HDRIs from Poly Haven can be switched; at most four live viewers at a time.
