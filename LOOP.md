# /loop 11h — direct mode + UV + materials (2026-11-02 18:59 → 2026-11-03 05:59)

Goal: best possible `--direct` (geometry + UV), and PBR materials inferred with Iris-3B help.
Test set: porcelain, metal, wood, lacquer, mixes. GPU on the Mac only through shared.sh, one job at a time.

## Benchmarks (keep comparable)
- cat (Idol, white bg): `out/auto/maneki_white/maneki_white.ply` (y_down) — QA vs its 87 athenea renders.
- render-path baseline for cat: PSNR 24.1, comps 1?, nm 0, charts 243, 12 min.
- direct v1 (r320 l0.3, solid fill): PSNR 26.3, comps 58, nm 89, charts 1267, 41 s.

## Plan
1. Direct topology: one closed manifold shell after decimation (comps → 1, nm → 0).
2. UV: fewer, larger charts at same atlas use (xatlas chart options; measure charts, stretch, fill).
3. Colour bake sharpness in direct mode (texel → gaussians weighting).
4. Test set via Iris (local) → Idol (ixwin) → direct: porcelain, metal, wood, lacquer, mixed.
5. Materials: prompt-informed PBR (text says the materials) + colour clustering + highlight removal → roughness/metallic maps; check by athenea relit render.

## Log
- 19:20 direct v2: topology-preserving decimation (Open3D quadric, fs agg3 fallback) → cat: 1 component, watertight,
  0 non-manifold, 108 UV charts (v1 1267), PSNR 27.05 (render path 24.1), 39 s.
- 19:35 Iris coherence. Same seed + lighting prompt from scratch: structure changes (tail/coin/paw move; edge-SSIM 0.36).
  img2img (start from base at σ): σ=0.833 edge-SSIM 0.89, σ=0.625 0.96, but lighting barely changes (Δlum 0.01–0.02).
  Turbo's 4-step schedule leaves no middle ground at 0.833→0.625. Next: σ=0.938, embedding mix, and "flat lighting/no
  highlights" variant as a delit albedo candidate.
- 19:55 Test set (Iris local, seed 11, white bg): porcelain teapot, brass candlestick, walnut owl, red lacquer bowl,
  mixed lamp (oak/steel/porcelain), black lacquer box (brass, nacre), aluminium robot, celadon crackle vase with kintsugi.
- 20:05 UV sweep on direct cat mesh (scripts/uv_sweep.py): xatlas defaults 103 charts, util 0.57, stretch σlog 0.096;
  max_cost ≥ 4 → 109 charts, stretch 0.081 (slightly better); other weights worse. Chart count is already low on the
  smooth direct mesh: the big UV win came from the clean decimated shell, not from xatlas tuning.
- User: commit + push as we go.
- 20:15 Owl (walnut, carved): 1284 charts. 773 charts have ≤4 faces: xatlas cuts at folds (981 dihedral angles >120°)
  of a crumpled surface. Smoothed-proxy unwrapping: barely fewer charts, stretch explodes (rejected). Blurring the
  solid before marching cubes removes folds: cat blur 0.8→1.2: folds 34→6, charts 108→65, PSNR 27.05→26.86.
  New default blur 1.2. Owl at 1.2: 990 charts (real carved detail). xatlas max_cost 4 was worse inside the pipeline
  (308 vs 108 charts on the cat; the uv_sweep ran on a seam-split mesh) → back to xatlas defaults.
- 20:10 Materials v1. Colour clusters (Lab k-means, ΔE merge) of the baked texture → overlay/crop per cluster →
  Qwen3-VL-4B (Iris's text encoder, local, MPS). Multiple-choice/free-text prompts collapsed to "porcelain" for all
  regions (prompt bias). Yes/no probing per material on the zoomed crop works: gold 0.98, red lacquer/glaze 0.998,
  white ceramic 0.85. apply(): soft per-texel assignment → roughness/metallic maps, highlight removal on glossy
  dielectrics, USD wired (roughness/metallic UsdUVTexture). athenea path-traced matte vs PBR: gold coin/bell metallic.
  athenea rt fails (ShaderFailure, internal compiler error) on our plain UsdPreviewSurface with constant float
  roughness/metallic; textured inputs work → reported.
- ENEAS (speridlabs segmentation): HF Space needs PRO; Windows native blocked by App Control (spaCy DLL); installing
  in ixwin WSL Ubuntu-24.04 (uv sync).
- 21:00 Web is USD-only (three.js USDZLoader; ASCII usdz, bare texture names, single matrix xformOp). Verified in Node.
- 21:10 User: owl and lamp broken on the web. Cause: low-opacity gaussians on Idol's back/side fall under the solid
  threshold (0.3) → craters. Level 0.15: owl clean (folds 228→9, charts 990→91), lamp base fixed (shade flap remains,
  from the splat). Cat at 0.15: PSNR 26.86→26.96, charts 65→148. New default level 0.15. Rebuilding test set (v3).
- ENEAS in ixwin WSL works: first call 28 min (download 9 GB), then ~13 s per phrase. One instance per phrase
  ("bell and coin" → bell only). Overlapping masks: smallest wins. scripts/materials_eneas.py = semi-automatic
  materials from a short description "material: part, part; …".
- 21:30 Material library: 58 PBR presets (metallic, roughness, range, ior, metal F0, keywords) — catalogue values.
- 21:40 ENEAS 3D: 24-view orbit of the textured mesh (white bg, texel index per pixel), phrases tracked across views,
  masks back-projected and voted per texel. First run grounded on the BACK view (bad frame choice) → bell = whole cat.
  Fixed: annotation frame chosen by colour match of object crops (front = frame 6). Rerunning.
- Test set v3 (level 0.15): owl and lamp base fixed; published (web v3). VLM labels still wrong for owl (no wood) and
  black lacquer box (→ gold): cases for ENEAS 3D with a description.

## /loop 2 — until 06:00 (set 21:15)
Complete, coherent result with the segmentation flow (ENEAS 3D / Qwen3-VL); more test objects if we run out. At the end:
USDZ for every model, everything committed + pushed, complete tutorial, and an artifact for a LinkedIn post with video.
- 21:35 Qwen3-VL writes per-part material descriptions from image + Iris prompt using library names
  (scripts/describe_materials.py): owl → varnished wood, box → urushi + mother-of-pearl + brass, lamp → wood/steel/glazed.
- ENEAS 3D on cat with the front annotation frame: bell, coin, collar (all 24 views), ears coherent; "golden flower
  branches" → red flowers; "body" → lower body only. Texel share porcelain 68 %, gold 22 %, lacquer 9.5 %.
- Qwen3-VL direct PBR numbers (scripts/qwen_pbr_probe.py): stereotyped (#d4af37 for any gold, #ffffff), unstable
  between crops, red collar → gold metal / white ceramic, glazed porcelain roughness 0.7–0.8. Not usable → numbers
  stay from the library, colour from the measured texture.
- 21:55 Lamp fixed at grid 448 (v4). Lacquer box breaks in direct mode (walls caved, 710 folds; blur 2.0 does not
  help) → automatic fallback: --direct uses the render path when folds > 150 (others: 4–18). Box → render path.
- Toaster and car images regenerated (seed 47) before Idol reached them.
- Web viewer: 4 CC0 Poly Haven HDRIs (studio, photo studio, sunset, sky), buttons + auto-cycle every 8 s.
- 22:15 User: odd stains (screenshot) at material borders. Cause: ENEAS phrase masks spill (flowers, porcelain
  around gold branches) → metallic halos. Fix: refine_by_regions — each colour region only allows material classes
  with ≥25 % of its votes; texel votes decide among them. Gold now follows the branches exactly.
- Library: transmission / subsurface / thin_walled; finish_maps: metals' base colour = library F0 (polished metals:
  F0 only — the baked colour is frozen reflections; matte metals keep 0.35 of the texture detail; dark ink on metal
  stays non-metal), glass: uniform tint; transmission/subsurface/opacity maps; MaterialX standard_surface (mtlx
  context, srgb_texture) next to UsdPreviewSurface. athenea renders it (porcelain SSS, gold).
- HD: 80K faces, 4096² atlas (chunked bake; 153 s; RSS was 15 GB before chunking). Orbit 16 views for ENEAS.
- 22:38 Autonomous night run (scripts/night_run.sh): Idol test set 3 (bulb, wine glass, perfume, candle, jade
  bangle, blue bottle, silver teapot, lantern) → flow for sets 1–3 into out/*_v5 (render fallback, Qwen parts,
  ENEAS 3D, PBR, Cryptomatte parts).
- 23:50 athenea fixed the rt --denoise failure on constant PreviewSurface inputs (main-vdb 35a6c5a, build athenea-ptaux); hair-run still lacks it — set ATHENEA_BUILD=~/luc/athenea-ptaux/build/macos-arm64-release if needed.
- 00:05 Flow running on render94 (ENEAS) + ixwin (Idol set 3) + render94 (Idol set 4, fixed watch/telescope images).
  Fixes: apostrophes in part phrases (owl: masks not found → crash); fallback when ENEAS finds nothing; prompt is the
  authority on the specific material ("gold" → "brass"; family missing from the prompt → the prompt's unused
  material, e.g. red lacquer bowl described as porcelain → urushi lacquer); keyword matching by whole word
  ("pla" no longer matches "plain"). Owl and bowl queued for redo.
- 01:10 DISK FULL on the Mac (159 MB free of 926 GB; project ~6 GB, the rest is other data) → silver teapot and
  lantern failed in set 3. Freed ~5 GB of superseded outputs (v2–v4, pawn/2DGS tests, intermediate EXR views);
  texel_class.npy now int16 (134 MB → 17 MB per object). Redo of both on ixwin queued. User warned.
- Video queue bug: the turntable script read stdin and ate the heredoc list ("ine_glass") → children now < /dev/null.
- 01:30 Sets 1–3 done; set 4 flow on render94. Extra automatic rules: thin parts (rim, wire, hinge…) capped at 15 %
  of the object; secondary part > 50 % dropped; missing ENEAS frames padded; colour priors for unfound grey metals
  (low chroma, bright), coloured metals (F0 hue) and rubber (near black). Redos queued: owl, bowl, box, vase,
  cloisonné, boot, car. Web v6: 33 objects. Disk ~4 GB free.
- 02:10 All 4 sets + redos done. Results of the last rules: owl → varnished wood; bowl → urushi + gold rim (colour
  prior); vase → crackle glaze + kintsugi gold; boot → leather + rubber sole; car → paint + chrome bumpers + rubber
  tyres; Tiffany → coloured glass shade 73 % + bronze; cloisonné → enamel/copper/bronze. Dropped masks keep their
  class (empty) so colour priors can still claim it. Transparent cap only for colourless glass (median chroma ≤ 20);
  reassigned bright texels prefer a non-metal. Known misses: watch crystal over the dial (glass 38 %), lacquer box
  nacre panel, jade dragon's rosewood stand.
- 02:25 Disk full again (0–2 GB; something outside the project keeps filling it). Set 5 images generated (data/testset5) but not processed; priority to the final deliverables.
- 03:35 Car windows: tried splitting colour regions by height so dark windows on top would not become rubber — reverted: object-specific rule, not generic (user: «eso es hacer trampa»). Car windows stay a known limitation: ENEAS does not find them, the dark-colour prior takes them as rubber.
