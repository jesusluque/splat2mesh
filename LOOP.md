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
