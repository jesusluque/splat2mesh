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
