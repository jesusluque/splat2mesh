"""Build a static web gallery of splat2mesh results: per model a GLB (browser / <model-viewer>), a USDZ
(AR Quick Look) and the source image; index.html shows them with <model-viewer>.

    python scripts/export_web.py OUTDIR NAME=ASSET_DIR[:TITLE[:SOURCE_IMAGE]] [...]
ASSET_DIR holds asset.usda + basecolor.png (+ report.json in it or its parent).
"""
import html, json, shutil, sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from pxr import Usd, UsdGeom, UsdUtils

out = Path(sys.argv[1]); (out / "models").mkdir(parents=True, exist_ok=True)
cards = []
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    src = Path(parts[0]); title = parts[1] if len(parts) > 1 and parts[1] else name
    img = Path(parts[2]) if len(parts) > 2 and parts[2] else None
    st = Usd.Stage.Open(str(src / "asset.usda"))
    m = UsdGeom.Mesh(st.GetPrimAtPath("/Asset/Geom"))
    V = np.asarray(m.GetPointsAttr().Get(), np.float64)
    F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
    uv = np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar("st").Get(), np.float64)
    tex = Image.open(src / "basecolor.png").convert("RGB")
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.6)
    mesh = trimesh.Trimesh(V, F, visual=trimesh.visual.TextureVisuals(uv=uv, material=mat), process=False)
    # centre on the floor, scale to ~0.3 m tall so viewers frame it the same way
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    mesh.apply_scale(0.3 / max(hi[1] - lo[1], 1e-9))
    mesh.export(out / "models" / f"{name}.glb")
    UsdUtils.CreateNewARKitUsdzPackage(str((src / "asset.usda").resolve()), str((out / "models" / f"{name}.usdz").resolve()))
    if img and img.exists():
        Image.open(img).convert("RGB").resize((512, 512), Image.LANCZOS).save(out / "models" / f"{name}_input.jpg", quality=85)
    rep = {}
    for r in (src / "report.json", src.parent / "report.json"):
        if r.exists():
            j = json.loads(r.read_text()); rep = j.get("mesh", j); rep["psnr"] = j.get("qa", {}).get("psnr_mean"); break
    cards.append((name, title, bool(img and img.exists()), rep, len(F)))
    print(name, len(F), "faces")

items = []
for name, title, has_img, rep, nf in cards:
    stats = f"{nf:,} faces · {rep.get('uv_charts', '?')} UV charts"
    if rep.get("boundary_edges") == 0 and rep.get("non_manifold_edges") == 0:
        stats += " · closed manifold"
    if rep.get("psnr"):
        stats += f" · PSNR {rep['psnr']:.1f} dB"
    src_img = f'<img class="src" src="models/{name}_input.jpg" alt="input image">' if has_img else ""
    items.append(f'''<article>
  <model-viewer src="models/{name}.glb" ios-src="models/{name}.usdz" camera-controls auto-rotate ar
                shadow-intensity="0.6" exposure="1.0" environment-image="neutral" alt="{html.escape(title)}"></model-viewer>
  <div class="meta">{src_img}<div><h2>{html.escape(title)}</h2><p>{stats}</p>
  <p class="dl"><a href="models/{name}.glb" download>GLB</a> · <a href="models/{name}.usdz" download>USDZ</a></p></div></div>
</article>''')

(out / "index.html").write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>splat2mesh gallery</title>
<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>
:root {{ --bg:#f6f5f2; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e4e2dd; --accent:#b8322a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#eee; --muted:#a0a0a6; --card:#1f1f21; --line:#333; --accent:#ff6b5e; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,sans-serif }}
header {{ padding:32px 16px 8px; max-width:1200px; margin:auto }} h1 {{ margin:0 0 4px; font-size:28px }}
header p {{ margin:0; color:var(--muted) }} header a {{ color:var(--accent) }}
main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; padding:16px; max-width:1200px; margin:auto }}
article {{ background:var(--card); border:1px solid var(--line); border-radius:14px; overflow:hidden }}
model-viewer {{ width:100%; height:340px; background:linear-gradient(#fff0,#0000000a) }}
.meta {{ display:flex; gap:12px; padding:12px 14px; align-items:flex-start }} .src {{ width:64px; height:64px; object-fit:cover; border-radius:8px; border:1px solid var(--line) }}
h2 {{ font-size:16px; margin:0 0 2px }} .meta p {{ margin:0; color:var(--muted); font-size:13px }} .dl a {{ color:var(--accent) }}
footer {{ color:var(--muted); font-size:13px; padding:16px; max-width:1200px; margin:auto }}
</style></head><body>
<header><h1>splat2mesh</h1>
<p>3D Gaussian splats → UV-textured meshes (USD / GLB). Image → <b>Idol</b> splat → <b>splat2mesh</b> mesh.
Drag to orbit · AR on phones · <a href="https://github.com/jesusluque/splat2mesh">code (Apache-2.0)</a></p></header>
<main>
{chr(10).join(items)}
</main>
<footer>Textures are baked colour only (no gloss yet). Generated with Iris-3B Turbo / Idol (speridlabs) and splat2mesh.</footer>
</body></html>
''')
print("site:", out / "index.html")
