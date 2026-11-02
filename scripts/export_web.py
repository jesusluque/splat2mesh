"""Build a static web gallery of splat2mesh results, USD only: per model one USDZ (ASCII .usda inside, texture
paths equal to the archive entry names) that the page shows with three.js' USDZLoader and that iOS opens in
AR Quick Look; plus the source image.

    python scripts/export_web.py OUTDIR NAME=ASSET_DIR[:TITLE[:SOURCE_IMAGE]] [...]
ASSET_DIR holds asset.usda + its textures (+ report.json in it or its parent).
"""
import html, json, shutil, sys, tempfile
from pathlib import Path
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils


def make_usdz(src: Path, dst: Path) -> dict:
    """Copy asset.usda + textures into a temp dir, rewrite texture paths to bare file names, package as USDZ."""
    tmp = Path(tempfile.mkdtemp())
    stage = Usd.Stage.Open(str(src / "asset.usda"))
    stage.GetRootLayer().Export(str(tmp / "asset.usda"))
    st = Usd.Stage.Open(str(tmp / "asset.usda"))
    info = {"pbr": False}
    for prim in st.Traverse():
        if prim.GetTypeName() == "Shader":
            sh = UsdShade.Shader(prim)
            f = sh.GetInput("file")
            if f and f.Get():
                p = Path(f.Get().path)
                srcfile = (src / p).resolve()
                shutil.copy(srcfile, tmp / srcfile.name)
                f.Set(Sdf.AssetPath(srcfile.name))
                if srcfile.name in ("roughness.png", "metallic.png"):
                    info["pbr"] = True
    # place on the floor, Y up, ~0.3 m tall, so every model is framed alike
    geom = UsdGeom.Mesh(st.GetPrimAtPath("/Asset/Geom"))
    lo, hi = geom.GetExtentAttr().Get()
    s = 0.3 / max(hi[1] - lo[1], 1e-9)
    xf = UsdGeom.Xformable(st.GetPrimAtPath("/Asset"))
    xf.ClearXformOpOrder()
    t = Gf.Vec3d(-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2)
    M = Gf.Matrix4d(1.0).SetScale(s)                       # one matrix op: three's USDZLoader reads only xformOp:transform
    M.SetTranslateOnly(t * s)                              # row vectors: p' = (p + t) · s
    xf.AddTransformOp().Set(M)
    st.GetRootLayer().Save()
    if dst.exists():
        dst.unlink()
    UsdUtils.CreateNewUsdzPackage(str(tmp / "asset.usda"), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    return info


out = Path(sys.argv[1]); (out / "models").mkdir(parents=True, exist_ok=True)
cards = []
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    src = Path(parts[0]); title = parts[1] if len(parts) > 1 and parts[1] else name
    img = Path(parts[2]) if len(parts) > 2 and parts[2] else None
    info = make_usdz(src, out / "models" / f"{name}.usdz")
    if img and img.exists():
        Image.open(img).convert("RGB").resize((512, 512), Image.LANCZOS).save(out / "models" / f"{name}_input.jpg", quality=85)
    rep = {}
    for r in (src / "report.json", src.parent / "report.json"):
        if r.exists():
            j = json.loads(r.read_text()); rep = dict(j.get("mesh", j)); rep["psnr"] = j.get("qa", {}).get("psnr_mean"); break
    rep["pbr"] = info["pbr"]
    cards.append((name, title, bool(img and img.exists()), rep))
    print(name, "pbr" if info["pbr"] else "colour")

items = []
for name, title, has_img, rep in cards:
    stats = f"{rep.get('faces', 0):,} faces · {rep.get('uv_charts', '?')} UV charts"
    if rep.get("boundary_edges") == 0 and rep.get("non_manifold_edges") == 0:
        stats += " · closed manifold"
    if rep.get("pbr"):
        stats += " · PBR materials"
    if rep.get("psnr"):
        stats += f" · PSNR {rep['psnr']:.1f} dB"
    src_img = f'<img class="src" src="models/{name}_input.jpg" alt="input image">' if has_img else ""
    items.append(f'''<article>
  <div class="view" data-usdz="models/{name}.usdz"></div>
  <div class="meta">{src_img}<div><h2>{html.escape(title)}</h2><p>{stats}</p>
  <p class="dl"><a href="models/{name}.usdz" download>USDZ</a> · <a rel="ar" href="models/{name}.usdz">AR (iOS)</a></p></div></div>
</article>''')

(out / "index.html").write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>splat2mesh gallery</title>
<script type="importmap">{{"imports": {{"three": "https://unpkg.com/three@0.170.0/build/three.module.js",
 "three/addons/": "https://unpkg.com/three@0.170.0/examples/jsm/"}}}}</script>
<style>
:root {{ --bg:#f6f5f2; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e4e2dd; --accent:#b8322a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#eee; --muted:#a0a0a6; --card:#1f1f21; --line:#333; --accent:#ff6b5e; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,sans-serif }}
header {{ padding:32px 16px 8px; max-width:1200px; margin:auto }} h1 {{ margin:0 0 4px; font-size:28px }}
header p {{ margin:0; color:var(--muted) }} header a {{ color:var(--accent) }}
main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; padding:16px; max-width:1200px; margin:auto }}
article {{ background:var(--card); border:1px solid var(--line); border-radius:14px; overflow:hidden }}
.view {{ position:relative; width:100%; height:340px; background:linear-gradient(#ffffff00,#0000000d) }} .view canvas {{ display:block; width:100%; height:100% }}
.view .msg {{ position:absolute; inset:0; display:grid; place-items:center; color:var(--muted); font-size:13px }}
.meta {{ display:flex; gap:12px; padding:12px 14px; align-items:flex-start }} .src {{ width:64px; height:64px; object-fit:cover; border-radius:8px; border:1px solid var(--line) }}
h2 {{ font-size:16px; margin:0 0 2px }} .meta p {{ margin:0; color:var(--muted); font-size:13px }} .dl a {{ color:var(--accent) }}
footer {{ color:var(--muted); font-size:13px; padding:16px; max-width:1200px; margin:auto }}
</style></head><body>
<header><h1>splat2mesh</h1>
<p>3D Gaussian splats → UV-textured <b>USD</b> meshes. Image → <b>Idol</b> splat → <b>splat2mesh</b>. Each viewer below renders the USDZ
itself (three.js USDZLoader); on iPhone/iPad the AR link opens it in Quick Look. <a href="https://github.com/jesusluque/splat2mesh">Code (Apache-2.0)</a></p></header>
<main>
{chr(10).join(items)}
</main>
<footer>Models marked “PBR materials” carry inferred roughness/metallic maps (Qwen3-VL); the others are baked colour only.
Generated with Iris-3B Turbo / Idol (speridlabs) and splat2mesh.</footer>
<script type="module">
import * as THREE from "three";
import {{ USDZLoader }} from "three/addons/loaders/USDZLoader.js";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ RoomEnvironment }} from "three/addons/environments/RoomEnvironment.js";
const loader = new USDZLoader();
function view(el) {{
  const msg = document.createElement("div"); msg.className = "msg"; msg.textContent = "loading USDZ…"; el.appendChild(msg);
  const r = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
  r.setPixelRatio(Math.min(devicePixelRatio, 2)); r.toneMapping = THREE.ACESFilmicToneMapping; r.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(r.domElement);
  const scene = new THREE.Scene();
  scene.environment = new THREE.PMREMGenerator(r).fromScene(new RoomEnvironment(), 0.04).texture;
  const cam = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
  const ctl = new OrbitControls(cam, r.domElement); ctl.enableDamping = true; ctl.autoRotate = true; ctl.autoRotateSpeed = 1.2;
  const resize = () => {{ const w = el.clientWidth, h = el.clientHeight; r.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }};
  new ResizeObserver(resize).observe(el); resize();
  loader.load(el.dataset.usdz, (g) => {{
    msg.remove(); scene.add(g);
    const box = new THREE.Box3().setFromObject(g), c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()).length();
    ctl.target.copy(c); cam.position.copy(c).add(new THREE.Vector3(0.35, 0.18, 1).normalize().multiplyScalar(s * 1.6));
    cam.near = s / 100; cam.far = s * 20; cam.updateProjectionMatrix();
  }}, undefined, (e) => {{ msg.textContent = "could not load " + el.dataset.usdz; console.error(e); }});
  let visible = false; new IntersectionObserver(([en]) => visible = en.isIntersecting).observe(el);
  r.setAnimationLoop(() => {{ if (!visible) return; ctl.update(); r.render(scene, cam); }});
}}
document.querySelectorAll(".view").forEach(view);
</script>
</body></html>
''')
print("site:", out / "index.html")
