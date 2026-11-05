"""Build a static web gallery of splat2mesh results, USD only: per model one USDZ (ASCII .usda inside, texture
paths equal to the archive entry names) that the page shows with three.js' USDZLoader and that iOS opens in
AR Quick Look; plus the source image.

    python scripts/export_web.py OUTDIR NAME=ASSET_DIR[:TITLE[:SOURCE_IMAGE[:POSTER]]] [...] [--materials texture|parts]
--materials texture (default): one mesh, materials per texel (transmission, roughness, metalness maps); parts: one
mesh prim per part with its own material (faces take their centre texel's part: jagged borders on small features).
ASSET_DIR holds asset.usda + its textures (+ report.json in it or its parent).
"""
import html, json, shutil, sys, tempfile
from pathlib import Path
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils


def _decimate_web(stage, keep: float):
    """Web copy only: decimate every Mesh to `keep` of its faces, keeping its UV charts. The mesh is decimated
    with the UV seams left open (seam duplicates are separate vertices, so no edge crosses a chart border) and
    every surviving vertex takes the UV of the original vertices collapsed into it; normals are recomputed."""
    import numpy as np, trimesh, fast_simplification as fs
    from pxr import UsdGeom, Vt
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        m = UsdGeom.Mesh(prim)
        V = np.array(m.GetPointsAttr().Get(), np.float64)          # writable copies: the binding refuses views
        F = np.array(m.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3).copy()
        pv = UsdGeom.PrimvarsAPI(m).GetPrimvar("st")
        if len(F) < 2000 or not pv or pv.GetInterpolation() != UsdGeom.Tokens.vertex:
            continue
        st = np.asarray(pv.Get(), np.float64)
        _, _, col = fs.simplify(V, F, 1 - keep, return_collapses=True)
        V2, F2, mp = fs.replay_simplification(V, F, col)
        ok = mp >= 0
        uv = np.zeros((len(V2), 2)); cnt = np.zeros(len(V2))
        first = np.full(len(V2), -1)
        idx = np.nonzero(ok)[0]
        first[mp[idx[::-1]]] = idx[::-1]                         # one original vertex per survivor
        uv = st[np.where(first >= 0, first, 0)]
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(V2, np.float32)))
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(F2), 3, np.int32)))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.asarray(F2, np.int32).ravel()))
        nrm = np.asarray(trimesh.Trimesh(V2, F2, process=False).vertex_normals, np.float32)
        m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(nrm)); m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
        for sub in prim.GetChildren():                          # face subsets no longer match: drop them
            if sub.GetTypeName() == "GeomSubset":
                stage.RemovePrim(sub.GetPath())


def _quantize_web(stage):
    """Web copy only: round the ASCII geometry to what is visible. Normals to 3 decimals (< 0.1°), UVs to
    1/20000 (a third of a texel at 6144²), points to 1e-5 of the object size. No vertex moves, so no seam opens."""
    import numpy as np
    from pxr import UsdGeom, Vt
    rng = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    size = max((rng.GetMax() - rng.GetMin()).GetLength(), 1e-9)
    q = 10 ** np.floor(np.log10(size * 1e-5))
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        m = UsdGeom.Mesh(prim)
        V = m.GetPointsAttr().Get()
        if V:
            m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy((np.round(np.asarray(V) / q) * q).astype(np.float32)))
        N = m.GetNormalsAttr().Get()
        if N:
            m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.round(np.asarray(N), 3).astype(np.float32)))
        pv = UsdGeom.PrimvarsAPI(m).GetPrimvar("st")
        if pv and pv.Get():
            pv.Set(Vt.Vec2fArray.FromNumpy((np.round(np.asarray(pv.Get()) * 20000) / 20000).astype(np.float32)))


def make_usdz(src: Path, dst: Path, usd_name: str = "asset.usda") -> dict:
    """Copy asset.usda + textures into a temp dir, rewrite texture paths to bare file names, package as USDZ."""
    tmp = Path(tempfile.mkdtemp())
    stage = Usd.Stage.Open(str(src / usd_name))
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
                name = srcfile.name
                im = Image.open(srcfile)
                if name.startswith(("basecolor", "interior")):           # colour: PNG (three's USDZLoader reads PNG only), ≤ 3072²
                    if not (tmp / name).exists():
                        w = min(im.width, 3072)
                        im.convert("RGB").resize((w, w), Image.LANCZOS).save(tmp / name, optimize=True)
                else:                                      # data maps: half size is plenty
                    if not (tmp / name).exists():
                        w = max(im.width // 2, 512)
                        im.resize((w, w), Image.BILINEAR).save(tmp / name, optimize=True)
                f.Set(Sdf.AssetPath(name))
                if srcfile.name in ("roughness.png", "metallic.png"):
                    info["pbr"] = True
    # three's USDZLoader resolves a material only inside the Xform it is building: copy each part's material
    # under its own Xform and rebind (web copy only; the master USD keeps /Asset/Materials)
    for grp in ("/Asset/Parts", "/Asset/MaterialGroups"):
        gp = st.GetPrimAtPath(grp)
        if not gp:
            continue
        layer = st.GetRootLayer()
        for xf_prim in gp.GetChildren():
            geom = xf_prim.GetChild("Geom")
            if not geom:
                continue
            rel = UsdShade.MaterialBindingAPI(geom).GetDirectBindingRel()
            tg = rel.GetTargets()
            if not tg:
                continue
            mdst = xf_prim.GetPath().AppendChild("Mat")
            Sdf.CopySpec(layer, tg[0], layer, mdst)
            # fix internal connections that still point at the old material path
            for p in Usd.PrimRange(st.GetPrimAtPath(mdst)):
                for a in p.GetAttributes():
                    srcs = a.GetConnections()
                    if srcs:
                        a.SetConnections([Sdf.Path(str(c).replace(str(tg[0]), str(mdst))) for c in srcs])
            rel.SetTargets([mdst])
    # place on the floor, Y up, ~0.3 m tall, so every model is framed alike
    rng = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(st.GetPrimAtPath("/Asset")).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    s = 0.3 / max(hi[1] - lo[1], 1e-9)
    xf = UsdGeom.Xformable(st.GetPrimAtPath("/Asset"))
    xf.ClearXformOpOrder()
    t = Gf.Vec3d(-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2)
    M = Gf.Matrix4d(1.0).SetScale(s)                       # one matrix op: three's USDZLoader reads only xformOp:transform
    M.SetTranslateOnly(t * s)                              # row vectors: p' = (p + t) · s
    xf.AddTransformOp().Set(M)
    st.GetRootLayer().Save()
    # the web copy is ASCII (three's USDZLoader reads no crate): 17-digit floats triple the geometry size;
    # 6 significant digits are far below a texel or a visible normal change
    import re
    usda_f = tmp / "asset.usda"
    usda_f.write_text(re.sub(r"-?\d+\.\d{7,}(?:e-?\d+)?", lambda m: f"{float(m.group(0)):.6g}", usda_f.read_text()))
    st.GetRootLayer().Reload(force=True)                  # the packager reads the cached layer, not the file
    if dst.exists():
        dst.unlink()
    UsdUtils.CreateNewUsdzPackage(str(tmp / "asset.usda"), str(dst))
    if dst.stat().st_size > 20_000_000:                   # keep well under Cloudflare's 25 MiB per file
        for f in tmp.glob("basecolor*.png"):
            im = Image.open(f); im.resize((2048, 2048), Image.LANCZOS).save(f, optimize=True)
        dst.unlink(); UsdUtils.CreateNewUsdzPackage(str(tmp / "asset.usda"), str(dst))
    if dst.stat().st_size > 20_000_000:                   # still too big: quantise the ASCII geometry (lossless to the eye)
        _quantize_web(st)
        st.GetRootLayer().Save()
        usda_f.write_text(re.sub(r"-?\d+\.\d{7,}(?:e-?\d+)?", lambda m: f"{float(m.group(0)):.6g}", usda_f.read_text()))
        st.GetRootLayer().Reload(force=True)
        info["web_quantized"] = True
        dst.unlink(); UsdUtils.CreateNewUsdzPackage(str(tmp / "asset.usda"), str(dst))
    print(f"{dst.name}: {dst.stat().st_size/1e6:.1f} MB after quantising", file=sys.stderr)
    if dst.stat().st_size > 23_500_000:                   # last resort: decimate the copy (may tear at UV seams)
        keep = max(0.2, 0.85 * 18_000_000 / dst.stat().st_size)
        _decimate_web(st, keep)
        st.GetRootLayer().Save()
        usda_f.write_text(re.sub(r"-?\d+\.\d{7,}(?:e-?\d+)?", lambda m: f"{float(m.group(0)):.6g}", usda_f.read_text()))
        st.GetRootLayer().Reload(force=True)
        info["web_decimated"] = round(keep, 2)
        dst.unlink(); UsdUtils.CreateNewUsdzPackage(str(tmp / "asset.usda"), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    return info


MATERIALS = "texture"
if "--materials" in sys.argv:
    i = sys.argv.index("--materials"); MATERIALS = sys.argv[i + 1]; del sys.argv[i:i + 2]
out = Path(sys.argv[1]); (out / "models").mkdir(parents=True, exist_ok=True)
HDRS = [("Studio", "studio_small_09"), ("Photo studio", "brown_photostudio_02"), ("Sunset", "venice_sunset"),
        ("Sky", "kloofendal_48d_partly_cloudy_puresky")]           # CC0, polyhaven.com, 1k
hdr_src = Path(__file__).resolve().parents[1] / "out" / "hdri"
(out / "hdri").mkdir(exist_ok=True)
for _, h in HDRS:
    if (hdr_src / f"{h}.hdr").exists():
        shutil.copy(hdr_src / f"{h}.hdr", out / "hdri" / f"{h}.hdr")
cards = []
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    src = Path(parts[0]); title = parts[1] if len(parts) > 1 and parts[1] else name
    img = Path(parts[2]) if len(parts) > 2 and parts[2] else None
    poster = Path(parts[3]) if len(parts) > 3 and parts[3] else None
    # materials per texture (default: one mesh, transmission/roughness/metalness maps per texel, borders follow the
    # texture) or per part (--materials parts: one mesh prim per part, a face takes its centre texel's part)
    use_parts = MATERIALS == "parts" and (src / "asset_parts.usda").exists() and (src / "parts.json").exists()
    info = make_usdz(src, out / "models" / f"{name}.usdz", "asset_parts.usda" if use_parts else "asset.usda")
    if use_parts:
        shutil.copy(src / "parts.json", out / "models" / f"{name}_parts.json")
    use_maps = not use_parts and (src / "orm.png").exists()
    if use_maps:
        fin = json.loads((src / "mat" / "finish.json").read_text()) if (src / "mat" / "finish.json").exists() else {}
        glass = [v for v in fin.values() if v.get("transmission", 0) > 0]
        mp = {"orm": f"models/{name}_orm.png"}
        Image.open(src / "orm.png").convert("RGB").resize((1024, 1024), Image.LANCZOS).save(out / "models" / f"{name}_orm.png", optimize=True)
        if glass and (src / "transmission.png").exists():
            Image.open(src / "transmission.png").convert("L").resize((1024, 1024), Image.LANCZOS).save(out / "models" / f"{name}_transmission.png", optimize=True)
            mp.update(transmission=f"models/{name}_transmission.png", has_transmission=True,
                      tint=next((v["tint"] for v in glass if v.get("tint")), None), ior=next((v["ior"] for v in glass if v.get("ior")), 1.5))
        (out / "models" / f"{name}_maps.json").write_text(json.dumps(mp))
    if poster and poster.exists():
        Image.open(poster).convert("RGB").resize((640, 640), Image.LANCZOS).save(out / "models" / f"{name}_poster.jpg", quality=86)
    if img and img.exists():
        Image.open(img).convert("RGB").resize((512, 512), Image.LANCZOS).save(out / "models" / f"{name}_input.jpg", quality=85)
    rep = {}
    for r in (src / "report.json", src.parent / "report.json"):
        if r.exists():
            j = json.loads(r.read_text()); rep = dict(j.get("mesh", j)); rep["psnr"] = j.get("qa", {}).get("psnr_mean"); break
    rep["pbr"] = info["pbr"]
    rep["poster"] = bool(poster and poster.exists())
    rep["parts"] = use_parts
    rep["maps"] = use_maps
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
  <div class="view" data-usdz="models/{name}.usdz" data-parts="{'models/' + name + '_parts.json' if rep.get('parts') else ''}" data-maps="{'models/' + name + '_maps.json' if rep.get('maps') else ''}" tabindex="0" role="button" aria-label="Load the 3D model of {html.escape(title)}">
    {f'<img class="poster" src="models/{name}_poster.jpg" alt="{html.escape(title)}" loading="lazy">' if rep.get("poster") else ""}
    <span class="hint">Click to load 3D</span></div>
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
.view {{ cursor:pointer }} .view.live {{ cursor:grab }}
.view .poster {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#cccccc }}
.view .hint {{ position:absolute; left:50%; bottom:14px; transform:translateX(-50%); padding:6px 14px; border-radius:20px;
  background:var(--card); color:var(--fg); border:1px solid var(--line); font-size:13px }}
.view.live .poster, .view.live .hint {{ display:none }}
.view .msg {{ position:absolute; inset:0; display:grid; place-items:center; color:var(--muted); font-size:13px }}
.meta {{ display:flex; gap:12px; padding:12px 14px; align-items:flex-start }} .src {{ width:64px; height:64px; object-fit:cover; border-radius:8px; border:1px solid var(--line) }}
h2 {{ font-size:16px; margin:0 0 2px }} .meta p {{ margin:0; color:var(--muted); font-size:13px }} .dl a {{ color:var(--accent) }}
footer {{ color:var(--muted); font-size:13px; padding:16px; max-width:1200px; margin:auto }}
.hdr {{ margin-top:10px !important }} .hdr button {{ font:inherit; font-size:13px; padding:4px 10px; margin:2px; border-radius:20px;
  border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer }} .hdr button.on {{ background:var(--accent); color:#fff; border-color:var(--accent) }}
</style></head><body>
<header><h1>splat2mesh</h1>
<p>3D Gaussian splats → UV-textured <b>USD</b> meshes. Image → <b>Idol</b> splat → <b>splat2mesh</b>. Each viewer below renders the USDZ
itself (three.js USDZLoader); on iPhone/iPad the AR link opens it in Quick Look. <a href="https://github.com/jesusluque/splat2mesh">Code (Apache-2.0)</a></p>
<p class="hdr">Lighting: {" ".join(f'<button data-hdr="{h}">{t}</button>' for t, h in HDRS)} <button data-hdr="auto" class="on">Cycle</button></p></header>
<main>
{chr(10).join(items)}
</main>
<footer>Models marked “PBR materials” carry inferred roughness/metallic maps (Qwen3-VL); the others are baked colour only.
Generated with Iris-3B Turbo / Idol (speridlabs) and splat2mesh. HDRIs: Poly Haven (CC0).</footer>
<script type="module">
import * as THREE from "three";
import {{ USDZLoader }} from "three/addons/loaders/USDZLoader.js";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ RGBELoader }} from "three/addons/loaders/RGBELoader.js";
const loader = new USDZLoader();
const HDRS = {json.dumps([h for _, h in HDRS])};
const hdrCache = {{}};   // name → Promise<DataTexture>
const loadHdr = (n) => hdrCache[n] ??= new RGBELoader().loadAsync(`hdri/${{n}}.hdr`).then(t => {{ t.mapping = THREE.EquirectangularReflectionMapping; return t; }});
const viewers = [];
function view(el) {{
  const msg = document.createElement("div"); msg.className = "msg"; msg.textContent = "loading USDZ…"; el.appendChild(msg);
  const r = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
  r.setPixelRatio(Math.min(devicePixelRatio, 2)); r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 0.92; r.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(r.domElement);
  const scene = new THREE.Scene();
  const pm = new THREE.PMREMGenerator(r);
  const setHdr = async (n) => {{ const t = await loadHdr(n); const env = pm.fromEquirectangular(t).texture;
    if (scene.environment) scene.environment.dispose(); scene.environment = env; scene.background = env;
    scene.backgroundBlurriness = 0.6; scene.backgroundIntensity = 0.6; scene.environmentIntensity = 0.8; }};
  viewers.push(setHdr); setHdr(window.__hdr || HDRS[0]);
  const cam = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
  const ctl = new OrbitControls(cam, r.domElement); ctl.enableDamping = true; ctl.autoRotate = true; ctl.autoRotateSpeed = 1.2;
  const resize = () => {{ const w = el.clientWidth, h = el.clientHeight; r.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }};
  new ResizeObserver(resize).observe(el); resize();
  const partsP = el.dataset.parts ? fetch(el.dataset.parts).then(r => r.json()).catch(() => ({{}})) : Promise.resolve({{}});
  loader.load(el.dataset.usdz, async (g) => {{
    const parts = await partsP;
    g.traverse(o => {{ if (o.isMesh && o.name === "Asset" && Object.keys(parts).length) o.material = new THREE.MeshBasicMaterial({{ visible: false }}); }});   // loader duplicates the first part at the root (its children are the parts: keep them)
    g.traverse(o => {{
      if (!o.isMesh) return;
      const p = parts[o.name] || (o.parent && parts[o.parent.name]);
      if (!p) return;
      const m = o.material;
      if (p.transmission > 0 && p.patterned) {{   // stained / multi-coloured glass: its colours are the texture
        o.material = new THREE.MeshPhysicalMaterial({{ map: m.map, metalness: 0, roughness: Math.min(p.roughness ?? 0.05, 0.1),
          transmission: p.transmission, ior: p.ior || 1.5, thickness: 0.02, side: THREE.DoubleSide, envMapIntensity: 1.0 }});
      }} else if (p.transmission > 0) {{          // glass / crystal: physical transmission (USDZLoader has no opacity)
        // tint from the part's texture (median colour), thin-ish glass, absorption tint over a short distance
        let tint = new THREE.Color(0xffffff);
        if (p.tint) tint = new THREE.Color().setRGB(p.tint[0], p.tint[1], p.tint[2], THREE.LinearSRGBColorSpace);
        else if (m.map && m.map.image) {{ try {{ const c = document.createElement("canvas"); c.width = c.height = 16;
          const x = c.getContext("2d"); x.drawImage(m.map.image, 0, 0, 16, 16); const d = x.getImageData(0, 0, 16, 16).data;
          let r = 0, g2 = 0, b = 0; for (let i = 0; i < d.length; i += 4) {{ r += d[i]; g2 += d[i + 1]; b += d[i + 2]; }}
          const n = d.length / 4, mx = Math.max(r, g2, b) / n || 1; tint = new THREE.Color(r / n / mx, g2 / n / mx, b / n / mx); }} catch (e) {{}} }}
        o.material = new THREE.MeshPhysicalMaterial({{ color: tint.clone().lerp(new THREE.Color(1, 1, 1), 0.35), metalness: 0, roughness: Math.min(p.roughness ?? 0.03, 0.08),
          transmission: 1, ior: p.ior || 1.5, thickness: 0.04, attenuationColor: tint, attenuationDistance: 0.25,
          side: THREE.DoubleSide, specularIntensity: 1, envMapIntensity: 1.2 }});
      }} else if (p.subsurface > 0) {{    // jade, wax, marble: a soft sheen of transmission as a cheap subsurface look
        o.material = new THREE.MeshPhysicalMaterial({{ map: m.map, roughness: 0.35, metalness: 0,
          transmission: Math.min(0.5, p.subsurface * 0.6), thickness: 0.15, ior: p.ior || 1.5,
          attenuationColor: new THREE.Color(0.75, 0.9, 0.75), attenuationDistance: 0.3, sheen: 0.3, sheenRoughness: 0.6 }});
      }}
    }});
    if (el.dataset.maps) {{   // per-texel materials on one mesh: transmission / roughness / metalness follow the texture
      const mp = await fetch(el.dataset.maps).then(r => r.json()).catch(() => null);
      if (mp) {{
        const tl = new THREE.TextureLoader();
        const ld = (u) => new Promise(res => u ? tl.load(u, t => res(t), undefined, () => res(null)) : res(null));
        const [orm, tr] = await Promise.all([ld(mp.orm), ld(mp.transmission)]);
        g.traverse(o => {{
          if (!o.isMesh) return;
          const m = o.material, fy = m.map ? m.map.flipY : false;
          for (const t of [orm, tr]) if (t) {{ t.flipY = fy; t.colorSpace = THREE.NoColorSpace; t.needsUpdate = true; }}
          const glass = !!(tr && mp.has_transmission);
          const tint = mp.tint ? new THREE.Color().setRGB(mp.tint[0], mp.tint[1], mp.tint[2], THREE.LinearSRGBColorSpace) : new THREE.Color(1, 1, 1);
          o.material = new THREE.MeshPhysicalMaterial({{ map: m.map, roughnessMap: orm, metalnessMap: orm, roughness: 1, metalness: 1,
            transmissionMap: glass ? tr : null, transmission: glass ? 1 : 0, ior: mp.ior || 1.5, thickness: glass ? 0.04 : 0,
            attenuationColor: tint, attenuationDistance: glass ? 0.25 : Infinity, side: glass ? THREE.DoubleSide : THREE.FrontSide,
            specularIntensity: 1, envMapIntensity: 1.1 }});
        }});
      }}
    }}
    msg.remove(); scene.add(g);
    const box = new THREE.Box3().setFromObject(g), c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()).length();
    ctl.target.copy(c); cam.position.copy(c).add(new THREE.Vector3(0.35, 0.18, 1).normalize().multiplyScalar(s * 1.6));
    cam.near = s / 100; cam.far = s * 20; cam.updateProjectionMatrix();
  }}, undefined, (e) => {{ msg.textContent = "could not load " + el.dataset.usdz; console.error(e); }});
  let visible = false; new IntersectionObserver(([en]) => visible = en.isIntersecting).observe(el);
  r.setAnimationLoop(() => {{ if (!visible) return; ctl.update(); r.render(scene, cam); }});
  el.__dispose = () => {{ r.setAnimationLoop(null); r.dispose(); r.forceContextLoss(); const i = viewers.indexOf(setHdr); if (i >= 0) viewers.splice(i, 1); }};
}}
const live = [];
document.querySelectorAll(".view").forEach(el => {{
  const go = () => {{ if (el.classList.contains("live")) return;
    if (live.length >= 4) {{ const old = live.shift(); old.__dispose && old.__dispose(); old.classList.remove("live");
      old.querySelectorAll("canvas, .msg").forEach(n => n.remove()); }}
    el.classList.add("live"); live.push(el); view(el); }};
  el.addEventListener("click", go);
  el.addEventListener("keydown", e => {{ if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); go(); }} }});
}});
let cur = 0, auto = true;
const pick = (n) => {{ window.__hdr = n; viewers.forEach(f => f(n)); document.querySelectorAll(".hdr button").forEach(b => b.classList.toggle("on", b.dataset.hdr === (auto ? "auto" : n))); }};
document.querySelectorAll(".hdr button").forEach(b => b.onclick = () => {{ auto = b.dataset.hdr === "auto"; if (!auto) pick(b.dataset.hdr); else pick(HDRS[cur]); }});
setInterval(() => {{ if (auto) {{ cur = (cur + 1) % HDRS.length; pick(HDRS[cur]); }} }}, 8000);
</script>
</body></html>
''')
print("site:", out / "index.html")
