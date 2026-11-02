"""Prompt/vision-informed PBR maps for a baked asset.

1. prepare(): cluster the baked texture's colours (Lab k-means) into material candidates; for each cluster
   write an overlay of the Idol input image with only that cluster in colour (for the VLM to look at).
2. scripts/material_vlm.py (Qwen3-VL-4B, Iris's text encoder) labels each cluster: material, metallic, roughness.
3. apply(): per texel → nearest cluster (soft, in Lab) → roughness & metallic maps; highlights baked into
   glossy dielectric regions are removed from the base colour (they will come back from the light);
   USD gets roughness/metallic textures, a GLB gets a metallicRoughness texture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def _srgb_to_lab(rgb):
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), 1)]
    for _ in range(1, k):                                   # k-means++
        d = ((X[:, None] - C[None]) ** 2).sum(-1).min(1)
        C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())]])
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return C, lab


def _texture_and_mask(asset_dir: Path):
    tex = np.asarray(Image.open(asset_dir / "basecolor.png").convert("RGB"), np.float32)[::-1] / 255
    from .rerender import load_usd_textured
    from .texture import rasterize_uv
    V, F, st, _ = load_usd_textured(str(asset_dir / "asset.usda"))
    import trimesh
    n = trimesh.Trimesh(V, F, process=False).vertex_normals
    _, _, mask = rasterize_uv(V.astype(np.float64), F, st, n, tex.shape[0])
    return tex, mask


def prepare(asset_dir, input_image, out_json, prompt: str = "", k: int = 6, min_share: float = 0.01,
            merge_de: float = 12.0) -> dict:
    asset_dir = Path(asset_dir); out_json = Path(out_json)
    work = out_json.parent; work.mkdir(parents=True, exist_ok=True)
    tex, mask = _texture_and_mask(asset_dir)
    lab_t = _srgb_to_lab(tex[mask])
    sel = np.random.default_rng(0).choice(len(lab_t), min(60000, len(lab_t)), replace=False)
    C, _ = _kmeans(lab_t[sel], k)
    # merge clusters closer than merge_de (ΔE) — same material, slightly different shade
    keep = []
    for c in C:
        if all(np.linalg.norm(c - C[j]) > merge_de for j in keep):
            keep.append(int(np.where((C == c).all(1))[0][0]))
    C = C[keep]
    lab_all = ((lab_t[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    share = np.bincount(lab_all, minlength=len(C)) / len(lab_all)
    C = C[share >= min_share]; share = share[share >= min_share]
    # overlays on the input image
    img = np.asarray(Image.open(input_image).convert("RGB"), np.float32) / 255
    obj = ndi.binary_fill_holes(img.max(-1) > 0.03)
    lab_i = _srgb_to_lab(img)
    lab_px = ((lab_i[..., None, :] - C[None, None]) ** 2).sum(-1).argmin(-1)
    grey = img.mean(-1, keepdims=True) * 0.35 + 0.45
    regions = []
    for j in range(len(C)):
        on = obj & (lab_px == j)
        on = ndi.binary_opening(on, iterations=1)
        ov = np.where(on[..., None], img, np.where(obj[..., None], grey, 0.0))
        path = work / f"region_{j}.png"
        Image.fromarray((ov * 255).astype(np.uint8)).save(path)
        ys, xs = np.nonzero(on)                       # zoomed crop of the region for the VLM
        if len(xs) > 20:
            y0, y1, x0, x1 = (np.percentile(ys, 2), np.percentile(ys, 98), np.percentile(xs, 2), np.percentile(xs, 98))
            pad = 0.15 * max(y1 - y0, x1 - x0) + 16
            box = (int(max(x0 - pad, 0)), int(max(y0 - pad, 0)), int(min(x1 + pad, ov.shape[1])), int(min(y1 + pad, ov.shape[0])))
            crop = Image.fromarray((ov * 255).astype(np.uint8)).crop(box)
            crop.thumbnail((512, 512))
            crop.save(work / f"crop_{j}.png")
        rgb = tex[mask][lab_all == j].mean(0) if (lab_all == j).any() else np.zeros(3)
        regions.append({"id": j, "overlay": str(path), "lab": C[j].tolist(), "colour": rgb.tolist(),
                        "share": round(float(share[j]), 4)})
    spec = {"image": str(input_image), "prompt": prompt, "asset": str(asset_dir), "regions": regions}
    out_json.write_text(json.dumps(spec, indent=1))
    return spec


def apply(spec_json, out_dir=None, highlight_sigma: float = 2.0) -> dict:
    spec = json.loads(Path(spec_json).read_text())
    asset = Path(spec["asset"]); out_dir = Path(out_dir or asset)
    tex, mask = _texture_and_mask(asset)
    regs = spec["regions"]
    C = np.array([r["lab"] for r in regs])
    rough = np.array([r.get("roughness", 0.5) for r in regs])
    metal = np.array([r.get("metallic", 0.0) for r in regs])
    lab = _srgb_to_lab(tex)
    d2 = ((lab[..., None, :] - C[None, None]) ** 2).sum(-1)
    w = np.exp(-(d2 - d2.min(-1, keepdims=True)) / (2 * 8.0 ** 2))      # soft assignment, ΔE scale 8
    w /= w.sum(-1, keepdims=True)
    R = (w * rough).sum(-1); M = (w * metal).sum(-1)
    R = ndi.median_filter(R, 3); M = ndi.median_filter(M, 3)
    # highlight removal on glossy dielectrics: texels much brighter and less saturated than their cluster
    hard = d2.argmin(-1)
    base = tex.copy()
    L = lab[..., 0]; chroma = np.hypot(lab[..., 1], lab[..., 2])
    spec_mask = np.zeros(mask.shape, bool)
    for j, r in enumerate(regs):
        sel = mask & (hard == j)
        if r.get("metallic", 0) >= 0.5 or sel.sum() < 50 or r.get("roughness", 0.5) > 0.5:
            continue
        mu, sd = L[sel].mean(), L[sel].std() + 1e-6
        spec_mask |= sel & (L > mu + highlight_sigma * sd) & (chroma < np.median(chroma[sel]) + 5)
    spec_mask = ndi.binary_dilation(spec_mask, iterations=2) & mask
    if spec_mask.any():
        from .texture import inpaint
        base = inpaint(base, mask & ~spec_mask, iters=128)
        base = np.where(mask[..., None], base, tex)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((base[::-1].clip(0, 1) * 255).astype(np.uint8)).save(out_dir / "basecolor_delit.png")
    Image.fromarray((R[::-1].clip(0, 1) * 255).astype(np.uint8)).save(out_dir / "roughness.png")
    Image.fromarray((M[::-1].clip(0, 1) * 255).astype(np.uint8)).save(out_dir / "metallic.png")
    orm = np.stack([np.ones_like(R), R, M], -1)                       # glTF: G = roughness, B = metallic
    Image.fromarray((orm[::-1].clip(0, 1) * 255).astype(np.uint8)).save(out_dir / "orm.png")
    _usd_add_maps(asset / "asset.usda", out_dir)
    return {"highlight_texels": int(spec_mask.sum()), "regions": [(r["material"], r["metallic"], r["roughness"], r["share"])
                                                                   for r in regs]}


def _usd_add_maps(usd_path: Path, maps_dir: Path):
    from pxr import Sdf, Usd, UsdShade
    stage = Usd.Stage.Open(str(usd_path))
    root = usd_path.parent
    rel = lambda p: "./" + str(Path(p).resolve().relative_to(root.resolve())) if Path(p).resolve().is_relative_to(root.resolve()) else str(Path(p).resolve())
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Shader" and UsdShade.Shader(prim).GetIdAttr().Get() == "UsdPreviewSurface":
            pbr = UsdShade.Shader(prim)
            mat_path = prim.GetPath().GetParentPath()
            reader = UsdShade.Shader(stage.GetPrimAtPath(mat_path.AppendChild("stReader")))
            base = UsdShade.Shader(stage.GetPrimAtPath(mat_path.AppendChild("BaseColor")))
            base.GetInput("file").Set(rel(maps_dir / "basecolor_delit.png"))
            for name, inp in (("Roughness", "roughness"), ("Metallic", "metallic")):
                t = UsdShade.Shader.Define(stage, mat_path.AppendChild(name))
                t.CreateIdAttr("UsdUVTexture")
                t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(maps_dir / f"{inp}.png"))
                t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                t.CreateOutput("r", Sdf.ValueTypeNames.Float)
                pbr.CreateInput(inp, Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "r")
    stage.GetRootLayer().Save()


# material library (splat2mesh/material_library.json): name → metallic, roughness, ior, f0, keywords
LIB = json.loads((Path(__file__).with_name("material_library.json")).read_text())["materials"]
# the coarse classes the VLM yes/no probe answers → a library entry
VLM_TO_LIB = {"gold metal": "gold", "silver metal": "aluminium", "wood": "oiled hardwood", "lacquer": "gloss lacquer",
              "glazed ceramic": "glazed ceramic", "painted plastic": "glossy plastic", "matte": "matte paint",
              "glass": "glass", "mother-of-pearl": "mother-of-pearl"}
PRESET = {k: (float(v["metallic"]), float(v["roughness"])) for k, v in LIB.items()}
PRESET.update({k: PRESET[v] for k, v in VLM_TO_LIB.items()})
KEYWORDS = {k: v["keywords"] for k, v in LIB.items()}


def lookup(text: str):
    """Best library entry for a phrase: the longest keyword found wins ('brushed steel' beats 'steel')."""
    import re
    t = text.lower(); best, blen = None, 0
    for name, ws in KEYWORDS.items():
        for w in ws:
            if len(w) > blen and re.search(r"\b" + re.escape(w) + r"\b", t):
                best, blen = name, len(w)
    return best


def decide(spec_json, mentioned_weight: float = 1.0, other_weight: float = 0.25) -> dict:
    """Re-decide each region's material from the VLM yes-scores × a prompt prior: classes whose words appear
    in the prompt keep their score, the others are down-weighted (still possible when the VLM is sure)."""
    import re
    p = Path(spec_json); spec = json.loads(p.read_text())
    text = spec.get("prompt", "").lower()
    lib_named = {k for k, ws in KEYWORDS.items() if any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in ws)}
    lib_to_vlm = {}
    for v, l in VLM_TO_LIB.items():
        lib_to_vlm.setdefault(LIB[l]["class"], set()).add(v)
    named = set()
    for l in lib_named:                       # a named library material enables the VLM classes of its family
        named |= lib_to_vlm.get(LIB[l]["class"], set())
        named |= {v for v, ll in VLM_TO_LIB.items() if ll == l}
    for r in spec["regions"]:
        ys = r.get("yes_scores")
        if not ys:
            continue
        post = {k: v * (mentioned_weight if (not named or k in named) else other_weight) for k, v in ys.items()}
        best = max(post, key=post.get)
        r["vlm_material"] = r.get("vlm_material", r.get("material"))
        r.update(material=best, metallic=PRESET[best][0], roughness=PRESET[best][1], prior=sorted(named))
    p.write_text(json.dumps(spec, indent=1))
    return {"named": sorted(named), "regions": [(r["material"], r["share"]) for r in spec["regions"]]}


def parse_description(desc: str) -> list[tuple[str, str]]:
    """'gold: the bell, the coin; red lacquer: the collar; porcelain: the body' →
    [('gold metal', 'the bell'), ('gold metal', 'the coin'), ('lacquer', 'the collar'), ('glazed ceramic', 'the body')]"""
    import re
    out = []
    for part in re.split(r"[;\n]+", desc):
        if ":" not in part:
            continue
        mat, things = part.split(":", 1)
        m = mat.strip().lower()
        cls = m if m in LIB else (lookup(m) or "glazed ceramic")
        for t in things.split(","):
            if t.strip():
                out.append((cls, t.strip()))
    return out


def from_masks(spec_json, masks: dict, min_overlap: float = 0.15) -> dict:
    """Semi-automatic: masks = {material class: HxW bool mask on the input image} (e.g. from ENEAS text prompts).
    Each colour region takes the material whose mask covers most of that region's pixels in the input image;
    regions no mask covers well keep the VLM/prompt decision."""
    p = Path(spec_json); spec = json.loads(p.read_text())
    img = np.asarray(Image.open(spec["image"]).convert("RGB"), np.float32) / 255
    obj = ndi.binary_fill_holes(img.max(-1) > 0.03)
    C = np.array([r["lab"] for r in spec["regions"]])
    lab_px = ((_srgb_to_lab(img)[..., None, :] - C[None, None]) ** 2).sum(-1).argmin(-1)
    # overlapping masks: the most specific (smallest) wins each pixel ("the body" must not swallow "the coin")
    label = np.full(obj.shape, -1)
    names = list(masks)
    for i in sorted(range(len(names)), key=lambda i: -masks[names[i]].sum()):
        label[masks[names[i]]] = i
    for j, r in enumerate(spec["regions"]):
        px = obj & (lab_px == j)
        n = px.sum()
        if n == 0:
            continue
        cover = {cls: float(((label == i) & px).sum() / n) for i, cls in enumerate(names)}
        best = max(cover, key=cover.get) if cover else None
        r["mask_cover"] = {k: round(v, 3) for k, v in cover.items()}
        if best and cover[best] >= min_overlap:
            cls = best.split("|")[0]                     # keys may be "class|phrase"
            r.update(material=cls, metallic=PRESET[cls][0], roughness=PRESET[cls][1], source="mask", mask=best)
    p.write_text(json.dumps(spec, indent=1))
    return {"regions": [(r["material"], r.get("source", "vlm"), r["share"]) for r in spec["regions"]]}


def render_orbit(asset_dir, out_dir, n: int = 24, elevation: float = 15.0, size: int = 768):
    """Unlit renders of the textured mesh on an orbit (a 'video' for ENEAS) + per-pixel texel index maps."""
    import open3d as o3d
    from .rerender import load_usd_textured
    asset_dir, out_dir = Path(asset_dir), Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    V, F, st, tex = load_usd_textured(str(asset_dir / "asset.usda"))
    R = tex.shape[0]
    lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; rad = np.linalg.norm(hi - lo) / 2
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(V.astype(np.float32), F.astype(np.uint32))
    f = size / 2 / np.tan(np.radians(15)); dist = rad / np.sin(np.radians(15)) * 1.1
    u, v = np.meshgrid(np.arange(size) + .5, np.arange(size) + .5)
    texel_maps = []
    for i in range(n):
        a, e = 2 * np.pi * i / n, np.radians(elevation)
        eye = c + dist * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        fw = c - eye; fw /= np.linalg.norm(fw); rt = np.cross(fw, [0, 1, 0]); rt /= np.linalg.norm(rt); dn = np.cross(fw, rt)
        d = np.stack([(u - size / 2) / f, (v - size / 2) / f, np.ones_like(u)], -1) @ np.stack([rt, dn, fw])
        res = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eye, d.shape), d], -1).reshape(-1, 6).astype(np.float32)))
        pid = res["primitive_ids"].numpy().reshape(size, size); bu = res["primitive_uvs"].numpy().reshape(size, size, 2)
        hit = pid != o3d.t.geometry.RaycastingScene.INVALID_ID
        ff = F[pid[hit]]; b1, b2 = bu[hit, 0:1], bu[hit, 1:2]
        uvp = st[ff[:, 0]] * (1 - b1 - b2) + st[ff[:, 1]] * b1 + st[ff[:, 2]] * b2
        px = np.clip((uvp * R).astype(int), 0, R - 1)
        img = np.full((size, size, 3), 1.0, np.float32)            # white background, as Idol likes
        img[hit] = tex[px[:, 1], px[:, 0]]
        tmap = np.full((size, size), -1, np.int64); tmap[hit] = px[:, 1] * R + px[:, 0]
        Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).save(out_dir / "frames" / f"frame_{i:04d}.jpg", quality=95)
        texel_maps.append(tmap)
    np.save(out_dir / "texel_maps.npy", np.stack(texel_maps))
    return out_dir


def from_view_masks(spec_json, orbit_dir, masks: dict) -> dict:
    """masks = {"class|phrase": [n_views HxW bool]}: back-project every view's masks to texels (smallest mask wins
    per pixel), vote per texel; texels no view saw take the majority class of their colour region."""
    spec_p = Path(spec_json); spec = json.loads(spec_p.read_text())
    tmaps = np.load(Path(orbit_dir) / "texel_maps.npy")
    tex, mask = _texture_and_mask(Path(spec["asset"]))
    R = tex.shape[0]
    names = list(masks)
    votes = np.zeros((len(names), R * R), np.float32)
    for vi in range(tmaps.shape[0]):
        tm = tmaps[vi]
        label = np.full(tm.shape, -1)
        for i in sorted(range(len(names)), key=lambda i: -masks[names[i]][vi].sum()):
            label[masks[names[i]][vi]] = i
        ok = (tm >= 0) & (label >= 0)
        np.add.at(votes, (label[ok], tm[ok]), 1.0)
    seen = votes.sum(0) > 0
    cls_tex = np.where(seen, votes.argmax(0), -1).reshape(R, R)
    # fill unseen texels: majority over the texel's colour region
    C = np.array([r["lab"] for r in spec["regions"]])
    reg = ((_srgb_to_lab(tex)[..., None, :] - C[None, None]) ** 2).sum(-1).argmin(-1)
    for j in range(len(C)):
        sel = mask & (reg == j)
        known = cls_tex[sel]; known = known[known >= 0]
        if len(known):
            maj = np.bincount(known, minlength=len(names)).argmax()
            fill = sel & (cls_tex < 0)
            cls_tex[fill] = maj
    # every colour region also gets the majority class of its texels (fallback for apply())
    for j, r in enumerate(spec["regions"]):
        sel = mask & (reg == j)
        known = cls_tex[sel]; known = known[known >= 0]
        if len(known):
            cls = names[np.bincount(known, minlength=len(names)).argmax()].split("|")[0]
        else:
            cls = r.get("material", "glazed ceramic")
        r.update(material=cls, metallic=PRESET[cls][0], roughness=PRESET[cls][1])
    out = Path(spec["asset"])
    classes = [n.split("|")[0] for n in names]
    np.save(out / "mat" / "texel_class.npy", cls_tex)
    spec["texel_classes"] = names
    spec_p.write_text(json.dumps(spec, indent=1))
    share = {n: round(float((cls_tex[mask] == i).mean()), 4) for i, n in enumerate(names)}
    return {"seen_texels": round(float(seen.reshape(R, R)[mask].mean()), 3), "share": share, "classes": classes}


def apply_texel_classes(spec_json, out_dir=None) -> dict:
    """Roughness/metallic maps from the per-texel classes of from_view_masks (falls back to regions where unknown)."""
    spec = json.loads(Path(spec_json).read_text())
    asset = Path(spec["asset"]); out_dir = Path(out_dir or asset)
    res = apply(spec_json, out_dir)                      # region-based maps + de-lit base colour as the fallback
    cls_tex = np.load(asset / "mat" / "texel_class.npy")
    names = spec["texel_classes"]
    R = np.asarray(Image.open(out_dir / "roughness.png"), np.float32)[::-1] / 255
    M = np.asarray(Image.open(out_dir / "metallic.png"), np.float32)[::-1] / 255
    for i, n in enumerate(names):
        cls = n.split("|")[0]
        sel = cls_tex == i
        R[sel] = PRESET[cls][1]; M[sel] = PRESET[cls][0]
    R = ndi.median_filter(R, 3); M = ndi.median_filter(M, 3)
    Image.fromarray((R[::-1].clip(0, 1) * 255).astype(np.uint8)).save(out_dir / "roughness.png")
    Image.fromarray((M[::-1].clip(0, 1) * 255).astype(np.uint8)).save(out_dir / "metallic.png")
    orm = np.stack([np.ones_like(R), R, M], -1)
    Image.fromarray((orm[::-1].clip(0, 1) * 255).astype(np.uint8)).save(out_dir / "orm.png")
    return res
