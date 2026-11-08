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


def _lab_to_srgb(lab):
    """Inverse of _srgb_to_lab (same D65 white, same matrix)."""
    fy = (lab[..., 0] + 16) / 116; fx = fy + lab[..., 1] / 500; fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.9505, 1.0, 1.089])
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    c = np.clip(xyz @ np.linalg.inv(M).T, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


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


# finish-pass constants
GLASS_PATTERN_SPREAD = 12.0   # Lab a*b* median distance; above it the glass is multi-coloured (stained glass)
INK_MAX_WIDTH = 4             # texels (half-width) of a dark stroke still read as ink rather than a reflection
METAL_DETAIL_SIGMA = 8.0      # texels: metal albedo keeps luminance detail finer than this (hammering, engraving)
METAL_DETAIL_MIN = 0.75       # ...and may darken by at most this factor (deeper darks are reflections)
DARK_METAL_LUM = 0.08        # a metal part whose median linear luminance is below this and whose median colour
DARK_METAL_CHROMA = 10.0     # is greyer than this is blackened or painted metal (a black iron cap: median 0.037),
                             # not a reflection (a brass frame's dark reflections keep its hue: chroma 25)


def metal_f0(f0, lin_px: np.ndarray, lab_px: np.ndarray) -> np.ndarray:
    """The F0 to use for a metal part: the library's, or, for a dark colourless part (blackened iron, black paint
    on metal), the library's hue scaled to the part's own median luminance. lin_px: linear RGB [n, 3]; lab_px: Lab."""
    f0 = np.asarray(f0, np.float64)
    if not len(lin_px):
        return f0
    lum = float(np.median(lin_px.mean(1))); chroma = float(np.hypot(*np.median(lab_px[:, 1:], 0)))
    if lum < DARK_METAL_LUM and chroma < DARK_METAL_CHROMA:
        return f0 * max(lum, 0.02) / f0.mean()
    return f0
GLASS_SMOOTH_ITERS = 120      # Taubin iterations on glass/crystal/jade/wax vertices
GLASS_WALL_MIN = 0.08         # glass thicker than this × object size is really a hollow wall (bulb, bowl, bottle)
GLOSS_EVIDENCE_MAX_ROUGH = 0.45  # materials whose range reaches this low may be polished; the texture decides
GLOSS_EVIDENCE_FRAC = 0.03       # share of pale-highlight texels that marks a polished finish
BILATERAL_ITERS = 60           # glossy shading normals: bilateral face-normal passes
BILATERAL_SIGMA = 0.3          # 1 - cos below which neighbours average (≈ 45°: edges above it stay)
WHITE_PATCH_MIN = 0.002        # white patch (share of the atlas) a coloured metal gives to an unfound non-metal
SEEN_LIGHT_MAX = 0.03         # share of light texels on the seen side above which light behind is kept
GLOSS_UNIFORM_SPREAD = 10.0   # Lab a*b* spread below which a glossy dielectric piece is one colour (paint, lacquer)
GLASS_WALL_TINT = 0.35        # share of the glass tint applied per wall a ray crosses (older rule, unused)
GLASS_WALLS = 2               # walls a ray crosses through a glass shell: per-wall colour = tint ** (1 / walls), so
                              # the colour seen through the piece is the tint measured on it
_RASTER = {}


def _raster(asset_dir: Path, R: int):
    """(pos, nrm, mask) of the UV atlas, memoised per asset.usda mtime — rasterising 80k faces into 4096² is the
    slowest step and every stage needs it."""
    import os
    key = (str(Path(asset_dir).resolve()), R, os.path.getmtime(Path(asset_dir) / "asset.usda"))
    if key not in _RASTER:
        from .rerender import load_usd_textured
        from .texture import rasterize_uv
        import trimesh
        V, F, st, _ = load_usd_textured(str(Path(asset_dir) / "asset.usda"))
        n = trimesh.Trimesh(V, F, process=False).vertex_normals
        _RASTER.clear()
        _RASTER[key] = rasterize_uv(V.astype(np.float64), F, st, n, R)
    return _RASTER[key]


def _texture_and_mask(asset_dir: Path):
    tex = np.asarray(Image.open(asset_dir / "basecolor.png").convert("RGB"), np.float32)[::-1] / 255
    return tex, _raster(asset_dir, tex.shape[0])[2]



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
            if not stage.GetPrimAtPath(mat_path.AppendChild("BaseColor")) or "M_interior" in str(mat_path):
                continue                                  # glass panes / interior: own materials, not the baked maps
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


def render_orbit(asset_dir, out_dir, n: int = 16, elevation: float = 15.0, size: int = 768):
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


def refine_by_regions(tex, mask, votes, spec, classes, min_share: float = 0.25, pos=None, home_share: float = 0.6):
    """Material borders follow colour borders. In each colour region only the material classes holding at least
    min_share of that region's votes are allowed; each texel takes the allowed part with most votes at that
    texel (or the region's majority part when unseen). White texels cannot become gold because a phrase mask
    spilled over them; two parts of the same colour can still keep different materials."""
    K = votes.shape[0]
    C = np.array([r["lab"] for r in spec["regions"]])
    reg = ((_srgb_to_lab(tex)[..., None, :] - C[None, None]) ** 2).sum(-1).argmin(-1)
    out = np.full(mask.shape, -1)
    # a class is also allowed in its home regions: the fewest regions that together hold home_share of its own
    # votes. A small part (horns) or one sharing its colour with a bigger one (lavender crystal next to lavender
    # wings, a steel helmet next to grey chainmail) is never a majority anywhere, but it lives somewhere; with
    # coarse colour regions its votes may also be split over two of them (two greys), so one region is not enough
    cls_names = sorted(set(classes))
    home = {}
    for c in cls_names:
        ks = [k for k in range(K) if classes[k] == c]
        per_reg = np.array([votes[ks][:, mask & (reg == j)].sum() for j in range(len(C))])
        if per_reg.sum() <= 0:
            continue
        order = np.argsort(-per_reg); cum = np.cumsum(per_reg[order]) / per_reg.sum()
        home[c] = {int(j) for j in order[:int(np.searchsorted(cum, home_share)) + 1]}
    for j in range(len(C)):
        sel = mask & (reg == j)
        if not sel.any():
            continue
        v = votes[:, sel]                                   # K × n
        per_part = v.sum(1)
        if per_part.sum() == 0:
            continue
        per_cls = {}
        for k in range(K):
            per_cls[classes[k]] = per_cls.get(classes[k], 0) + per_part[k]
        tot = sum(per_cls.values())
        allowed_cls = {c for c, w in per_cls.items() if w >= min_share * tot} or {max(per_cls, key=per_cls.get)}
        allowed_cls |= {c for c, h in home.items() if j in h and per_cls.get(c, 0) > 0}
        allowed = np.array([classes[k] in allowed_cls for k in range(K)])
        vv = np.where(allowed[:, None], v, -1)
        lab_t = vv.argmax(0)
        unseen = v.sum(0) == 0
        lab_t[unseen] = int(np.argmax(np.where(allowed, per_part, -1)))
        if pos is not None and unseen.any() and allowed.sum() > 1:
            # an unvoted texel takes the allowed part of the nearest voted texel in 3D, not the region's majority:
            # dark reflections on the back of a silver pot are next to silver, far from the wooden handle
            from scipy.spatial import cKDTree
            P = pos[sel]; known = ~unseen & allowed[lab_t]
            if known.sum() > 50:
                kidx = np.nonzero(known)[0][::max(1, known.sum() // 400_000)]
                _, nn = cKDTree(P[kidx]).query(P[unseen], k=1, workers=-1)
                lab_t[unseen] = lab_t[kidx[nn]]
        out[sel] = lab_t
    return out


def refine_by_colour(tex, mask, votes, cls_tex, conf: float = 0.75, prior_w: float = 2.0):
    """Mask votes are blocky (renders at 768 px, phrase masks spill over neighbours). Learn each class's colour
    (Lab Gaussian) from confidently voted texels, then label every texel by colour likelihood + vote prior, so
    material borders follow colour edges: a white porcelain texel cannot stay "gold" because a mask covered it."""
    K = votes.shape[0]
    lab = _srgb_to_lab(tex)
    tot = votes.sum(0)
    frac = votes / np.maximum(tot, 1e-6)
    ll = np.full((K,) + mask.shape, -np.inf, np.float32)
    for k in range(K):
        sure = mask & (tot > 0) & (frac[k] >= conf)
        if sure.sum() < 30:
            continue
        X = lab[sure]
        mu = X.mean(0); cov = np.cov(X.T) + np.eye(3) * 4.0          # ΔE ≥ 2 floor
        icov = np.linalg.inv(cov); _, logdet = np.linalg.slogdet(cov)
        d = lab[mask] - mu
        m2 = np.einsum("ni,ij,nj->n", d, icov, d)
        prior = np.log(frac[k][mask] + 0.05) * prior_w * (tot[mask] > 0)
        ll[k][mask] = -0.5 * (m2 + logdet) + prior
    best = ll.argmax(0)
    ok = np.isfinite(ll.max(0)) & mask
    out = cls_tex.copy()
    out[ok] = best[ok]
    return out


def from_view_masks(spec_json, orbit_dir, masks: dict, votes=None, cap_transparent: bool = True) -> dict:
    """masks = {"class|phrase": [n_views HxW bool]}: back-project every view's masks to texels (smallest mask wins
    per pixel), vote per texel; texels no view saw take the majority class of their colour region.
    votes: [n_phrases, R*R] texel votes computed elsewhere (gs_materials.texel_votes: from the splat's gaussians);
    the back-projection is then skipped and only the rules below run.
    cap_transparent: the 25 % cap on a secondary transparent part; off when the votes come from a splat whose
    masks were already settled (a lantern's chimney is a large glass part, and the splat keeps it)."""
    spec_p = Path(spec_json); spec = json.loads(spec_p.read_text())
    tex, mask = _texture_and_mask(Path(spec["asset"]))
    R = tex.shape[0]
    names = list(masks)
    tmaps = np.load(Path(orbit_dir) / "texel_maps.npy") if votes is None else np.zeros((0,))
    if votes is None:
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
    cls_tex = refine_by_regions(tex, mask, votes.reshape(len(names), R, R), spec, [n.split("|")[0] for n in names],
                                pos=_raster(Path(spec["asset"]), R)[0])
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
    # colour priors trust the colour, and the colour is only real where the input picture saw the surface; on the
    # side Idol invented (a rear tyre painted grey, a back window painted white) the ENEAS votes decide alone
    sv = spec.get("seen_view")
    if sv is None:
        sv = best_view(Path(spec["asset"]) / "mat" / "orbit", spec["image"])
    seen_in = _seen_mask(Path(spec["asset"]), R, sv)
    # thin metal parts ENEAS cannot find (kintsugi lines, gilt rims): a named metal with no texels claims the
    # colour regions whose hue matches its F0 colour (and that are saturated enough to be coloured metal)
    for i, n in enumerate(names):
        cls = n.split("|")[0]; L = _lib(cls)
        if L.get("metallic", 0) < 0.5 or not L.get("f0") or (cls_tex == i).any():
            continue
        if any((cls_tex == j).any() for j, m in enumerate(names) if m.split("|")[0] == cls):
            continue
        if np.ptp(L["f0"]) < 0.08:                       # achromatic metal (chrome, silver, steel, aluminium)
            for j, r in enumerate(spec["regions"]):
                if np.hypot(r["lab"][1], r["lab"][2]) < 10 and r["lab"][0] > 55:
                    cls_tex[mask & (reg == j) & seen_in] = i; print(f"colour prior: region {j} → {cls} (grey metal)")
            continue
        f0lab = _srgb_to_lab(np.where(np.array(L["f0"]) <= 0.0031308, np.array(L["f0"]) * 12.92,
                                      1.055 * np.power(np.array(L["f0"]), 1 / 2.4) - 0.055)[None])[0]
        h0 = np.degrees(np.arctan2(f0lab[2], f0lab[1]))
        for j, r in enumerate(spec["regions"]):
            a, b = r["lab"][1], r["lab"][2]
            chroma = np.hypot(a, b); h = np.degrees(np.arctan2(b, a))
            if chroma > 25 and abs((h - h0 + 180) % 360 - 180) < 20 and np.hypot(f0lab[1], f0lab[2]) > 15:
                sel = mask & (reg == j) & seen_in
                cls_tex[sel] = i
                print(f"colour prior: region {j} → {cls}")
    # unfound rubber / black parts: the near-black regions
    for i, n in enumerate(names):
        cls = n.split("|")[0]
        if cls in ("rubber", "matte plastic") and not (cls_tex == i).any():
            for j, r in enumerate(spec["regions"]):
                if r["lab"][0] < 20:
                    cls_tex[mask & (reg == j) & seen_in] = i; print(f"colour prior: region {j} → {cls} (dark)")
    # a coloured metal (gold, brass, copper) is never white: white, colourless regions it took on the seen side go
    # to a described opaque non-metal part that found no texels (the enamel dial of a gold watch)
    unfound = [i for i, n in enumerate(names) if not (cls_tex == i).any()
               and _lib(n.split("|")[0]).get("metallic", 0) < 0.5 and _lib(n.split("|")[0]).get("transmission", 0) <= 0
               and not any((cls_tex == k).any() for k, m in enumerate(names) if m.split("|")[0] == n.split("|")[0])]
    if unfound:
        tgt = unfound[0]
        for i, n in enumerate(names):
            L = _lib(n.split("|")[0])
            if L.get("metallic", 0) < 0.5 or not L.get("f0") or np.ptp(L["f0"]) < 0.08:
                continue
            for j, r in enumerate(spec["regions"]):
                if r["lab"][0] > 70 and np.hypot(r["lab"][1], r["lab"][2]) < 12:
                    # colour alone decides here (a gold piece cannot be white on any side); only large patches,
                    # so pale glints of a frozen reflection on the metal stay metal
                    sel = mask & (reg == j) & (cls_tex == i)
                    lab_, nl = ndi.label(sel)
                    if nl:
                        sz = np.bincount(lab_.ravel()); sz[0] = 0
                        sel = sz[lab_] > WHITE_PATCH_MIN * mask.sum()
                    if sel.any():
                        cls_tex[sel] = tgt
                        print(f"colour prior: white region {j} is not {n.split('|')[0]} → {names[tgt].split('|')[0]}")
    # a secondary transparent part may not swallow the object (watch "crystal" over the dial): above 25 % its
    # texels go back to the best-voted opaque part
    first_cls = names[0].split("|")[0] if names else None
    labT = _srgb_to_lab(tex)
    for i, n in enumerate(names):
        cls = n.split("|")[0]
        if not cap_transparent or _lib(cls).get("transmission", 0) <= 0 or cls == first_cls:
            continue
        sel = (cls_tex == i) & mask
        if sel.sum() <= 0.25 * mask.sum():
            continue
        if np.median(np.hypot(labT[sel][:, 1], labT[sel][:, 2])) > 20:   # coloured glass (stained glass): keep
            continue
        opaque = [k for k, m in enumerate(names) if _lib(m.split("|")[0]).get("transmission", 0) <= 0]
        if not opaque:
            continue
        bright = labT[sel][:, 0].mean() > 60
        pref = [k for k in opaque if _lib(names[k].split("|")[0]).get("metallic", 0) < 0.5] if bright else []
        cand = pref or opaque
        v = votes[cand][:, sel.reshape(-1)]
        best = cand[int(np.argmax(v.sum(1)))] if v.sum() > 0 else cand[0]
        cls_tex[sel] = best
        print(f"transparent cap: {n} → {names[best]}")
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
    np.save(out / "mat" / "texel_class.npy", cls_tex.astype(np.int16))
    spec["texel_classes"] = names
    spec_p.write_text(json.dumps(spec, indent=1))
    share = {n: round(float((cls_tex[mask] == i).mean()), 4) for i, n in enumerate(names)}
    return {"seen_texels": round(float(seen.reshape(R, R)[mask].mean()), 3), "share": share, "classes": classes}


def apply_texel_classes(spec_json, out_dir=None) -> dict:
    """Roughness/metallic maps from the per-texel classes of from_view_masks (falls back to regions where unknown)."""
    spec = json.loads(Path(spec_json).read_text())
    asset = Path(spec["asset"]); out_dir = Path(out_dir or asset)
    res = apply(spec_json, out_dir)                      # region-based maps + de-lit base colour as the fallback
    try:
        sv = spec.get("seen_view")
        if sv is None:
            sv = best_view(asset / "mat" / "orbit", spec["image"])
        res["lifted_texels"] = fix_invented_side(asset, np.load(asset / "mat" / "texel_class.npy"), spec["texel_classes"], sv)
    except Exception as e:
        res["lifted_texels"] = f"skipped: {e}"
    cls_tex = np.load(asset / "mat" / "texel_class.npy")
    names = spec["texel_classes"]
    try:                                   # described glass the splat left as an opening (car windows)
        from .openings import glass_openings
        tot = max(int((cls_tex >= 0).sum()), 1)
        share = {n: float((cls_tex == i).sum()) / tot for i, n in enumerate(names)}
        res["glass_openings"] = glass_openings(asset, names, share)
        gi = [i for i, n in enumerate(names) if _lib(n.split("|")[0]).get("transmission", 0) > 0]
        if gi and (asset / "glass_openings.npz").exists():
            from .openings import surface_glass_texels
            sg = surface_glass_texels(asset, cls_tex.shape[0]) & (cls_tex >= 0)
            if sg.any():
                cls_tex = cls_tex.copy(); cls_tex[sg] = gi[0]
                res["glass_openings"]["surface_glass_texels"] = int(sg.sum())
        gi = [i for i, n in enumerate(names) if _lib(n.split("|")[0]).get("transmission", 0) > 0]

    except Exception as e:
        res["glass_openings"] = f"skipped: {e}"
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
    res["finish"] = finish_maps(asset, cls_tex, names)
    res["glass_smoothed"] = smooth_glass(asset, cls_tex, names)
    res["shading_smooth_iters"] = smooth_shading_normals(asset, cls_tex, names)
    res["glass_wall_faces"] = glass_walls(asset, cls_tex, names)
    return res


def glass_walls(asset_dir, cls_tex, names, min_thickness: float = GLASS_WALL_MIN) -> int:
    """Direct mode builds solids, but a glass bulb, bowl or bottle is a thin wall around air: refracting through
    the whole filled volume shows its inner lumps as grain. Glass faces where the solid is thick (inward ray to
    the opposite side > min_thickness × object size) are bound to a thin-walled copy of the material through a
    GeomSubset; thin glass (a stem, a foot, a rod) stays solid and refracts. Returns the number of wall faces."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt
    import open3d as o3d
    asset = Path(asset_dir)
    _, mask = _texture_and_mask(asset)
    R = mask.shape[0]
    glass_cls = [i for i, n in enumerate(names) if _lib(n.split("|")[0]).get("transmission", 0) > 0]
    st = Usd.Stage.Open(str(asset / "asset.usda"))
    geom = st.GetPrimAtPath("/Asset/Geom"); m = UsdGeom.Mesh(geom)
    for c in geom.GetChildren():                          # idempotent: drop an earlier split
        if c.GetName() == "glass_walls":
            st.RemovePrim(c.GetPath())
    mat_prim = UsdShade.MaterialBindingAPI(geom).ComputeBoundMaterial()[0].GetPrim()   # the object's own material
    thin_path = Sdf.Path(str(mat_prim.GetPath()) + "_thin")
    if st.GetPrimAtPath(thin_path):
        st.RemovePrim(thin_path)
    n_wall = 0
    if glass_cls:
        V = np.asarray(m.GetPointsAttr().Get(), np.float32)
        F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
        uv = np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar("st").Get(), np.float64)
        c = uv[F].mean(1)
        fcls = cls_tex[np.clip((c[:, 1] * R).astype(int), 0, R - 1), np.clip((c[:, 0] * R).astype(int), 0, R - 1)]
        fg = np.nonzero(np.isin(fcls, glass_cls))[0]
        if len(fg):
            tri = V[F[fg]]
            cen = tri.mean(1)
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(V, F.astype(np.uint32))
            size = float(np.linalg.norm(V.max(0) - V.min(0)))
            eps = 1e-4 * size
            def cast(o, d, off=eps):
                rays = np.concatenate([o + d * off, d], 1).astype(np.float32)
                return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            d = -nrm
            t_in = cast(cen, d)                                 # outward normals: the opposite side is inward
            if np.isinf(t_in).mean() > 0.5:                     # inward-facing winding
                d = nrm; t_in = cast(cen, d)
            hit = np.isfinite(t_in)
            t_next = np.full(len(cen), np.inf, np.float32)      # beyond the first exit: air, then another wall?
            t_next[hit] = cast(cen[hit] + d[hit] * t_in[hit, None], d[hit], off=2e-3 * size)
            filled = hit & (t_in > min_thickness * size)        # a filled solid (bulb made solid by direct mode)
            shell = hit & ~filled & np.isfinite(t_next) & (t_next > min_thickness * size)   # thin wall, air behind
            score = (filled | shell).astype(np.float64)
            # coherent regions, not a patchwork: average the decision over neighbouring faces (via vertices)
            Fg = F[fg]; nv = len(V)
            for _ in range(8):
                acc = np.zeros(nv); cnt = np.zeros(nv)
                np.add.at(acc, Fg.ravel(), np.repeat(score, 3)); np.add.at(cnt, Fg.ravel(), 1)
                score = (acc / np.maximum(cnt, 1))[Fg].mean(1)
            thick = score > 0.5
            wall = fg[thick]
            n_wall = int(len(wall))
            if n_wall:
                Sdf.CopySpec(st.GetRootLayer(), mat_prim.GetPath(), st.GetRootLayer(), thin_path)
                for p in Usd.PrimRange(st.GetPrimAtPath(thin_path)):     # CopySpec already remaps internal paths
                    if p.GetName() == "StandardSurface":
                        ssh = UsdShade.Shader(p)
                        ssh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
                        # a thin-walled sheet is already two interfaces; the wall's inner and outer surfaces are
                        # both sheets here, so each carries half the reflectance (one real wall = two interfaces)
                        ssh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.5)
                sub = UsdGeom.Subset.Define(st, geom.GetPath().AppendChild("glass_walls"))
                sub.CreateElementTypeAttr(UsdGeom.Tokens.face)
                sub.CreateIndicesAttr(Vt.IntArray.FromNumpy(wall.astype(np.int32)))
                sub.CreateFamilyNameAttr(UsdShade.Tokens.materialBind)
                UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(UsdShade.Material(st.GetPrimAtPath(thin_path)))
                UsdGeom.Subset.SetFamilyType(m, UsdShade.Tokens.materialBind, UsdGeom.Tokens.nonOverlapping)
    st.GetRootLayer().Save()
    return n_wall


def export_parts(asset_dir, out_name: str = "asset_parts.usda", group: str = "part") -> dict:
    """Split the textured mesh into one Mesh prim per part (ENEAS phrase, else material class of the face's colour
    region), each with its own material — /Asset/Parts/<part>, /Asset/Materials/M_<part> — so Cryptomatte
    (CryptoObject from the prim path, e.g. athenea) yields one matte per part. Textures are shared."""
    import re
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    from .rerender import load_usd_textured
    asset = Path(asset_dir)
    spec = json.loads((asset / "mat" / "regions.json").read_text())
    V, F, st, _ = load_usd_textured(str(asset / "asset.usda"))
    src = Usd.Stage.Open(str(asset / "asset.usda"))
    N = np.asarray(UsdGeom.Mesh(src.GetPrimAtPath("/Asset/Geom")).GetNormalsAttr().Get(), np.float32)
    tex, mask = _texture_and_mask(asset)
    fin = json.loads((asset / "mat" / "finish.json").read_text()) if (asset / "mat" / "finish.json").exists() else {}
    R = tex.shape[0]
    cen = st[F].mean(1)
    px = np.clip((cen * R).astype(int), 0, R - 1)
    tc_path = asset / "mat" / "texel_class.npy"
    if tc_path.exists() and spec.get("texel_classes"):
        names = spec["texel_classes"]
        lab = np.load(tc_path)[px[:, 1], px[:, 0]]
        part_names = [n.split("|", 1)[1] if "|" in n else n for n in names]
        part_cls = [n.split("|", 1)[0] for n in names]
    else:
        C = np.array([r["lab"] for r in spec["regions"]])
        lab = ((_srgb_to_lab(tex[px[:, 1], px[:, 0]])[:, None] - C[None]) ** 2).sum(-1).argmin(-1)
        part_cls = [r.get("material", "glazed ceramic") for r in spec["regions"]]
        part_names = part_cls
    lab = np.where(lab < 0, int(np.bincount(lab[lab >= 0]).argmax()) if (lab >= 0).any() else 0, lab)
    if group == "material":          # one prim (and one material) per material class: Cryptomatte per material
        classes = sorted(set(part_cls))
        lab = np.array([classes.index(part_cls[i]) for i in lab])
        part_names, part_cls = classes, classes
    # files referenced by the single-material version
    files = {}
    for prim in src.Traverse():
        if prim.GetTypeName() == "Shader" and "M_interior" not in str(prim.GetPath()) and "M_glass_openings" not in str(prim.GetPath()):
            f = UsdShade.Shader(prim).GetInput("file")
            if f and f.Get():
                files[prim.GetName()] = f.Get().path
    st_out = Usd.Stage.CreateNew(str(asset / out_name))
    UsdGeom.SetStageUpAxis(st_out, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st_out, 1.0)
    root = UsdGeom.Xform.Define(st_out, "/Asset"); st_out.SetDefaultPrim(root.GetPrim())
    slug = lambda s: re.sub(r"[^A-Za-z0-9_]", "_", re.sub(r"^the\s+", "", s.strip().lower())).strip("_")[:48] or "part"
    used, parts = set(), {}
    for i in np.unique(lab):
        name = slug(part_names[i]); k = name; n = 2
        while k in used:
            k = f"{name}_{n}"; n += 1
        used.add(k)
        fi = np.nonzero(lab == i)[0]
        vid, inv = np.unique(F[fi].ravel(), return_inverse=True)
        grp = f"/Asset/{'Parts' if group == 'part' else 'MaterialGroups'}"
        UsdGeom.Scope.Define(st_out, grp)
        UsdGeom.Xform.Define(st_out, f"{grp}/{k}")
        mesh = UsdGeom.Mesh.Define(st_out, f"{grp}/{k}/Geom")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V[vid].astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(fi), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(inv.astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N[vid])); mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        lo, hi = V[vid].min(0), V[vid].max(0)
        mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(st[vid].astype(np.float32)))
        cls = part_cls[i]
        mpath = f"/Asset/Materials/M_{k}"
        mat = UsdShade.Material.Define(st_out, mpath)
        pbr = UsdShade.Shader.Define(st_out, mpath + "/PreviewSurface"); pbr.CreateIdAttr("UsdPreviewSurface")
        rd = UsdShade.Shader.Define(st_out, mpath + "/stReader"); rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        for shname, inp, typ, out in (("BaseColor", "diffuseColor", Sdf.ValueTypeNames.Color3f, "rgb"),
                                      ("Roughness", "roughness", Sdf.ValueTypeNames.Float, "r"),
                                      ("Metallic", "metallic", Sdf.ValueTypeNames.Float, "r")):
            if shname not in files:
                continue
            t = UsdShade.Shader.Define(st_out, f"{mpath}/{shname}"); t.CreateIdAttr("UsdUVTexture")
            t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(files[shname])
            t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if shname == "BaseColor" else "raw")
            t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
            t.CreateOutput(out, Sdf.ValueTypeNames.Float3 if out == "rgb" else Sdf.ValueTypeNames.Float)
            pbr.CreateInput(inp, typ).ConnectToSource(t.ConnectableAPI(), out)
        ior = (LIB.get(cls) or LIB.get(VLM_TO_LIB.get(cls, ""), {})).get("ior")
        if ior:
            pbr.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        mesh.GetPrim().SetCustomDataByKey("splat2mesh:material", cls)
        L = {**_lib(cls), **fin.get(cls, {})}
        parts[k] = {"material": cls, "faces": int(len(fi)), "transmission": L.get("transmission", 0.0),
                    "patterned": bool(L.get("patterned", False)), "tint": L.get("tint"),
                    "subsurface": L.get("subsurface", 0.0), "ior": L.get("ior"), "roughness": L.get("roughness"),
                    "metallic": L.get("metallic", 0)}
    if (asset / "glass_openings.npz").exists():   # glass sheets over openings (see openings.py)
        from .openings import define_glass
        grp = f"/Asset/{'Parts' if group == 'part' else 'MaterialGroups'}"
        UsdGeom.Xform.Define(st_out, f"{grp}/glass_panes")
        gm = define_glass(st_out, f"{grp}/glass_panes/Geom", asset, "/Asset/Materials/M_glass_panes")
        parts["glass_panes"] = {"material": "glass", "faces": len(gm.GetFaceVertexCountsAttr().Get()), "transmission": 1.0,
                                "patterned": False, "subsurface": 0.0, "ior": 1.5,
                                "tint": [float(x) for x in np.load(asset / "glass_openings.npz").get("tint", [0.95, 0.97, 1.0])],
                                "roughness": 0.03, "metallic": 0}
    if (asset / "interior.npz").exists():      # inner structure of transparent parts (see interior.py)
        from .interior import define_interior
        grp = f"/Asset/{'Parts' if group == 'part' else 'MaterialGroups'}"
        UsdGeom.Xform.Define(st_out, f"{grp}/interior")
        im = define_interior(st_out, f"{grp}/interior/Geom", asset, "/Asset/Materials/M_interior")
        parts["interior"] = {"material": "interior", "faces": len(im.GetFaceVertexCountsAttr().Get()), "transmission": 0.0,
                             "patterned": False, "tint": None, "subsurface": 0.0, "ior": None, "roughness": 0.45,
                             "metallic": 0}
    st_out.GetRootLayer().Save()
    (asset / ("parts.json" if group == "part" else "material_groups.json")).write_text(json.dumps(parts, indent=1))
    return parts


def _lib(cls):
    return LIB.get(cls) or LIB.get(VLM_TO_LIB.get(cls, ""), {})


def best_view(orbit_dir, image) -> int:
    """Index of the orbit frame that looks most like the input picture (the side the generator really saw)."""
    def crop(path, bg_dark):
        a = np.asarray(Image.open(path).convert("RGB"), np.float32)
        fg = a.max(-1) > 10 if bg_dark else a.min(-1) < 245
        ys, xs = np.nonzero(fg)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((96, 96)), np.float32)
    ref = crop(image, True)
    frames = sorted((Path(orbit_dir) / "frames").glob("*.jpg"))
    return int(np.argmax([-np.abs(crop(f, False) - ref).mean() for f in frames]))


def _seen_mask(asset: Path, R: int, seen_view: int) -> np.ndarray:
    """Texels whose surface faces the input picture's camera (orbit frame seen_view: azimuth 2πi/16, elevation
    15°). Geometric, so it is dense at any atlas size (pixel maps of a 768² view are not)."""
    _, nrm, msk = _raster(asset, R)
    n_views = len(list((asset / "mat" / "orbit" / "frames").glob("*.jpg"))) or 16
    a, e = 2 * np.pi * seen_view / n_views, np.radians(15.0)
    d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    return msk & ((nrm @ d) > 0.15)


def fix_invented_side(asset_dir, cls_tex, names, seen_view: int, ratio: float = 0.65) -> int:
    """Idol invents what the input picture did not show and often paints it darker (dark backs, black bases).
    For each material class, texels seen in the input's own view set the reference; texels of the same class
    that the view never saw and that are much darker than that reference get their luminance lifted to it
    (chroma kept), so a bust is not black behind. Returns the number of texels changed."""
    asset = Path(asset_dir)
    base = np.asarray(Image.open(asset / "basecolor_delit.png").convert("RGB"), np.float32)[::-1] / 255
    R = base.shape[0]
    # "seen" = the texel's surface faces the input picture's camera (orbit frame seen_view: azimuth 2πi/16,
    # elevation 15°). Geometric, so it is dense at any atlas size (pixel maps of a 768² view are not).
    seen = _seen_mask(asset, R, seen_view)
    lab = _srgb_to_lab(base)
    changed = 0
    for i in range(len(names)):
        cls_mask = cls_tex == i
        ref = cls_mask & seen
        if ref.sum() < 200:
            continue
        Lref = np.median(lab[..., 0][ref])
        bad = cls_mask & ~seen & (lab[..., 0] < ratio * Lref)
        # a one-colour piece (rubber, paint) invented much LIGHTER behind is brought down the same way
        abr = lab[..., 1:][ref]
        # unless the seen side itself shows that lightness (an inlay, a panel, a label): then light texels behind
        # are the same features continuing (nacre panels of a black box), not an invented lighter colour
        seen_light = float((lab[..., 0][ref] > Lref / ratio).mean())
        if (np.median(np.hypot(*(abr - np.median(abr, 0)).T)) < GLOSS_UNIFORM_SPREAD and Lref < 70
                and seen_light < SEEN_LIGHT_MAX):
            bad |= cls_mask & ~seen & (lab[..., 0] > Lref / ratio)
        if bad.sum() == 0:
            continue
        gain = np.clip(Lref / np.maximum(lab[..., 0][bad], 1.0), 0.15, 4.0)
        lin = np.where(base[bad] <= 0.04045, base[bad] / 12.92, ((base[bad] + 0.055) / 1.055) ** 2.4)
        lin = np.clip(lin * (gain[:, None] ** 1.6), 0, 1)
        base[bad] = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
        changed += int(bad.sum())
    Image.fromarray((base[::-1].clip(0, 1) * 255).astype(np.uint8)).save(asset / "basecolor_delit.png")
    return changed


def _orig_points(asset: Path, m):
    """Vertex positions as built (before any finish smoothing), saved once so re-applying stays idempotent."""
    from pxr import Vt
    f = asset / "mat" / "points_orig.npy"
    V = np.asarray(m.GetPointsAttr().Get(), np.float64)
    if f.exists():
        V0 = np.load(f)
        if V0.shape == V.shape:
            return V0
    ply = asset / "mesh.ply"                     # written by the geometry stage, before any finish pass
    if ply.exists():
        import trimesh
        tm = trimesh.load(ply, process=False)
        F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
        if np.asarray(tm.vertices).shape == V.shape and np.array_equal(np.asarray(tm.faces), F):
            V = np.asarray(tm.vertices, np.float64)
    f.parent.mkdir(exist_ok=True); np.save(f, V)
    return V


def smooth_glass(asset_dir, cls_tex, names, iterations: int = GLASS_SMOOTH_ITERS, min_share: float = 0.02) -> bool:
    """Glass, crystal, jade and wax show every bump of the fused surface (grain on a wine glass). Their vertices are
    Taubin-smoothed — only theirs, with a soft falloff at the border so metal or wood parts keep their shape —
    starting from the original positions every time (idempotent). Seam duplicates merged; UVs untouched."""
    from pxr import Usd, UsdGeom, Vt
    import scipy.sparse as sp
    import trimesh
    asset = Path(asset_dir)
    tex, mask = _texture_and_mask(asset)
    R = mask.shape[0]
    glass_cls = [i for i, n in enumerate(names)
                 if _lib(n.split("|")[0]).get("transmission", 0) > 0 or _lib(n.split("|")[0]).get("subsurface", 0) >= 0.5]
    st = Usd.Stage.Open(str(asset / "asset.usda"))
    m = UsdGeom.Mesh(st.GetPrimAtPath("/Asset/Geom"))
    V = _orig_points(asset, m)
    if not glass_cls:
        return False
    F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
    uv = np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar("st").Get(), np.float64)
    c = uv[F].mean(1)
    fcls = cls_tex[np.clip((c[:, 1] * R).astype(int), 0, R - 1), np.clip((c[:, 0] * R).astype(int), 0, R - 1)]
    fglass = np.isin(fcls, glass_cls)
    if fglass.mean() < min_share:
        return False
    key = np.round(V / (np.ptp(V, 0).max() * 1e-6)).astype(np.int64)
    _, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1); Fu = inv[F]; n = len(uid)
    w = np.zeros(n); np.add.at(w, Fu[fglass].ravel(), 1.0); cnt = np.zeros(n); np.add.at(cnt, Fu.ravel(), 1.0)
    w = (w / np.maximum(cnt, 1))                      # share of glass faces around each vertex: 1 inside, 0 outside
    e = np.concatenate([Fu[:, [0, 1]], Fu[:, [1, 2]], Fu[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    D = np.asarray(A.sum(1)).ravel()
    P = V[uid].copy()
    for _ in range(iterations):                       # Taubin λ|μ, weighted per vertex
        for f in (0.5, -0.53):
            P += (f * w)[:, None] * ((A @ P) / np.maximum(D, 1)[:, None] - P)
    Vn = P[inv]
    m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(Vn.astype(np.float32)))
    nrm = np.asarray(trimesh.Trimesh(P, Fu, process=False).vertex_normals)[inv]
    m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
    st.GetRootLayer().Save()
    return True


def _bilateral_normals(V, F, iters: int, sigma: float):
    """Edge-preserving smoothing of face normals (bilateral over edge-adjacent faces), then area-weighted
    vertex normals. Positions are not touched."""
    import trimesh, scipy.sparse as sp
    tm = trimesh.Trimesh(V, F, process=False)
    n = tm.face_normals.copy(); a = tm.area_faces
    adj = tm.face_adjacency
    i = np.r_[adj[:, 0], adj[:, 1]]; j = np.r_[adj[:, 1], adj[:, 0]]
    for _ in range(iters):
        w = np.exp(-((1 - (n[i] * n[j]).sum(1)) / sigma) ** 2) * a[j]
        m = sp.csr_matrix((w, (i, j)), shape=(len(F), len(F))) @ n + a[:, None] * n
        n = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], n * a[:, None])
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def smooth_shading_normals(asset_dir, cls_tex, names) -> int:
    """Reflections expose every small bump of the fused surface. Keep the positions (silhouette, detail) and write
    vertex normals taken from a Taubin-smoothed copy of the mesh, so glossy porcelain, lacquer and polished metal
    reflect like finished pieces. Stronger when the object is mostly glossy. Returns the iterations used."""
    from pxr import Usd, UsdGeom, Vt
    import trimesh
    asset = Path(asset_dir)
    _, mask = _texture_and_mask(asset)
    # glossiness from the finished roughness map (it includes finishes decided from the texture, e.g. polish)
    Rm = np.asarray(Image.open(asset / "roughness.png"), np.float32)[::-1] / 255
    if Rm.ndim == 3:
        Rm = Rm[..., 0]
    if Rm.shape != mask.shape:
        Rm = np.asarray(Image.fromarray((Rm * 255).astype(np.uint8)).resize(mask.shape[::-1]), np.float32) / 255
    glossy = ((Rm < 0.32) & mask).sum() / max(mask.sum(), 1)
    mirror = ((Rm < 0.2) & mask).sum() / max(mask.sum(), 1)
    it = 40 if glossy > 0.5 else 20
    st = Usd.Stage.Open(str(asset / "asset.usda"))
    m = UsdGeom.Mesh(st.GetPrimAtPath("/Asset/Geom"))
    V = np.asarray(m.GetPointsAttr().Get(), np.float64)
    F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
    key = np.round(V / (np.ptp(V, 0).max() * 1e-6)).astype(np.int64)
    _, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    if glossy > 0.5 or mirror > 0.5:
        # glossy: bilateral filtering of the face normals. Ripples and terracing of the fused surface (a few
        # degrees) are averaged away over a wide area while edges and corners (tens of degrees) stay sharp;
        # Taubin on the positions either leaves the ripples or rounds the corners
        nrm = _bilateral_normals(V[uid], inv[F], BILATERAL_ITERS, BILATERAL_SIGMA)[inv]
        it = BILATERAL_ITERS
    else:
        tm = trimesh.Trimesh(V[uid], inv[F], process=False)
        trimesh.smoothing.filter_taubin(tm, lamb=0.5, nu=0.53, iterations=it)
        nrm = np.asarray(tm.vertex_normals)[inv]
    m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st.GetRootLayer().Save()
    return it




def finish_maps(asset_dir, cls_tex, names, metal_detail: float = 0.35) -> dict:
    """Second pass over the per-texel classes: physical base colour for metals (library F0 × the texture's own
    luminance detail, so engraving/hammering stays), transmission & subsurface maps, opacity for the preview
    surface, and a MaterialX standard_surface next to the UsdPreviewSurface."""
    asset = Path(asset_dir)
    tex, mask = _texture_and_mask(asset)
    base = np.asarray(Image.open(asset / "basecolor_delit.png").convert("RGB"), np.float32)[::-1] / 255
    lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    srgb = lambda c: np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)
    T = np.zeros(mask.shape, np.float32); S = np.zeros(mask.shape, np.float32)
    hp_cache = {}
    Rmap = np.asarray(Image.open(asset / "roughness.png"), np.float32)[::-1] / 255
    Mmap = np.asarray(Image.open(asset / "metallic.png"), np.float32)[::-1] / 255
    info = {}
    for i, n in enumerate(names):
        cls = n.split("|")[0]; L = _lib(cls); sel = (cls_tex == i) & mask
        if not sel.any():
            continue
        if L.get("metallic", 0) >= 0.5 and L.get("f0"):
            lum0 = lin(base[sel]).mean(-1)
            labm = _srgb_to_lab(base[sel])
            dark = (lum0 < 0.08) & (np.hypot(labm[:, 1], labm[:, 2]) < 10)   # black ink/enamel (colourless): not metal; tinted shadows stay metal
            if dark.any():
                ys, xs = np.nonzero(sel)
                # ink and engraving are thin strokes; a wide dark patch on a metal is a frozen reflection of a dark
                # surrounding (chrome toaster, silver teapot): it stays metal, only thin strokes become non-metal
                dimg = np.zeros(sel.shape, bool); dimg[ys[dark], xs[dark]] = True
                blob = ndi.binary_dilation(ndi.binary_opening(dimg, iterations=INK_MAX_WIDTH), iterations=2)
                dark &= ~blob[ys, xs]
                Mmap[ys[dark], xs[dark]] = 0.0; Rmap[ys[dark], xs[dark]] = 0.3
                sel = sel.copy(); sel[ys[dark], xs[dark]] = False
            # keep only the FINE luminance detail (hammering, engraving, grain): high-pass of log-luminance; the
            # broad variation is frozen reflections/lighting and is discarded
            if hp_cache.get("hp") is None:
                lum_all = np.log(lin(base).mean(-1) + 1e-3)
                hp_cache["hp"] = lum_all - ndi.gaussian_filter(lum_all, METAL_DETAIL_SIGMA)
            f0 = metal_f0(L["f0"], lin(base[sel]), _srgb_to_lab(base[sel])).astype(np.float32)
            detail = np.exp(0.9 * hp_cache["hp"][sel])
            detail = np.clip(detail, METAL_DETAIL_MIN, min(1.0 / f0.max(), 2.0))
            base[sel] = srgb(f0[None] * detail[:, None])
        glossy_ev = False
        if (L.get("metallic", 0) < 0.5 and L.get("transmission", 0) <= 0 and L.get("roughness", 0.5) > 0.4
                and L.get("range", [1])[0] <= GLOSS_EVIDENCE_MAX_ROUGH and sel.sum() > 2000):
            # a material that can be either satin or polished (leather, varnished wood, satin paint): the texture
            # decides. Frozen specular highlights (paler and brighter than the piece's own colour, on a single-colour
            # piece) are evidence of a polished finish → the low end of its roughness range, highlights removed below
            lab_s = _srgb_to_lab(base[sel])
            ab0 = np.median(lab_s[:, 1:], 0); L0 = np.median(lab_s[:, 0]); c0 = float(np.hypot(*ab0))
            spread = float(np.median(np.hypot(*(lab_s[:, 1:] - ab0).T)))
            frac = float(((np.hypot(lab_s[:, 1], lab_s[:, 2]) < 0.8 * c0) & (lab_s[:, 0] > L0 + 12)).mean())
            if c0 > 15 and spread < GLOSS_UNIFORM_SPREAD and frac > GLOSS_EVIDENCE_FRAC:
                glossy_ev = True
                r_lo = max(0.25, float(L["range"][0]) - 0.1)     # polished: below the range's satin end
                Rmap[sel] = np.minimum(Rmap[sel], r_lo)
                print(f"polish {cls}: {frac:.1%} highlight texels → roughness {r_lo}")
        if (L.get("metallic", 0) < 0.5 and L.get("transmission", 0) <= 0 and (L.get("roughness", 0.5) <= 0.4 or glossy_ev)
                and sel.sum() > 2000):
            # glossy single-colour dielectric (paint, lacquer, plastic): a frozen reflection mixes the colour with
            # white — paler and brighter than the piece's own colour. Uniform pieces only (patterned porcelain
            # keeps its patterns): those texels go back to the median colour, keeping some fine detail
            lab_s = _srgb_to_lab(base[sel])
            ab0 = np.median(lab_s[:, 1:], 0); L0 = np.median(lab_s[:, 0]); c0 = float(np.hypot(*ab0))
            spread = float(np.median(np.hypot(*(lab_s[:, 1:] - ab0).T)))
            if c0 > 15 and spread < GLOSS_UNIFORM_SPREAD:
                c = np.hypot(lab_s[:, 1], lab_s[:, 2])
                hl = (c < 0.8 * c0) & (lab_s[:, 0] > L0)
                if hl.any():
                    fix = lab_s[hl].copy()
                    fix[:, 0] = L0 + 0.3 * (fix[:, 0] - L0)
                    fix[:, 1:] = ab0[None]
                    ys, xs = np.nonzero(sel)
                    base[ys[hl], xs[hl]] = _lab_to_srgb(fix)
                    print(f"gloss {cls}: {int(hl.sum())} pale-highlight texels back to the piece colour")
        trans = L.get("transmission", 0.0); patterned = False; tint = None
        if trans > 0:      # glass: what the splat shows is mostly the background seen through it
            ab = _srgb_to_lab(base[sel])[:, 1:]
            spread = float(np.median(np.hypot(*(ab - np.median(ab, 0)).T)))   # hue diversity of the glass texels
            patterned = spread > GLASS_PATTERN_SPREAD     # stained / multi-coloured glass: the colours are the object
            print(f"glass {cls}: colour spread {spread:.1f} → {'patterned (colour kept)' if patterned else 'uniform tint'}")
            if patterned:
                trans = trans * 0.5                       # coloured glass lets less light through, shows its colours
            elif (asset / "glass_openings.npz").exists() and "tint" in np.load(asset / "glass_openings.npz").files:
                tint = np.load(asset / "glass_openings.npz")["tint"].astype(np.float64)   # same glass as the panes
            else:
                tint = np.median(lin(base[sel]), 0)
                tint = tint / max(tint.max(), 1e-4) * 0.95        # keep the hue, drop the frozen brightness
                base[sel] = srgb(tint)[None]
        T[sel] = trans; S[sel] = L.get("subsurface", 0.0)
        info[cls] = {"transmission": trans, "subsurface": L.get("subsurface", 0.0), "patterned": patterned,
                     "tint": None if tint is None else [round(float(x), 4) for x in tint],   # linear RGB
                     "thin_walled": L.get("thin_walled", False), "ior": L.get("ior")}
    from .texture import inpaint
    # spread the maps past the chart borders (nearest texel) so filtering at UV seams does not draw opaque lines
    _, (iy, ix) = ndi.distance_transform_edt(~mask, return_indices=True)
    T, S = T[iy, ix], S[iy, ix]
    for name, img in (("transmission", T), ("subsurface", S)):
        Image.fromarray((img[::-1].clip(0, 1) * 255).astype(np.uint8)).save(asset / f"{name}.png")
    Image.fromarray(((1 - 0.85 * T)[::-1].clip(0, 1) * 255).astype(np.uint8)).save(asset / "opacity.png")
    # transmission colour: the glass tint applies at every wall a ray crosses (two for a shell), so the per-wall
    # colour is the measured tint's square root; opaque texels stay white (no effect)
    tc = np.ones(base.shape, np.float32)
    lb = lin(base)
    tc = np.where((T > 0)[..., None], np.clip(lb, 1e-3, 1) ** (1 / GLASS_WALLS), 1.0)[iy, ix]
    Image.fromarray((srgb(np.clip(tc, 0, 1))[::-1] * 255).astype(np.uint8)).save(asset / "transmission_color.png")
    for name, img in (("roughness", Rmap), ("metallic", Mmap)):
        Image.fromarray((img[::-1].clip(0, 1) * 255).astype(np.uint8)).save(asset / f"{name}.png")
    Image.fromarray((np.stack([np.ones_like(Rmap), Rmap, Mmap], -1)[::-1].clip(0, 1) * 255).astype(np.uint8)).save(asset / "orm.png")
    Image.fromarray((base[::-1].clip(0, 1) * 255).astype(np.uint8)).save(asset / "basecolor_delit.png")
    has_t, has_s = bool((T > 0).any()), bool((S > 0).any())
    iors = [v["ior"] for v in info.values() if v["ior"]]
    _usd_add_mtlx(asset / "asset.usda", has_t, has_s, ior=float(np.median(iors)) if iors else 1.5,
                  thin=any(v["thin_walled"] and v["transmission"] > 0 for v in info.values()))
    (asset / "mat" / "finish.json").write_text(json.dumps(info, indent=1, default=float))
    return {"transmissive": has_t, "subsurface": has_s, "classes": info}


def _usd_add_mtlx(usd_path: Path, has_t: bool, has_s: bool, ior: float = 1.5, thin: bool = False):
    """MaterialX standard_surface (render context 'mtlx') beside the UsdPreviewSurface; same textures."""
    from pxr import Sdf, Usd, UsdShade
    stage = Usd.Stage.Open(str(usd_path))
    mat_prim = next(p for p in stage.Traverse() if p.GetTypeName() == "Material")
    mp = mat_prim.GetPath(); mat = UsdShade.Material(mat_prim)
    pbr = UsdShade.Shader(stage.GetPrimAtPath(mp.AppendChild("PreviewSurface")))
    if has_t:                                     # preview surface: per-texel opacity + ior
        t = UsdShade.Shader.Define(stage, mp.AppendChild("Opacity")); t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./opacity.png")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        rd = UsdShade.Shader(stage.GetPrimAtPath(mp.AppendChild("stReader")))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        t.CreateOutput("r", Sdf.ValueTypeNames.Float)
        pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "r")
        pbr.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    ng = mp.AppendChild("MtlxGraph")
    def img(name, file, typ="float"):
        s = UsdShade.Shader.Define(stage, ng.AppendChild(name))
        s.CreateIdAttr(f"ND_image_{typ}")
        fi = s.CreateInput("file", Sdf.ValueTypeNames.Asset); fi.Set(file)
        fi.GetAttr().SetColorSpace("srgb_texture" if typ == "color3" else "raw")
        out = s.CreateOutput("out", Sdf.ValueTypeNames.Color3f if typ == "color3" else Sdf.ValueTypeNames.Float)
        return s
    ss = UsdShade.Shader.Define(stage, mp.AppendChild("StandardSurface")); ss.CreateIdAttr("ND_standard_surface_surfaceshader")
    bc = img("base_color", "./basecolor_delit.png", "color3")
    ss.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(bc.ConnectableAPI(), "out")
    ss.CreateInput("base", Sdf.ValueTypeNames.Float).Set(1.0)
    for inp, f in (("specular_roughness", "./roughness.png"), ("metalness", "./metallic.png")):
        ss.CreateInput(inp, Sdf.ValueTypeNames.Float).ConnectToSource(img(inp, f).ConnectableAPI(), "out")
    ss.CreateInput("specular_IOR", Sdf.ValueTypeNames.Float).Set(ior)
    if has_t:
        ss.CreateInput("transmission", Sdf.ValueTypeNames.Float).ConnectToSource(img("transmission", "./transmission.png").ConnectableAPI(), "out")
        tcol = img("transmission_color", "./transmission_color.png", "color3")                 # per-wall glass tint
        ss.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(tcol.ConnectableAPI(), "out")
        ss.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(bool(thin))
    if has_s:
        ss.CreateInput("subsurface", Sdf.ValueTypeNames.Float).ConnectToSource(img("subsurface", "./subsurface.png").ConnectableAPI(), "out")
        ss.CreateInput("subsurface_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(bc.ConnectableAPI(), "out")
        ss.CreateInput("subsurface_radius", Sdf.ValueTypeNames.Color3f).Set((1.0, 0.6, 0.4))
    ss.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mtlx").ConnectToSource(ss.ConnectableAPI(), "out")
    stage.GetRootLayer().Save()
