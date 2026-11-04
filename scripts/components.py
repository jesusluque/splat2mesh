"""Per-component pipeline: one reference picture of the whole object + the list of its components → one closed mesh
per component, each generated on its own (Idol) and placed where it is in the whole object.

    python scripts/components.py REFERENCE.png WHOLE_ASSET_DIR OUT_DIR "the bulb=glass" "the screw base=brass" \
           "the base=brass: the thread; black glass: the insulator"

Each component is "phrase=materials": one library material (the component is all of it) or a material description
"material: part, part; material: part" (the component gets its own material stage: ENEAS on its orbit, per-texel
maps, like any asset).

WHOLE_ASSET_DIR is the object made the usual way (splat2usd + materials_eneas3d): it gives the layout (where each
part is, at what scale) through its per-texel material votes. Steps, each cached in OUT_DIR:
 1. ENEAS segments every component phrase on the reference picture itself (ixwin).
 2. Each component is cut out of the picture (its mask on white) → component image.
 3. Idol + direct meshing per component image (splat2usd --direct) → a closed mesh with its own UVs and texture.
 4. The component mesh is placed in its box from the reference mask (the whole mesh faces the picture), then ICP
    with scale (open3d). Components ENEAS / Idol could not produce keep the whole mesh's faces for that phrase.
 5. One USD: /Asset/Components/<part> (mesh + material from the library: metal F0, glass transmission, ...).
"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from splat2mesh.materials import parse_description, _lib

HOST = "ixwin-tb"
MIN_MASK = 0.004          # component mask below this share of the picture: too small for its own generation
PAD = 0.12                # margin around the component cut-out


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower().replace("the ", "")).strip("_")


def eneas_on_reference(ref: Path, phrases, out_npz: Path):
    """ENEAS on a one-frame 'sequence' (the reference picture), one mask per phrase."""
    if out_npz.exists() and set(np.load(out_npz).files) >= set(phrases):
        return
    win = "C:/splat2mesh/eneas_jobs/components_" + ref.stem; wsl = "/mnt/c/" + win[3:]
    fr = out_npz.parent / "ref_frames"; fr.mkdir(exist_ok=True)
    Image.open(ref).convert("RGB").save(fr / "frame_0000.jpg", quality=95)
    subprocess.run(["ssh", HOST, f'rmdir /s /q "{win}" 2>nul & mkdir "{win}\\frames"'], check=False)
    subprocess.run(["scp", "-q", str(fr / "frame_0000.jpg"), f"{HOST}:{win}/frames/"], check=True)
    subprocess.run(["scp", "-q", str(ROOT / "scripts/eneas_seq.py"), f"{HOST}:{win}/"], check=True)
    q = " ".join("'" + p.replace("'", "") + "'" for p in phrases)
    inner = f"~/.local/bin/uv run python {wsl}/eneas_seq.py {wsl}/frames {wsl}/masks.npz frame_0000.jpg {q}"
    print(subprocess.run(["ssh", HOST, f'wsl -d Ubuntu-24.04 -- bash -lc "cd ~/splat2mesh/eneas && {inner}"'],
                         capture_output=True, text=True, timeout=3600).stdout.replace("\x00", "")[-800:])
    subprocess.run(["scp", "-q", f"{HOST}:{win}/masks.npz", str(out_npz)], check=True)


def cut_out(ref: Path, mask: np.ndarray, out_png: Path):
    """The component on white, square, with a margin (Idol's own matting takes it from there)."""
    img = np.asarray(Image.open(ref).convert("RGB")).copy()
    if mask.shape != img.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(img.shape[1::-1])) > 127
    from scipy import ndimage as ndi
    m = ndi.binary_dilation(mask, iterations=2)
    img[~m] = 255
    ys, xs = np.nonzero(m)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    h = max(np.ptp(ys), np.ptp(xs)) * (1 + 2 * PAD) / 2
    canvas = Image.new("RGB", (int(2 * h), int(2 * h)), (255, 255, 255))
    canvas.paste(Image.fromarray(img), (int(h - cx), int(h - cy)))
    canvas.resize((1024, 1024), Image.LANCZOS).save(out_png)


def face_labels(asset: Path):
    """Per face of the whole asset: the index of the phrase its texels voted for (majority at the face's UV centre)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(asset / "asset.usda"))
    m = UsdGeom.Mesh(st.GetPrimAtPath("/Asset/Geom"))
    V = np.asarray(m.GetPointsAttr().Get(), np.float64)
    F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
    uv = np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar("st").Get(), np.float64)
    cls = np.load(asset / "mat" / "texel_class.npy")
    R = cls.shape[0]
    c = uv[F].mean(1)
    px = np.clip((c[:, 0] * R).astype(int), 0, R - 1); py = np.clip(((1 - c[:, 1]) * R).astype(int), 0, R - 1)
    lab = cls[py, px]
    if (lab < 0).mean() > 0.5:          # texel_class stored bottom-up
        lab = cls[R - 1 - py, px]
    return V, F, lab


def ref_box(ref: Path, mask: np.ndarray, V: np.ndarray):
    """Where the component is in the whole object, from the reference picture: the whole mesh is made from that
    picture seen from the front (x right, y up), so the mask's extent inside the object's silhouette maps linearly
    onto the mesh's x and y extent; depth keeps the whole object's range."""
    img = np.asarray(Image.open(ref).convert("RGB")).astype(np.float32)
    bg = np.median(np.r_[img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3), img[:, :4].reshape(-1, 3),
                         img[:, -4:].reshape(-1, 3)], 0)
    sil = np.abs(img - bg).max(-1) > 18
    sil = ndi.binary_opening(sil, iterations=2)
    lab_, n = ndi.label(sil)
    if n:
        sil = lab_ == (np.argmax(ndi.sum(sil, lab_, range(1, n + 1))) + 1)
    if mask.shape != sil.shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(sil.shape[::-1])) > 127
    ys, xs = np.nonzero(sil); my, mx = np.nonzero(mask)
    lo, hi = V.min(0), V.max(0)
    fx = lambda x: lo[0] + (x - xs.min()) / max(np.ptp(xs), 1) * (hi[0] - lo[0])
    fy = lambda y: hi[1] - (y - ys.min()) / max(np.ptp(ys), 1) * (hi[1] - lo[1])
    blo = np.array([fx(np.percentile(mx, 1)), fy(np.percentile(my, 99)), lo[2]])
    bhi = np.array([fx(np.percentile(mx, 99)), fy(np.percentile(my, 1)), hi[2]])
    return blo, bhi


ENVELOPE_SHARE = 0.5      # of another component's mask inside this one's: this one is a hollow envelope
GLASS_SMOOTH_ITERS = 40   # Taubin iterations on transmissive components
BOX_SCALE_TOL = 0.25       # ICP may change the box-fitted scale by at most this much


def place(part_V, part_F, target_pts, box):
    """Similarity transform of the component mesh into its box from the reference picture (x/y extent fitted, depth
    centred), refined by scaled ICP on the whole mesh's surface inside the box. Fitness: share of the component's
    points that land on that surface."""
    import open3d as o3d
    blo, bhi = box
    lo, hi = part_V.min(0), part_V.max(0)
    s = np.mean((bhi[:2] - blo[:2]) / np.maximum(hi[:2] - lo[:2], 1e-9))
    T0 = np.eye(4); T0[:3, :3] *= s; T0[:3, 3] = (blo + bhi) / 2 - s * (lo + hi) / 2
    diag = np.linalg.norm(bhi - blo)
    src = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(part_V), o3d.utility.Vector3iVector(part_F))
    src_pc = src.sample_points_uniformly(20000)
    pad = 0.08 * diag
    inside = np.all((target_pts > blo - pad) & (target_pts < bhi + pad), 1)
    tp = target_pts[inside] if inside.sum() > 200 else target_pts
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tp))
    thr = 0.1 * diag
    reg = o3d.pipelines.registration.registration_icp(
        src_pc, dst, thr, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T = reg.transformation
    s1 = np.cbrt(abs(np.linalg.det(T[:3, :3])))
    if abs(s1 / s - 1) > BOX_SCALE_TOL:                       # ICP drifted away from the picture: keep the box fit
        T = T0
    ev = o3d.pipelines.registration.evaluate_registration(src_pc, dst, thr, T)
    return T, float(ev.fitness), float(ev.inlier_rmse) / max(diag, 1e-9)


CONTACT_DILATE = 6         # px: reference masks this close touch, so the parts must touch in 3D
CONTACT_OVERLAP = 0.01     # of the object size: the closed joint overlaps a little, no visible seam


def close_contacts(parts, geo, report):
    """Generated components are placed one by one and can float apart. Two components whose masks touch in the
    reference picture touch in the object: the smaller one moves along the line between their picture centres
    (x right, y up) until its extent meets the bigger one's."""
    names = sorted(geo, key=lambda n: -geo[n][1].sum())
    shift = {n: np.zeros(3) for n in names}
    allp = np.concatenate([geo[n][0] for n in names]) if names else np.zeros((1, 3))
    size = np.linalg.norm(np.ptp(allp, 0))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma, mb = geo[a][1], geo[b][1]
            if not (ndi.binary_dilation(ma, iterations=CONTACT_DILATE) & mb).any():
                continue
            ca = np.argwhere(ma).mean(0); cb = np.argwhere(mb).mean(0)
            d = np.array([cb[1] - ca[1], ca[0] - cb[0], 0.0]); d /= max(np.linalg.norm(d), 1e-9)   # a → b
            pa = (geo[a][0] + shift[a]) @ d; pb = (geo[b][0] + shift[b]) @ d
            gap = np.percentile(pb, 1) - np.percentile(pa, 99)         # > 0: b floats beyond a
            mv = -(gap + CONTACT_OVERLAP * size) * d
            shift[b] = shift[b] + mv
            report[b]["contact_shift"] = round(float(-(gap + CONTACT_OVERLAP * size) / size), 4)
    outp = []
    for name, cls, V, F, uv, tex in parts:
        t = shift.get(name)
        if t is None or not t.any():
            outp.append((name, cls, V, F, uv, tex)); continue
        if cls is None:                       # referenced component: V is its 4x4 transform
            T = V.copy(); T[:3, 3] += t; outp.append((name, cls, T, F, uv, tex))
        else:
            outp.append((name, cls, V + t, F, uv, tex))
    return outp


def write_usd(out: Path, parts, thin=()):
    """parts: [(name, material class, V, F, uv or None, texture path or None)]"""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    st = Usd.Stage.CreateNew(str(out / "asset.usda"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Asset"); st.SetDefaultPrim(root.GetPrim())
    for name, cls, V, F, uv, tex in parts:
        xf = f"/Asset/Components/{name}"
        if cls is None:                    # own material stage: reference its asset, placed by the ICP transform
            T, src = V, F
            x = UsdGeom.Xform.Define(st, xf)
            x.AddTransformOp().Set(Gf.Matrix4d(T.T.tolist()))      # USD row vectors
            x.GetPrim().GetReferences().AddReference("./" + str((Path(src) / "asset_materials.usda"
                                                                  if (Path(src) / "asset_materials.usda").exists()
                                                                  else Path(src) / "asset.usda").relative_to(out)))
            continue
        L = _lib(cls)
        UsdGeom.Xform.Define(st, xf)
        m = UsdGeom.Mesh.Define(st, xf + "/Geom")
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(F.astype(np.int32).ravel()))
        m.GetSubdivisionSchemeAttr().Set("none")
        import trimesh
        nrm = trimesh.Trimesh(V, F, process=False).vertex_normals
        m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(nrm, np.float32))); m.SetNormalsInterpolation("vertex")
        mat = UsdShade.Material.Define(st, xf + "/Mat")
        sh = UsdShade.Shader.Define(st, xf + "/Mat/PBR"); sh.CreateIdAttr("UsdPreviewSurface")
        metal = float(L.get("metallic", 0)); rough = float(L.get("roughness", 0.5)); trans = float(L.get("transmission", 0))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        if L.get("ior"):
            sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(L["ior"]))
        if trans > 0:
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.25)
        if metal >= 0.5 and L.get("f0"):            # metal: the library's F0, not the baked reflections
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, L["f0"])))
        elif trans > 0:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.95, 0.95))
        elif tex is not None and uv is not None:
            pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
            rd = UsdShade.Shader.Define(st, xf + "/Mat/stReader"); rd.CreateIdAttr("UsdPrimvarReader_float2")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(st, xf + "/Mat/BaseColor"); tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./" + str(Path(tex).relative_to(out)))
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        # MaterialX standard_surface for path tracers (glass transmission, metal F0)
        ss = UsdShade.Shader.Define(st, xf + "/Mat/StandardSurface"); ss.CreateIdAttr("ND_standard_surface_surfaceshader")
        ss.CreateInput("metalness", Sdf.ValueTypeNames.Float).Set(metal)
        ss.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(rough)
        ss.CreateInput("transmission", Sdf.ValueTypeNames.Float).Set(trans)
        if L.get("ior"):
            ss.CreateInput("specular_IOR", Sdf.ValueTypeNames.Float).Set(float(L["ior"]))
        if metal >= 0.5 and L.get("f0"):
            ss.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, L["f0"])))
        elif trans > 0:
            ss.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
        if trans > 0 and name in thin:
            # an envelope with parts inside (a bulb, a jar): a thin shell, not a solid lump of glass
            ss.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
        mat.CreateSurfaceOutput("mtlx").ConnectToSource(ss.ConnectableAPI(), "out")
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    st.GetRootLayer().Save()


def main():
    ref, whole, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    # components: "phrase=materials"
    comps = []
    for a in sys.argv[4:]:
        ph, mats = a.split("=", 1)
        comps.append((ph.strip(), mats.strip()))
    items_c = [(m.split(":")[0].strip() if ":" in m else m, ph) for ph, m in comps]
    # 1. components on the reference picture
    phrases = [ph for ph, _ in comps]
    eneas_on_reference(ref, phrases, out / "masks_ref.npz")
    z = np.load(out / "masks_ref.npz")
    # layout: the whole object's faces per phrase (its material votes)
    names_whole = json.loads((whole / "mat" / "regions.json").read_text()).get("names")
    V, F, lab = face_labels(whole)
    wnames = names_whole or [f"{c}|{t}" for c, t in parse_description((whole.parent / "materials.txt").read_text())]
    parts, report, geo = [], {}, {}
    for (ph, mats), (cls, _) in zip(comps, items_c):
        name = slug(ph)
        multi = ":" in mats
        sub = parse_description(mats) if multi else []
        m = z[ph][0] if ph in z.files else None
        share = float(m.mean()) if m is not None else 0.0
        # where the component is in the whole object: faces the whole asset voted for this phrase (or for any of
        # its sub-parts' phrases)
        ks = [i for i, n in enumerate(wnames) if n.split("|", 1)[1] in [ph] + [t for _, t in sub]]
        tgt_faces = np.nonzero(np.isin(lab, ks))[0] if ks else np.array([], int)
        rep = {"material": cls, "mask_share": round(share, 4), "whole_faces": int(len(tgt_faces))}
        placed = False
        if m is not None and share >= MIN_MASK and len(tgt_faces) >= 50:
            img = out / "parts" / f"{name}.png"; img.parent.mkdir(exist_ok=True)
            if not img.exists():
                cut_out(ref, m, img)
            run = out / "runs"
            usd_dir = run / name / "usd"
            if not (usd_dir / "asset.usda").exists():
                subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/splat2usd.py"), str(img), "--out", str(run),
                                "--direct", "--faces", "40000", "--tex", "2048"], check=False)
            if (usd_dir / "asset.usda").exists():
                from pxr import Usd, UsdGeom
                pst = Usd.Stage.Open(str(usd_dir / "asset.usda"))          # keep the stage alive while reading
                pm = UsdGeom.Mesh(pst.GetPrimAtPath("/Asset/Geom"))
                PV = np.asarray(pm.GetPointsAttr().Get(), np.float64)
                PF = np.asarray(pm.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
                PUV = np.asarray(UsdGeom.PrimvarsAPI(pm).GetPrimvar("st").Get(), np.float64)
                T, fit, rmse = place(PV, PF, V[F].mean(1), ref_box(ref, m, V))
                rep.update(icp_fitness=round(fit, 3), icp_rmse=round(rmse, 4))
                if fit > 0.3:
                    PV2 = (np.c_[PV, np.ones(len(PV))] @ T.T)[:, :3]
                    if multi:
                        # several materials: the component's own material stage, then it is referenced placed
                        if not (usd_dir / "parts.json").exists():
                            subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/materials_eneas3d.py"),
                                            str(usd_dir), str(img), mats], check=False)
                        parts.append((name, None, T, usd_dir, None, None))
                    else:
                        if float(_lib(cls).get("transmission", 0)) > 0:
                            # glass is smooth: Idol's lumps refract like stones, so the surface is relaxed
                            import trimesh
                            # smoothed on the welded surface: UV seams duplicate vertices, which would split apart
                            key = np.round(PV2 / (1e-6 * np.linalg.norm(np.ptp(PV2, 0)))).astype(np.int64)
                            _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
                            inv = inv.ravel()
                            tm = trimesh.Trimesh(PV2[first].copy(), inv[PF], process=False)
                            trimesh.smoothing.filter_taubin(tm, iterations=GLASS_SMOOTH_ITERS)
                            PV2 = np.asarray(tm.vertices)[inv]
                        parts.append((name, cls, PV2, PF, PUV, usd_dir / "basecolor.png"))
                    placed = True
                    geo[name] = (PV2, m)
                    rep["source"] = "generated" + (" (own materials)" if multi else "")
        if not placed and len(tgt_faces):
            # keep the whole mesh's faces for this component
            used = np.unique(F[tgt_faces]); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
            parts.append((name, cls, V[used], remap[F[tgt_faces]], None, None)); rep["source"] = "whole mesh"
        report[name] = rep
        print(name, rep, flush=True)
    parts = close_contacts(parts, geo, report)
    # envelopes: a component whose picture mask holds a good share of another component's mask (the filament seen
    # through the bulb) is hollow
    thin = set()
    for ph, _ in comps:
        if ph not in z.files:
            continue
        a = z[ph][0]
        for q, _ in comps:
            if q != ph and q in z.files and z[q][0].sum() < a.sum() and (a & z[q][0]).sum() > ENVELOPE_SHARE * z[q][0].sum():
                thin.add(slug(ph)); report[slug(ph)]["envelope"] = True
    write_usd(out, parts, thin)
    (out / "components.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
