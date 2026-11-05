"""Post process: quad retopology of every mesh of a USD asset with CurioMesh, UVs remade and every texture carried over.

CurioMesh (https://github.com/CurioCrafter/CurioMesh, GPL-3.0-or-later) is run as a separate program, headless Blender
with its own agent scripts, on an OBJ of each mesh; nothing of it is imported here. Its own bake keeps only normal and
colour, and its UVs are Blender's, so this script does the UVs itself:

1. CurioMesh reduces the mesh to quads (Use, Detail), then its Fix tools repair bad faces (kept only if fewer).
   If the new vertices stray from the old surface (p95 over SHAPE_MAX), it runs again with half the Detail.
2. The quads are triangulated for xatlas (splat2mesh.texture.unwrap) and put back together: a quad whose diagonal is
   a UV seam (its two triangles in different charts, or the diagonal's corners at different UVs) stays two triangles,
   so no face spans two islands. st is written faceVarying and indexed.
3. Every texture of the mesh's material (base colour, roughness, metallic, opacity, transmission...) is carried from
   the old UVs to the new ones texel by texel: each texel's 3D point is projected onto the original surface (nearest
   point, or along the normal where the nearest point is on a back-facing sheet), its old st interpolated, the old
   texture sampled there. Vertex normals are read from the original surface the same way (smooth shading kept). Uncovered texels take the nearest covered one (padding for mip-maps). A tangent-space normal
   map cannot be carried this way: such a mesh keeps its original geometry.
4. UV checks per mesh, written to the report: charts, quads split at seams, st outside [0, 1], flipped triangles,
   overlapping texels, stretch (3D area over UV area, p5/p95 against the median), and the transfer distance.
   A mesh that fails them (overlap or out-of-range st or flips above UV_MAX) keeps its original geometry.

    python scripts/curio_post.py ASSET.usda OUT_DIR [--use GAME] [--detail 6] [--min-faces 3000] [--blender PATH]
Writes OUT_DIR/asset.usda (copy of the asset folder with the new meshes and textures) and OUT_DIR/curio_report.json.
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from splat2mesh.texture import unwrap, rasterize_uv

CURIO = ROOT / "third_party/curiomesh"
SKILL = CURIO / "skills/curiomesh-retopo/scripts"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
SHAPE_MAX = 0.001       # p95 distance of the new vertices to the old surface, share of the diagonal
UV_MAX = {"overlap": 0.002, "outside": 0.0, "flipped": 0.002}     # shares of texels / corners / triangles


def write_obj(path, V, F):
    with open(path, "w") as f:
        f.write("".join(f"v {a:.7g} {b:.7g} {c:.7g}\n" for a, b, c in V))
        f.write("".join("f " + " ".join(str(i + 1) for i in p) + "\n" for p in F))


def read_obj(path):
    V, P = [], []
    for l in open(path):
        if l.startswith("v "):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "):
            P.append([int(t.split("/")[0]) - 1 for t in l.split()[1:]])
    return np.asarray(V, np.float64), P


def blender(args, log):
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python", *args], capture_output=True, text=True, timeout=3600)
    Path(log).write_text(r.stdout + r.stderr)
    rep = [l for l in r.stdout.splitlines() if l.startswith("CURIOMESH_REPORT ")]
    return json.loads(rep[-1][len("CURIOMESH_REPORT "):]) if rep else None


def retopo(V, F, work, name, use, detail):
    src, low, fixed = work / f"{name}.obj", work / f"{name}_low.obj", work / f"{name}_fixed.obj"
    write_obj(src, V, F)
    rep = blender([str(SKILL / "curiomesh_run.py"), "--", str(src), "--use", use, "--detail", str(detail), "--out", str(low),
                   "--report", str(work / f"{name}_run.json"), "--addon", str(CURIO)], work / f"{name}_run.log")
    if not low.exists():
        return None, rep, None
    fix = blender([str(SKILL / "curiomesh_fix.py"), "--", str(low), "--original", str(src), "--fix", "ALL", "--out", str(fixed),
                   "--report", str(work / f"{name}_fix.json"), "--addon", str(CURIO)], work / f"{name}_fix.log")
    return read_obj(fixed if fixed.exists() else low), rep, fix


def polys_to_tris(P):
    T, owner = [], []
    for k, p in enumerate(P):
        for i in range(1, len(p) - 1):
            T.append([p[0], p[i], p[i + 1]]); owner.append(k)
    return np.asarray(T), np.asarray(owner)


def uv_polys(V, P, res):
    """New UVs on the polygons: xatlas on their triangles, polygons kept whole where all their triangles share one
    UV per corner (else split into those triangles). Returns new polygons (vertex ids), corner st ids, st values."""
    T, owner = polys_to_tris(P)
    vmap, tf, uvs, charts = unwrap(trimesh.Trimesh(V, T, process=False), res)
    out_p, out_c, split = [], [], 0
    by_poly = {}
    for t, k in enumerate(owner):
        by_poly.setdefault(k, []).append(t)
    for k, p in enumerate(P):
        ts = by_poly[k]; corner = {}
        ok = True
        for t in ts:
            for a in tf[t]:
                v = int(vmap[a])
                if corner.setdefault(v, int(a)) != int(a):
                    ok = False
        if ok:
            out_p.append(list(p)); out_c.append([corner[v] for v in p])
        else:
            split += 1
            for t in ts:
                out_p.append([int(vmap[a]) for a in tf[t]]); out_c.append([int(a) for a in tf[t]])
    return out_p, out_c, uvs, charts, split, vmap, tf


def uv_checks(Vn, tf, uvs, res):
    """Vn: positions of the xatlas vertices; tf: their triangles; uvs: their st."""
    A, B, C = uvs[tf[:, 0]], uvs[tf[:, 1]], uvs[tf[:, 2]]
    sa = ((B - A)[:, 0] * (C - A)[:, 1] - (B - A)[:, 1] * (C - A)[:, 0]) / 2
    a3 = np.linalg.norm(np.cross(Vn[tf[:, 1]] - Vn[tf[:, 0]], Vn[tf[:, 2]] - Vn[tf[:, 0]]), axis=1) / 2
    # flipped: against the majority of the triangle's own chart (xatlas may mirror a whole chart, which is harmless
    # for colour maps; a triangle turned over inside its chart is a fold in the texture)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nt = len(tf); e = np.repeat(np.arange(nt), 3)
    g = coo_matrix((np.ones(3 * nt), (e, tf.ravel())), shape=(nt, len(uvs))).tocsr()
    _, chart = connected_components(g @ g.T, directed=False)
    pos_share = np.bincount(chart, weights=(sa > 0).astype(float)) / np.bincount(chart)
    major = np.where(pos_share[chart] >= 0.5, 1, -1)
    flipped = float(np.mean(np.sign(sa) != major)) if nt else 0.0
    ok = np.abs(sa) > 1e-12
    st_ = a3[ok] / np.abs(sa[ok]); st_ = st_ / np.median(st_)
    # texel overlap: coverage count on a half-resolution grid, texel centres strictly inside
    R = max(res // 2, 256); cnt = np.zeros((R, R), np.int32)
    px = uvs * R - 0.5
    for f in tf:
        a, b, c = px[f]
        x0, x1 = max(int(np.ceil(min(a[0], b[0], c[0]))), 0), min(int(np.floor(max(a[0], b[0], c[0]))), R - 1)
        y0, y1 = max(int(np.ceil(min(a[1], b[1], c[1]))), 0), min(int(np.floor(max(a[1], b[1], c[1]))), R - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        p = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
        v0, v1 = b - a, c - a; den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        d = p - a; w1 = (d[:, 0] * v1[1] - v1[0] * d[:, 1]) / den; w2 = (v0[0] * d[:, 1] - d[:, 0] * v0[1]) / den
        inn = (w1 > 1e-6) & (w2 > 1e-6) & (1 - w1 - w2 > 1e-6)
        cnt[p[inn, 1].astype(int), p[inn, 0].astype(int)] += 1
    cov = (cnt > 0).sum()
    return {"outside": float(np.mean((uvs < 0) | (uvs > 1))), "flipped": flipped,
            "overlap": float((cnt > 1).sum() / max(cov, 1)), "atlas_used": float(cov / R / R),
            "stretch_p5": float(np.percentile(st_, 5)) if len(st_) else 0.0,
            "stretch_p95": float(np.percentile(st_, 95)) if len(st_) else 0.0}


def project(Vs, Fs, pos, nrm):
    """Nearest point of the original surface for each texel point; where that point's face looks the other way (the
    far sheet of a thin part), the hit along the normal instead. Returns face ids, barycentrics, distances."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(Vs.astype(np.float32), Fs.astype(np.uint32))
    q = o3d.core.Tensor(pos.astype(np.float32))
    r = sc.compute_closest_points(q)
    fid = r["primitive_ids"].numpy().astype(np.int64); uv = r["primitive_uvs"].numpy().astype(np.float64)
    pt = r["points"].numpy()
    fn = np.cross(Vs[Fs[:, 1]] - Vs[Fs[:, 0]], Vs[Fs[:, 2]] - Vs[Fs[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    bad = np.einsum("ij,ij->i", fn[fid], nrm) < 0
    if bad.any():
        diag = np.linalg.norm(Vs.max(0) - Vs.min(0)); best = np.full(bad.sum(), np.inf)
        bf, bu = fid[bad].copy(), uv[bad].copy()
        for sgn in (1, -1):        # along the normal both ways, from a little outside
            o = pos[bad] + sgn * nrm[bad] * 0.02 * diag
            h = sc.cast_rays(o3d.core.Tensor(np.c_[o, -sgn * nrm[bad]].astype(np.float32)))
            t = h["t_hit"].numpy(); hid = h["primitive_ids"].numpy().astype(np.int64)
            good = np.isfinite(t) & (hid < len(Fs))
            good[good] &= np.einsum("ij,ij->i", fn[hid[good]], nrm[bad][good]) > 0
            d = np.abs(t - 0.02 * diag); take = good & (d < best)
            bf[take], bu[take], best[take] = hid[take], h["primitive_uvs"].numpy()[take], d[take]
        fid[bad], uv[bad] = bf, bu
        pt[bad] = (np.einsum("i,ij->ij", 1 - bu[:, 0] - bu[:, 1], Vs[Fs[bf, 0]]) + np.einsum("i,ij->ij", bu[:, 0], Vs[Fs[bf, 1]])
                   + np.einsum("i,ij->ij", bu[:, 1], Vs[Fs[bf, 2]]))
    w = np.c_[1 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]]
    return fid, w, np.linalg.norm(pt - pos, axis=1)


def sample(img, st):
    """Bilinear lookup of image rows/cols at st (v up: row 0 is v = 1)."""
    H, W = img.shape[:2]
    x = st[:, 0] * W - 0.5; y = (1 - st[:, 1]) * H - 0.5
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    c = lambda yy, xx: img[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)].reshape(len(st), -1).astype(np.float32)
    return (c(y0, x0) * (1 - fx) * (1 - fy) + c(y0, x0 + 1) * fx * (1 - fy) + c(y0 + 1, x0) * (1 - fx) * fy
            + c(y0 + 1, x0 + 1) * fx * fy)


def textures(stage, prim):
    """(material prim, {shader input attribute: resolved file path}) for the mesh's bound material."""
    from pxr import UsdShade, Sdf
    mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not mat:
        return None, {}
    files = {}
    for p in Usd_descendants(mat.GetPrim()):
        sh = UsdShade.Shader(p)
        if not sh:
            continue
        for i in sh.GetInputs():
            v = i.Get()
            if isinstance(v, Sdf.AssetPath) and v.path:
                files[i.GetAttr().GetPath()] = v.resolvedPath or str(Path(stage.GetRootLayer().realPath).parent / v.path)
    return mat, files


def Usd_descendants(p):
    from pxr import Usd
    return list(Usd.PrimRange(p))


def normal_mapped(stage, mat):
    from pxr import UsdShade
    for p in Usd_descendants(mat.GetPrim()):
        sh = UsdShade.Shader(p)
        if sh and any(i.GetBaseName() in ("normal", "normalmap") and i.HasConnectedSource() for i in sh.GetInputs()):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asset"); ap.add_argument("out")
    ap.add_argument("--use", default="GAME"); ap.add_argument("--detail", type=float, default=6.0)
    ap.add_argument("--min-faces", type=int, default=3000, help="meshes with fewer faces are left as they are")
    ap.add_argument("--blender", default=None)
    a = ap.parse_args()
    if a.blender:
        global BLENDER
        BLENDER = a.blender
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt
    src_dir, out = Path(a.asset).resolve().parent, Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src_dir, out, ignore=shutil.ignore_patterns("curio_work"))
    work = out / "curio_work"; work.mkdir()
    stage = Usd.Stage.Open(str(out / Path(a.asset).name)); layer = stage.GetRootLayer()
    report = {"asset": a.asset, "use": a.use, "detail": a.detail, "curiomesh": subprocess.run(
        ["git", "-C", str(CURIO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), "meshes": {}}
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdGeom.Mesh) or not layer.GetAttributeAtPath(prim.GetPath().AppendProperty("points")):
            continue                       # copies made by reference follow their prototype
        m = UsdGeom.Mesh(prim); name = str(prim.GetPath()).strip("/").replace("/", "_")
        V = np.asarray(m.GetPointsAttr().Get(), np.float64); cnt = np.asarray(m.GetFaceVertexCountsAttr().Get())
        I = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        r = {"faces_before": int(len(cnt))}; report["meshes"][str(prim.GetPath())] = r
        if len(cnt) < a.min_faces or not (cnt == 3).all():
            r["kept"] = "few faces" if len(cnt) < a.min_faces else "not triangles"; print(name, r, flush=True); continue
        F = I.reshape(-1, 3)
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        st_src = np.asarray(pv.Get(), np.float64) if pv and pv.Get() is not None else None
        mat, files = textures(stage, prim)
        if st_src is not None and (pv.GetInterpolation() != "vertex" or pv.IsIndexed()):
            r["kept"] = f"st is {pv.GetInterpolation()}{' indexed' if pv.IsIndexed() else ''}"; continue
        if mat and files and normal_mapped(stage, mat):
            r["kept"] = "tangent-space normal map"; continue
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        detail = a.detail
        for _ in range(3):        # finer while the new surface strays from the old (a turned tyre goes polygonal)
            got, run, fix = retopo(V, F, work, name, a.use, detail)
            if got is None:
                break
            T, _ = polys_to_tris(got[1])
            _, _, d_new = project(V, F, got[0], np.asarray(trimesh.Trimesh(got[0], T, process=False).vertex_normals))
            if np.percentile(d_new, 95) / diag <= SHAPE_MAX:
                break
            detail /= 2
        r["detail"] = detail
        r["curiomesh"] = {k: (run or {}).get(k) for k in ("faces", "quad_pct", "view_changed_pct", "seconds", "verdict")}
        r["fix"] = {k: v for k, v in fix.items() if not isinstance(v, (list, dict)) or k in ("before", "after")} if fix else None
        if got is None:
            r["kept"] = "CurioMesh gave nothing"; continue
        Vn, P = got
        # shape: how far the new surface lies from the old (share of the diagonal)
        T, _ = polys_to_tris(P)
        new_tm = trimesh.Trimesh(Vn, T, process=False)
        vfid, vw, d_new = project(V, F, Vn, np.asarray(new_tm.vertex_normals))
        r["shape_p95"] = float(np.percentile(d_new, 95) / diag)
        # shading: the original's own normals, read at the new vertices (smooth where it was smooth)
        N0 = np.asarray(m.GetNormalsAttr().Get() or [], np.float64)
        if len(N0) == len(V) and m.GetNormalsInterpolation() == "vertex":
            Nn = np.einsum("ij,ijk->ik", vw, N0[F[vfid]])
        else:
            Nn = np.asarray(new_tm.vertex_normals)
        Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-12)
        if st_src is None:
            counts, idx, st_vals, st_idx = [len(p) for p in P], [i for p in P for i in p], None, None
        else:
            res = max([Image.open(f).size[0] for f in files.values() if Path(f).exists()] or [1024])
            P2, C2, uvs, charts, split, vmap, tf = uv_polys(Vn, P, res)
            Vx = Vn[vmap]
            chk = uv_checks(Vx, tf, uvs, res); chk.update(charts=int(charts), split_quads=int(split))
            r["uv"] = chk
            if any(chk[k] > UV_MAX[k] for k in UV_MAX):
                r["kept"] = "UV checks failed"; print(name, r, flush=True); continue
            nx = np.asarray(trimesh.Trimesh(Vx, tf, process=False).vertex_normals)
            pos, nrm, mask = rasterize_uv(Vx, tf, uvs, nx, res)
            sel = np.nonzero(mask)
            fid, w, dist = project(V, F, pos[sel], nrm[sel])
            st_old = np.einsum("ij,ijk->ik", w, st_src[F[fid]])
            r["transfer_p95"] = float(np.percentile(dist, 95) / diag)
            _, near = ndi.distance_transform_edt(~mask, return_indices=True)
            new_files = {}
            for attr, f in files.items():
                if not Path(f).exists():
                    continue
                im = Image.open(f); mode = im.mode
                arr = np.asarray(im.convert("RGBA" if "A" in mode else ("L" if mode in ("L", "I", "I;16", "F") else "RGB")))
                arr = arr.reshape(*arr.shape[:2], -1)
                vals = sample(arr, st_old)
                outi = np.zeros((res, res, arr.shape[2]), np.float32); outi[sel] = vals
                outi = outi[near[0], near[1]]                        # padding: nearest covered texel
                outi = outi[::-1]                                    # rasteriser row = v; image row 0 = v 1
                dst = out / f"{Path(f).stem}__{name}.png"
                Image.fromarray(np.clip(outi, 0, 255).round().astype(np.uint8).squeeze()).save(dst)
                new_files[attr] = "./" + dst.name
            counts, idx = [len(p) for p in P2], [i for p in P2 for i in p]
            st_vals, st_idx = uvs, [c for cs in C2 for c in cs]
            # the material, copied for this mesh and pointed at its new textures
            newmat, k = Sdf.Path(f"{mat.GetPath()}_{prim.GetParent().GetName()}"), 0
            while stage.GetPrimAtPath(newmat):
                k += 1; newmat = Sdf.Path(f"{mat.GetPath()}_{prim.GetParent().GetName()}_{k}")
            Sdf.CopySpec(layer, mat.GetPath(), layer, newmat)
            for attr, rel in new_files.items():
                ap_ = Sdf.Path(str(attr).replace(str(mat.GetPath()), str(newmat), 1))
                stage.GetAttributeAtPath(ap_).Set(Sdf.AssetPath(rel))
            UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(newmat)))
            r["material"] = str(newmat); r["textures"] = sorted(new_files.values())
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(Vn.astype(np.float32)))
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts)); m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(idx))
        m.GetExtentAttr().Set(Vt.Vec3fArray([tuple(Vn.min(0)), tuple(Vn.max(0))]))
        m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(Nn.astype(np.float32))); m.SetNormalsInterpolation("vertex")
        for nm in ("primvars:normals",):
            if prim.HasAttribute(nm):
                prim.RemoveProperty(nm)
        if st_vals is not None:
            pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            pv.Set(Vt.Vec2fArray.FromNumpy(st_vals.astype(np.float32))); pv.SetIndices(Vt.IntArray(st_idx))
        r["faces_after"] = len(counts); r["quads"] = int(sum(c == 4 for c in counts))
        print(name, json.dumps(r), flush=True)
    stage.GetRootLayer().Save()
    (out / "curio_report.json").write_text(json.dumps(report, indent=1))
    print(out / "curio_report.json")


if __name__ == "__main__":
    main()
