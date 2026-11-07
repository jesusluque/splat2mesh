"""A splat with per-gaussian materials (gs_materials.write_ply / add_albedo) as a relightable USD for athenea.

Standard OpenUSD UsdVolParticleField3DGaussianSplat (positions, orientations, linear scales, linear opacities,
degree-0 spherical harmonics) plus athenea's AtheneaSplatLightingAPI primvars (third_party/athenea/docs/operations.md
§4.3): relight = 1 (the colour is an albedo), linear = 1, per-gaussian metallic / roughness / transmission, and one
ior per prim, so gaussians are grouped into one ParticleField per distinct ior among the transmitting ones.

    python -m splat2mesh.athenea_splat SPLAT_MATERIALS.ply OUT.usda [--y-down] [--radiance]
--radiance: keep Idol's baked colour (f_dc_*) instead of the albedo, and do not relight (shows the splat as captured).
Then:  athenea stage OUT.usda ... --technique rt   (transmitting gaussians refract)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ply import read_vertex

SH_C0 = 0.28209479177387814


def _quat_from_frames(t1, t2, n):
    """Rotation matrices with columns (t1, t2, n) → quaternions w, x, y, z."""
    R = np.stack([t1, t2, n], -1)                       # [N, 3, 3], columns
    w = np.sqrt(np.clip(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], 1e-12, None)) / 2
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    q = np.stack([w, x, y, z], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


LOOSE_VOXELS = 2.0          # opaque gaussians this far in front of the mesh surface are dropped (stray glints)
SOLID_GLASS_COVER = 0.12    # athenea: a solid dielectric's surface cover that reads as "seen through" in raster too


def surface_normals(asset_dir, xyz):
    """The mesh's shading normal at the surface point nearest to each gaussian, and the distance to it. Without a
    normal athenea takes a gaussian's shortest axis, random on Idol's near-round gaussians (stray glints)."""
    from scipy.spatial import cKDTree
    from .materials import _raster
    from PIL import Image
    asset = Path(asset_dir)
    pos, nrm, mask = _raster(asset, Image.open(asset / "basecolor.png").size[0])
    P, N = pos[mask], nrm[mask]
    step = max(1, len(P) // 2_000_000); P, N = P[::step], N[::step]
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    d, i = cKDTree(P).query(xyz)
    outside = np.einsum("ij,ij->i", xyz - P[i], N[i]) > 0             # in front of the surface, not under it
    return N[i].astype(np.float32), np.where(outside, d, -d)


def glass_surfels(asset_dir, xyz, q, scl, op, sel_glass, ior=None, thin=None, tol_voxels: float = 2.5,
                  cell_voxels: float = 1.5, clear_voxels: float = 2.0):
    """Idol leaves glass as a haze: thousands of faint gaussians through a shell centimetres thick, which a
    refracting renderer reads as smoke. Keep only the glass gaussians on the mesh's surface and make them surfels:
    the surface normal (athenea:splat:normal), flat along it, wide enough to close the surface, nearly opaque.
    Returns (keep mask over all gaussians, normals for the kept glass ones) and edits q, scl, op in place."""
    from scipy.spatial import cKDTree
    from .materials import _raster
    from PIL import Image
    asset = Path(asset_dir)
    R = Image.open(asset / "basecolor.png").size[0]
    pos, nrm, mask = _raster(asset, R)
    P, N = pos[mask], nrm[mask]
    step = max(1, len(P) // 2_000_000); P, N = P[::step], N[::step]
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    rep = json.loads((asset / "report.json").read_text()) if (asset / "report.json").exists() else {}
    tol = tol_voxels * float(rep.get("voxel") or 0.0025)
    vox = float(rep.get("voxel") or 0.0025)
    gi = np.nonzero(sel_glass)[0]
    d, idx = cKDTree(P).query(xyz[gi])
    on = d < tol
    # not against an opaque part (glass the masks left by the brass base)
    opq = np.nonzero(~sel_glass)[0]
    if len(opq):
        dop, _ = cKDTree(xyz[opq]).query(xyz[gi])
        on &= dop > clear_voxels * vox
    cand = gi[on]; cidx = idx[on]
    # one layer per wall: each surfel sits ON the surface, one per cell (a stack of surfels would act as several
    # sheets); the kept one is the most opaque gaussian of its cell
    sp_pos = P[cidx]; cell = cell_voxels * vox
    key = np.floor(sp_pos / cell).astype(np.int64)
    order = np.lexsort((-op[cand], key[:, 2], key[:, 1], key[:, 0]))
    ks = key[order]; first = np.ones(len(order), bool); first[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    pick = order[first]
    k = cand[pick]
    keep = np.ones(len(xyz), bool); keep[gi] = False; keep[k] = True
    # on the surface point nearest to the cell's centre: close to a regular grid, so the overlapping discs sum to an
    # even cover (a jittered layout sums unevenly: speckle)
    centre = (key[pick] + 0.5) * cell
    _, ci = cKDTree(P).query(centre)
    xyz[k] = P[ci]
    n = N[ci].astype(np.float32)
    a = np.where(np.abs(n[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1, axis=1, keepdims=True); t2 = np.cross(n, t1)
    q[k] = _quat_from_frames(t1, t2, n)
    SIGMA = 0.75                                          # disc sigma / spacing: overlapping discs sum flat
    st = np.full(len(k), SIGMA * cell, np.float32)
    scl[k] = np.stack([st, st, np.maximum(st * 0.05, 1e-5)], 1)
    # athenea's thin wall: opacity 2·R0/(1+R0) (its render scales the reflection by 1/R0), so the blend gives a
    # sheet's reflectance and the rest passes straight (mesh2splat's m2sGlassCovers)
    i_ = np.asarray(ior[k] if ior is not None else np.full(len(k), 1.5), np.float64)
    r0 = ((i_ - 1) / (i_ + 1)) ** 2
    # that value is the COVER the overlapping discs add up to; a disc of sigma s at spacing h covers 2π·s²/h² cells.
    # A solid dielectric (thin = False: a crystal, a glass block) is a real surface instead: cover SOLID_GLASS_COVER,
    # the light goes through by transmission, tinted at each of its two crossings by the albedo
    tw = np.ones(len(k), bool) if thin is None else thin[k]
    cover = np.where(tw, 2 * r0 / (1 + r0), SOLID_GLASS_COVER)
    op[k] = (cover / (2 * np.pi * SIGMA ** 2)).astype(np.float32)
    return keep, k, n, tw


def write_splat_usd(ply, out, y_down: bool = True, radiance: bool = False, glass_surface=None) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol, Vt
    v = read_vertex(ply)
    n = len(v)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)          # w x y z (3DGS)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scl = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    op = (1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))).astype(np.float32)
    have_alb = "albedo_r" in v.dtype.names and not radiance
    if have_alb:
        alb = np.stack([v["albedo_r"], v["albedo_g"], v["albedo_b"]], 1)              # linear
        dc = ((alb - 0.5) / SH_C0).astype(np.float32)
    else:
        dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    if y_down:      # Idol / COLMAP: turn 180° about X in the data itself (some readers ignore prim transforms)
        xyz = xyz * np.array([1, -1, -1], np.float32)
        w, x, y, z = q.T                         # (0,1,0,0) ⊗ (w,x,y,z) = (-x, w, -z, y)
        q = np.stack([-x, w, -z, y], 1)
    attr = lambda k, d: np.asarray(v[k], np.float32) if k in v.dtype.names else np.full(n, d, np.float32)
    metal, rough, trans, ior = attr("metallic", 0), attr("roughness", 0.5), attr("transmission", 0), attr("ior", 1.5)
    normals = np.zeros_like(xyz); has_n = np.zeros(n, bool); thin = np.zeros(n, np.int32)
    drop = np.zeros(n, bool)
    if glass_surface is not None:
        normals, sd = surface_normals(glass_surface, xyz); has_n[:] = True
        # loose gaussians floating in front of the surface catch the light as stray glints: dropped (those under the
        # surface are hidden and stay)
        rep = json.loads((Path(glass_surface) / "report.json").read_text()) if (Path(glass_surface) / "report.json").exists() else {}
        drop = (trans <= 0) & (sd > LOOSE_VOXELS * float(rep.get("voxel") or 0.0025))
        print(f"loose gaussians off the surface: {int(drop.sum())}", flush=True)
    if glass_surface is not None and (trans > 0).any():
        # thin-walled or solid per material (the finish pass's flag), through the per-gaussian mat_id
        thin_g = np.ones(n, bool)
        fin = Path(glass_surface) / "mat" / "finish.json"; names_f = Path(ply).with_name("mat_names.json")
        if fin.exists() and names_f.exists() and "mat_id" in v.dtype.names:
            cls = json.loads(fin.read_text()); cls = cls.get("classes", cls)
            names = json.loads(names_f.read_text())["mat_id"]; mid = np.asarray(v["mat_id"]).astype(int)
            for i, nm in enumerate(names):
                thin_g[mid == i] = bool(cls.get(nm.split("|")[0], {}).get("thin_walled", True))
        keep, k, nk, tw = glass_surfels(glass_surface, xyz, q, scl, op, trans > 0, ior=ior, thin=thin_g)
        normals[k] = nk; thin[k] = tw.astype(np.int32)
        drop &= keep; keep &= ~drop
    elif drop.any():
        keep = ~drop
    if glass_surface is not None and (drop.any() or (trans > 0).any()):
        if (trans > 0).any():
            print(f"glass: {int((trans > 0).sum())} gaussians → {len(k)} surfels on the surface", flush=True)
        sel_all = np.nonzero(keep)[0]
        xyz, q, scl, op, dc = xyz[sel_all], q[sel_all], scl[sel_all], op[sel_all], dc[sel_all]
        metal, rough, trans, ior = metal[sel_all], rough[sel_all], trans[sel_all], ior[sel_all]
        normals, has_n, thin = normals[sel_all], has_n[sel_all], thin[sel_all]
        n = len(xyz)
    # one prim per ior among transmitting gaussians (athenea's ior is one per prim); opaque ones in a prim of their own
    key = np.where(trans > 0, np.round(ior, 3), -1.0)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Splat"); stage.SetDefaultPrim(root.GetPrim())
    groups = {}
    for gi, k in enumerate(sorted(np.unique(key))):
        sel = np.nonzero(key == k)[0]
        name = "Opaque" if k < 0 else f"Ior_{str(k).replace('.', '_')}"
        p = UsdVol.ParticleField3DGaussianSplat.Define(stage, f"/Splat/{name}")
        prim = p.GetPrim()
        for api in (UsdVol.ParticleFieldPositionAttributeAPI, UsdVol.ParticleFieldOrientationAttributeAPI,
                    UsdVol.ParticleFieldScaleAttributeAPI, UsdVol.ParticleFieldOpacityAttributeAPI,
                    UsdVol.ParticleFieldSphericalHarmonicsAttributeAPI, UsdVol.ParticleFieldKernelGaussianEllipsoidAPI):
            api.Apply(prim)
        prim.GetAttribute("positions").Set(Vt.Vec3fArray.FromNumpy(xyz[sel]))
        prim.GetAttribute("orientations").Set(Vt.QuatfArray([Gf.Quatf(float(a), float(b), float(c), float(d))
                                                               for a, b, c, d in q[sel]]))
        prim.GetAttribute("scales").Set(Vt.Vec3fArray.FromNumpy(scl[sel]))
        prim.GetAttribute("opacities").Set(Vt.FloatArray.FromNumpy(op[sel]))
        prim.GetAttribute("radiance:sphericalHarmonicsDegree").Set(0)
        prim.GetAttribute("radiance:sphericalHarmonicsCoefficients").Set(Vt.Vec3fArray.FromNumpy(dc[sel]))
        lo, hi = xyz[sel].min(0), xyz[sel].max(0)
        p.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
        pv = UsdGeom.PrimvarsAPI(prim)
        pv.CreatePrimvar("athenea:splat:relight", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(have_alb)
        pv.CreatePrimvar("athenea:splat:linear", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(have_alb)
        for nm, arr in (("metallic", metal), ("roughness", rough), ("transmission", trans)):
            pv.CreatePrimvar(f"athenea:splat:{nm}", Sdf.ValueTypeNames.FloatArray,
                             UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(arr[sel]))
        pv.CreatePrimvar("athenea:splat:ior", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant).Set(
            float(k) if k > 0 else 0.0)
        if has_n[sel].any():
            pv.CreatePrimvar("athenea:splat:normal", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.vertex).Set(
                Vt.Vec3fArray.FromNumpy(normals[sel].astype(np.float32)))
            pv.CreatePrimvar("athenea:splat:thinWalled", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex).Set(
                Vt.IntArray.FromNumpy(thin[sel]))
        groups[name] = int(len(sel))
    stage.GetRootLayer().Save()
    return {"gaussians": n, "prims": groups, "relight": have_alb}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply"); ap.add_argument("out"); ap.add_argument("--y-down", action="store_true")
    ap.add_argument("--radiance", action="store_true")
    ap.add_argument("--glass-surface", default=None, metavar="ASSET_DIR",
                    help="the mesh asset of the same splat: glass gaussians become surfels on its surface (normal, "
                         "flat, covering, thinWalled) instead of Idol's haze")
    a = ap.parse_args()
    print(write_splat_usd(a.ply, a.out, a.y_down, a.radiance, a.glass_surface))


if __name__ == "__main__":
    main()


def write_web_splat(ply, out, y_down: bool = True, max_gaussians: int = 0) -> dict:
    """The compact .splat web format (32 bytes a gaussian: position 3×f32, linear scale 3×f32, sRGB colour + opacity
    4×u8, rotation w,x,y,z 4×u8), Y-down turned like the meshes so a splat and its USD line up. Sorted by size ×
    opacity (largest first), the order web viewers expect; max_gaussians > 0 keeps only the largest ones."""
    v = read_vertex(ply)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    if y_down:
        xyz = xyz * np.array([1, -1, -1], np.float32)
        w, x, y, z = q.T
        q = np.stack([-x, w, -z, y], 1)
    scl = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    rgb = 0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)
    order = np.argsort(-(scl.prod(1) * op))
    if max_gaussians:
        order = order[:max_gaussians]
    rec = np.zeros(len(order), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = xyz[order]; rec["s"] = scl[order]
    rec["c"] = np.clip(np.concatenate([rgb[order], op[order, None]], 1) * 255, 0, 255).round().astype(np.uint8)
    rec["r"] = np.clip(q[order] * 128 + 128, 0, 255).round().astype(np.uint8)
    Path(out).write_bytes(rec.tobytes())
    return {"gaussians": int(len(order)), "bytes": Path(out).stat().st_size}
