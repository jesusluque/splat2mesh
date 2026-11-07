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
from pathlib import Path

import numpy as np

from .ply import read_vertex

SH_C0 = 0.28209479177387814


def write_splat_usd(ply, out, y_down: bool = True, radiance: bool = False) -> dict:
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
        groups[name] = int(len(sel))
    stage.GetRootLayer().Save()
    return {"gaussians": n, "prims": groups, "relight": have_alb}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply"); ap.add_argument("out"); ap.add_argument("--y-down", action="store_true")
    ap.add_argument("--radiance", action="store_true")
    a = ap.parse_args()
    print(write_splat_usd(a.ply, a.out, a.y_down, a.radiance))


if __name__ == "__main__":
    main()
