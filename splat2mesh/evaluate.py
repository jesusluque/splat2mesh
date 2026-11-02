"""Geometry metrics of a reconstructed mesh vs a reference USD.

    python -m splat2mesh.evaluate REF.usda RECON.(ply|usda|glb|obj) [--prim /World/Pawn] [--tau 0.0005]

Chamfer (mean, symmetric), F-score at tau, plus split into "visible" vs "hidden"
reference surface when --views is given (a ref point is visible if some view sees it).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def usd_to_trimesh(path: str, prim_path: str | None = None) -> trimesh.Trimesh:
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(path)
    root = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    cache = UsdGeom.XformCache()
    meshes = []
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(p)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        m = UsdGeom.Mesh(p)
        pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        tris, o = [], 0
        for c in counts:
            for k in range(1, c - 1):
                tris.append((idx[o], idx[o + k], idx[o + k + 1]))
            o += c
        M = np.asarray(cache.GetLocalToWorldTransform(p), np.float64)  # row-vector convention
        pw = np.c_[pts, np.ones(len(pts))] @ M
        meshes.append(trimesh.Trimesh(pw[:, :3], np.asarray(tris), process=False))
    return trimesh.util.concatenate(meshes)


def load_any(path: str, prim: str | None = None) -> trimesh.Trimesh:
    if path.endswith((".usd", ".usda", ".usdc", ".usdz")):
        return usd_to_trimesh(path, prim)
    m = trimesh.load(path, force="mesh", process=False)
    return m


def visible_mask(points: np.ndarray, views, tol: float) -> np.ndarray:
    vis = np.zeros(len(points), bool)
    for v in views:
        Pc = points @ v.w2c[:3, :3].T + v.w2c[:3, 3]
        z = Pc[:, 2]
        uvw = Pc @ v.K.T
        u = uvw[:, 0] / np.maximum(uvw[:, 2], 1e-12)
        y = uvw[:, 1] / np.maximum(uvw[:, 2], 1e-12)
        h, w = v.depth.shape
        ok = (z > 0) & (u >= 0) & (u < w) & (y >= 0) & (y < h)
        d = np.zeros(len(points))
        d[ok] = v.depth[y[ok].astype(int), u[ok].astype(int)]
        vis |= ok & (d > 0) & (np.abs(d - z) < tol)
    return vis


def compare(ref: trimesh.Trimesh, rec: trimesh.Trimesh, tau: float, n: int = 200000,
            views=None) -> dict:
    a, _ = trimesh.sample.sample_surface(ref, n, seed=0)
    b, _ = trimesh.sample.sample_surface(rec, n, seed=0)
    d_ab = cKDTree(b).query(a)[0]       # ref → rec (completeness)
    d_ba = cKDTree(a).query(b)[0]       # rec → ref (accuracy)
    prec, rec_ = (d_ba < tau).mean(), (d_ab < tau).mean()
    out = {
        "chamfer_mean": float((d_ab.mean() + d_ba.mean()) / 2),
        "accuracy_mean": float(d_ba.mean()),
        "completeness_mean": float(d_ab.mean()),
        "precision@tau": float(prec), "recall@tau": float(rec_),
        "fscore@tau": float(2 * prec * rec_ / max(prec + rec_, 1e-12)), "tau": tau,
        "ref_diag": float(np.linalg.norm(ref.bounds[1] - ref.bounds[0])),
    }
    if views is not None:
        vis = visible_mask(a, views, tol=tau)
        out["ref_visible_fraction"] = float(vis.mean())
        out["completeness_visible"] = float(d_ab[vis].mean()) if vis.any() else None
        out["completeness_hidden"] = float(d_ab[~vis].mean()) if (~vis).any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("rec")
    ap.add_argument("--prim", default=None)
    ap.add_argument("--rec-prim", default=None)
    ap.add_argument("--tau", type=float, default=None, help="default: 0.5%% of ref bbox diagonal")
    a = ap.parse_args()
    ref = load_any(a.ref, a.prim)
    rec = load_any(a.rec, a.rec_prim)
    tau = a.tau or 0.005 * float(np.linalg.norm(ref.bounds[1] - ref.bounds[0]))
    print(json.dumps(compare(ref, rec, tau), indent=2))


if __name__ == "__main__":
    main()
