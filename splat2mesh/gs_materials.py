"""Materials on the splat itself: part masks are voted per gaussian, then the mesh inherits them.

The mesh-based flow (materials.render_orbit + from_view_masks) runs ENEAS on renders of the textured mesh and votes
per texel. Few texels are hit directly (a 768 px orbit frame covers a 4096² atlas sparsely: ~6 % on a figurine), so
most texels get their class from colour regions. Here:

  1. render_orbit_gs: the splat is rendered with athenea (`render --splats`) on a 16-view orbit, on white, with
     depth; cameras in OpenCV convention (cameras.json) like the render path.
  2. ENEAS segments and tracks the phrases on those frames (scripts/materials_eneas3d.py --gs).
  3. vote_gaussians: every gaussian centre is projected into every view; where it is the visible surface (its depth
     matches the rendered depth) it takes a vote, weighted by its opacity, for the phrase whose mask covers that
     pixel (smallest mask wins, as per texel). Gaussians no view sees take their neighbours' votes.
  4. texel_votes: each texel of the mesh's atlas takes the votes of its nearest gaussians (Gaussian kernel over
     distance, times opacity), which is dense: every texel gets votes. from_view_masks then applies the same rules
     (colour regions, priors, caps) as before.
  5. write_ply: the splat is written back with per-gaussian material attributes (mat_id, roughness, metallic,
     transmission, subsurface, ior) plus mat_names.json, so any later mesh, or a splat renderer, can use them.

The gaussians are taken in the mesh's frame: Idol splats are Y-down and direct mode flips them (y_down=True).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .ply import read_vertex, write_vertex

FOCAL, APERTURE, SIZE = 50.0, 24.576, 1024
DEPTH_TOL = 0.015          # × object radius: a gaussian within this of the rendered depth is the visible surface
ROOT = Path(__file__).resolve().parents[1]


def _gaussians(ply, y_down=True):
    v = read_vertex(ply)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    if y_down:
        xyz = xyz * np.array([1.0, -1.0, -1.0])
    alpha = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    return v, xyz, alpha


def render_orbit_gs(ply, out_dir, n: int = 16, elevation: float = 15.0, y_down: bool = True) -> list[Path]:
    """athenea renders of the splat on an orbit; returns the white-background frames (JPEG) for ENEAS."""
    out = Path(out_dir); (out / "views").mkdir(parents=True, exist_ok=True); (out / "frames").mkdir(exist_ok=True)
    frames = [out / "frames" / f"f{i:02d}.jpg" for i in range(n)]
    if all(f.exists() for f in frames) and (out / "cameras.json").exists():
        return frames
    _, xyz, alpha = _gaussians(ply, y_down)
    p = xyz[alpha > 0.05]; lo, hi = np.percentile(p, 0.5, 0), np.percentile(p, 99.5, 0)
    C = (lo + hi) / 2; r = float(np.linalg.norm(hi - lo) / 2)
    D = r / math.sin(math.atan(APERTURE / 2 / FOCAL)) * 1.08
    UP = np.array([0, 1.0, 0]); E1 = np.array([0, 0, 1.0]); E2 = np.cross(UP, E1)
    views = []
    for i in range(n):
        e, t = math.radians(elevation), 2 * math.pi * i / n
        eye = C + D * (math.cos(e) * (math.cos(t) * E1 + math.sin(t) * E2) + math.sin(e) * UP)
        f = C - eye; f /= np.linalg.norm(f); rr = np.cross(f, UP); rr /= np.linalg.norm(rr); d = np.cross(f, rr)
        R = np.stack([rr, d, f])
        views.append({"name": f"f{i:02d}", "eye": eye.tolist(), "target": C.tolist(), "up": UP.tolist(),
                      "R_world_to_cam": R.tolist(), "t_world_to_cam": (-R @ eye).tolist()})
    fx = FOCAL / APERTURE * SIZE
    json.dump({"convention": "opencv", "intrinsics": {"width": SIZE, "height": SIZE, "fx": fx, "fy": fx,
               "cx": SIZE / 2, "cy": SIZE / 2}, "views": views, "centre": C.tolist(), "radius": r},
              open(out / "cameras.json", "w"), indent=1)
    with open(out / "cameras.txt", "w") as fo:
        for vv in views:
            fo.write(" ".join([vv["name"]] + [f"{x:.7g}" for x in vv["eye"] + vv["target"] + vv["up"]]) + "\n")
    from .athenea import render_splats
    render_splats(ply, [(vv["name"], vv["eye"], vv["target"], vv["up"]) for vv in views], out, size=SIZE,
                  rotate_x180=y_down, background=(1.0, 1.0, 1.0))
    for vv, fr in zip(views, frames):
        Image.open(out / "views" / f"{vv['name']}.png").convert("RGB").save(fr, quality=95)
    return frames


def _depth(exr):
    import OpenEXR
    z = OpenEXR.File(str(exr)).channels()["Z"].pixels.astype(np.float32)
    return np.where(np.isfinite(z), z, 0.0)


def vote_gaussians(ply, orbit_dir, masks: dict, y_down: bool = True, k_fill: int = 16) -> tuple[np.ndarray, list]:
    """votes [n_phrases, n_gaussians] from the orbit masks {"class|phrase": [n_views, H, W] bool}."""
    meta = json.loads((Path(orbit_dir) / "cameras.json").read_text())
    it = meta["intrinsics"]; tol = DEPTH_TOL * meta["radius"]
    _, xyz, alpha = _gaussians(ply, y_down)
    names = list(masks)
    votes = np.zeros((len(names), len(xyz)), np.float32)
    for vi, vv in enumerate(meta["views"]):
        R = np.asarray(vv["R_world_to_cam"]); t = np.asarray(vv["t_world_to_cam"])
        cam = xyz @ R.T + t; z = cam[:, 2]
        Z = _depth(Path(orbit_dir) / "views" / f"{vv['name']}.exr")
        H, W = Z.shape
        u = np.round(it["fx"] * cam[:, 0] / np.maximum(z, 1e-6) + it["cx"]).astype(int)
        v = np.round(it["fy"] * cam[:, 1] / np.maximum(z, 1e-6) + it["cy"]).astype(int)
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(ok)[0]
        zs = Z[v[idx], u[idx]]
        vis = idx[(zs > 0) & (np.abs(z[idx] - zs) < tol)]
        label = np.full((H, W), -1)
        for i in sorted(range(len(names)), key=lambda i: -masks[names[i]][vi].sum()):
            m = masks[names[i]][vi]
            if m.shape != (H, W):
                m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
            label[m] = i
        lab = label[v[vis], u[vis]]
        sel = lab >= 0
        np.add.at(votes, (lab[sel], vis[sel]), alpha[vis[sel]].astype(np.float32))
    # gaussians no view saw as the surface (inside, behind, between views): their seen neighbours' votes
    seen = votes.sum(0) > 0
    print(f"gaussians voted directly: {seen.mean():.1%} (opacity-weighted {alpha[seen].sum() / alpha.sum():.1%})", flush=True)
    if seen.any() and (~seen).any():
        tree = cKDTree(xyz[seen])
        _, nn = tree.query(xyz[~seen], k=min(k_fill, int(seen.sum())))
        nn = np.atleast_2d(nn)
        votes[:, ~seen] = votes[:, seen][:, nn].mean(-1)
    return votes, names


def texel_votes(asset_dir, ply, votes_g: np.ndarray, R: int, y_down: bool = True, k: int = 8) -> np.ndarray:
    """votes [n_phrases, R*R] for the atlas texels, from their nearest gaussians (all texels get votes)."""
    from .materials import _raster
    pos, _, mask = _raster(Path(asset_dir), R)
    _, xyz, alpha = _gaussians(ply, y_down)
    keep = alpha > 0.05
    xyz, alpha, vg = xyz[keep], alpha[keep], votes_g[:, keep]
    vg = vg / np.maximum(vg.sum(0, keepdims=True), 1e-9)              # each gaussian: a distribution over phrases
    tree = cKDTree(xyz)
    ids = np.nonzero(mask.reshape(-1))[0]
    P = pos.reshape(-1, 3)[ids]
    out = np.zeros((vg.shape[0], R * R), np.float32)
    h = None
    for a in range(0, len(P), 500_000):                 # chunks: 10 phrases × 9 M texels × k would not fit
        d, nn = tree.query(P[a:a + 500_000], k=k)
        if h is None:
            h = np.median(d[:, 0]) * 2 + 1e-9           # kernel width: twice the typical nearest-gaussian distance
        w = alpha[nn] * np.exp(-0.5 * (d / h) ** 2) + 1e-12
        out[:, ids[a:a + 500_000]] = np.einsum("pnk,nk->pn", vg[:, nn], w) / w.sum(1)
    return out


def write_ply(ply, out_ply, votes_g: np.ndarray, names: list, y_down: bool = True) -> dict:
    """The splat with per-gaussian material attributes; names → library values. Returns the class shares."""
    from .materials import _lib
    v, _, _ = _gaussians(ply, y_down)
    cls_names = [n.split("|")[0] for n in names]
    mid = votes_g.argmax(0).astype(np.int32)
    lib = [_lib(c) for c in cls_names]
    extra = {"mat_id": mid.astype(np.float32)}
    for key, default in (("roughness", 0.5), ("metallic", 0.0), ("transmission", 0.0), ("subsurface", 0.0),
                         ("ior", 1.5)):
        vals = np.array([float(L.get(key) if L.get(key) is not None else default) for L in lib], np.float32)
        extra[key] = vals[mid]
    dt = v.dtype.descr + [(k_, "<f4") for k_ in extra]
    out = np.empty(len(v), dtype=dt)
    for n_ in v.dtype.names:
        out[n_] = v[n_]
    for k_, a in extra.items():
        out[k_] = a
    write_vertex(out_ply, out)
    Path(out_ply).with_name("mat_names.json").write_text(json.dumps({"mat_id": names}, indent=1))
    return {n: round(float((mid == i).mean()), 4) for i, n in enumerate(names)}


def add_albedo(asset_dir, ply_in, ply_out=None, y_down: bool = True) -> dict:
    """Per-gaussian albedo from the finished base colour of the mesh (basecolor_delit.png: highlights removed,
    metals at their F0 colour): each gaussian takes the texel nearest to its centre. Written as albedo_r/g/b
    (linear, 0..1) next to Idol's f_dc_*, which is baked radiance (light included) and so not for relighting."""
    from .materials import _raster
    asset = Path(asset_dir)
    tex = np.asarray(Image.open(asset / "basecolor_delit.png").convert("RGB"), np.float32)[::-1] / 255.0
    R = tex.shape[0]
    pos, _, mask = _raster(asset, R)
    P = pos[mask]; C = tex[mask]
    step = max(1, len(P) // 2_000_000)
    tree = cKDTree(P[::step]); C = C[::step]
    v = read_vertex(ply_in)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) * (np.array([1.0, -1.0, -1.0]) if y_down else 1)
    d, nn = tree.query(xyz, k=4)
    w = 1.0 / np.maximum(d, 1e-6)
    srgb = (C[nn] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
    names = [n for n in v.dtype.names if n not in ("albedo_r", "albedo_g", "albedo_b")]
    out = np.empty(len(v), dtype=[(n, v.dtype[n]) for n in names] + [(c, "<f4") for c in ("albedo_r", "albedo_g", "albedo_b")])
    for n in names:
        out[n] = v[n]
    out["albedo_r"], out["albedo_g"], out["albedo_b"] = lin.T
    write_vertex(ply_out or ply_in, out)
    return {"gaussians": len(v), "surface_distance_median": float(np.median(d[:, 0]))}
