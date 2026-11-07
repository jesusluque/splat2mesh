"""The finish pass on the splat itself, with no mesh: what athenea needs to relight a splat with per-gaussian materials
(gs_materials.write_ply), taken from the splat and its own athenea renders only.

  1. normals: every orbit view's depth (the EXR's Z from gs_materials.render_orbit_gs) gives a normal per pixel;
     each gaussian that is the visible surface in a view takes it, weighted by its opacity and by how squarely the
     view faces it. Gaussians no view sees take their neighbours' normals. Idol's gaussians are nearly round, so
     their own shortest axis is no normal (athenea would draw random glints).
  2. floaters: a gaussian that is never the visible surface, yet sits in front of it in several views, is too faint
     to show in the depth: a loose speck that relit catches the light. Flagged (loose = 1) and left out of the USD.
  3. albedo: Idol's colour is radiance (light baked in). Per material, as the mesh's finish pass does per texel:
     frozen highlights on glossy dielectrics go back to the piece's colour, metals take the library F0 times their
     fine luminance detail, uniform glass takes its measured tint per wall. Neighbourhoods are the gaussians' own
     nearest neighbours of the same material instead of a texture.
  4. finish.json next to the PLY: per class transmission, tint, patterned, thin_walled and ior (athenea_splat
     reads it for the glass).

    python -m splat2mesh.gs_finish SPLAT_MATERIALS.ply ORBIT_DIR [--y-down]
Writes nx, ny, nz (in the PLY's own frame), loose and albedo_r/g/b into the PLY, and finish.json beside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .gs_materials import DEPTH_TOL, SH_C0, _depth, _gaussians
from .ply import read_vertex, write_vertex

K = 32                   # neighbours of the same material: a gaussian's "piece" for colour statistics
BIG = 0.08              # × object radius: a gaussian wider than this is background or a veil, not the object
FRONT_VIEWS = 3          # in front of the visible surface in this many views, never on it: a floater
EDGE_TOL = 3.0           # × DEPTH_TOL: a depth step this large between pixels is an edge, no normal there
DEPTH_BLUR = 1.5         # px: the rendered depth is smoothed before its slopes are taken


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def depth_normals(Z, fx, fy, cx, cy, tol):
    """Camera-space normals [H, W, 3] (OpenCV axes, facing the camera) from a z-depth map; zero where unknown."""
    H, W = Z.shape
    ok = Z > 0
    # blur only inside the object: normalised convolution, so the silhouette does not pull towards 0
    zb = ndi.gaussian_filter(np.where(ok, Z, 0.0), DEPTH_BLUR) / np.maximum(ndi.gaussian_filter(ok * 1.0, DEPTH_BLUR), 1e-6)
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    X = np.stack([(u - cx) * zb / fx, (v - cy) * zb / fy, zb], -1)
    du = np.zeros_like(X); dv = np.zeros_like(X)
    du[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / 2; dv[1:-1] = (X[2:] - X[:-2]) / 2
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    n = np.where((np.einsum("hwi,hwi->hw", n, X) > 0)[..., None], -n, n)          # towards the camera
    edge = np.zeros_like(ok)
    dz_u = np.abs(np.diff(Z, axis=1)); dz_v = np.abs(np.diff(Z, axis=0))
    edge[:, 1:] |= dz_u > EDGE_TOL * tol; edge[:, :-1] |= dz_u > EDGE_TOL * tol
    edge[1:] |= dz_v > EDGE_TOL * tol; edge[:-1] |= dz_v > EDGE_TOL * tol
    good = ok & ~ndi.binary_dilation(edge | ~ok, iterations=1)
    return np.where(good[..., None], n, 0.0), good


def normals_and_floaters(ply, orbit_dir, y_down: bool = True, k_fill: int = 16):
    """Per-gaussian unit normals in the orbit's (USD) frame, and the floater flag."""
    meta = json.loads((Path(orbit_dir) / "cameras.json").read_text())
    it = meta["intrinsics"]; tol = DEPTH_TOL * meta["radius"]
    _, xyz, alpha = _gaussians(ply, y_down)
    acc = np.zeros_like(xyz); seen = np.zeros(len(xyz), np.int32); front = np.zeros(len(xyz), np.int32)
    for vv in meta["views"]:
        R = np.asarray(vv["R_world_to_cam"]); t = np.asarray(vv["t_world_to_cam"])
        Z = _depth(Path(orbit_dir) / "views" / f"{vv['name']}.exr")
        H, W = Z.shape
        Nc, good = depth_normals(Z, it["fx"], it["fy"], it["cx"], it["cy"], tol)
        cam = xyz @ R.T + t; z = cam[:, 2]
        u = np.round(it["fx"] * cam[:, 0] / np.maximum(z, 1e-6) + it["cx"]).astype(int)
        v = np.round(it["fy"] * cam[:, 1] / np.maximum(z, 1e-6) + it["cy"]).astype(int)
        idx = np.nonzero((z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H))[0]
        zs = Z[v[idx], u[idx]]
        hit = zs > 0
        vis = idx[hit & (np.abs(z[idx] - zs) < tol)]
        front[idx[hit & (z[idx] < zs - tol)]] += 1
        seen[vis] += 1
        g = good[v[vis], u[vis]]
        vis = vis[g]
        nc = Nc[v[vis], u[vis]]                                        # camera space
        ray = cam[vis] / np.linalg.norm(cam[vis], axis=1, keepdims=True)
        facing = np.clip(-(nc * ray).sum(1), 0, 1)                      # squarely seen views count more
        acc[vis] += (nc @ R) * (alpha[vis] * facing ** 2)[:, None]      # back to world: Rᵀ n
    have = np.linalg.norm(acc, axis=1) > 1e-9
    n = np.zeros_like(acc); n[have] = acc[have] / np.linalg.norm(acc[have], axis=1, keepdims=True)
    if have.any() and (~have).any():
        _, nn = cKDTree(xyz[have]).query(xyz[~have], k=min(k_fill, int(have.sum())))
        m = n[have][np.atleast_2d(nn)].sum(1)
        n[~have] = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    v_ = read_vertex(ply)
    big = np.exp(np.stack([v_[f"scale_{i}"] for i in range(3)], 1)).max(1) > BIG * meta["radius"]
    loose = ((seen == 0) & (front >= FRONT_VIEWS)) | big
    print(f"normals: {have.mean():.1%} from the views, the rest from neighbours; floaters {int(loose.sum())} ({int(big.sum())} oversized)", flush=True)
    return n.astype(np.float32), loose


def _neighbours(xyz, sel, k=K):
    s = np.nonzero(sel)[0]
    _, kn = cKDTree(xyz[s]).query(xyz[s], k=min(k, len(s)))
    return s, np.atleast_2d(kn)


def finish(ply, orbit_dir, y_down: bool = True) -> dict:
    from .materials import (GLASS_PATTERN_SPREAD, GLASS_WALLS, GLOSS_EVIDENCE_FRAC, GLOSS_EVIDENCE_MAX_ROUGH,
                            GLOSS_UNIFORM_SPREAD, METAL_DETAIL_MIN, _lab_to_srgb, _lib, _srgb_to_lab)
    ply = Path(ply)
    v = read_vertex(ply)
    names = json.loads(ply.with_name("mat_names.json").read_text())["mat_id"]
    mid = np.asarray(v["mat_id"]).astype(int)
    _, xyz, alpha = _gaussians(ply, y_down)
    nrm, loose = normals_and_floaters(ply, orbit_dir, y_down)
    srgb = np.clip(0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1), 0, 1).astype(np.float64)
    base = srgb.copy()
    rough = np.asarray(v["roughness"], np.float32).copy(); metal = np.asarray(v["metallic"], np.float32).copy()
    trans = np.asarray(v["transmission"], np.float32).copy()
    info = {}
    for i, name in enumerate(names):
        cls = name.split("|")[0]; L = _lib(cls); sel = (mid == i) & ~loose
        if sel.sum() < 50:
            continue
        s, kn = _neighbours(xyz, sel)
        lab = _srgb_to_lab(srgb[s])
        if L.get("metallic", 0) >= 0.5 and L.get("f0"):
            # metal: the library F0 times the fine luminance detail (hammering, engraving); the broad variation is
            # frozen reflections. Dark colourless specks are ink or enamel, not metal
            lum = np.log(_lin(srgb[s]).mean(1) + 1e-3)
            hp = lum - lum[kn].mean(1)
            f0 = np.asarray(L["f0"], np.float64)
            detail = np.clip(np.exp(0.9 * hp), METAL_DETAIL_MIN, 1.0 / f0.max())
            base[s] = _srgb(f0[None] * detail[:, None])
            ink = (_lin(srgb[s]).mean(1) < 0.08) & (np.hypot(lab[:, 1], lab[:, 2]) < 10)
            ink &= ink[kn].mean(1) < 0.5                                  # thin strokes; a wide dark patch is a reflection
            metal[s[ink]] = 0.0; rough[s[ink]] = 0.3; base[s[ink]] = srgb[s[ink]]
        dielectric = L.get("metallic", 0) < 0.5 and L.get("transmission", 0) <= 0
        ab0 = np.median(lab[:, 1:], 0); L0 = float(np.median(lab[:, 0])); c0 = float(np.hypot(*ab0))
        spread = float(np.median(np.hypot(*(lab[:, 1:] - ab0).T)))
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        pale = (chroma < 0.8 * c0) & (lab[:, 0] > L0)
        glossy = dielectric and L.get("roughness", 0.5) <= 0.4
        if (dielectric and not glossy and L.get("roughness", 0.5) > 0.4
                and L.get("range", [1])[0] <= GLOSS_EVIDENCE_MAX_ROUGH and len(s) > 2000):
            frac = float((pale & (lab[:, 0] > L0 + 12)).mean())
            if c0 > 15 and spread < GLOSS_UNIFORM_SPREAD and frac > GLOSS_EVIDENCE_FRAC:
                glossy = True
                r_lo = max(0.25, float(L["range"][0]) - 0.1)
                rough[s] = np.minimum(rough[s], r_lo)
                print(f"polish {cls}: {frac:.1%} highlight gaussians → roughness {r_lo}", flush=True)
        if glossy and c0 > 15 and spread < GLOSS_UNIFORM_SPREAD and pale.any():
            fix = lab[pale].copy(); fix[:, 0] = L0 + 0.3 * (fix[:, 0] - L0); fix[:, 1:] = ab0[None]
            base[s[pale]] = _lab_to_srgb(fix)
            print(f"gloss {cls}: {int(pale.sum())} pale-highlight gaussians back to the piece colour", flush=True)
        elif dielectric and L.get("roughness", 0.5) <= 0.5:
            # patterned glossy pieces: only the clear highlights (much brighter, paler than their neighbours) go
            # to the median of their non-highlight neighbours
            Ln = lab[:, 0][kn]
            hl = (lab[:, 0] > Ln.mean(1) + 2 * Ln.std(1) + 1e-6) & (chroma < np.median(chroma) + 5)
            if hl.any():
                ok = ~hl[kn]
                base[s[hl]] = np.array([np.median(srgb[s][kn[j][ok[j]]], 0) if ok[j].any() else srgb[s[j]]
                                        for j in np.nonzero(hl)[0]])
        patterned, tint = False, None
        if L.get("transmission", 0) > 0:
            patterned = spread > GLASS_PATTERN_SPREAD
            if patterned:
                trans[s] *= 0.5
            else:
                # the more opaque half: less of the background seen through it
                strong = alpha[s] > np.percentile(alpha[s], 50)
                tint = np.median(_lin(srgb[s][strong]), 0); tint = tint / max(tint.max(), 1e-4) * 0.95
                # a thin wall: the shell's two walls share the tint; a solid: the tint at each of its two
                # crossings, which is what the captured colour looks like through the whole piece
                base[s] = _srgb(tint ** (1.0 / GLASS_WALLS) if L.get("thin_walled", False) else tint)[None]
            print(f"glass {cls}: colour spread {spread:.1f} → {'patterned' if patterned else 'uniform tint'}", flush=True)
        info[cls] = {"transmission": float(trans[s].max()), "subsurface": L.get("subsurface", 0.0),
                     "patterned": patterned, "tint": None if tint is None else [round(float(x), 4) for x in tint],
                     "thin_walled": L.get("thin_walled", False), "ior": L.get("ior")}
    alb = _lin(np.clip(base, 0, 1)).astype(np.float32)
    n_ply = nrm * (np.array([1, -1, -1], np.float32) if y_down else 1)               # back to the PLY's frame
    keep_names = [n for n in v.dtype.names if n not in ("albedo_r", "albedo_g", "albedo_b", "nx", "ny", "nz", "loose",
                                                        "roughness", "metallic", "transmission")]
    extra = {"roughness": rough, "metallic": metal, "transmission": trans, "albedo_r": alb[:, 0],
             "albedo_g": alb[:, 1], "albedo_b": alb[:, 2], "nx": n_ply[:, 0], "ny": n_ply[:, 1], "nz": n_ply[:, 2],
             "loose": loose.astype(np.float32)}
    out = np.empty(len(v), dtype=[(n, v.dtype[n]) for n in keep_names] + [(k, "<f4") for k in extra])
    for n in keep_names:
        out[n] = v[n]
    for k, a in extra.items():
        out[k] = a
    write_vertex(ply, out)
    ply.with_name("finish.json").write_text(json.dumps(info, indent=1, default=float))
    return {"gaussians": len(v), "floaters": int(loose.sum()), "classes": list(info)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply"); ap.add_argument("orbit"); ap.add_argument("--y-down", action="store_true")
    a = ap.parse_args()
    print(finish(a.ply, a.orbit, a.y_down))


if __name__ == "__main__":
    main()
