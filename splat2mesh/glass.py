"""Automatic glass material estimation (IOR + absorption tint) from images of a transparent object
in front of a known background.

Forward model per pixel inside the object's mask:
  camera ray → refract in (Snell, Fresnel transmittance) → trace inside the closed mesh
  (exit refraction, or total internal reflection, up to `max_bounces`) → refract out
  → hit the background → sample its texture.
  colour = T_fresnel · exp(-σ · L_inside) · background(hit)
Fit: golden-section search on IOR (σ solved in closed form for each candidate by least squares in
log space), robust loss over all views. Output: {"ior", "sigma_rgb", "transmission_per_unit", "loss"}.

Background description (JSON), e.g. a textured plane:
  {"type": "plane", "origin": [x,y,z], "u": [..], "v": [..], "texture": "checker.png"}
  point(s,t) = origin + s·u + t·v, s,t ∈ [0,1], texture row 0 at t = 1 (image top).
or a list of planes (cube faces) under {"type": "planes", "planes": [...]}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image


class Background:
    def __init__(self, desc: dict, base: Path):
        planes = desc["planes"] if desc["type"] == "planes" else [desc]
        self.planes = []
        for p in planes:
            tex = np.asarray(Image.open(base / p["texture"]).convert("RGB"), np.float32) / 255.0
            tex = tex ** 2.2 if p.get("srgb", True) else tex     # work in linear light
            self.planes.append((np.array(p["origin"], float), np.array(p["u"], float), np.array(p["v"], float), tex))

    def sample(self, o: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.zeros((len(o), 3))
        best = np.full(len(o), np.inf)
        for O, U, V, tex in self.planes:
            n = np.cross(U, V)
            den = d @ n
            t = ((O - o) @ n) / np.where(np.abs(den) < 1e-12, np.nan, den)
            p = o + d * t[:, None] - O
            # solve p = s U + t V
            G = np.array([[U @ U, U @ V], [U @ V, V @ V]])
            st = np.linalg.solve(G, np.stack([p @ U, p @ V])).T
            ok = (t > 1e-6) & (t < best) & np.all((st >= 0) & (st <= 1), 1)
            h, w = tex.shape[:2]
            x = (st[ok, 0] * (w - 1)).astype(int)
            y = ((1 - st[ok, 1]) * (h - 1)).astype(int)
            col[ok] = tex[y, x]
            best[ok] = t[ok]
        return col, np.isfinite(best)


def _refract(d, n, eta):
    """d, n unit; n faces against d. eta = n1/n2. Returns (dir, tir_mask)."""
    cos_i = -(d * n).sum(1)
    k = 1 - eta ** 2 * (1 - cos_i ** 2)
    tir = k < 0
    t = eta * d + (eta * cos_i - np.sqrt(np.maximum(k, 0)))[:, None] * n
    return t / np.linalg.norm(t, axis=1, keepdims=True), tir, cos_i


def _fresnel_t(cos_i, eta):
    sin_t2 = eta ** 2 * (1 - cos_i ** 2)
    cos_t = np.sqrt(np.clip(1 - sin_t2, 0, 1))
    rs = ((eta * cos_i - cos_t) / (eta * cos_i + cos_t + 1e-12)) ** 2
    rp = ((eta * cos_t - cos_i) / (eta * cos_t + cos_i + 1e-12)) ** 2
    return np.where(sin_t2 >= 1, 0.0, 1 - 0.5 * (rs + rp))


class GlassTracer:
    def __init__(self, mesh: trimesh.Trimesh, background: Background, max_bounces: int = 6):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
        self.fn = mesh.face_normals
        self.bg = background
        self.max_bounces = max_bounces

    def _hit(self, o, d):
        r = self.scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
        t = r["t_hit"].numpy().astype(np.float64)
        pid = r["primitive_ids"].numpy()
        ok = np.isfinite(t)
        n = np.zeros_like(o)
        n[ok] = self.fn[pid[ok]]
        return t, n, ok

    def trace(self, o, d, ior):
        """Returns (background colour, fresnel throughput, path length inside, valid)."""
        N = len(o)
        thr = np.ones(N)
        L = np.zeros(N)
        t, n, hit = self._hit(o, d)
        valid = hit.copy()
        p = o + d * np.where(hit, t, 0)[:, None]
        n = np.where(((d * n).sum(1) > 0)[:, None], -n, n)
        d2, tir, cos_i = _refract(d, n, 1.0 / ior)
        thr *= _fresnel_t(cos_i, 1.0 / ior)
        d = d2
        o = p + d * 1e-5
        inside = valid.copy()
        for _ in range(self.max_bounces):
            if not inside.any():
                break
            t, n, h = self._hit(o, d)
            lost = inside & ~h
            valid &= ~lost
            inside &= h
            L += np.where(inside, t, 0)
            p = o + d * np.where(inside, t, 0)[:, None]
            n = np.where(((d * n).sum(1) > 0)[:, None], -n, n)   # face the incoming ray
            d_out, tir, cos_i = _refract(d, n, ior)
            ft = _fresnel_t(cos_i, ior)
            leaving = inside & ~tir
            refl = d - 2 * (d * n).sum(1, keepdims=True) * n
            thr = np.where(leaving, thr * ft, thr)
            d = np.where(leaving[:, None], d_out, np.where(inside[:, None], refl, d))
            o = p + d * 1e-5
            inside &= tir                                        # TIR rays keep bouncing
        valid &= ~inside                                          # never got out
        col, bg_hit = self.bg.sample(o, d)
        return col, thr, L, valid & bg_hit


def _fit_sigma(obs, pred, L, w):
    """obs ≈ pred · exp(-σ L) per channel → σ by weighted LS on log ratio (σ ≥ 0)."""
    eps = 1e-3
    r = np.log((obs + eps) / (pred + eps))           # = -σ L
    sig = np.zeros(3)
    for c in range(3):
        ww = w * (pred[:, c] > 0.05) * (L > 1e-6)
        den = (ww * L * L).sum()
        sig[c] = max(0.0, -(ww * L * r[:, c]).sum() / den) if den > 0 else 0.0
    return sig


def estimate(mesh: trimesh.Trimesh, views, background: Background, ior_range=(1.2, 2.0),
             samples_per_view: int = 4000, iters: int = 18, seed: int = 0) -> dict:
    """views: splat2mesh.views.View list with linear-ish rgb, alpha mask, K, w2c."""
    rng = np.random.default_rng(seed)
    O, D, OBS = [], [], []
    for v in views:
        ys, xs = np.nonzero(v.alpha > 0.5)
        if len(xs) == 0:
            continue
        sel = rng.choice(len(xs), min(samples_per_view, len(xs)), replace=False)
        x, y = xs[sel] + 0.5, ys[sel] + 0.5
        dc = np.stack([(x - v.K[0, 2]) / v.K[0, 0], (y - v.K[1, 2]) / v.K[1, 1], np.ones_like(x)], 1)
        d = dc @ v.c2w[:3, :3].T
        D.append(d / np.linalg.norm(d, axis=1, keepdims=True))
        O.append(np.broadcast_to(v.c2w[:3, 3], d.shape))
        OBS.append(v.rgb[ys[sel], xs[sel]] ** 2.2)          # sRGB → linear
    o, d, obs = np.concatenate(O), np.concatenate(D), np.concatenate(OBS)
    tracer = GlassTracer(mesh, background)

    def loss(ior):
        col, thr, L, ok = tracer.trace(o, d, ior)
        pred = col * thr[:, None]
        sig = _fit_sigma(obs[ok], pred[ok], L[ok], np.ones(ok.sum()))
        model = pred[ok] * np.exp(-L[ok, None] * sig)
        res = np.abs(model - obs[ok]).sum(1)
        miss = (~ok).mean()
        return float(np.mean(np.minimum(res, 0.5)) + 0.5 * miss), sig, float(miss)

    # coarse scan then golden-section refinement
    grid = np.linspace(*ior_range, 17)
    vals = [loss(g)[0] for g in grid]
    k = int(np.argmin(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (5 ** 0.5 - 1) / 2
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = loss(c)[0], loss(e)[0]
    for _ in range(iters):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a); fc = loss(c)[0]
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a); fe = loss(e)[0]
    ior = (a + b) / 2
    f, sig, miss = loss(ior)
    return {"ior": float(ior), "sigma_rgb": sig.tolist(),
            "transmission_per_unit": np.exp(-sig).tolist(), "loss": f, "rays_lost": miss,
            "scan": {"ior": grid.tolist(), "loss": vals}, "samples": int(len(o))}


def load_background(path: str | Path) -> Background:
    path = Path(path)
    return Background(json.loads(path.read_text()), path.parent)
