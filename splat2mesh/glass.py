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


class RepeatPlane(Background):
    """athenea planes.json entry: centre ± side/2 along u_axis, v_axis; texture repeats uv_repeat times,
    u = frac(s), v = frac(t), v = 0 at the image's bottom row. Texture already linear if linear=True."""
    def __init__(self, p: dict, tex: np.ndarray):
        self.c = np.array(p["centre"], float); self.u = np.array(p["u_axis"], float); self.v = np.array(p["v_axis"], float)
        self.side = float(p["side"]); self.rep = float(p["uv_repeat"]); self.tex = tex

    def sample(self, o, d):
        n = np.cross(self.u, self.v)
        den = d @ n
        t = ((self.c - o) @ n) / np.where(np.abs(den) < 1e-12, np.nan, den)
        p = o + d * t[:, None] - self.c
        a = p @ self.u / self.side + 0.5
        b = p @ self.v / self.side + 0.5
        ok = (t > 1e-6) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
        h, w = self.tex.shape[:2]
        uu = np.mod(a * self.rep, 1.0); vv = np.mod(b * self.rep, 1.0)
        col = np.zeros((len(o), 3))
        col[ok] = self.tex[((1 - vv[ok]) * (h - 1)).astype(int), (uu[ok] * (w - 1)).astype(int)]
        return col, ok


def load_athenea_planes(path) -> list:
    """planes.json → one RepeatPlane per view (same order as cameras.json views)."""
    path = Path(path)
    d = json.loads(path.read_text())
    tex = np.asarray(Image.open(path.parent / d["texture"]).convert("RGB"), np.float32) / 255.0
    tex = np.where(tex <= 0.04045, tex / 12.92, ((tex + 0.055) / 1.055) ** 2.4)   # sRGB → linear, as athenea
    return [RepeatPlane(p, tex) for p in d["planes"]]


class GlassTracer:
    def __init__(self, mesh: trimesh.Trimesh, background: Background, max_bounces: int = 6,
                 smooth_normals: bool = True):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
        self.fn = mesh.face_normals
        self.faces = mesh.faces
        self.vn = mesh.vertex_normals if smooth_normals else None   # shading normals, as a renderer uses
        self.bg = background
        self.max_bounces = max_bounces

    def _hit(self, o, d):
        r = self.scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
        t = r["t_hit"].numpy().astype(np.float64)
        pid = r["primitive_ids"].numpy()
        ok = np.isfinite(t)
        n = np.zeros_like(o)
        if self.vn is None:
            n[ok] = self.fn[pid[ok]]
        else:
            bu = r["primitive_uvs"].numpy().astype(np.float64)
            f = self.faces[pid[ok]]
            b1, b2 = bu[ok, 0:1], bu[ok, 1:2]
            nn = self.vn[f[:, 0]] * (1 - b1 - b2) + self.vn[f[:, 1]] * b1 + self.vn[f[:, 2]] * b2
            n[ok] = nn / np.linalg.norm(nn, axis=1, keepdims=True)
        return t, n, ok

    def trace(self, o, d, ior):
        """General dielectric path: at every hit the side is decided by the geometric normal
        (entering: n1=1→n2=ior, leaving: ior→1), so inner shells and several pieces are handled.
        Returns (background colour, fresnel throughput, path length inside, valid). Rays that leave
        without meeting the background see black (valid, colour 0), as in a scene with no other light."""
        N = len(o)
        thr = np.ones(N)
        L = np.zeros(N)
        nref = np.zeros(N)
        depth = np.zeros(N)                    # number of media we are inside (0 = air)
        alive = np.ones(N, bool)
        for _ in range(self.max_bounces * 2 + 2):
            if not alive.any():
                break
            t, n, h = self._hit(o, d)
            esc = alive & ~h
            alive &= h
            if not alive.any():
                break
            L += np.where(alive & (depth > 0), t, 0)
            p = o + d * np.where(alive, t, 0)[:, None]
            entering = (d * n).sum(1) < 0
            nf = np.where(entering[:, None], n, -n)          # facing the ray
            eta = np.where(entering, 1.0 / ior, ior)
            cos_i = -(d * nf).sum(1)
            k = 1 - eta ** 2 * (1 - cos_i ** 2)
            tir = k < 0
            tdir = eta[:, None] * d + (eta * cos_i - np.sqrt(np.maximum(k, 0)))[:, None] * nf
            tdir /= np.linalg.norm(tdir, axis=1, keepdims=True)
            refl = d + 2 * cos_i[:, None] * nf
            ft = _fresnel_t(cos_i, eta)
            go = alive & ~tir
            thr = np.where(go, thr * ft, thr)
            nref += go
            depth = np.where(go, depth + np.where(entering, 1, -1), depth)
            d = np.where(go[:, None], tdir, np.where(alive[:, None], refl, d))
            o = np.where(alive[:, None], p + d * 1e-5, o)
        stuck = alive                                     # still bouncing: energy lost
        self.last_nref = nref
        col, bg_hit = self.bg.sample(o, d)
        col = np.where((bg_hit & ~stuck)[:, None], col, 0.0)
        valid = nref > 0                                  # ray actually went through glass
        return col, thr, L, valid


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


def estimate(mesh: trimesh.Trimesh, views, background, ior_range=(1.2, 2.0),
             samples_per_view: int = 4000, iters: int = 18, seed: int = 0, linear_obs=None,
             absorption: str = "auto", clip: float = 1.0) -> dict:
    """views: splat2mesh.views.View list (rgb, alpha mask, K, w2c).
    background: one Background for all views, or a list with one per view.
    linear_obs: optional list of HxWx3 linear radiance images (e.g. EXR) used instead of sRGB rgb.
    absorption: "beer" (exp(-σ·length inside)), "interface" (tint per refraction, athenea's surface model) or "auto"."""
    rng = np.random.default_rng(seed)
    bgs = background if isinstance(background, list) else [background] * len(views)
    groups = []
    for i, v in enumerate(views):
        ys, xs = np.nonzero(v.alpha > 0.5)
        if len(xs) == 0:
            continue
        sel = rng.choice(len(xs), min(samples_per_view, len(xs)), replace=False)
        x, y = xs[sel] + 0.5, ys[sel] + 0.5
        dc = np.stack([(x - v.K[0, 2]) / v.K[0, 0], (y - v.K[1, 2]) / v.K[1, 1], np.ones_like(x)], 1)
        d = dc @ v.c2w[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.broadcast_to(v.c2w[:3, 3], d.shape).copy()
        obs = linear_obs[i][ys[sel], xs[sel]] if linear_obs is not None else v.rgb[ys[sel], xs[sel]] ** 2.2
        groups.append((o, d, np.minimum(obs, clip), GlassTracer(mesh, bgs[i])))

    def evaluate(ior, model):
        P, OB, X = [], [], []
        miss_n = tot = 0
        for o, d, obs, tr in groups:
            col, thr, L, ok = tr.trace(o, d, ior)
            P.append(col[ok] * thr[ok, None]); OB.append(obs[ok])
            X.append(L[ok] if model == "beer" else tr.last_nref[ok])
            miss_n += (~ok).sum(); tot += len(ok)
        pred, obs, x = np.concatenate(P), np.concatenate(OB), np.concatenate(X)
        sig = _fit_sigma(obs, pred, x, np.ones(len(obs)))
        res = np.abs(pred * np.exp(-x[:, None] * sig) - obs).sum(1)
        miss = miss_n / max(tot, 1)
        return float(np.mean(np.minimum(res, 0.5)) + 0.5 * miss), sig, float(miss)

    def fit(model):
        loss = lambda i: evaluate(i, model)[0]
        grid = np.linspace(*ior_range, 17)
        vals = [loss(g) for g in grid]
        k = int(np.argmin(vals))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        g = (5 ** 0.5 - 1) / 2
        c, e = hi - g * (hi - lo), lo + g * (hi - lo)
        fc, fe = loss(c), loss(e)
        for _ in range(iters):
            if fc < fe:
                hi, e, fe = e, c, fc; c = hi - g * (hi - lo); fc = loss(c)
            else:
                lo, c, fc = c, e, fe; e = lo + g * (hi - lo); fe = loss(e)
        ior = (lo + hi) / 2
        f, sig, miss = evaluate(ior, model)
        return {"model": model, "ior": float(ior), "sigma_rgb": sig.tolist(),
                "tint": np.exp(-sig).tolist(), "loss": f, "rays_lost": miss,
                "scan": {"ior": grid.tolist(), "loss": vals}}

    models = ["beer", "interface"] if absorption == "auto" else [absorption]
    fits = [fit(m) for m in models]
    best = min(fits, key=lambda r: r["loss"])
    best = dict(best)
    best["transmission_per_unit"] = best["tint"]          # kept for set_glass / older callers
    best["alternatives"] = [{k: f[k] for k in ("model", "ior", "tint", "loss")} for f in fits]
    best["samples"] = int(sum(len(g[0]) for g in groups))
    return best


def load_background(path: str | Path) -> Background:
    path = Path(path)
    return Background(json.loads(path.read_text()), path.parent)
