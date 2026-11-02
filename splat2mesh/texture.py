"""UV unwrap (xatlas) + multi-view back-projection bake with occlusion test.

For every texel we know its 3D position and normal. Each view contributes its
pixel colour if the texel is visible (depth test against the view's depth map)
with weight cos(theta)^4 * (1 / dist^2), where theta is the angle between the
surface normal and the direction to the camera. Uncovered texels are filled by
dilation + diffusion inpainting so mip-mapping and seams don't bleed black.
"""
from __future__ import annotations

import numpy as np
import trimesh
import xatlas

from .views import View


def unwrap(tm: trimesh.Trimesh, resolution: int, padding: int = 4):
    """xatlas unwrap scaled to fill one `resolution`² atlas.

    xatlas' own estimate of texels-per-unit from `resolution` undershoots badly on
    noisy scan meshes (many small charts → the atlas came out 515² for a 2048 target).
    We set texels_per_unit from the surface area and back off until it fits one atlas.
    """
    # xatlas drops faces below an absolute area epsilon (gives them UV (0,0)); small
    # assets in metres hit it (20% of a 4 cm pawn). UVs are scale-free → normalise first.
    scale = 1.0 / max(float(np.linalg.norm(tm.bounds[1] - tm.bounds[0])), 1e-12)
    verts = (tm.vertices - tm.bounds[0]) * scale
    area = tm.area * scale ** 2
    fill = 0.55
    while True:
        atlas = xatlas.Atlas()
        atlas.add_mesh(verts.astype(np.float32), tm.faces.astype(np.uint32))
        po = xatlas.PackOptions()
        po.resolution = resolution
        po.padding = padding
        po.bilinear = True
        po.texels_per_unit = float(np.sqrt(fill * resolution ** 2 / area))
        co = xatlas.ChartOptions()
        co.max_cost = 4.0          # uv_sweep: lower stretch (σlog 0.096 → 0.081) at the same chart count
        atlas.generate(co, po)
        if atlas.atlas_count == 1 or fill < 0.05:
            break
        fill *= 0.85
    vmap, faces, uvs = atlas[0]
    return vmap, faces, uvs, atlas.chart_count


def rasterize_uv(verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray, normals: np.ndarray, res: int):
    """CPU barycentric rasteriser in UV space. Returns (pos HxWx3, nrm HxWx3, mask HxW)."""
    pos = np.zeros((res, res, 3), np.float32)
    nrm = np.zeros((res, res, 3), np.float32)
    mask = np.zeros((res, res), bool)
    px = uvs * res - 0.5          # texel centres at integer coords
    for f in faces:
        a, b, c = px[f]
        xmin = max(int(np.floor(min(a[0], b[0], c[0]))), 0)
        xmax = min(int(np.ceil(max(a[0], b[0], c[0]))), res - 1)
        ymin = max(int(np.floor(min(a[1], b[1], c[1]))), 0)
        ymax = min(int(np.ceil(max(a[1], b[1], c[1]))), res - 1)
        if xmax < xmin or ymax < ymin:
            continue
        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        p = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        d = p - a
        w1 = (d[:, 0] * v1[1] - v1[0] * d[:, 1]) / den
        w2 = (v0[0] * d[:, 1] - d[:, 0] * v0[1]) / den
        w0 = 1 - w1 - w2
        eps = -0.5 / max(abs(den), 1.0) ** 0.5   # conservative: include edge texels
        ok = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not ok.any():
            continue
        bc = np.stack([w0[ok], w1[ok], w2[ok]], 1)
        yy, xx = p[ok, 1].astype(int), p[ok, 0].astype(int)
        pos[yy, xx] = bc @ verts[f]
        nrm[yy, xx] = bc @ normals[f]
        mask[yy, xx] = True
    n = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = np.where(n > 0, nrm / np.maximum(n, 1e-12), 0)
    return pos, nrm, mask


def _samples(P, N, v, depth_tol, cos_power):
    """Colour and weight of every texel as seen from view v (weight 0 = not usable)."""
    h, w = v.depth.shape
    Pc = P @ v.w2c[:3, :3].T + v.w2c[:3, 3]
    z = Pc[:, 2]
    uvw = Pc @ v.K.T
    u = uvw[:, 0] / np.maximum(uvw[:, 2], 1e-9) - 0.5      # pixel centres at integer coords
    y = uvw[:, 1] / np.maximum(uvw[:, 2], 1e-9) - 0.5
    inside = (z > 1e-6) & (u >= 0) & (u < w - 1) & (y >= 0) & (y < h - 1)
    x0, y0 = np.floor(u).astype(int).clip(0, w - 2), np.floor(y).astype(int).clip(0, h - 2)
    dz = v.depth[np.rint(y).astype(int).clip(0, h - 1), np.rint(u).astype(int).clip(0, w - 1)]
    visible = inside & (dz > 0) & (np.abs(dz - z) < depth_tol)
    if v.alpha is not None:   # all four bilinear taps must be on the object
        a = v.alpha > 0.5
        visible &= a[y0, x0] & a[y0, x0 + 1] & a[y0 + 1, x0] & a[y0 + 1, x0 + 1]
    to_cam = v.center - P
    dist = np.linalg.norm(to_cam, axis=1)
    cos = np.einsum("ij,ij->i", N, to_cam / dist[:, None])
    wgt = np.where(visible & (cos > 0.05), np.maximum(cos, 0) ** cos_power / dist ** 2, 0.0)
    fx, fy = (u - x0)[:, None].clip(0, 1), (y - y0)[:, None].clip(0, 1)
    col = (v.rgb[y0, x0] * (1 - fx) * (1 - fy) + v.rgb[y0, x0 + 1] * fx * (1 - fy)
           + v.rgb[y0 + 1, x0] * (1 - fx) * fy + v.rgb[y0 + 1, x0 + 1] * fx * fy)
    return col, wgt


def bake(pos, nrm, mask, views: list[View], depth_tol: float, cos_power: float = 4.0,
         robust_iters: int = 2, sigma: float = 0.08):
    """Weighted multi-view average, then IRLS passes that down-weight samples far from the
    current estimate (Cauchy weight). Removes speculars, refraction and mis-registered
    silhouette samples that a plain mean smears into the texture."""
    res = pos.shape[0]
    P = pos[mask].astype(np.float64)
    N = nrm[mask].astype(np.float64)
    est = None
    for it in range(robust_iters + 1):
        acc = np.zeros((len(P), 3))
        wsum = np.zeros(len(P))
        for v in views:
            col, wgt = _samples(P, N, v, depth_tol, cos_power)
            if est is not None:
                r2 = ((col - est) ** 2).sum(1)
                wgt = wgt / (1.0 + r2 / sigma ** 2)
            acc += col * wgt[:, None]
            wsum += wgt
        good = wsum > 0
        new = np.zeros_like(acc)
        new[good] = acc[good] / wsum[good, None]
        est = new
    tex = np.zeros((res, res, 3), np.float32)
    covered = np.zeros((res, res), bool)
    idx = np.argwhere(mask)
    tex[idx[good, 0], idx[good, 1]] = est[good].astype(np.float32)
    covered[idx[good, 0], idx[good, 1]] = True
    return tex, covered


def inpaint(tex: np.ndarray, known: np.ndarray, iters: int = 64) -> np.ndarray:
    """Push-pull style fill: iterative 3x3 mean of known neighbours."""
    from scipy.ndimage import uniform_filter
    t, k = tex.copy(), known.astype(np.float32)
    for _ in range(iters):
        if k.min() > 0:
            break
        s = np.stack([uniform_filter(t[..., c] * k, 3) for c in range(3)], -1)
        cnt = uniform_filter(k, 3)
        grow = (k == 0) & (cnt > 0)
        t[grow] = s[grow] / cnt[grow, None]
        k = np.where(grow, 1.0, k)
    return t
