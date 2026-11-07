"""Direct mode: 3DGS .ply → mesh without rendering.

Geometry: the splat's opacity field on a voxel grid,
    O(x) = 1 - Π_i (1 - α_i · G_i(x))      (accumulated as  -Σ log(1 - α_i G_i))
with G_i the anisotropic Gaussian of splat i, then marching cubes at O = level. Only gaussians above an
opacity threshold contribute; kernels are evaluated within 3σ.
Colour: each texel takes the SH-DC colour of the nearby gaussians, weighted by α·G at the texel and by how
close they are to the surface (interior ones barely count), so no views are needed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from .ply import read_vertex
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814


def load_gaussians(path, y_down: bool = False, min_opacity: float = 0.02):
    v = read_vertex(path)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    scale = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float64)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    alpha = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    rgb = np.clip(0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
    keep = alpha > min_opacity
    xyz, scale, q, alpha, rgb = xyz[keep], scale[keep], q[keep], alpha[keep], rgb[keep]
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    if y_down:                                   # rotate 180° about X: (x, y, z) → (x, -y, -z)
        F = np.diag([1.0, -1.0, -1.0])
        xyz = xyz @ F
        R = F @ R
    # inverse covariance: Σ⁻¹ = R diag(1/s²) Rᵀ
    inv = np.einsum("nij,nj,nkj->nik", R, 1 / scale ** 2, R)
    return xyz, scale, R, inv, alpha, rgb


def opacity_field(xyz, scale, inv, alpha, res: int = 320, pad: float = 0.05, max_sigma: float = 3.0):
    lo, hi = np.percentile(xyz, 0.2, 0), np.percentile(xyz, 99.8, 0)
    ext = hi - lo
    lo, hi = lo - pad * ext.max(), hi + pad * ext.max()
    step = (hi - lo).max() / (res - 1)
    n = np.ceil((hi - lo) / step).astype(int) + 1
    acc = np.zeros(int(np.prod(n)), np.float64)          # Σ -log(1 - α G)
    rad = np.ceil(max_sigma * scale.max(1) / step).astype(int)
    rad = np.clip(rad, 1, 12)                            # huge gaussians are capped (floaters/background)
    centre = np.rint((xyz - lo) / step).astype(int)
    for r in np.unique(rad):
        sel = np.nonzero(rad == r)[0]
        o = np.arange(-r, r + 1)
        off = np.stack(np.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
        for chunk in np.array_split(sel, max(1, len(sel) * len(off) // 4_000_000 + 1)):
            vox = centre[chunk, None, :] + off[None]                       # c × k × 3
            p = lo + vox * step
            d = p - xyz[chunk, None, :]
            m = np.einsum("cki,cij,ckj->ck", d, inv[chunk], d)
            g = alpha[chunk, None] * np.exp(-0.5 * m)
            ok = (m < max_sigma ** 2) & np.all((vox >= 0) & (vox < n), -1) & (g > 1e-4)
            flat = np.ravel_multi_index(tuple(vox[ok].T), n)
            np.add.at(acc, flat, -np.log1p(-np.minimum(g[ok], 0.999)))
    occ = 1 - np.exp(-acc.reshape(n))
    return occ, lo, step


def extract(occ, lo, step, level: float = 0.5, close_iters: int = 3, blur: float = 0.8,
            fill: bool = True) -> trimesh.Trimesh:
    """Solid = (occ > level), closed (bridges gaps between flat gaussians), interior cavities filled,
    only the largest connected solid kept; the surface is taken on a lightly blurred solid so it stays
    smooth, then marching cubes. Result: one closed shell."""
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes
    solid = occ > level
    if close_iters:
        solid = ndi.binary_closing(np.pad(solid, close_iters), iterations=close_iters)[
            close_iters:-close_iters, close_iters:-close_iters, close_iters:-close_iters]
    if fill:
        solid = ndi.binary_fill_holes(solid)
    lab, n = ndi.label(solid)
    if n > 1:
        sizes = ndi.sum(solid, lab, range(1, n + 1))
        keep = np.nonzero(sizes >= 0.02 * sizes.max())[0] + 1      # main pieces, not specks
        solid = np.isin(lab, keep)
    field = ndi.gaussian_filter(solid.astype(np.float32), blur)
    if field.max() < 0.55:          # thin shell that could not be filled (open surface splat): no blur
        field = solid.astype(np.float32)
    v, f, _, _ = marching_cubes(np.pad(field, 2), 0.5)
    v = (v - 2) * step + lo
    return trimesh.Trimesh(v, f[:, ::-1], process=True)


def bake_direct(pos, nrm, mask, xyz, inv, alpha, rgb, k: int = 12, chunk: int = 400_000):
    """Texel colour from nearby gaussians: weight = α·G(texel) · exp(-(signed depth below surface)/h).
    Processed in chunks so 4096² atlases stay within a few GB."""
    P_all = pos[mask].astype(np.float64)
    N_all = nrm[mask].astype(np.float64)
    tree = cKDTree(xyz)
    h = None
    cols = np.zeros((len(P_all), 3), np.float32)
    for a in range(0, len(P_all), chunk):
        P, N = P_all[a:a + chunk], N_all[a:a + chunk]
        dist, idx = tree.query(P, k=k, workers=-1)
        if h is None:
            h = np.median(dist[:, 0]) * 2 + 1e-12
        d = xyz[idx] - P[:, None, :]
        m = np.einsum("tki,tkij,tkj->tk", d, inv[idx], d)
        depth_below = -np.einsum("tki,ti->tk", d, N)
        w = alpha[idx] * np.exp(-0.5 * np.minimum(m, 50)) * np.exp(-np.maximum(depth_below, 0) / h)
        w += 1e-12 * np.exp(-dist / h)
        cols[a:a + chunk] = (w[..., None] * rgb[idx]).sum(1) / w.sum(1, keepdims=True)
    res = pos.shape[0]
    tex = np.zeros((res, res, 3), np.float32)
    tex[mask] = cols
    return tex, mask.copy()


def run_direct(ply, out: Path, faces: int = 80000, tex_res: int = 4096, y_down: bool = False,
               res: int = 448, level: float = 0.15, smooth: float = 0.0, blur: float = 1.2) -> dict:
    from PIL import Image
    from . import geometry, texture
    from .usd_export import write_usd
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    xyz, scale, R, inv, alpha, rgb = load_gaussians(ply, y_down)
    occ, lo, step = opacity_field(xyz, scale, inv, alpha, res)
    t_field = time.time() - t0
    raw = extract(occ, lo, step, level, blur=blur)
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(raw.vertices), o3d.utility.Vector3iVector(raw.faces))
    tm = geometry.clean(om)
    tm = geometry.smooth(tm, smooth)
    tm = geometry.decimate(tm, faces)
    tm = geometry.cap_holes(tm)
    tm = geometry.smooth(tm, smooth * 0.3)
    t_geo = time.time() - t0
    vmap, uv_faces, uvs, charts = texture.unwrap(tm, tex_res)
    verts, normals = tm.vertices[vmap], tm.vertex_normals[vmap]
    pos, nrm, mask = texture.rasterize_uv(verts, uv_faces, uvs, normals, tex_res)
    col, covered = bake_direct(pos, nrm, mask, xyz, inv, alpha, rgb)
    col = texture.inpaint(col, covered)
    Image.fromarray((col[::-1].clip(0, 1) * 255).astype(np.uint8)).save(out / "basecolor.png")
    write_usd(out / "asset.usda", verts, uv_faces, uvs, normals, "./basecolor.png")
    trimesh.Trimesh(verts, uv_faces, process=False).export(out / "mesh.ply")
    rep = geometry.stats(trimesh.Trimesh(tm.vertices, tm.faces, process=False))
    rep.update(mode="direct", grid=res, blur=blur, folds=int((tm.face_adjacency_angles > np.radians(120)).sum()), voxel=float(step), level=level, smooth=smooth, gaussians=int(len(xyz)),
               uv_charts=int(charts), atlas_used=float(mask.mean()), texels_covered=1.0,
               t_field_s=round(t_field, 1), t_geometry_s=round(t_geo, 1), t_total_s=round(time.time() - t0, 1))
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    return rep
