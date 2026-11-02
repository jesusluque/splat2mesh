"""splat2mesh: multi-view RGB-D (rendered from a splat) → UV-textured USD.

    python -m splat2mesh.cli VIEWS_DIR OUT_DIR --faces 20000 --tex 2048
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from . import geometry, texture
from .usd_export import write_usd
from .views import load_views


def run(views, out: Path, faces: int, tex_res: int, voxel: float | None, up_axis: str = "Y",
        hull_clamp: bool = True, cap: bool = True, smooth: float = 0.0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # scene extent from valid depth
    pts = []
    for v in views[:: max(1, len(views) // 8)]:
        ys, xs = np.nonzero(v.depth)
        if len(xs) == 0:
            continue
        sel = np.random.default_rng(0).choice(len(xs), min(2000, len(xs)), replace=False)
        z = v.depth[ys[sel], xs[sel]]
        cam = np.stack([(xs[sel] + 0.5 - v.K[0, 2]) / v.K[0, 0] * z,
                        (ys[sel] + 0.5 - v.K[1, 2]) / v.K[1, 1] * z, z], 1)
        pts.append(cam @ v.c2w[:3, :3].T + v.c2w[:3, 3])
    pts = np.concatenate(pts)
    diag = float(np.linalg.norm(np.percentile(pts, 99, 0) - np.percentile(pts, 1, 0)))
    voxel = voxel or diag / 384
    if hull_clamp and all(v.alpha is not None for v in views):
        lo, hi = pts.min(0), pts.max(0)
        pad = 0.1 * (hi - lo).max()
        hull = geometry.visual_hull(views, lo - pad, hi + pad, res=320)
        hull.export(out / "visual_hull.ply")
        geometry.clamp_depth_to_hull(views, hull, margin=2 * voxel)
    raw = geometry.tsdf_fuse(views, voxel)
    tm = geometry.clean(raw)
    tm = geometry.smooth(tm, smooth)           # before decimation: high-res bumps
    tm = geometry.decimate(tm, faces)
    if cap:
        tm = geometry.cap_holes(tm)
    tm = geometry.smooth(tm, smooth * 0.3)     # light pass after decimation/capping
    t_geo = time.time() - t0

    vmap, uv_faces, uvs, charts = texture.unwrap(tm, tex_res)
    verts = tm.vertices[vmap]
    normals = tm.vertex_normals[vmap]
    pos, nrm, mask = texture.rasterize_uv(verts, uv_faces, uvs, normals, tex_res)
    col, covered = texture.bake(pos, nrm, mask, views, depth_tol=voxel * 3)
    col = texture.inpaint(col, covered)
    # PNG rows go top-down; USD/GL st has v up → flip vertically
    Image.fromarray((col[::-1].clip(0, 1) * 255).astype(np.uint8)).save(out / "basecolor.png")
    write_usd(out / "asset.usda", verts, uv_faces, uvs, normals, "./basecolor.png", up_axis=up_axis)
    trimesh.Trimesh(verts, uv_faces, process=False).export(out / "mesh.ply")

    report = geometry.stats(trimesh.Trimesh(tm.vertices, tm.faces, process=False))
    report.update(smooth=smooth, voxel=voxel, uv_charts=int(charts), texels_covered=float(covered.sum() / max(mask.sum(), 1)),
                  atlas_used=float(mask.mean()), t_geometry_s=round(t_geo, 2),
                  t_total_s=round(time.time() - t0, 2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("views")
    ap.add_argument("out")
    ap.add_argument("--faces", type=int, default=20000)
    ap.add_argument("--tex", type=int, default=2048)
    ap.add_argument("--voxel", type=float, default=None)
    ap.add_argument("--up", default="Y", choices=["Y", "Z"])
    ap.add_argument("--smooth", type=float, default=0.0, help="Taubin smoothing factor X (0 = off, 1 = 10 iterations)")
    a = ap.parse_args()
    print(json.dumps(run(load_views(a.views), Path(a.out), a.faces, a.tex, a.voxel, a.up, smooth=a.smooth), indent=2))


if __name__ == "__main__":
    main()
