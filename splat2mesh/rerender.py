"""Re-render a UV-textured mesh (unlit base colour) at given cameras by ray casting; PSNR vs views."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from PIL import Image
from pxr import Usd, UsdGeom


def load_usd_textured(usd_path: str, prim="/Asset/Geom"):
    stage = Usd.Stage.Open(usd_path)
    m = UsdGeom.Mesh(stage.GetPrimAtPath(prim))
    V = np.asarray(m.GetPointsAttr().Get(), np.float32)
    F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.uint32).reshape(-1, 3)
    st = np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar("st").Get(), np.float32)
    tex = np.asarray(Image.open(usd_path.rsplit("/", 1)[0] + "/basecolor.png").convert("RGB"), np.float32)[::-1] / 255
    return V, F, st, tex


def render(V, F, st, tex, v):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(V, F)
    h, w = v.depth.shape
    u, y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d = np.stack([(u - v.K[0, 2]) / v.K[0, 0], (y - v.K[1, 2]) / v.K[1, 1], np.ones_like(u)], -1) @ v.c2w[:3, :3].T
    rays = np.concatenate([np.broadcast_to(v.c2w[:3, 3], d.shape), d], -1).reshape(-1, 6).astype(np.float32)
    r = scene.cast_rays(o3d.core.Tensor(rays))
    pid = r["primitive_ids"].numpy().reshape(h, w)
    bu = r["primitive_uvs"].numpy().reshape(h, w, 2)
    hit = pid != o3d.t.geometry.RaycastingScene.INVALID_ID
    img = np.zeros((h, w, 3), np.float32)
    f = F[pid[hit]]
    b1, b2 = bu[hit, 0], bu[hit, 1]
    uv = st[f[:, 0]] * (1 - b1 - b2)[:, None] + st[f[:, 1]] * b1[:, None] + st[f[:, 2]] * b2[:, None]
    R = tex.shape[0]
    px = (uv * R - 0.5)
    x0 = np.floor(px).astype(int).clip(0, R - 2); fx = (px[:, 0] - x0[:, 0])[:, None].clip(0, 1); fy = (px[:, 1] - x0[:, 1])[:, None].clip(0, 1)
    c = (tex[x0[:, 1], x0[:, 0]] * (1 - fx) * (1 - fy) + tex[x0[:, 1], x0[:, 0] + 1] * fx * (1 - fy)
         + tex[x0[:, 1] + 1, x0[:, 0]] * (1 - fx) * fy + tex[x0[:, 1] + 1, x0[:, 0] + 1] * fx * fy)
    img[hit] = c
    return img, hit


def psnr(a, b, m):
    mse = ((a[m] - b[m]) ** 2).mean()
    return float(10 * np.log10(1 / max(mse, 1e-12)))


def qa(asset_dir, views, every: int = 5, tiles_at=(0.25, 0.5, 0.97)) -> dict:
    """Re-render the textured USD at every `every`-th view: PSNR vs the splat renders + compare.png."""
    from pathlib import Path
    asset_dir = Path(asset_dir)
    V, F, st, tex = load_usd_textured(str(asset_dir / "asset.usda"))
    idx = list(range(0, len(views), every))
    want = {idx[min(int(t * len(idx)), len(idx) - 1)] for t in tiles_at}
    ps, tiles = [], []
    for i in idx:
        img, hit = render(V, F, st, tex, views[i])
        m = hit & (views[i].alpha > 0.5)
        ps.append(psnr(img, views[i].rgb, m))
        if i in want:
            tiles.append(np.concatenate([views[i].rgb, np.where(hit[..., None], img, views[i].rgb)], 1))
    Image.fromarray((np.concatenate(tiles, 0) * 255).clip(0, 255).astype(np.uint8)).resize((1024, 512 * len(tiles))).save(asset_dir / "compare.png")
    return {"psnr_mean": float(np.mean(ps)), "psnr_min": float(np.min(ps)), "psnr_max": float(np.max(ps)), "qa_views": len(idx)}
