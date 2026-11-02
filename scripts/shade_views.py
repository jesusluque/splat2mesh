"""Grey shaded views of mesh files (geometry check). usage: shade_views.py OUT.png mesh1.usda|ply [mesh2 ...]"""
import sys
import numpy as np, trimesh, open3d as o3d
from PIL import Image
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from splat2mesh.evaluate import load_any
tiles = []
for path in sys.argv[2:]:
    m = load_any(path, "/Asset/Geom" if path.endswith(".usda") else None)
    V, F = m.vertices, m.faces
    c = m.bounds.mean(0); r = np.linalg.norm(m.bounds[1] - m.bounds[0])
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(V.astype(np.float32), F.astype(np.uint32))
    row = []
    for az in (0, 90, 180, 270):
        a = np.radians(az); eye = c + r * 1.4 * np.array([np.sin(a), 0.25, np.cos(a)])
        f = c - eye; f /= np.linalg.norm(f); rr = np.cross(f, [0, 1, 0]); rr /= np.linalg.norm(rr); dn = np.cross(f, rr)
        W = 256; u, v = np.meshgrid(np.arange(W) + .5, np.arange(W) + .5)
        d = np.stack([(u - 128) / 400, (v - 128) / 400, np.ones_like(u)], -1) @ np.stack([rr, dn, f])
        res = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eye, d.shape), d], -1).reshape(-1, 6).astype(np.float32)))
        n = res["primitive_normals"].numpy().reshape(W, W, 3); hit = np.isfinite(res["t_hit"].numpy().reshape(W, W))
        n = np.where(((n * d).sum(-1) > 0)[..., None], -n, n)
        img = np.where(hit, np.clip(np.abs(n @ (-f * 0.6 + np.array([0, 0.5, 0]))) * 0.85 + 0.15, 0, 1), 1.0)
        row.append((img * 255).astype(np.uint8))
    tiles.append(np.concatenate(row, 1))
Image.fromarray(np.concatenate(tiles, 0)).save(sys.argv[1])
