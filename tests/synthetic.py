"""End-to-end check on a synthetic object with known geometry and colour.

Renders RGB-D views of a procedurally coloured mesh by ray casting, runs the
pipeline, then measures Chamfer vs ground truth and re-render PSNR on held-out views.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splat2mesh.cli import run  # noqa: E402
from splat2mesh.views import View  # noqa: E402


def gt_mesh():
    m = trimesh.creation.torus(major_radius=0.35, minor_radius=0.12, major_sections=128, minor_sections=64)
    m.apply_transform(trimesh.transformations.rotation_matrix(0.6, [1, 0, 0]))
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.18)
    return trimesh.util.concatenate([m, sphere])


def colour(p):
    """Procedural albedo: stripes + checker so texture errors are visible."""
    s = 0.5 + 0.5 * np.sin(p[:, 0] * 40)
    c = ((np.floor(p[:, 1] * 12) + np.floor(p[:, 2] * 12)) % 2)
    return np.stack([0.2 + 0.7 * s, 0.3 + 0.5 * c, 0.8 - 0.6 * s * c], 1).clip(0, 1)


def look_at(eye, target=np.zeros(3), up=np.array([0, 1, 0.0])):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, [1, 0, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)                    # OpenCV: y down
    R = np.stack([r, d, f])
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ eye
    return w2c


def render(scene, mesh, w2c, K, w, h):
    c2w = np.linalg.inv(w2c)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dirs_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    dirs = dirs_c @ c2w[:3, :3].T
    orig = np.broadcast_to(c2w[:3, 3], dirs.shape)
    rays = o3d.core.Tensor(np.concatenate([orig, dirs], -1).reshape(-1, 6).astype(np.float32))
    hit = scene.cast_rays(rays)
    t = hit["t_hit"].numpy().reshape(h, w)
    ok = np.isfinite(t)
    depth = np.where(ok, t, 0).astype(np.float32)   # dirs have z=1 → t is z-depth
    p = orig + dirs * np.where(ok, t, 0)[..., None]
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[ok] = colour(p[ok])
    return rgb, depth


def fib_sphere(n, radius):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return radius * np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)


def main(out="out/synthetic"):
    gt = gt_mesh()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(gt.vertices.astype(np.float32), gt.faces.astype(np.uint32))
    w = h = 512
    K = np.array([[600, 0, w / 2], [0, 600, h / 2], [0, 0, 1.0]])
    eyes = fib_sphere(80, 1.6)
    views = []
    for e in eyes:
        w2c = look_at(e)
        rgb, d = render(scene, gt, w2c, K, w, h)
        views.append(View(rgb, None, d, K, w2c))
    out = Path(out)
    rep = run(views, out, faces=20000, tex_res=2048, voxel=None)

    # Chamfer (symmetric, mean) vs GT
    rec = trimesh.load(out / "mesh.ply")
    a, _ = trimesh.sample.sample_surface(gt, 100000, seed=0)
    b, _ = trimesh.sample.sample_surface(rec, 100000, seed=0)
    from scipy.spatial import cKDTree
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    rep["chamfer_mean"] = float((d_ab.mean() + d_ba.mean()) / 2)
    rep["fscore_1mm@unit"] = float(2 / (1 / max((d_ab < 0.005).mean(), 1e-9) + 1 / max((d_ba < 0.005).mean(), 1e-9)))

    # Texture accuracy: sample texture at reconstructed surface points vs procedural GT colour
    from PIL import Image
    texim = np.asarray(Image.open(out / "basecolor.png"), np.float32)[::-1] / 255.0
    uvmesh = trimesh.load(out / "mesh.ply", process=False)
    import pxr.Usd as Usd, pxr.UsdGeom as UsdGeom
    stage = Usd.Stage.Open(str(out / "asset.usda"))
    st = np.array(UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/Asset/Geom")).GetPrimvar("st").Get())
    pts, fid = trimesh.sample.sample_surface(uvmesh, 50000, seed=1)
    bc = trimesh.triangles.points_to_barycentric(uvmesh.triangles[fid], pts)
    uv = np.einsum("ij,ijk->ik", bc, st[uvmesh.faces[fid]])
    res = texim.shape[0]
    px = (uv * res).astype(int).clip(0, res - 1)
    pred = texim[px[:, 1], px[:, 0]]
    # GT colour at the closest GT point
    closest, _, _ = trimesh.proximity.closest_point(gt, pts)
    mse = float(((pred - colour(closest)) ** 2).mean())
    rep["texture_psnr_db"] = float(10 * np.log10(1 / max(mse, 1e-12)))
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
