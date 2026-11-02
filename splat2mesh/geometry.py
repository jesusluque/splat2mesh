"""Depth fusion → clean, decimated triangle mesh (all permissive licenses)."""
from __future__ import annotations

import numpy as np
import open3d as o3d
import trimesh

from .views import View


def tsdf_fuse(views: list[View], voxel: float, trunc_mult: float = 4.0,
              depth_max: float = 1e4) -> o3d.geometry.TriangleMesh:
    """TSDF fusion with Open3D's tensor VoxelBlockGrid (the legacy ScalableTSDFVolume
    returns empty meshes on Open3D 0.20 / arm64)."""
    import open3d.core as o3c
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"), attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1), (1)), voxel_size=voxel, block_resolution=16,
        block_count=200000, device=o3c.Device("CPU:0"))
    for v in views:
        dep = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(v.depth)))
        K, E = o3c.Tensor(v.K), o3c.Tensor(np.ascontiguousarray(v.w2c))
        coords = vbg.compute_unique_block_coordinates(dep, K, E, 1.0, depth_max, trunc_mult)
        vbg.integrate(coords, dep, K, E, 1.0, depth_max, trunc_mult)
    return vbg.extract_triangle_mesh().to_legacy()


def clean(mesh: o3d.geometry.TriangleMesh, keep_ratio: float = 0.02) -> trimesh.Trimesh:
    """Drop floaters (components < keep_ratio of the largest), fix non-manifold edges."""
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    small = counts[labels] < keep_ratio * counts.max()
    mesh.remove_triangles_by_mask(small)
    mesh.remove_unreferenced_vertices()
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=True)
    trimesh.repair.fix_normals(tm)
    trimesh.repair.fill_holes(tm)
    return tm


def decimate(tm: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric decimation that keeps the topology: Open3D first (keeps a closed manifold shell
    closed), fast-simplification with low aggressiveness as fallback; the first result whose
    component count and boundary did not get worse wins."""
    if len(tm.faces) <= target_faces:
        return tm
    import fast_simplification
    before = stats(tm)

    def ok(m):
        s = stats(m)
        return (s["components"] <= before["components"] and s["boundary_edges"] <= before["boundary_edges"]
                and s["non_manifold_edges"] <= before["non_manifold_edges"])

    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices), o3d.utility.Vector3iVector(tm.faces))
    q = om.simplify_quadric_decimation(int(target_faces))
    out = trimesh.Trimesh(np.asarray(q.vertices), np.asarray(q.triangles), process=True)
    if not ok(out):
        for agg in (3, 1):
            v, f = fast_simplification.simplify(tm.vertices.astype(np.float32), tm.faces.astype(np.int32),
                                                target_reduction=1.0 - target_faces / len(tm.faces), agg=agg)
            cand = trimesh.Trimesh(v, f, process=True)
            if ok(cand):
                out = cand
                break
    trimesh.repair.fix_normals(out)
    return out


def stats(tm: trimesh.Trimesh) -> dict:
    edges = tm.edges_sorted
    _, cnt = np.unique(edges, axis=0, return_counts=True)
    return {
        "faces": int(len(tm.faces)),
        "vertices": int(len(tm.vertices)),
        "components": int(len(tm.split(only_watertight=False))),
        "non_manifold_edges": int((cnt > 2).sum()),
        "boundary_edges": int((cnt == 1).sum()),
        "watertight": bool(tm.is_watertight),
    }


def visual_hull(views: list[View], bounds_min: np.ndarray, bounds_max: np.ndarray, res: int = 256,
                min_views_frac: float = 1.0) -> trimesh.Trimesh:
    """Silhouette carving from the view masks (v.alpha) → outer-bound mesh.

    Splat depth is unreliable on transparent/refractive materials (it lands behind
    the glass), but silhouettes stay exact; the hull bounds the surface from outside.
    """
    from skimage.measure import marching_cubes
    ext = bounds_max - bounds_min
    step = ext.max() / (res - 1)
    n = np.ceil(ext / step).astype(int) + 1
    g = np.stack(np.meshgrid(*[bounds_min[i] + np.arange(n[i]) * step for i in range(3)], indexing="ij"), -1)
    P = g.reshape(-1, 3)
    inside = np.zeros(len(P), np.int32)
    seen = np.zeros(len(P), np.int32)
    for v in views:
        m = v.alpha > 0.5
        Pc = P @ v.w2c[:3, :3].T + v.w2c[:3, 3]
        uvw = Pc @ v.K.T
        u = uvw[:, 0] / uvw[:, 2]
        y = uvw[:, 1] / uvw[:, 2]
        h, w = m.shape
        ok = (Pc[:, 2] > 0) & (u >= 0) & (u < w) & (y >= 0) & (y < h)
        seen += ok
        hit = np.zeros(len(P), bool)
        hit[ok] = m[y[ok].astype(int), u[ok].astype(int)]
        inside += hit
    occ = (inside >= np.ceil(min_views_frac * seen)) & (seen > 0)
    vol = occ.reshape(n).astype(np.float32)
    vol = np.pad(vol, 1)
    verts, faces, _, _ = marching_cubes(vol, 0.5)
    verts = (verts - 1) * step + bounds_min
    return trimesh.Trimesh(verts, faces[:, ::-1], process=True)


def clamp_depth_to_hull(views: list[View], hull: trimesh.Trimesh, margin: float) -> None:
    """In place: depth = min(depth, hull front depth + margin) inside the mask."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(hull.vertices.astype(np.float32), hull.faces.astype(np.uint32))
    for v in views:
        h, w = v.depth.shape
        u, y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        dirs = np.stack([(u - v.K[0, 2]) / v.K[0, 0], (y - v.K[1, 2]) / v.K[1, 1], np.ones_like(u)], -1)
        c2w = v.c2w
        dw = dirs @ c2w[:3, :3].T
        rays = np.concatenate([np.broadcast_to(c2w[:3, 3], dw.shape), dw], -1).reshape(-1, 6)
        t = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy().reshape(h, w)
        lim = np.where(np.isfinite(t), t + margin, np.inf)          # dirs have z=1 → t is z-depth
        bad = (v.depth > 0) & (v.depth > lim)
        v.depth[bad] = lim[bad]


def cap_holes(tm: trimesh.Trimesh, max_loops: int = 64) -> trimesh.Trimesh:
    """Close every boundary loop with a fan around its centroid (unseen undersides,
    e.g. the base of an object resting on the ground), then re-orient normals."""
    import networkx as nx
    edges = tm.edges_sorted
    uniq, cnt = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[cnt == 1]
    if len(boundary) == 0:
        return tm
    g = nx.Graph()
    g.add_edges_from(map(tuple, boundary))
    V = [tm.vertices]
    F = [tm.faces]
    n = len(tm.vertices)
    for loop in sorted(nx.cycle_basis(g), key=len, reverse=True)[:max_loops]:
        if len(loop) < 3:
            continue
        c = tm.vertices[loop].mean(0)
        V.append(c[None])
        F.append(np.array([[loop[i], loop[(i + 1) % len(loop)], n] for i in range(len(loop))]))
        n += 1
    out = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=True)
    trimesh.repair.fix_normals(out)
    return out


def smooth(tm: trimesh.Trimesh, factor: float) -> trimesh.Trimesh:
    """Taubin λ|μ smoothing (no shrinkage): removes the high-frequency bumps of depth fusion that
    show through glass and fragment the UV charts. factor 0 = off, 1 = 10 iterations, 2 = 20, …"""
    if factor <= 0:
        return tm
    out = tm.copy()
    trimesh.smoothing.filter_taubin(out, lamb=0.5, nu=0.53, iterations=max(1, int(round(10 * factor))))
    return out
