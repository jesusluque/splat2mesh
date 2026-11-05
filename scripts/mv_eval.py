"""Distance of a reconstructed model to a ground-truth model, after aligning them (rotation, scale, position).

Both are normalised (bounding-box centre, diagonal 1); the rotation is searched over the 24 axis-aligned frames x
yaw steps around each up axis, each refined by ICP with scaling; the best by symmetric Chamfer is kept.
    python scripts/mv_eval.py GT.usda name=MODEL.usda [name=MODEL.usda ...] [--json OUT.json]
Prints, per model: Chamfer mean and p95 (both directions, share of the GT diagonal), and the share of GT surface
farther than 2 % of the diagonal from the model (missing or wrong parts).
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay import load

N = 20000


def sample(V, F, n=N, seed=0):
    import trimesh
    return np.asarray(trimesh.sample.sample_surface(trimesh.Trimesh(V, F, process=False), n, seed=seed)[0])


def norm(P):
    lo, hi = P.min(0), P.max(0)
    return (P - (lo + hi) / 2) / np.linalg.norm(hi - lo)


def frames():
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (j, sg) in enumerate(zip(perm, s)):
                R[i, j] = sg
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def chamfer(A, B):
    from scipy.spatial import cKDTree
    da, _ = cKDTree(B).query(A); db, _ = cKDTree(A).query(B)
    return da, db


def align(P, G):
    import open3d as o3d
    pg = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(G))
    best = None
    for R0 in frames():
        for yaw in np.radians(np.arange(0, 360, 30)):
            c, s = np.cos(yaw), np.sin(yaw)
            Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            Q = P @ (Ry @ R0).T
            pq = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q[::8]))
            r = o3d.pipelines.registration.registration_icp(
                pq, pg, 0.05, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            Qa = Q @ r.transformation[:3, :3].T + r.transformation[:3, 3]
            da, db = chamfer(Qa[::4], G[::4])
            cm = da.mean() + db.mean()
            if best is None or cm < best[0]:
                best = (cm, Qa)
    return best[1]


def main():
    args = sys.argv[1:]; js = None
    if "--json" in args:
        i = args.index("--json"); js = args[i + 1]; args = args[:i] + args[i + 2:]
    Vg, Fg, _, _ = load(args[0]); G = norm(sample(Vg, Fg))
    res = {}
    for it in args[1:]:
        name, path = it.split("=", 1)
        V, F, _, _ = load(path); P = norm(sample(V, F))
        Q = align(P, G)
        da, db = chamfer(Q, G)          # da: model → GT (extra stuff), db: GT → model (missing stuff)
        res[name] = {"chamfer_mean": float((da.mean() + db.mean()) / 2), "p95": float(max(np.percentile(da, 95), np.percentile(db, 95))),
                     "gt_missing_2pct": float((db > 0.02).mean()), "model_extra_2pct": float((da > 0.02).mean())}
        print(name, json.dumps({k: round(v, 4) for k, v in res[name].items()}), flush=True)
    if js:
        Path(js).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
