"""Cut one object out of a captured 3DGS scene (a 3DGS training output: point_cloud.ply + cameras.json) and put it in
the frame splat2mesh expects from Idol: centred, Y down, unit radius, so the splat-only material flow
(scripts/materials_eneas3d.py --gs ... --splat-only) can run on it.

    python scripts/gs_crop.py point_cloud.ply cameras.json OUT.ply [--radius R] [--ground H] [--top H] [--preview P.png]

- The object is where the training cameras look: the point nearest to all their optical axes (least squares).
- Up is the mean of the cameras' up vectors (OpenCV: -y of each camera).
- Kept: gaussians within --radius of that point across the ground, and between --ground and --top along up (both
  relative to the focus point, in scene units; by default the ground is found as the densest height below it).
- Rotated so up is -Y (Idol's convention); positions, rotations and scales are moved with it. View-dependent colour
  (f_rest) is dropped: higher SH bands would have to be rotated, and the materials replace them anyway.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from splat2mesh.ply import read_vertex, write_vertex

ap = argparse.ArgumentParser()
ap.add_argument("ply"); ap.add_argument("cameras"); ap.add_argument("out")
ap.add_argument("--radius", type=float, default=None, help="scene units; default: from the cameras' distance")
ap.add_argument("--ground", type=float, default=None, help="height of the cut below the focus point (negative)")
ap.add_argument("--top", type=float, default=None)
ap.add_argument("--min-opacity", type=float, default=0.05)
ap.add_argument("--preview", default=None)
a = ap.parse_args()

cams = json.loads(Path(a.cameras).read_text())
C = np.array([c["position"] for c in cams], np.float64)
R = np.array([c["rotation"] for c in cams], np.float64)          # camera to world (columns: x right, y down, z fwd)
D = R[:, :, 2]
# point nearest to every optical axis: sum (I - d d^T) (p - c) = 0
A = np.zeros((3, 3)); b = np.zeros(3)
for c, d in zip(C, D):
    M = np.eye(3) - np.outer(d, d); A += M; b += M @ c
focus = np.linalg.solve(A, b)
up = -R[:, :, 1].mean(0); up /= np.linalg.norm(up)
dist = np.median(np.linalg.norm(C - focus, axis=1))
print(f"focus {focus.round(3)}, up {up.round(3)}, median camera distance {dist:.3f}")

v = read_vertex(a.ply)
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
alpha = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
rel = xyz - focus
h = rel @ up
across = np.linalg.norm(rel - np.outer(h, up), axis=1)
radius = a.radius or 0.35 * dist
if a.ground is None:
    near = (across < radius) & (h < 0) & (alpha > 0.5)
    hist, edges = np.histogram(h[near], bins=200)
    # the ground is the densest layer below the focus; cut a little above it
    i = int(np.argmax(hist)); a.ground = edges[i + 1] + 0.01 * dist
    print(f"ground at {edges[i]:.3f}..{edges[i+1]:.3f}, cut at {a.ground:.3f}")
top = a.top if a.top is not None else radius * 2
keep = (across < radius) & (h > a.ground) & (h < top) & (alpha > a.min_opacity)
print(f"kept {keep.sum()} of {len(keep)} gaussians")
v = v[keep]; rel = rel[keep]

# basis: new -Y = up; new Z = horizontal direction towards the mean camera (front); X = Y × Z
front = (C - focus).mean(0); front -= (front @ up) * up; front /= np.linalg.norm(front)
ey = -up; ez = front; ex = np.cross(ey, ez); ex /= np.linalg.norm(ex)
B = np.stack([ex, ey, ez])                                       # world → new
p = rel @ B.T
centre = (p.max(0) + p.min(0)) / 2; p -= centre
s = 1.0 / np.linalg.norm(p, axis=1).max()
p *= s

def qmul(q1, q2):
    w1, x1, y1, z1 = q1.T; w2, x2, y2, z2 = q2.T
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], 1)

def mat2q(M):
    if np.linalg.det(M) < 0: M = -M
    w = np.sqrt(max(0.0, 1 + M[0, 0] + M[1, 1] + M[2, 2])) / 2
    return np.array([w, (M[2, 1] - M[1, 2]) / (4 * w), (M[0, 2] - M[2, 0]) / (4 * w), (M[1, 0] - M[0, 1]) / (4 * w)])

out = v.copy()
out["x"], out["y"], out["z"] = p[:, 0], p[:, 1], p[:, 2]
q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
q /= np.linalg.norm(q, axis=1, keepdims=True)
qn = qmul(np.broadcast_to(mat2q(B), q.shape), q)
for i in range(4): out[f"rot_{i}"] = qn[:, i]
for i in range(3): out[f"scale_{i}"] = np.asarray(v[f"scale_{i}"]) + np.log(s)
for n in out.dtype.names:
    if n.startswith("f_rest_"): out[n] = 0
for n in ("nx", "ny", "nz"):
    if n in out.dtype.names: out[n] = 0
write_vertex(a.out, out)
json.dump({"focus": focus.tolist(), "up": up.tolist(), "basis": B.tolist(), "centre": centre.tolist(), "scale": s,
           "radius": radius, "ground": a.ground, "top": top, "kept": int(keep.sum())},
          open(Path(a.out).with_suffix(".json"), "w"), indent=1)

if a.preview:                                                    # centres seen from the front, for a quick check
    from PIL import Image
    W = 768; img = np.full((W, W, 3), 128, np.uint8)
    col = np.clip(0.5 + 0.28209479 * np.stack([out["f_dc_0"], out["f_dc_1"], out["f_dc_2"]], 1), 0, 1)
    o = np.argsort(-p[:, 2]) if False else np.argsort(p[:, 2])  # far first (camera at +Z)
    u = ((p[o, 0] + 1) / 2 * (W - 1)).astype(int); w_ = ((p[o, 1] + 1) / 2 * (W - 1)).astype(int)
    img[w_, u] = (col[o] * 255).astype(np.uint8)
    Image.fromarray(img).save(a.preview)
print("wrote", a.out)
