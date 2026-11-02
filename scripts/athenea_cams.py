"""Cameras framing any 3DGS .ply for athenea's `render --splats` (OpenCV cameras.json + cameras.txt lines).

    python scripts/athenea_cams.py IN.ply OUTDIR [--up y|z]
Rings: -30 (12), 0 (24), 25 (24), 50 (18), 75 (8) degrees + zenith = 87 views (~6 min at 4 s/view, under shared.sh's 10 min).
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from plyfile import PlyData

ap = argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("out"); ap.add_argument("--up", default="y", choices=["y", "z"])
ap.add_argument("--rotate-x-180", action="store_true", help="cloud is Y-down (COLMAP/Idol); pair with athenea --rotate-x 180")
a = ap.parse_args()
v = PlyData.read(a.ply)["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1)
if a.rotate_x_180:
    xyz = xyz * np.array([1, -1, -1])
op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
p = xyz[op > 0.05]
lo, hi = np.percentile(p, 0.5, 0), np.percentile(p, 99.5, 0)
C = (lo + hi) / 2; r = float(np.linalg.norm(hi - lo) / 2)
W = 1024; FOCAL, APERTURE = 50.0, 24.576
half = math.atan(APERTURE / 2 / FOCAL)
D = r / math.sin(half) * 1.08
UP = np.array([0, 1.0, 0]) if a.up == "y" else np.array([0, 0, 1.0])
E1 = np.array([0, 0, 1.0]) if a.up == "y" else np.array([0, -1.0, 0])   # azimuth 0 direction (front)
E2 = np.cross(UP, E1)

def view(name, el, az):
    e, t = math.radians(el), math.radians(az)
    eye = C + D * (math.cos(e) * (math.cos(t) * E1 + math.sin(t) * E2) + math.sin(e) * UP)
    up = UP if abs(el) < 89 else -E1
    f = C - eye; f /= np.linalg.norm(f); rr = np.cross(f, up); rr /= np.linalg.norm(rr); d = np.cross(f, rr)
    R = np.stack([rr, d, f]); tt = -R @ eye
    return {"name": name, "elevation_deg": el, "azimuth_deg": az, "eye": eye.tolist(), "target": C.tolist(), "up": up.tolist(),
            "R_world_to_cam": R.tolist(), "t_world_to_cam": tt.tolist()}

views = []
for el, n in ((-30, 12), (0, 24), (25, 24), (50, 18), (75, 8)):
    views += [view(f"v{len(views) + k:03d}", el, 360.0 * k / n) for k in range(n)]
views.append(view(f"v{len(views):03d}", 90, 0))
fx = FOCAL / APERTURE * W
out = Path(a.out); (out / "views").mkdir(parents=True, exist_ok=True)
json.dump({"convention": "opencv", "intrinsics": {"width": W, "height": W, "fx": fx, "fy": fx, "cx": W / 2, "cy": W / 2},
           "views": views, "centre": C.tolist(), "radius": r}, open(out / "cameras.json", "w"), indent=1)
with open(out / "cameras.txt", "w") as fo:
    for vv in views:
        fo.write(" ".join([vv["name"]] + [f"{x:.7g}" for x in vv["eye"] + vv["target"] + vv["up"]]) + "\n")
print(len(views), "views; radius", r, "distance", D)
