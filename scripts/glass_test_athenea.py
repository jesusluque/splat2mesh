"""Blind glass estimate on athenea's path-traced test (data/glass_test): mesh + 24 views + per-view planes.
Does NOT read ../glass_test_truth. Writes data/glass_test/estimate.json."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, OpenEXR
from PIL import Image
from splat2mesh.views import View
from splat2mesh.evaluate import usd_to_trimesh
from splat2mesh.glass import estimate, load_athenea_planes

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data/glass_test")
meta = json.loads((D / "cameras.json").read_text()); it = meta["intrinsics"]
K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1.0]])
views, lin = [], []
for v in meta["views"]:
    n = v["name"]
    ch = OpenEXR.File(str(D / f"{n}.exr")).channels()
    key = next(k for k in ("RGBA", "RGB") if k in ch)
    rgb = ch[key].pixels.astype(np.float32)[..., :3]
    mask = np.asarray(Image.open(D / f"{n}_mask.png").convert("L")) > 127
    from scipy.ndimage import binary_erosion
    mask = binary_erosion(mask, iterations=2)             # silhouette pixels mix pawn and plane
    E = np.eye(4); E[:3, :3] = v["R_world_to_cam"]; E[:3, 3] = v["t_world_to_cam"]
    views.append(View(np.clip(rgb, 0, 1), mask.astype(np.float32), np.zeros(mask.shape, np.float32), K, E)); lin.append(rgb)
mesh = usd_to_trimesh("out/glass_pawn_20k/asset.usda", "/Asset/Geom"); mesh.merge_vertices()
t0 = time.time()
r = estimate(mesh, views, load_athenea_planes(D / "planes.json"), linear_obs=lin, samples_per_view=3000)
r["seconds"] = round(time.time() - t0, 1)
(D / "estimate.json").write_text(json.dumps(r, indent=2))
print(json.dumps({k: r[k] for k in ("model", "ior", "tint", "loss", "rays_lost", "alternatives", "seconds")}, indent=1))
