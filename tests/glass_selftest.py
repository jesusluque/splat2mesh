"""Self-consistency: render a glass sphere + pawn-like solid in front of the checker with the forward model
at a hidden IOR/σ, then check `estimate` recovers them. (Validates the optimiser, not the physics;
athenea's path-traced renders do that.)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, trimesh
from splat2mesh.glass import Background, GlassTracer, estimate
from splat2mesh.views import View
sys.path.insert(0, str(Path(__file__).parent))
from synthetic import look_at

TRUE_IOR, TRUE_SIG = 1.52, np.array([0.30, 0.08, 0.15])
mesh = trimesh.creation.icosphere(subdivisions=5, radius=0.5)
desc = {"type": "plane", "origin": [-2, -2, -1.5], "u": [4, 0, 0], "v": [0, 4, 0], "texture": "checker_ref.png"}
tmp = Path(__file__).parent / "_assets"; tmp.mkdir(exist_ok=True)
if not (tmp / "checker_ref.png").exists():   # procedural 8x8 coloured checker
    from PIL import Image
    pal = np.array([[87, 145, 135], [242, 225, 155], [240, 170, 120], [225, 95, 85]], np.uint8)
    ij = np.add.outer(np.arange(8), np.arange(8)) % 4
    Image.fromarray(np.kron(pal[ij], np.ones((56, 56, 1), np.uint8))).save(tmp / "checker_ref.png")
bg = Background(desc, tmp)
tr = GlassTracer(mesh, bg)
W = 256; K = np.array([[300, 0, W / 2], [0, 300, W / 2], [0, 0, 1.0]])
views = []
for az in np.radians([-20, -10, 0, 10, 20]):
    eye = np.array([3 * np.sin(az), 0.2, 3 * np.cos(az)])
    w2c = look_at(eye); c2w = np.linalg.inv(w2c)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(W) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3) @ c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True); o = np.broadcast_to(c2w[:3, 3], d.shape).copy()
    col, thr, L, ok = tr.trace(o, d, TRUE_IOR)
    img = (col * thr[:, None] * np.exp(-L[:, None] * TRUE_SIG)).clip(0, 1) ** (1 / 2.2)
    _, _, hit = tr._hit(o, d)
    views.append(View(img.reshape(W, W, 3).astype(np.float32), (hit & ok).reshape(W, W).astype(np.float32),
                      np.zeros((W, W), np.float32), K, w2c))
r = estimate(mesh, views, bg, samples_per_view=3000)
print(json.dumps({k: r[k] for k in ["ior", "sigma_rgb", "loss", "rays_lost"]}, indent=1))
print("IOR error", abs(r["ior"] - TRUE_IOR), "sigma error", np.abs(np.array(r["sigma_rgb"]) - TRUE_SIG).max())
