"""Path (ii): gsplat 2DGS median depth (trained on ixwin) → same fusion/bake/USD, colours from the splat renders."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from splat2mesh.cli import run
from splat2mesh.views import load_athenea
from splat2mesh.evaluate import usd_to_trimesh, compare, load_any

src = Path(sys.argv[1] if len(sys.argv) > 1 else "out/ixwin_pawn_2dgs/views")
hull = "--hull" in sys.argv
out = Path("out/pawn_2dgs_20k" + ("_hull" if hull else ""))
views = load_athenea("data/pawn")
for i, v in enumerate(views):
    d = np.load(src / f"v{i:03d}_depth.npy").astype(np.float32)
    v.depth = np.where(v.alpha > 0.5, d, 0).astype(np.float32)
rep = run(views, out, faces=20000, tex_res=2048, voxel=None, hull_clamp=hull)
ref = usd_to_trimesh("data/pawn/mesh/mesh.usda", "/World/Pawn")
tau = 0.005 * float(np.linalg.norm(ref.bounds[1] - ref.bounds[0]))
rep["geometry_vs_original"] = compare(ref, load_any(str(out / "mesh.ply")), tau, views=views)
(out / "report.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
