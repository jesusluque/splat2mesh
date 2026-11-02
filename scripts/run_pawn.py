"""Path (iii): splat renders (RGB-D) → TSDF → 20K mesh → UV bake → USD, evaluated vs the original pawn."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from splat2mesh.cli import run
from splat2mesh.views import load_athenea
from splat2mesh.evaluate import usd_to_trimesh, compare, load_any

faces = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
out = Path(f"out/pawn_tsdf_{faces // 1000}k")
views = load_athenea("data/pawn")
sm = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
out = Path(str(out) + f"_s{sm:g}")
rep = run(views, out, faces=faces, tex_res=2048, voxel=None, smooth=sm)
ref = usd_to_trimesh("data/pawn/mesh/mesh.usda", "/World/Pawn")
tau = 0.005 * float(np.linalg.norm(ref.bounds[1] - ref.bounds[0]))
rep["geometry_vs_original"] = compare(ref, load_any(str(out / "mesh.ply")), tau, views=views)
(out / "report.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
