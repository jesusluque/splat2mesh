"""Direct mode on athenea's pawn splat vs the original mesh (same metrics as run_pawn.py)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from splat2mesh.direct import run_direct
from splat2mesh.evaluate import usd_to_trimesh, compare, load_any
from splat2mesh.views import load_athenea
res = int(sys.argv[1]) if len(sys.argv) > 1 else 320
level = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
out = Path(f"out/pawn_direct_r{res}_l{level:g}")
rep = run_direct("data/pawn/splat/pawn_3dgs.ply", out, res=res, level=level)
ref = usd_to_trimesh("data/pawn/mesh/mesh.usda", "/World/Pawn")
tau = 0.005 * float(np.linalg.norm(ref.bounds[1] - ref.bounds[0]))
rep["geometry_vs_original"] = compare(ref, load_any(str(out / "mesh.ply")), tau, views=load_athenea("data/pawn"))
(out / "report.json").write_text(json.dumps(rep, indent=2))
g = rep["geometry_vs_original"]
print(f"res {res} level {level}: chamfer {g['chamfer_mean']*1e3:.3f} mm  acc {g['accuracy_mean']*1e3:.3f}  F {g['fscore@tau']:.3f}  "
      f"vis {g['completeness_visible']*1e3:.3f}  comps {rep['components']} nm {rep['non_manifold_edges']} open {rep['boundary_edges']}  "
      f"charts {rep['uv_charts']}  field {rep['t_field_s']}s total {rep['t_total_s']}s")
