"""Generic: rendered views folder (render_views.py output) → USD. Optional GT mesh for metrics."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splat2mesh.cli import run
from splat2mesh.views import load_athenea
views_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
faces = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
sm = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
rep = run(load_athenea(views_dir), out, faces=faces, tex_res=2048, voxel=None, smooth=sm)
print(json.dumps(rep, indent=2))
