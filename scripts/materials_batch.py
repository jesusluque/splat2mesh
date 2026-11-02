"""Re-run direct mode on Idol splats and infer PBR materials for each, then (optionally) the web gallery.
    python scripts/materials_batch.py SRC_DIR OUT_DIR PROMPTS.txt
SRC_DIR/NN/NN.ply + idol_input.png (splat2usd output); prompt NN = line NN of PROMPTS.txt.
"""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from splat2mesh.direct import run_direct
from splat2mesh.materials import prepare, apply
src, out = Path(sys.argv[1]), Path(sys.argv[2]); prompts = Path(sys.argv[3]).read_text().splitlines()
jobs = []
for d in sorted(p for p in src.iterdir() if p.is_dir()):
    ply = d / f"{d.name}.ply"
    if not ply.exists():
        continue
    o = out / d.name / "usd"
    t = time.time()
    rep = run_direct(ply, o, y_down=True)
    pr = prompts[int(d.name)] if d.name.isdigit() and int(d.name) < len(prompts) else ""
    prepare(o, d / "idol_input.png", o / "mat" / "regions.json", prompt=pr)
    jobs.append((d.name, o, pr))
    print(d.name, f"direct {time.time() - t:.0f}s comps {rep['components']} nm {rep['non_manifold_edges']} charts {rep['uv_charts']}", flush=True)
vlm = ["bash", str(Path.home() / "luc/athenea-gpu/shared.sh"), str(ROOT / "third_party/iris-3b/.venv/bin/python"),
       str(ROOT / "scripts/material_vlm_yesno.py")] + [str(o / "mat" / "regions.json") for _, o, _ in jobs]
subprocess.run(vlm, check=True)
summary = {}
for name, o, pr in jobs:
    r = apply(o / "mat" / "regions.json")
    summary[name] = {"prompt": pr, **r}
    print(name, [(m, met, ro, sh) for m, met, ro, sh in r["regions"]], flush=True)
(out / "materials_summary.json").write_text(json.dumps(summary, indent=1))
