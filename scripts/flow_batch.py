"""Coherent materials flow for a set of direct-mode assets:
   Qwen3-VL describes parts by material (image + Iris prompt) → ENEAS tracks each part on a 24-view orbit (ixwin WSL)
   → masks voted per texel → library PBR (roughness/metallic) + measured base colour → USD.

    python scripts/flow_batch.py SRC_DIR OUT_DIR PROMPTS.txt [--only 00,03]
SRC_DIR/NN/{NN.ply, idol_input.png} as written by splat2usd; OUT_DIR/NN/usd is (re)built with direct mode.
"""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from splat2mesh.direct import run_direct
src, out = Path(sys.argv[1]), Path(sys.argv[2]); prompts = Path(sys.argv[3]).read_text().splitlines()
only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
items = [d for d in sorted(src.iterdir()) if d.is_dir() and (d / f"{d.name}.ply").exists() and (not only or d.name in only)]
# 1) direct meshes (skip when present)
for d in items:
    o = out / d.name / "usd"
    if not (o / "asset.usda").exists():
        r = run_direct(d / f"{d.name}.ply", o, y_down=True)
        print(d.name, "direct", r["components"], r["non_manifold_edges"], r["uv_charts"], flush=True)
# 2) descriptions with Qwen3-VL (one model load for all)
need = [d for d in items if not (out / d.name / "materials.txt").exists()]
if need:
    imgs = [str(d / "idol_input.png") for d in need]
    pf = out / "_prompts.txt"; pf.write_text("\n".join(prompts[int(d.name)] for d in need))
    subprocess.run(["bash", str(Path.home() / "luc/athenea-gpu/shared.sh"), str(ROOT / "third_party/iris-3b/.venv/bin/python"),
                    str(ROOT / "scripts/describe_materials.py"), *imgs, "--prompts-file", str(pf)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for d in need:
        (out / d.name / "materials.txt").write_text(Path(d / "idol_input.materials.txt").read_text())
# 3) ENEAS 3D + PBR per object
summary = {}
for d in items:
    desc = (out / d.name / "materials.txt").read_text().strip()
    t = time.time()
    p = subprocess.run([sys.executable, str(ROOT / "scripts/materials_eneas3d.py"), str(out / d.name / "usd"),
                        str(d / "idol_input.png"), desc], capture_output=True, text=True)
    tail = [l for l in p.stdout.splitlines() if l.startswith("{'seen_texels'")]
    summary[d.name] = {"prompt": prompts[int(d.name)], "description": desc, "result": tail[-1] if tail else p.stderr[-500:],
                       "seconds": round(time.time() - t)}
    print(d.name, summary[d.name]["seconds"], "s |", desc[:120], "|", (tail[-1] if tail else "ERROR " + p.stderr[-300:])[:300], flush=True)
    (out / "flow_summary.json").write_text(json.dumps(summary, indent=1))
