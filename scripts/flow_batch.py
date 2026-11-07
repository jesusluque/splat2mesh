"""Coherent materials flow for a set of direct-mode assets:
   Qwen3-VL describes parts by material (image + Iris prompt) → ENEAS tracks each part on a 16-view orbit (ENEAS host)
   → masks voted per texel → library PBR (roughness/metallic) + measured base colour → USD.

    python scripts/flow_batch.py SRC_DIR OUT_DIR PROMPTS.txt [--only 00,03] [--gs] [--reuse-masks]
--reuse-masks: keep the ENEAS masks already saved for the same phrases; ENEAS runs only for new ones.
--gs: parts segmented on renders of the splat and voted per gaussian (splat2mesh/gs_materials.py); the mesh inherits
them. The splat with per-gaussian materials is written to OUT_DIR/NN/usd/mat/splat_materials.ply.
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
    prev = d / "usd"                               # splat2usd already built an HD mesh here? reuse it
    if not (o / "asset.usda").exists() and (prev / "asset.usda").exists() and (d / "report.json").exists():
        rp = json.loads((d / "report.json").read_text()).get("mesh", {})
        if rp.get("faces", 0) >= 60000:
            import shutil
            o.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(prev, o)
            print(d.name, "reused HD mesh", rp.get("faces"), flush=True)
    if not (o / "asset.usda").exists():
        r = run_direct(d / f"{d.name}.ply", o, y_down=True)
        print(d.name, "direct", r["components"], r["non_manifold_edges"], r["uv_charts"], "folds", r["folds"], flush=True)
        if r["folds"] > 300:                       # crumpled walls: render path (athenea) instead
            import shutil
            tmp = out / d.name / "render_path"
            subprocess.run([sys.executable, str(ROOT / "scripts/splat2usd.py"), str(d / f"{d.name}.ply"), "--y-down",
                            "--no-antialias", "--out", str(tmp)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(o); shutil.copytree(tmp / d.name / "usd", o)
            print(d.name, "→ render path", flush=True)
# 2) descriptions: DeepSeek (vision) first; Qwen3-VL-4B locally when DeepSeek is not reachable
from splat2mesh.describe import describe_ds, part_materials_from_prompt

from splat2mesh.materials import LIB
DESCRIBED_BY = {}
need = [d for d in items if not (out / d.name / "materials.txt").exists()]
for d in list(need):
    try:
        (out / d.name / "materials.txt").write_text(describe_ds(d / "idol_input.png", prompts[int(d.name)]))
        DESCRIBED_BY[d.name] = "deepseek"; need.remove(d)
        print(d.name, "described by DeepSeek", flush=True)
    except Exception as e:
        print(d.name, "DeepSeek description failed, Qwen3-VL instead:", e, flush=True)
if need:
    imgs = [str(d / "idol_input.png") for d in need]
    pf = out / "_prompts.txt"; pf.write_text("\n".join(prompts[int(d.name)] for d in need))
    from splat2mesh.config import gpu_prefix, iris_python
    subprocess.run(gpu_prefix() + [iris_python(),
                    str(ROOT / "scripts/describe_materials.py"), *imgs, "--prompts-file", str(pf)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for d in need:
        (out / d.name / "materials.txt").write_text(Path(d / "idol_input.materials.txt").read_text())
def complete_parts(desc, image, max_new=3):
    """DeepSeek checks the description's coverage: it sees the picture and the part list and adds up to max_new large
    visible parts that are missing (a 4B describer leaves out boots or a cloak), with library material names.
    Unchanged when DeepSeek is not reachable."""
    import re
    from PIL import Image
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from splat2mesh.deepseek import ask as ds, img_url
        have = [t.strip() for c in desc.split(";") if ":" in c for t in c.split(":", 1)[1].split(",")]
        q = ("This is a picture of an object and the list of its parts used to assign materials: " + "; ".join(have) +
             f". Name up to {max_new} LARGE visible parts that are missing from the list (each a short noun phrase "
             "starting with 'the' that a segmentation model can find; not a part already listed under another name, "
             "not tiny details). For each give its material, using only these names: " + ", ".join(LIB) +
             ". If nothing large is missing, return an empty list. Answer JSON: {\"missing\": [{\"part\": ..., "
             "\"material\": ...}]}")
        ans = ds([{"type": "image_url", "image_url": {"url": img_url(Image.open(image))}}, {"type": "text", "text": q}])
    except Exception as e:
        print("coverage check skipped:", e, flush=True)
        return desc
    low = {h.lower() for h in have}
    for m in ans.get("missing", [])[:max_new]:
        part, mat = str(m.get("part", "")).strip().lower(), str(m.get("material", "")).strip().lower()
        if not part or mat not in LIB or part in low:
            continue
        part = part if part.startswith("the ") else "the " + part
        print(f"coverage check adds: {mat}: {part}", flush=True)
        desc = desc + f"; {mat}: {part}"; low.add(part)
    return desc


# 2b) the generation prompt is the better authority on WHICH metal/wood/glass: swap the VLM's generic family
#     member for the specific one the prompt names ("gold" → "brass" when the prompt says brass)
from splat2mesh.materials import LIB, lookup
for d in items:
    f = out / d.name / "materials.txt"
    pr = prompts[int(d.name)].lower(); desc = f.read_text()
    import re
    named = {k for k, v in LIB.items() if any(re.search(r"\b" + re.escape(w) + r"\b", pr) for w in v["keywords"])}
    new = []
    for part in desc.split(";"):
        if ":" not in part:
            continue
        mat, things = part.split(":", 1); m = mat.strip()
        cls = m if m in LIB else lookup(m)
        if cls and cls not in named:
            fam = [k for k in named if LIB[k]["class"] == LIB[cls]["class"]]
            if len(fam) == 1:
                m = fam[0]
            elif not fam:            # family absent from the prompt: take the prompt material the description misses
                used = {lookup(x.split(":")[0]) or x.split(":")[0].strip() for x in desc.split(";") if ":" in x}
                missing = [k for k in named if k not in used and LIB[k]["class"] not in {LIB[u]["class"] for u in used if u in LIB}]
                if len(missing) == 1:
                    m = missing[0]
        new.append(f"{m}: {things.strip()}")
    full = "; ".join(new) if DESCRIBED_BY.get(d.name) == "deepseek" else complete_parts("; ".join(new), d / "idol_input.png")
    f.write_text(part_materials_from_prompt(full, pr))
# 3) ENEAS 3D + PBR per object
summary = {}
for d in items:
    desc = (out / d.name / "materials.txt").read_text().strip()
    t = time.time()
    p = subprocess.run([sys.executable, str(ROOT / "scripts/materials_eneas3d.py"), str(out / d.name / "usd"),
                        str(d / "idol_input.png"), desc] + (["--gs", str(d / f"{d.name}.ply")] if "--gs" in sys.argv else [])
                       + (["--reuse-masks"] if "--reuse-masks" in sys.argv else []),
                       capture_output=True, text=True)
    (out / d.name / "materials.log").write_text(p.stdout + p.stderr)
    from splat2mesh.interior import extract_interior          # filaments, wicks… inside glass parts
    try:
        print(d.name, "interior", extract_interior(out / d.name / "usd", d / f"{d.name}.ply", y_down=True), flush=True)
    except Exception as e:
        print(d.name, "interior skipped:", e, flush=True)
    tail = [l for l in p.stdout.splitlines() if l.startswith("{'seen_texels'")]
    summary[d.name] = {"prompt": prompts[int(d.name)], "description": desc, "result": tail[-1] if tail else p.stderr[-500:],
                       "seconds": round(time.time() - t)}
    print(d.name, summary[d.name]["seconds"], "s |", desc[:120], "|", (tail[-1] if tail else "ERROR " + p.stderr[-300:])[:300], flush=True)
    (out / "flow_summary.json").write_text(json.dumps(summary, indent=1))
