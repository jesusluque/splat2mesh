"""Apply the prompt-authority rule to existing descriptions and stored texel class names (no ENEAS)."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splat2mesh.materials import LIB, lookup

def refine(desc, pr):
    pr = pr.lower()
    named = {k for k, v in LIB.items() if any(re.search(r"\b" + re.escape(w) + r"\b", pr) for w in v["keywords"])}
    used = {lookup(x.split(":")[0]) or x.split(":")[0].strip() for x in desc.split(";") if ":" in x}
    mapping = {}
    for part in desc.split(";"):
        if ":" not in part:
            continue
        m = part.split(":", 1)[0].strip(); cls = m if m in LIB else lookup(m)
        if not cls or cls in named:
            continue
        fam = [k for k in named if LIB[k]["class"] == LIB[cls]["class"]]
        if len(fam) == 1:
            mapping[m] = fam[0]
        elif not fam:
            missing = [k for k in named if k not in used and LIB[k]["class"] not in {LIB[u]["class"] for u in used if u in LIB}]
            if len(missing) == 1:
                mapping[m] = missing[0]
    return mapping

for setn in ("testset", "testset2", "testset3", "testset4"):
    prompts = Path(f"data/{setn}/prompts.txt").read_text().splitlines()
    for d in sorted(Path(f"out/{setn}_v5").glob("0*")):
        mt = d / "materials.txt"; spec = d / "usd" / "mat" / "regions.json"
        if not mt.exists() or not spec.exists():
            continue
        desc = mt.read_text(); mp = refine(desc, prompts[int(d.name)])
        if not mp:
            continue
        for a, b in mp.items():
            desc = re.sub(r"(^|;\s*)" + re.escape(a) + r"\s*:", lambda m_: m_.group(1) + b + ":", desc)
        mt.write_text(desc)
        sp = json.loads(spec.read_text())
        sp["texel_classes"] = [(mp.get(n.split("|")[0], n.split("|")[0]) + "|" + n.split("|", 1)[1]) if "|" in n else n for n in sp.get("texel_classes", [])]
        spec.write_text(json.dumps(sp, indent=1))
        print(setn, d.name, mp)
