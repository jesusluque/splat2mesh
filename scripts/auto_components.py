"""Per-component reconstruction with no hand-written part list: deepseek-flash looks at the reference picture and
writes its bill of materials (each kind of part, how many, its library material, turned or not), then components.py makes
each one (ENEAS mask → Iris clean picture chosen by deepseek → Idol → placed by its mask) and assembles them.

    python scripts/auto_components.py REF.png WHOLE_ASSET_DIR OUT_DIR
Writes OUT_DIR/parts_list.json (the model's answer) and everything components.py writes.
"""
import base64, io, json, subprocess, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from splat2mesh import deepseek as _deepseek

MAX_PARTS = 8
PROMPT = ("This photo shows one object. Write its bill of materials (BOM) for a 3D artist who models each kind of "
          "separate physical part once: parts that are visibly distinct pieces (different material, or a piece mounted "
          "on another), each large enough to be seen clearly (not tiny details or decals). One line per KIND of part, "
          "biggest first, at most {n} lines; the first line is the main body the rest is mounted on. For each: 'phrase', "
          "a short English noun phrase starting with 'the' that a segmentation model can find in the photo (plural for "
          "a repeated part, e.g. 'the wheels', so that all its visible copies are found); 'count', how many IDENTICAL copies "
          "the whole object has, counting the ones hidden on the far side (a car has 4 identical wheels); parts that "
          "differ in shape (a windscreen and a side window) are not copies: count 1, or separate lines; 'material', chosen ONLY from this "
          "list: {mats}; 'revolution', true if the part is a body of revolution (round all around one axis, like a "
          "wheel, a tyre, a bottle, a cup or a bulb). Answer JSON: {{\"bom\": [{{\"phrase\": ..., \"count\": n, "
          "\"material\": ..., \"revolution\": true|false}}]}}")


def list_parts(ref: Path):
    mats = list(json.loads((ROOT / "splat2mesh/material_library.json").read_text())["materials"])
    b = io.BytesIO(); Image.open(ref).convert("RGB").save(b, "JPEG", quality=90)
    url = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    body = {"model": "", "temperature": 0.1, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": PROMPT.format(n=MAX_PARTS, mats=", ".join(mats))}]}]}
    import urllib.request
    j = _deepseek.ask(body["messages"][0]["content"])
    parts, seen = [], set()
    for p in (j.get("bom") or j.get("parts") or [])[:MAX_PARTS]:
        ph, m = str(p.get("phrase", "")).strip().lower(), str(p.get("material", "")).strip().lower()
        if not ph or ph in seen or m not in mats:
            continue
        seen.add(ph); parts.append((ph if ph.startswith("the ") else "the " + ph, m + ("@rev" if p.get("revolution") is True else "")
                      + (f"#{int(p['count'])}" if str(p.get("count", 1)).isdigit() and int(p["count"]) > 1 else "")))
    return parts


if __name__ == "__main__":
    ref, whole, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    lf = out / "parts_list.json"
    parts = [tuple(p) for p in json.loads(lf.read_text())] if lf.exists() else list_parts(ref)
    lf.write_text(json.dumps(parts, indent=1))
    print("parts:", parts, flush=True)
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/components.py"), str(ref), str(whole), str(out),
                    *[f"{ph}={m}" for ph, m in parts]], check=True)
