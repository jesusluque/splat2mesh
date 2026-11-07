"""Visual QA: Qwen3-VL compares renders of each published model with the picture it was made from.

    python scripts/qa_visual.py render out/web/models out/qa        # 4 path-traced views per USDZ → contact sheets
    python scripts/qa_visual.py judge out/qa            # deepseek-flash (vision API), score 1–5 + defects
    bash ~/luc/athenea-gpu/shared.sh third_party/iris-3b/.venv/bin/python scripts/qa_visual.py judge-qwen out/qa

render: for every NAME.usdz with a NAME_input.jpg next to it, four views (front three-quarter, back three-quarter,
        side, from above) on the athenea path tracer, composed with the input picture into out/qa/NAME.jpg.
judge:  Qwen3-VL-4B (local, the Iris venv) gets the photo and the four views and one yes/no question per defect
        (torn surface, holes, false transparency, lost colour, stains, wavy surface, dull finish); the score is
        P(yes). Results in out/qa/qa.json and out/qa/qa.txt, worst first.
"""
import json, re, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
VIEWS = [("front 3/4", (2.0, 1.0, 3.1)), ("back 3/4", (-2.3, 0.9, -2.8)), ("side", (3.8, 0.4, 0.0)),
         ("top", (0.6, 3.6, 1.3))]          # eye in bounding-sphere radii: the whole object in frame
SIZE, SPP = 384, 32


def render(models: Path, out: Path, pairs=None):
    """pairs: [(name, master asset.usda, input picture)]. The master USD carries the physical materials (glass
    transmission); a web USDZ renders its glass opaque outside the web viewer, which adds it from parts.json."""
    out.mkdir(parents=True, exist_ok=True)
    todo = pairs or [(u.stem, u, models / f"{u.stem}_input.jpg") for u in sorted(models.glob("*.usdz"))]
    for name, usdz, inp in todo:
        usdz, inp = Path(usdz), Path(inp)
        sheet = out / f"{name}.jpg"
        if not inp.exists() or (sheet.exists() and sheet.stat().st_mtime > usdz.stat().st_mtime):
            continue
        tiles = []
        with tempfile.TemporaryDirectory() as t:
            for i, (_, (x, y, z)) in enumerate(VIEWS):
                png = Path(t) / f"v{i}.png"
                subprocess.run(["env", "VIEW_HDRI=brown_photostudio_02", "bash", str(ROOT / "scripts/athenea_view.sh"), str(usdz), str(png), str(x), str(y),
                                str(z), str(SIZE), str(SPP)], capture_output=True)
                tiles.append(Image.open(png).convert("RGB") if png.exists() else Image.new("RGB", (SIZE, SIZE), "black"))
        S = SIZE
        im = Image.new("RGB", (S * 4, S * 2), (255, 255, 255))
        im.paste(Image.open(inp).convert("RGB").resize((2 * S, 2 * S)), (0, 0))
        d = ImageDraw.Draw(im)
        for i, tile in enumerate(tiles):
            im.paste(tile, (2 * S + (i % 2) * S, (i // 2) * S))
            d.text((2 * S + (i % 2) * S + 6, (i // 2) * S + 4), VIEWS[i][0], fill=(0, 0, 0))
        d.text((6, 4), "reference photo", fill=(0, 0, 0))
        im.save(sheet, quality=90)
        print("sheet", sheet, flush=True)


# one yes/no question per defect; the answer is P(yes) from the first generated token, so the scores are
# continuous and thresholds can be calibrated on known-broken models (a 4B VLM asked for a grade says 5 to all)
QUESTIONS = {
    "torn": "Is the surface of the 3D model torn or shredded, with jagged broken triangles, cracks or "
            "fragments, anywhere?",
    "holes": "Does the 3D model have holes or missing pieces of surface?",
    "transparency": "Is some part of the 3D model see-through like glass although that part of the object in the "
                    "photo is solid and opaque?",
    "colour": "Does a part of the 3D model have a clearly different colour from the same part in the photo (for "
              "example a white area that became gold or dark)?",
    "stains": "Does the 3D model show blotches, smears or stains that the object in the photo does not have?",
    "shape": "Is the surface of the 3D model wavy, rippled or lumpy where the object in the photo is smooth?",
    "finish": "Does the 3D model look dull or matte while the object in the photo is shiny and reflective?",
}
INTRO = ("The first image is a photo of an object. The next images are renders of a 3D model generated "
         "automatically from that photo, from several sides; sides not seen in the photo may be invented. ")


ANALYSIS = ("Compare the 3D model with the photo the way a strict quality inspector would. Describe only the real "
            "differences you can point at: damaged or broken surface, missing parts, wrong materials (for example "
            "glass where the photo is metal), colours that changed, marks that are not in the photo, rough surfaces "
            "where the photo is smooth. One short line per defect, 'part: problem (minor/major)', at most 6 lines, "
            "worst first, no repetitions. If the model matches the photo well, answer 'none'.")


def judge_qwen(out: Path):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    mid = "Qwen/Qwen3-VL-4B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    proc = AutoProcessor.from_pretrained(mid)
    model = Qwen3VLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")]
    res_f = out / "qa.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for sheet in sorted(out.glob("*.jpg")):
        name = sheet.stem
        if name in res and res[name].get("mtime") == sheet.stat().st_mtime:
            continue
        im = Image.open(sheet).convert("RGB"); S = im.height // 2
        photo = im.crop((0, 0, 2 * S, 2 * S)).resize((S, S))
        views = [im.crop((2 * S + (i % 2) * S, (i // 2) * S, 2 * S + (i % 2 + 1) * S, (i // 2 + 1) * S)) for i in range(4)]
        r = {"mtime": sheet.stat().st_mtime}
        for k, q in QUESTIONS.items():
            content = [{"type": "image", "image": photo}] + [{"type": "image", "image": v} for v in views]
            content.append({"type": "text", "text": INTRO + q + " Answer yes or no."})
            inp = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                lg = model(**inp).logits[0, -1].float()
            py = torch.logsumexp(lg[yes], 0); pn = torch.logsumexp(lg[no], 0)
            r[k] = round(float(torch.sigmoid(py - pn)), 3)
        # free analysis: the model lists the defects it sees, part by part
        content = [{"type": "image", "image": photo}] + [{"type": "image", "image": v} for v in views]
        content.append({"type": "text", "text": INTRO + ANALYSIS})
        inp = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                       add_generation_prompt=True, return_dict=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=220, do_sample=False, repetition_penalty=1.15)
        txt = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        seen_l, keep = set(), []
        for l in txt.splitlines():                           # drop repeated lines (small VLMs loop)
            k = l.strip().lower()
            if k and k not in seen_l:
                seen_l.add(k); keep.append(l.strip())
        r["analysis"] = "\n".join(keep[:6])
        res[name] = r
        res_f.write_text(json.dumps(res, indent=1))
        print(name, {k: r[k] for k in QUESTIONS}, flush=True)
    rows = []
    for n, r in res.items():
        worst = max(QUESTIONS, key=lambda k: r.get(k, 0))
        rows.append((max(r.get(k, 0) for k in QUESTIONS), n, worst, r))
    rows.sort(reverse=True)
    lines = [f"{m:.2f}\t{n}\tworst={w}\t" + " ".join(f"{k}={r.get(k, 0):.2f}" for k in QUESTIONS)
             + "\n    " + r.get("analysis", "").replace("\n", "\n    ") for m, n, w, r in rows]
    (out / "qa.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


DS_PROMPT = (
    "Left half: a reference photo of an object. Right half: four renders (front 3/4, back 3/4, side, top) of a 3D "
    "model generated automatically from it; sides not seen in the photo may be invented. Act as a strict quality "
    "inspector. List only real visible defects of the 3D model versus the photo (broken/torn surface, holes, missing "
    "parts, wrong material e.g. glass instead of metal or the reverse, changed colours, stains, lumpy surfaces, wrong "
    "finish). Answer JSON: {\"score\": 1-5 (5 ready to publish, 1 broken), \"defects\": [{\"part\": ..., "
    "\"problem\": ..., \"severity\": \"minor\"|\"major\"}]}")


def judge(out: Path):
    """deepseek-flash (vision) inspects each sheet: score 1–5 and the defects, part by part. Calibrated on known
    breakages (torn web car 2, glass watch 1, rippled box 2); DeepSeek settings in splat2mesh.config."""
    import base64, io, urllib.request
    from splat2mesh import deepseek as _deepseek
    res_f = out / "qa.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for sheet in sorted(out.glob("*.jpg")):
        name = sheet.stem
        if name in res and res[name].get("mtime") == sheet.stat().st_mtime and "score" in res[name]:
            continue
        b = io.BytesIO(); Image.open(sheet).convert("RGB").save(b, "JPEG", quality=88)
        url = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
        body = {"model": "", "temperature": 0.1, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                          {"type": "text", "text": DS_PROMPT}]}]}
        try:
            j = _deepseek.ask(body["messages"][0]["content"])
        except Exception as e:
            print(name, "failed", e, flush=True); continue
        res[name] = {"score": j.get("score"), "defects": j.get("defects", []), "mtime": sheet.stat().st_mtime}
        res_f.write_text(json.dumps(res, indent=1))
        print(name, j.get("score"), flush=True)
    rows = sorted(res.items(), key=lambda kv: kv[1].get("score") or 0)
    lines = []
    for n, r in rows:
        lines.append(f"{r.get('score')}\t{n}")
        for d in r.get("defects", []):
            if isinstance(d, dict):
                lines.append(f"    {d.get('severity')}: {d.get('part')} — {d.get('problem')}")
    (out / "qa.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "render":
        render(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "render-assets":            # render-assets OUT name=asset.usda:input.png ...
        render(None, Path(sys.argv[2]), [(a.split("=")[0], *a.split("=", 1)[1].split(":")) for a in sys.argv[3:]])
    elif sys.argv[1] == "judge-qwen":
        judge_qwen(Path(sys.argv[2]))
    else:
        judge(Path(sys.argv[2]))
