"""Can Qwen3-VL-4B estimate PBR values directly? For each part (ENEAS mask on the front orbit frame) it gets the
whole view with the part outlined plus a zoomed crop, and must answer numbers. Two prompts are compared:
  A) free numbers, B) numbers with a short physical rubric. Repeated twice with different crops to check stability.

    python scripts/qwen_pbr_probe.py ASSET_DIR [FRAME_INDEX]   (Iris venv, under shared.sh)
Reads ASSET/mat/masks.npz + ASSET/mat/orbit/frames, writes ASSET/mat/qwen_pbr.json.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

asset = Path(sys.argv[1]); fi = int(sys.argv[2]) if len(sys.argv) > 2 else 6
z = np.load(asset / "mat" / "masks.npz")
frame = Image.open(asset / "mat" / "orbit" / "frames" / f"frame_{fi:04d}.jpg").convert("RGB")
MID = "Qwen/Qwen3-VL-4B-Instruct"; dev = "mps" if torch.backends.mps.is_available() else "cuda"
proc = AutoProcessor.from_pretrained(MID)
model = Qwen3VLForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()

PROMPT_A = ("The left image shows an object; the part outlined in red is shown enlarged on the right. "
            "Estimate the physically based rendering (PBR) material of that part. Reply ONLY with JSON: "
            '{"material": "...", "metallic": <0..1>, "roughness": <0..1>, "base_color": "#rrggbb"}')
PROMPT_B = ("The left image shows an object; the part outlined in red is shown enlarged on the right. "
            "Estimate its PBR material for a metallic-roughness shader.\n"
            "metallic: 1 only for bare metal (gold, brass, steel, chrome, aluminium…), 0 for everything else "
            "(paint, lacquer, ceramic, wood, plastic, stone, glass), even if shiny or gold-coloured paint.\n"
            "roughness: 0.0 mirror/chrome, 0.05-0.15 glossy lacquer, glaze or polished metal, 0.2-0.4 satin or "
            "brushed, 0.5-0.7 oiled wood or matte plastic, 0.8-1.0 fabric, unglazed clay, chalky paint.\n"
            "base_color: the albedo without highlights or shadows.\n"
            'Reply ONLY with JSON: {"material": "...", "metallic": <0 or 1>, "roughness": <0..1>, "base_color": "#rrggbb"}')


def views(mask, pad):
    ys, xs = np.nonzero(mask)
    left = frame.copy(); d = ImageDraw.Draw(left)
    edge = mask ^ np.pad(mask, 1)[2:, 1:-1] | mask ^ np.pad(mask, 1)[1:-1, 2:]
    ey, ex = np.nonzero(edge)
    for y, x in zip(ey[::2], ex[::2]):
        d.point((x, y), fill=(255, 0, 0))
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    p = int(pad * max(y1 - y0, x1 - x0)) + 8
    crop = frame.crop((max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, frame.width), min(y1 + p, frame.height)))
    crop.thumbnail((448, 448))
    pair = Image.new("RGB", (896, 448), "white"); l2 = left.copy(); l2.thumbnail((448, 448))
    pair.paste(l2, (0, 0)); pair.paste(crop, (448, 0))
    return pair


def ask(img, prompt):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=80, do_sample=False)
    t = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"raw": t}


res = {}
for part in z.files:
    mask = z[part][fi]
    if mask.sum() < 50:
        continue
    res[part] = {"A": [ask(views(mask, pd), PROMPT_A) for pd in (0.15, 0.6)],
                 "B": [ask(views(mask, pd), PROMPT_B) for pd in (0.15, 0.6)]}
    print(part, json.dumps(res[part]), flush=True)
(asset / "mat" / "qwen_pbr.json").write_text(json.dumps(res, indent=1))
