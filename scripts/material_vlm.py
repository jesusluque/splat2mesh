"""Label material regions with Qwen3-VL-4B-Instruct (the vision-language model Iris-3B uses as its text encoder,
already on disk). Run with the Iris venv, under the GPU lock.

    python scripts/material_vlm.py regions.json
regions.json (written by splat2mesh.materials.prepare): {"image": "...png", "prompt": "...",
  "regions": [{"id": 0, "overlay": "...png", "colour": [r,g,b], "share": 0.4}, ...]}
Adds to each region: material, metallic (0/1), roughness (0..1), base_colour_hint, raw answer.
"""
import json, re, sys
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MID = "Qwen/Qwen3-VL-4B-Instruct"
dev = "mps" if torch.backends.mps.is_available() else "cuda"
spec_path = Path(sys.argv[1]); spec = json.loads(spec_path.read_text())
proc = AutoProcessor.from_pretrained(MID)
model = Qwen3VLForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()

# closed set of materials with PBR presets: (metallic, roughness)
PRESETS = {
    "A": ("glazed ceramic / porcelain", 0.0, 0.12),
    "B": ("gold, brass or bronze metal", 1.0, 0.25),
    "C": ("silver, steel, chrome or aluminium metal", 1.0, 0.30),
    "D": ("wood", 0.0, 0.65),
    "E": ("glossy lacquer / varnish", 0.0, 0.08),
    "F": ("glossy painted plastic", 0.0, 0.35),
    "G": ("matte paint, fabric or unglazed clay", 0.0, 0.85),
    "H": ("mother-of-pearl / nacre", 0.0, 0.20),
    "I": ("glass", 0.0, 0.03),
    "J": ("rubber or dark matte plastic", 0.0, 0.75),
}
OPTS = "\n".join(f"{k}) {v[0]}" for k, v in PRESETS.items())
Q = ("This is a close-up of part of an object; only the parts in colour matter, grey parts are masked out. "
     "Which material are the coloured parts made of? Choose exactly one letter:\n" + OPTS + "\nAnswer with the letter only.")


def crop_region(overlay_path, image_path):
    import numpy as np
    ov = np.asarray(Image.open(overlay_path).convert("RGB"), np.float32)
    im = np.asarray(Image.open(image_path).convert("RGB"), np.float32)
    on = np.abs(ov - im).sum(-1) < 6                      # pixels left in colour
    on &= im.max(-1) > 8
    ys, xs = np.nonzero(on)
    if len(xs) < 20:
        return Image.open(overlay_path).convert("RGB")
    y0, y1, x0, x1 = np.percentile(ys, 2), np.percentile(ys, 98), np.percentile(xs, 2), np.percentile(xs, 98)
    pad = 0.15 * max(y1 - y0, x1 - x0) + 16
    box = (int(max(x0 - pad, 0)), int(max(y0 - pad, 0)), int(min(x1 + pad, im.shape[1])), int(min(y1 + pad, im.shape[0])))
    return Image.open(overlay_path).convert("RGB").crop(box)


for r in spec["regions"]:
    hint = f'The object was described as: "{spec["prompt"]}". ' if spec.get("prompt") else ""
    pair = crop_region(r["overlay"], spec["image"])
    pair.thumbnail((512, 512))
    pair.save(Path(r["overlay"]).with_name(f"crop_{r['id']}.png"))
    msgs = [{"role": "user", "content": [{"type": "image", "image": pair}, {"type": "text", "text": Q}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False, output_scores=True, return_dict_in_generate=True)
    # probabilities over the option letters at the first generated token
    logits = out.scores[0][0].float()
    ids = {k: proc.tokenizer.convert_tokens_to_ids(k) for k in PRESETS}
    lp = torch.softmax(torch.stack([logits[i] for i in ids.values()]), 0).cpu().numpy()
    probs = dict(zip(ids.keys(), lp.tolist()))
    best = max(probs, key=probs.get)
    name, met, rough = PRESETS[best]
    r.update(letter=best, material=name, metallic=met, roughness=rough, confidence=round(probs[best], 3),
             probs={k: round(v, 3) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3]})
    print(r["id"], r["share"], best, name, f"p={probs[best]:.2f}", r["probs"], flush=True)
spec_path.write_text(json.dumps(spec, indent=1))
