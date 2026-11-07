"""Qwen3-VL-4B (Iris-3B's text encoder) writes the material description of an object for ENEAS:
"gold: the bell, the coin; porcelain: the body". It sees the image and, when there is one, the generation prompt,
and must use material names from splat2mesh/material_library.json.

    python scripts/describe_materials.py IMAGE [--prompt "..."] [--out desc.txt]   (Iris venv, under shared.sh)
Several images: pass IMAGE1 IMAGE2 ... with --prompts-file (one prompt per line, same order).
"""
import argparse, json, re
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
LIB = json.loads((ROOT / "splat2mesh/material_library.json").read_text())["materials"]
NAMES = ", ".join(LIB)
MID = "Qwen/Qwen3-VL-4B-Instruct"
MAX_PARTS = 12

ap = argparse.ArgumentParser(); ap.add_argument("images", nargs="+"); ap.add_argument("--prompt", default="")
ap.add_argument("--prompts-file", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
prompts = Path(a.prompts_file).read_text().splitlines() if a.prompts_file else [a.prompt] * len(a.images)
dev = "mps" if torch.backends.mps.is_available() else "cuda"
proc = AutoProcessor.from_pretrained(MID)
model = Qwen3VLForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()

INSTR = ("You prepare inputs for a segmentation model that finds one object part per short phrase. "
         "List the visible PARTS of this object grouped by MATERIAL.\n"
         "Rules:\n- Use only these material names: {names}.\n"
         "- Each part is a short noun phrase a segmenter can find, starting with 'the' (e.g. 'the lid', 'the handle', "
         "'the gold rim'). One part per phrase; do not join two parts with 'and'.\n"
         "- 3 to 12 parts in total. Cover the whole object: first the parts that make up most of its visible surface "
         "(the body, clothing, footwear, hands, face, hair, large panels or boards), then the smaller details; the largest part first "
         "within each material.\n"
         "- Always list visible glass or transparent parts (windows, lenses, dials' crystals, bulbs) as their own part, and "
         "use a transparent material only for parts you can see through (glossy is not transparent).\n"
         "- A painted figurine or model: name the material each painted part represents (steel, wood, leather, "
         "skin, hair / fur, cotton / linen...), not the paint.\n"
         "- Each part once; each material on one line.\n"
         "- Output ONLY lines of the form  material: part, part\n{hint}")

for img_path, pr in zip(a.images, prompts):
    hint = f'The image was generated from this description: "{pr}"' if pr else ""
    msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(img_path).convert("RGB")},
                                         {"type": "text", "text": INSTR.format(names=NAMES, hint=hint)}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=320, do_sample=False, repetition_penalty=1.15)
    text = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    # small VLMs loop ("shield rim, shield edge, shield rim...") and drop the article: one entry per part, "the ..."
    # parts, at most MAX_PARTS, materials merged in first-seen order
    groups, seen = {}, set()
    for l in re.split(r"[\n;]", text):
        m = re.match(r"\s*[-*]?\s*([^:]+):\s*(.+)", l)
        if not m:
            continue
        mat = m.group(1).strip().lower()
        for part in m.group(2).split(","):
            part = part.strip().strip(".").lower()
            if not part:
                continue
            part = part if part.startswith("the ") else "the " + part
            if part in seen or len(seen) >= MAX_PARTS:
                continue
            seen.add(part); groups.setdefault(mat, []).append(part)
    desc = "; ".join(f"{m}: {', '.join(ps)}" for m, ps in groups.items())
    print(f"{img_path}\t{desc}", flush=True)
    if a.out and len(a.images) == 1:
        Path(a.out).write_text(desc)
    else:
        Path(img_path).with_suffix(".materials.txt").write_text(desc)
