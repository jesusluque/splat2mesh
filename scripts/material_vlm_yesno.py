"""Per-region yes/no material questions to Qwen3-VL-4B (stronger than one multiple-choice question for a 4B VLM).
    python scripts/material_vlm_yesno.py regions.json [more.json ...]   (Iris venv, under shared.sh; model loaded once)
"""
import json, sys
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
MID = "Qwen/Qwen3-VL-4B-Instruct"; dev = "mps" if torch.backends.mps.is_available() else "cuda"
proc = AutoProcessor.from_pretrained(MID)
model = Qwen3VLForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
QS = {  # key: (question about the COLOURED parts, metallic, roughness if yes)
    "gold metal":       ("Are the coloured parts shiny gold, brass or bronze METAL?", 1.0, 0.25),
    "silver metal":     ("Are the coloured parts shiny silver, steel, chrome or aluminium METAL?", 1.0, 0.30),
    "wood":             ("Are the coloured parts WOOD (visible wood grain)?", 0.0, 0.6),
    "lacquer":          ("Are the coloured parts glossy LACQUER or varnish?", 0.0, 0.08),
    "glazed ceramic":   ("Are the coloured parts glazed CERAMIC or porcelain?", 0.0, 0.12),
    "painted plastic":  ("Are the coloured parts painted PLASTIC?", 0.0, 0.35),
    "matte":            ("Are the coloured parts MATTE (no shine at all, like fabric, rubber or unglazed clay)?", 0.0, 0.85),
    "glass":            ("Are the coloured parts transparent GLASS?", 0.0, 0.03),
    "mother-of-pearl":  ("Are the coloured parts iridescent MOTHER-OF-PEARL?", 0.0, 0.2),
}
yes = proc.tokenizer.convert_tokens_to_ids("Yes"); no = proc.tokenizer.convert_tokens_to_ids("No")
for spec_path in map(Path, sys.argv[1:]):
  spec = json.loads(spec_path.read_text())
  print("==", spec_path, flush=True)
  for r in spec["regions"]:
      crop = Path(r["overlay"]).with_name(f"crop_{r['id']}.png")
      img = Image.open(crop if crop.exists() else r["overlay"]).convert("RGB")
      scores = {}
      for k, (q, _, _) in QS.items():
          text = "Grey parts of this picture are masked out; look only at the parts in colour. " + q + " Answer Yes or No."
          msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
          inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(dev)
          with torch.no_grad():
              logits = model(**inp).logits[0, -1].float()
          scores[k] = float(torch.softmax(torch.stack([logits[yes], logits[no]]), 0)[0])
      best = max(scores, key=scores.get)
      r.update(material=best, metallic=QS[best][1], roughness=QS[best][2], confidence=round(scores[best], 3),
               yes_scores={k: round(v, 3) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])})
      print(r["id"], r["share"], best, list(r["yes_scores"].items())[:4], flush=True)
  spec_path.write_text(json.dumps(spec, indent=1))
