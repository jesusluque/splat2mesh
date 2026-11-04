"""Sheet of one component: its cut-out and the Iris candidates, the chosen one framed in green, with scores."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
out, slug = Path(sys.argv[1]), sys.argv[2]
S = 320
pick = json.loads((out / "pick_log.json").read_text()) if (out / "pick_log.json").exists() else {}
fs = [out / "parts" / f"{slug}_cut.png"] + [out / "iris" / f"{slug}_s{s}.png" for s in ("0.938", "0.833", "0.625")]
labs = ["recorte", "0.938", "0.833", "0.625"]
chosen = None
log = Path(sys.argv[3]) if len(sys.argv) > 3 else out.parent.parent / "components_car.log"
for l in log.read_text().splitlines():
    if l.startswith("the ") and "'chosen'" in l and f"/{slug}_s" in l:
        chosen = l.split("'chosen': ")[1].split(",")[0].strip("'")
im = Image.new("RGB", (S * 4, S + 24), "white"); d = ImageDraw.Draw(im)
for i, (f, l) in enumerate(zip(fs, labs)):
    if f.exists():
        im.paste(Image.open(f).convert("RGB").resize((S, S)), (i * S, 24))
    if chosen and str(f).endswith(Path(chosen).name):
        d.rectangle([i * S + 2, 26, i * S + S - 3, S + 21], outline=(0, 170, 0), width=6); l += "  ELEGIDA"
    d.text((i * S + 6, 6), l, fill="black")
p = out / "sheets" / f"{slug}.jpg"; im.save(p, quality=88); print(p)
