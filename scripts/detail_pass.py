"""Detail pass on a picture before Idol: the Iris-3B restorer's detail at the picture's own size, on the whole picture
(--all) and on the small regions that carry a character (faces, eyes).

Idol works on a 1024 px picture, so an upscale alone gives it nothing: the extra pixels are resized away. What helps
is sharper detail inside the same pixels. DeepSeek finds the faces; each one is cropped with a margin, restored at
4x by the Iris-3B restorer (scripts/iris_restore.py), brought back to its size and pasted with a feathered edge.
The rest of the picture is untouched.

    python scripts/detail_pass.py IN.png OUT.png [--all] [--regions "the face"] [--margin 0.35] [--scale 4]
--all: the whole picture through the restorer (no downscale first), brought back to its size; then the regions.
Writes OUT.png (same size as IN) and OUT_regions/ (each crop before, restored at 4x, and after).
The restorer runs in Iris-3B's environment (iris.dir / iris.python in the config), under the GPU lock.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def find_regions(im: Image.Image, what: str) -> list[list[float]]:
    from splat2mesh.deepseek import ask_image
    q = (f"Find {what} of every character in this picture. Answer JSON: {{\"boxes\": [[x0, y0, x1, y1], ...]}} with "
         "coordinates as fractions of the picture's width and height (0 to 1), each box tight around one region. "
         "An empty list if there is none.")
    boxes = ask_image(im, q).get("boxes", [])
    return [[float(v) for v in b] for b in boxes if len(b) == 4 and b[2] > b[0] and b[3] > b[1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--regions", default="the face (forehead to chin, with the eyes)")
    ap.add_argument("--margin", type=float, default=0.35, help="added around each box, as a fraction of its size")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--boxes", default=None, help="JSON list of boxes (fractions) instead of asking DeepSeek")
    ap.add_argument("--all", action="store_true", help="detail on the whole picture first")
    ap.add_argument("--no-regions", action="store_true", help="only the whole picture (with --all)")
    a = ap.parse_args()
    im = Image.open(a.inp).convert("RGB"); W, H = im.size
    rd = Path(a.out).with_name(Path(a.out).stem + "_regions"); rd.mkdir(parents=True, exist_ok=True)
    from splat2mesh.config import gpu_prefix, iris_python
    py = iris_python()
    if a.all:
        # the whole picture, at its own size into the restorer (4x), and back: detail everywhere, same pixels count
        big = rd / f"all_x{a.scale}.png"
        subprocess.run(gpu_prefix() + [py, str(ROOT / "scripts/iris_restore.py"), a.inp, str(big), "--scale",
                                       str(a.scale), "--no-budget"], check=True)
        im = Image.open(big).convert("RGB").resize((W, H), Image.LANCZOS)
        im.save(rd / "all_back.png")
        print("whole picture: detail added", flush=True)
    boxes = [] if a.no_regions else (json.loads(a.boxes) if a.boxes else find_regions(im, a.regions))
    print("regions:", boxes, flush=True)
    out = im.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
        side = max((x1 - x0) * W, (y1 - y0) * H) * (1 + 2 * a.margin)          # square, with the margin
        side = int(min(side, W, H))
        l = int(np.clip(cx - side / 2, 0, W - side)); t = int(np.clip(cy - side / 2, 0, H - side))
        crop = im.crop((l, t, l + side, t + side))
        src, big = rd / f"r{i}_before.png", rd / f"r{i}_x{a.scale}.png"
        crop.save(src)
        subprocess.run(gpu_prefix() + [py, str(ROOT / "scripts/iris_restore.py"), str(src), str(big),
                                       "--scale", str(a.scale)], check=True)
        det = Image.open(big).convert("RGB").resize((side, side), Image.LANCZOS)
        det.save(rd / f"r{i}_after.png")
        # feathered paste: full weight inside, fading over the outer quarter of the margin
        f = max(4, int(side * a.margin / 2))
        mask = Image.new("L", (side, side), 0)
        mask.paste(255, (f, f, side - f, side - f))
        mask = mask.filter(ImageFilter.GaussianBlur(f / 2))
        out.paste(det, (l, t), mask)
        print(f"region {i}: {side}px at ({l},{t}) restored x{a.scale}", flush=True)
    out.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
