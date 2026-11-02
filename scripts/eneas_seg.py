"""Text → mask with ENEAS on a single image. usage: eneas_seg.py IMAGE OUTDIR "phrase" ["phrase" ...]"""
import sys, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image
from eneas.segmentation import UniqueInstanceSegmenter
img, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
frames = out / "frames"; frames.mkdir(exist_ok=True)
Image.open(img).convert("RGB").save(frames / "frame_0000.jpg", quality=95)
t = time.time()
seg = UniqueInstanceSegmenter()
print("loaded", round(time.time() - t), "s", flush=True)
for ph in sys.argv[3:]:
    t = time.time()
    try:
        r = seg.segment(frames_path=str(frames), text=ph)
        m = list(r.masks.values())[0]
        m = (np.asarray(m) > 127).astype(np.uint8) * 255
        Image.fromarray(m).save(out / (ph.replace(" ", "_") + ".png"))
        print(ph, "->", round(m.mean() / 255, 4), "of image", round(time.time() - t, 1), "s", flush=True)
    except Exception as e:
        print(ph, "ERROR", repr(e)[:300], flush=True)
