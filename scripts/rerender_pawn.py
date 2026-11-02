import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from splat2mesh.views import load_athenea
from splat2mesh.rerender import load_usd_textured, render, psnr
d = sys.argv[1] if len(sys.argv) > 1 else "out/pawn_tsdf_20k"
vs = load_athenea(sys.argv[2] if len(sys.argv) > 2 else "data/pawn", erode_px=0)
V, F, st, tex = load_usd_textured(f"{d}/asset.usda")
ps, tiles = [], []
for i in range(0, len(vs), 5):
    img, hit = render(V, F, st, tex, vs[i]); m = hit & (vs[i].alpha > 0.5); ps.append(psnr(img, vs[i].rgb, m))
    if i in (len(vs) // 4 // 5 * 5, 45, 85): tiles.append(np.concatenate([vs[i].rgb, np.where(hit[..., None], img, vs[i].rgb)], 1))
print('PSNR mean %.2f min %.2f max %.2f' % (np.mean(ps), np.min(ps), np.max(ps)))
Image.fromarray((np.concatenate(tiles, 0) * 255).clip(0, 255).astype(np.uint8)).resize((1024, 1536)).save(f"{d}/compare.png")
