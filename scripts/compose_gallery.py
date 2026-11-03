"""Gallery image (8×4) from the athenea stills of the four test sets."""
import numpy as np
from PIL import Image
fs = [f"out/stills/{s}_v5_0{i}.png" for s in ("testset", "testset2", "testset3", "testset4") for i in range(8)]
cell, cols = 270, 8
g = Image.new("RGB", (cols * cell, 4 * cell), (204, 204, 204))
for k, f in enumerate(fs):
    im = Image.open(f).convert("RGB"); a = np.asarray(im).astype(int); bg = tuple(int(v) for v in a[2, 2])
    ys, xs = np.nonzero(np.abs(a - np.array(bg)).sum(-1) > 18)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    h = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.62 + 10)
    c = Image.new("RGB", (2 * h, 2 * h), bg); c.paste(im, (h - cx, h - cy))
    g.paste(c.resize((cell, cell), Image.LANCZOS), ((k % cols) * cell, (k // cols) * cell))
g.save("docs/img/gallery.jpg", quality=88)
for n, f in (("bulb", "out/stills/testset3_v5_00.png"), ("jade", "out/stills/testset3_v5_04.png")):
    Image.open(f).convert("RGB").resize((480, 480), Image.LANCZOS).save(f"docs/img/{n}.jpg", quality=88)
print("gallery ok")
