"""Composite an athenea EXR rendered under an HDRI dome onto a flat grey card, using depth as the object mask
(the dome counts as covered in alpha). Edges are softened over one pixel.

    python scripts/exr_over.py IN.exr OUT.png [--grey 0.8]
"""
import sys
import numpy as np
import OpenEXR, Imath
from PIL import Image
from scipy import ndimage as ndi

src, dst = sys.argv[1], sys.argv[2]
grey = float(sys.argv[sys.argv.index("--grey") + 1]) if "--grey" in sys.argv else 0.8
f = OpenEXR.InputFile(src)
dw = f.header()["dataWindow"]; w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
ch = lambda c: np.frombuffer(f.channel(c, FLOAT), np.float32).reshape(h, w)
rgb = np.stack([ch("R"), ch("G"), ch("B")], -1)
z = ch("Z")
obj = np.isfinite(z) & (z > 0)                     # athenea writes depth 0 where a ray reached the dome
a = ndi.gaussian_filter(obj.astype(np.float32), 0.7)[..., None]
srgb = lambda c: np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)
out = srgb(np.clip(rgb, 0, 1)) * a + grey * (1 - a)
Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8)).save(dst)
import os
if os.environ.get("OVER_MASK"):                    # the object mask itself (overlay.py: exact model silhouette)
    Image.fromarray(obj.astype(np.uint8) * 255).save(os.environ["OVER_MASK"])
print(dst, "object px", int(obj.sum()))
