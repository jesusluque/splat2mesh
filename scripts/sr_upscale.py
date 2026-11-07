"""Super-resolution with a permissively licensed model, loaded through spandrel (MIT). Run with a Python that has
PyTorch (the Iris or Idol environment); splat2mesh.upscale calls it.

    python scripts/sr_upscale.py IN.png OUT.png [--model realesrgan|swinir] [--back] [--tile 384] [--device auto]
--back: bring the result back to the input's size (sharper detail at the same size, for Idol, which works at 1024 px).

Models (weights downloaded once to ~/.cache/splat2mesh/sr):
  realesrgan  Real-ESRGAN x4plus (BSD-3-Clause), fast, robust on generated pictures
  swinir      SwinIR-L x4, real-world GAN (Apache-2.0), slower, finer textures
"""
import argparse
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS = {
    "realesrgan": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "swinir": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
}
CACHE = Path(os.environ.get("SPLAT2MESH_SR_CACHE") or Path.home() / ".cache/splat2mesh/sr")


def weights(name: str) -> Path:
    url = MODELS[name]; f = CACHE / Path(url).name
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, f.with_suffix(".part")); f.with_suffix(".part").replace(f)
    return f


def device(want: str) -> torch.device:
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def upscale(img: Image.Image, model, dev, tile: int = 384, pad: int = 16) -> Image.Image:
    """Tiled inference with overlap (pad on every side), so a large picture fits in memory."""
    x = torch.from_numpy(np.asarray(img.convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None]
    _, _, H, W = x.shape; s = model.scale
    out = torch.zeros(1, 3, H * s, W * s)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ya, xa, yb, xb = max(y0 - pad, 0), max(x0 - pad, 0), min(y1 + pad, H), min(x1 + pad, W)
            t = model(x[:, :, ya:yb, xa:xb].to(dev)).float().cpu()
            out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = t[:, :, (y0 - ya) * s:(y0 - ya + y1 - y0) * s,
                                                         (x0 - xa) * s:(x0 - xa + x1 - x0) * s]
    a = (out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--model", default="realesrgan", choices=list(MODELS))
    ap.add_argument("--back", action="store_true"); ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    from spandrel import ModelLoader
    dev = device(a.device)
    m = ModelLoader().load_from_file(str(weights(a.model))).eval().to(dev)
    img = Image.open(a.inp).convert("RGB")
    t = time.time()
    big = upscale(img, m, dev, tile=a.tile if a.model == "realesrgan" else min(a.tile, 256))
    if a.back:
        big = big.resize(img.size, Image.LANCZOS)
    big.save(a.out)
    print(f"{a.model} x{m.scale} on {dev}: {img.size} -> {big.size} in {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
