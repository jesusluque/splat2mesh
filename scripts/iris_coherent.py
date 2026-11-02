"""Coherent Iris-3B Turbo variations of ONE object (e.g. the same object under different lighting).

Two knobs keep the object fixed while the prompt changes:
  --start-sigma S   img2img (SDEdit): start from the base image noised to S (one of the Turbo schedule
                    levels 0.938 / 0.833 / 0.625) with the SAME noise, and run only the remaining steps.
                    Lower S = more of the base kept.
  --mix A           text-embedding blend: cond = (1-A)·base + A·variant (A=1: the variant prompt as is).

    python scripts/iris_coherent.py BASE_PROMPT @variants.txt OUTDIR [--start-sigma 0.833] [--mix 1.0] [--seed 7]
Writes OUTDIR/base.png, OUTDIR/NN_sS_aA.png and OUTDIR/coherence.json (edge-SSIM vs base, mean |Δ| luminance).
Run under the GPU lock on the Mac.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

IRIS = Path(__file__).resolve().parents[1] / "third_party/iris-3b"
from omegaconf import OmegaConf
from iris3b.config import inference_config
from iris3b.sampling import load_for_inference
from iris3b.models.dit import IrisDiT
from iris3b.registry import TEXT_ENCODERS
from torchvision.utils import save_image

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("variants"); ap.add_argument("out")
ap.add_argument("--start-sigma", type=float, nargs="+", default=[0.833])
ap.add_argument("--mix", type=float, nargs="+", default=[1.0])
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--size", default="1024x1024")
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
ap.add_argument("--checkpoint", default=str(IRIS / "weights/turbo"))
a = ap.parse_args()
W, H = map(int, a.size.lower().split("x"))
dev = torch.device(a.device)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
variants = [l.strip() for l in Path(a.variants[1:]).read_text().splitlines() if l.strip()] if a.variants.startswith("@") else [a.variants]

raw, weights = load_for_inference(a.checkpoint)
cfg = inference_config(raw, [])
SIG = list(OmegaConf.load(Path(a.checkpoint, "distill.yaml")).dmd.sigmas)
with torch.device("meta"):
    model = IrisDiT(cfg.model)
model.load_state_dict(weights, strict=True, assign=True)
model = model.eval().to(device=dev, dtype=torch.bfloat16)
del weights
import iris3b.text  # noqa: F401
te = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=dev)
T = cfg.flow.num_train_timesteps
pred = cfg.flow.prediction


def encode(p):
    e = te.encode([p])
    return e.embeddings.to(device=dev, dtype=torch.bfloat16), e.mask.to(device=dev)


def run(cond, mask, x, sigmas):
    sigma = torch.full((1, 1, 1, 1), sigmas[0], device=dev)
    for i in range(len(sigmas)):
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            o = model(x.to(torch.bfloat16), sigma.flatten() * T, cond, y_mask=mask).x.float()
        clean = o if pred == "x" else x - sigma * o
        if i + 1 < len(sigmas):
            implied = (x - (1 - sigma) * clean) / sigma
            sigma = torch.full_like(sigma, sigmas[i + 1])
            x = (1 - sigma) * clean + sigma * implied
    return clean.clamp(-1, 1)


def pad_to(e, m, n):
    if e.shape[1] >= n:
        return e, m
    pe = torch.zeros(e.shape[0], n - e.shape[1], *e.shape[2:], device=e.device, dtype=e.dtype)
    pm = torch.zeros(m.shape[0], n - m.shape[1], device=m.device, dtype=m.dtype)
    return torch.cat([e, pe], 1), torch.cat([m, pm], 1)


g = torch.Generator(device=dev).manual_seed(a.seed)
noise = torch.randn(1, model.cfg.in_channels, H, W, generator=g, device=dev)
t0 = time.time()
with torch.no_grad():
    cb, mb = encode(a.base)
    base = run(cb, mb, noise.clone(), SIG)
    save_image(base.float(), out / "base.png", normalize=True, value_range=(-1, 1))
    results = []
    for vi, v in enumerate(variants):
        cv, mv = encode(v)
        n = max(cb.shape[1], cv.shape[1])
        cb2, mb2 = pad_to(cb, mb, n); cv2, mv2 = pad_to(cv, mv, n)
        for s in a.start_sigma:
            k = min(range(len(SIG)), key=lambda i: abs(SIG[i] - s))
            for al in a.mix:
                cond = (1 - al) * cb2 + al * cv2
                mask = torch.maximum(mb2, mv2)
                x = (1 - SIG[k]) * base + SIG[k] * noise if k > 0 else noise.clone()
                img = run(cond, mask, x, SIG[k:])
                name = f"{vi:02d}_s{SIG[k]:g}_a{al:g}.png"
                save_image(img.float(), out / name, normalize=True, value_range=(-1, 1))
                results.append({"variant": v[-90:], "start_sigma": SIG[k], "mix": al, "file": name})
                print(name, flush=True)


# coherence: structure (edge SSIM) vs base and how much the light changed
def lum(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("L").resize((256, 256)), np.float32) / 255


def edges(x):
    from scipy import ndimage as ndi
    return np.hypot(ndi.sobel(x, 0), ndi.sobel(x, 1))


def ssim(x, y):
    from scipy.ndimage import uniform_filter as U
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx, my = U(x, 7), U(y, 7)
    sx, sy, sxy = U(x * x, 7) - mx ** 2, U(y * y, 7) - my ** 2, U(x * y, 7) - mx * my
    return float((((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2))).mean())


lb = lum(out / "base.png")
eb = edges(lb)
for r in results:
    l = lum(out / r["file"])
    r["edge_ssim"] = ssim(eb / (eb.max() + 1e-9), edges(l) / (edges(l).max() + 1e-9))
    r["light_change"] = float(np.abs(l - lb).mean())
(out / "coherence.json").write_text(json.dumps(results, indent=1))
for r in results:
    print(f"s={r['start_sigma']:<6g} a={r['mix']:<4g} edgeSSIM {r['edge_ssim']:.3f}  Δlum {r['light_change']:.3f}  {r['variant'][-45:]}")
print(f"{time.time() - t0:.0f}s")
