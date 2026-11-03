"""Coherent detail for an Iris image before Idol: crop → Iris restorer ×4 → ILVR re-invention with Iris Turbo
→ back to 1024, with coherence numbers against the original.

Idol only uses the object's crop (bbox × 1.2) resized to 1024, so that crop is what gets upscaled. Steps:
  1. crop the object (white background) like Idol's preprocess;
  2. Iris-3B restorer (one step per tile, wavelet colour fix): the crop at up to 2048, detail in Iris' own style;
  3. ILVR with Iris-3B Turbo on overlapping 1024 tiles, ONE noise field for the whole image: start from the
     restored image noised to --sigma, and after every step replace the state's low frequencies (below
     1/--lp of the tile) with those of the restored image at the same noise level — structure and colour are
     anchored by construction, only fine detail is invented. The prompt is the object's own, blended (--mix)
     with a material-detail phrase;
  4. low band from the restored image, high band from the ILVR result; area-downscale to 1024;
  5. coherence vs the original crop: PSNR at 256 px, mean ΔE (64×64 pool), luminance SSIM and Sobel-SSIM at
     512, silhouette IoU.

    python scripts/upscale_detail.py IMAGE "object prompt" OUTDIR [--detail "fine surface texture, ..."]
           [--sigma 0.833] [--lp 8] [--mix 0.3] [--seed 7]      (run under shared.sh on the Mac)
Writes OUTDIR/crop.png, restored.png, ilvr_s*_lp*.png (full res), idol_*.png (1024) and coherence.json.
"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IRIS = ROOT / "third_party/iris-3b"
sys.path.insert(0, str(IRIS / "src")); sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("image"); ap.add_argument("prompt"); ap.add_argument("out")
ap.add_argument("--detail", default="fine surface texture, visible material grain, crisp small details, micro scratches")
ap.add_argument("--sigma", type=float, nargs="+", default=[0.833])
ap.add_argument("--lp", type=int, nargs="+", default=[8])
ap.add_argument("--mix", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--margin", type=float, default=1.2)
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
ap.add_argument("--restore-only", default=None, metavar="OUT_PNG",
                help="only the restorer: write the 1024 Idol input to OUT_PNG and stop (the measured best: ILVR "
                     "re-invention adds grain that Idol turns into holes and stains on the invented side)")
a = ap.parse_args()
dev = torch.device(a.device)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def to_t(img):   # PIL → [1,3,H,W] in [0,1]
    return torch.from_numpy(np.asarray(img.convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None]


def to_img(t):
    return Image.fromarray((t.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))


# 1) crop like Idol's synthetic preprocess
src = Image.open(a.image).convert("RGB")
arr = np.asarray(src).astype(np.float32)
edge = np.concatenate([arr[:8].reshape(-1, 3), arr[-8:].reshape(-1, 3), arr[:, :8].reshape(-1, 3), arr[:, -8:].reshape(-1, 3)])
bgc = np.median(edge, 0)                                    # the background colour (white, grey or black)
fg = np.abs(arr - bgc).max(-1) > 30
ys, xs = np.nonzero(fg)
cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * a.margin)
bg = tuple(int(v) for v in bgc)
canvas = Image.new("RGB", (side, side), bg)
canvas.paste(src, (int(side / 2 - cx), int(side / 2 - cy)))
crop = canvas
crop.save(out / "crop.png")
print("crop side", side, "→ Idol would upscale", round(1024 / side, 2), "×", flush=True)

# 2) Iris restorer, bf16 on MPS
t0 = time.time()
from iris3b.downstream.restoration import Restorer, fit_budget, wavelet_color_fix, tiled, gaussian_window, tile_positions
rest = Restorer("speridlabs/iris-3b/upscaler", device=dev)
rest.model.to(torch.bfloat16); rest.embeddings = rest.embeddings.to(torch.bfloat16)
_orig = rest.restore_tile
def _tile_bf16(x):
    t = torch.full((x.shape[0],), rest.time, device=x.device)
    y = rest.embeddings.expand(len(x), *rest.embeddings.shape[1:])
    v = rest.model(x.to(torch.bfloat16), t, y, y_mask=rest.mask.expand(len(x), -1)).x
    return x.float() - rest.sigma * v.float()
rest.restore_tile = _tile_bf16
restored = rest(fit_budget(crop), color_fix=True)
restored.save(out / "restored.png")
print("restored", restored.size, round(time.time() - t0), "s", flush=True)
del rest; gc.collect(); torch.mps.empty_cache() if dev.type == "mps" else None
if a.restore_only:
    restored.resize((1024, 1024), Image.Resampling.BOX).save(a.restore_only)
    print("idol input", a.restore_only, flush=True)
    sys.exit(0)

# 3) ILVR with Iris Turbo
from omegaconf import OmegaConf
from iris3b.config import inference_config
from iris3b.sampling import load_for_inference
from iris3b.models.dit import IrisDiT
from iris3b.registry import TEXT_ENCODERS
ckpt = IRIS / "weights/turbo"
raw, weights = load_for_inference(str(ckpt))
cfg = inference_config(raw, [])
SIG = list(OmegaConf.load(ckpt / "distill.yaml").dmd.sigmas)
with torch.device("meta"):
    model = IrisDiT(cfg.model)
model.load_state_dict(weights, strict=True, assign=True)
model = model.eval().to(device=dev, dtype=torch.bfloat16); del weights
import iris3b.text  # noqa: F401
te = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=dev)
T = cfg.flow.num_train_timesteps; pred = cfg.flow.prediction


def encode(p):
    e = te.encode([p]); return e.embeddings.to(device=dev, dtype=torch.bfloat16), e.mask.to(device=dev)


with torch.no_grad():
    cb, mb = encode(a.prompt); cd, md = encode(a.prompt + ", " + a.detail)
    n = max(cb.shape[1], cd.shape[1])
    pad = lambda e, m: (torch.cat([e, torch.zeros(1, n - e.shape[1], *e.shape[2:], device=dev, dtype=e.dtype)], 1),
                        torch.cat([m, torch.zeros(1, n - m.shape[1], device=dev, dtype=m.dtype)], 1))
    cb, mb = pad(cb, mb); cd, md = pad(cd, md)
    cond = (1 - a.mix) * cb + a.mix * cd; mask = torch.maximum(mb, md)
del te; gc.collect()

base = to_t(restored).to(dev) * 2 - 1                      # [-1,1], full res
H, W = base.shape[-2:]
g = torch.Generator(device=dev).manual_seed(a.seed)
noise = torch.randn(1, 3, H, W, generator=g, device=dev)   # one noise field for every tile


def lowpass(x, k):
    return F.interpolate(F.avg_pool2d(x, k), size=x.shape[-2:], mode="bilinear", align_corners=False)


def ilvr_tile(b, e, k0, lp):
    sig = SIG[k0:]
    s = torch.full((1, 1, 1, 1), sig[0], device=dev)
    x = (1 - s) * b + s * e
    for i in range(len(sig)):
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            o = model(x.to(torch.bfloat16), s.flatten() * T, cond, y_mask=mask).x.float()
        clean = o if pred == "x" else x - s * o
        if i + 1 < len(sig):
            implied = (x - (1 - s) * clean) / s
            s = torch.full_like(s, sig[i + 1])
            x = (1 - s) * clean + s * implied
            ref = (1 - s) * b + s * e                       # the base at the same noise level
            x = x - lowpass(x, lp) + lowpass(ref, lp)        # ILVR: low band anchored to the base
    return clean.clamp(-1, 1)


results = []
win = gaussian_window(1024, dev)
for s_req in a.sigma:
    k0 = min(range(len(SIG)), key=lambda i: abs(SIG[i] - s_req))
    for lp in a.lp:
        t0 = time.time()
        acc = torch.zeros_like(base); wsum = torch.zeros_like(base[:, :1])
        with torch.no_grad():
            for top in tile_positions(H, 1024, 512):
                for left in tile_positions(W, 1024, 512):
                    sl = (..., slice(top, top + 1024), slice(left, left + 1024))
                    acc[sl] += ilvr_tile(base[sl], noise[sl], k0, lp) * win
                    wsum[sl] += win
        y = acc / wsum
        # 4) bands: low from the restored image, high from the re-invention
        y = (y - lowpass(y, lp) + lowpass(base, lp) + 1) / 2
        tag = f"s{SIG[k0]:g}_lp{lp}"
        full = to_img(y); full.save(out / f"ilvr_{tag}.png")
        full.resize((1024, 1024), Image.Resampling.BOX).save(out / f"idol_{tag}.png")
        results.append({"tag": tag, "seconds": round(time.time() - t0)})
        print(tag, round(time.time() - t0), "s", flush=True)
restored.resize((1024, 1024), Image.Resampling.BOX).save(out / "idol_restored.png")
crop.resize((1024, 1024), Image.Resampling.LANCZOS).save(out / "idol_original.png")

# 5) coherence vs the original crop
from splat2mesh.materials import _srgb_to_lab


def arr_(p, s):
    return np.asarray(Image.open(p).convert("RGB").resize((s, s), Image.Resampling.BOX), np.float64) / 255


def ssim(x, y):
    from scipy.ndimage import gaussian_filter as gf
    mx, my = gf(x, 1.5), gf(y, 1.5)
    vx, vy, cxy = gf(x * x, 1.5) - mx * mx, gf(y * y, 1.5) - my * my, gf(x * y, 1.5) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean())


def sobel(x):
    from scipy.ndimage import sobel as sb
    return np.hypot(sb(x, 0), sb(x, 1))


def coherence(p):
    o, u = arr_(out / "idol_original.png", 256), arr_(p, 256)
    psnr = 10 * np.log10(1 / max(((o - u) ** 2).mean(), 1e-10))
    o64, u64 = arr_(out / "idol_original.png", 64), arr_(p, 64)
    de = float(np.linalg.norm(_srgb_to_lab(o64) - _srgb_to_lab(u64), axis=-1).mean())
    o5, u5 = arr_(out / "idol_original.png", 512).mean(-1), arr_(p, 512).mean(-1)
    so, su = sobel(o5), sobel(u5)
    fo = (arr_(out / "idol_original.png", 512).min(-1) < 0.96); fu = (arr_(p, 512).min(-1) < 0.96)
    iou = float((fo & fu).sum() / max((fo | fu).sum(), 1))
    sharp = float(np.abs(sobel(arr_(p, 1024).mean(-1))).mean() / max(np.abs(sobel(arr_(out / "idol_original.png", 1024).mean(-1))).mean(), 1e-9))
    return {"psnr256": round(psnr, 2), "dE64": round(de, 2), "ssim512": round(ssim(o5, u5), 3),
            "sobel_ssim512": round(ssim(so / so.max(), su / max(su.max(), 1e-9)), 3), "iou": round(iou, 4),
            "sharpness_vs_original": round(sharp, 2)}


report = {"crop_side": side, "restored": coherence(out / "idol_restored.png"),
          **{r["tag"]: coherence(out / f"idol_{r['tag']}.png") for r in results}}
(out / "coherence.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
