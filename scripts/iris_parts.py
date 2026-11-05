"""Clean per-component pictures with Iris-3B Turbo img2img, from the component cut out of the reference picture.

A cut-out is the part as it was seen inside the whole object: occluded where other parts cover it, with other
parts showing through it (the filament through the bulb) and torn edges at the mask border. Iris starts from the
cut-out noised to a level of its Turbo schedule (SDEdit) and is prompted for that part alone, complete, on white:
lower start levels keep more of the cut-out (shape, colour), higher ones redraw more (complete the hidden side).

    python scripts/iris_parts.py OUT_DIR "phrase=material=cutout.png[=other part;other part]" ... [--start-sigma 0.833 0.625] [--seed 7]
Writes OUT_DIR/<slug>_s<sigma>.png. Run with the Iris venv under the GPU lock (shared.sh) on the Mac.
"""
import argparse, re, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

IRIS = Path(__file__).resolve().parents[1] / "third_party/iris-3b"
from omegaconf import OmegaConf
from iris3b.config import inference_config
from iris3b.sampling import load_for_inference
from iris3b.models.dit import IrisDiT
from iris3b.registry import TEXT_ENCODERS
from torchvision.utils import save_image

PROMPT = ("Product photo of {part} alone, made of {material}, a single separate component, complete and whole, "
          "nothing inside it and nothing attached to it, centred, isolated on a plain white background, soft studio "
          "light, sharp focus")

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("items", nargs="+")
ap.add_argument("--start-sigma", type=float, nargs="+", default=[0.833, 0.625])
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--neg-scale", type=float, nargs="+", default=[1.5],
                help="guidance away from the negative prompt (the other components): o = pos + w (pos - neg)")
ap.add_argument("--prompt", default=None, help="prompt template instead of PROMPT: {part}, {material}, {desc}")
ap.add_argument("--desc", default="", help="text for {desc} in --prompt")
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
ap.add_argument("--checkpoint", default=str(IRIS / "weights/turbo"))
ap.add_argument("--steps", type=int, default=50, help="full model (no distill.yaml): Euler steps over the whole schedule")
ap.add_argument("--cfg", type=float, nargs="+", default=[3.0], help="full model: classifier-free guidance")
ap.add_argument("--shift", type=float, default=4.0)
a = ap.parse_args()
dev = torch.device(a.device)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

raw, weights = load_for_inference(a.checkpoint)
cfg = inference_config(raw, [])
FULL = not Path(a.checkpoint, "distill.yaml").exists()
SIG = None if FULL else list(OmegaConf.load(Path(a.checkpoint, "distill.yaml")).dmd.sigmas)
with torch.device("meta"):
    model = IrisDiT(cfg.model)
model.load_state_dict(weights, strict=True, assign=True)
model = model.eval().to(device=dev, dtype=torch.bfloat16)
del weights
import iris3b.text  # noqa: F401
te = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=dev)
T = cfg.flow.num_train_timesteps
pred = cfg.flow.prediction


def run(cond, mask, x, sigmas, neg=None, w=0.0, hole=None, x0=None, noise=None):
    """hole (1 = redraw): inpainting — outside it the picture is put back, noised to each step's level."""
    sigma = torch.full((1, 1, 1, 1), sigmas[0], device=dev)
    for i in range(len(sigmas)):
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            o = model(x.to(torch.bfloat16), sigma.flatten() * T, cond, y_mask=mask).x.float()
            if neg is not None and w > 0:
                on = model(x.to(torch.bfloat16), sigma.flatten() * T, neg[0], y_mask=neg[1]).x.float()
                o = o + w * (o - on)
        clean = o if pred == "x" else x - sigma * o
        if i + 1 < len(sigmas):
            implied = (x - (1 - sigma) * clean) / sigma
            sigma = torch.full_like(sigma, sigmas[i + 1])
            x = (1 - sigma) * clean + sigma * implied
            if hole is not None:
                x = hole * x + (1 - hole) * ((1 - sigma) * x0 + sigma * noise)
    if hole is not None:
        clean = hole * clean + (1 - hole) * x0
    return clean.clamp(-1, 1)


def run_full(cond, mask, unc, umask, x, s0, w):
    """Full Iris-3B img2img: shifted flow schedule from s0 down to 0, Euler, real CFG against the negative."""
    t = torch.linspace(1, 0, a.steps + 1)
    sig = (a.shift * t / (1 + (a.shift - 1) * t)).tolist()
    sig = [s0] + [v for v in sig if v < s0]
    n = max(cond.shape[1], unc.shape[1])
    pad = lambda e, m_: (torch.cat([e, e.new_zeros(1, n - e.shape[1], *e.shape[2:])], 1),
                         torch.cat([m_, m_.new_zeros(1, n - m_.shape[1])], 1))
    c2, cm = pad(cond, mask); u2, um = pad(unc, umask)
    yy, ym = torch.cat([u2, c2]), torch.cat([um, cm])
    for i in range(len(sig) - 1):
        sg = torch.full((2,), sig[i] * T, device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            o = model(torch.cat([x, x]).to(torch.bfloat16), sg, yy, y_mask=ym).x.float()
        ou, oc = o.chunk(2)
        v = ou + w * (oc - ou)
        if pred == "x":
            v = (x - v) / max(sig[i], 1e-4)
        x = x + (sig[i + 1] - sig[i]) * v
    return x.clamp(-1, 1)


t0 = time.time()
with torch.no_grad():
    for it in a.items:
        phrase, material, src, *negs = it.split("=")
        neg = None
        if negs and negs[0].strip():
            # the other components, named: the part must not contain them (the filament without the bulb or base)
            en = te.encode([negs[0].replace(";", ", ")])
            ce, me = en.embeddings.to(device=dev, dtype=torch.bfloat16), en.mask.to(device=dev)
            neg = (ce, me)
        img = np.asarray(Image.open(src).convert("RGB").resize((1024, 1024), Image.LANCZOS), np.float32) / 127.5 - 1
        x0 = torch.from_numpy(img).permute(2, 0, 1)[None].to(dev)
        hp = Path(src[:-4] + "_hole.png")
        hole = None
        if hp.exists():          # recomposition: only the hidden part of the part is drawn
            hm = np.asarray(Image.open(hp).convert("L").resize((1024, 1024)), np.float32) / 255
            # only the empty (white) part of the extent is redrawn: the seen part is kept to its edge
            from scipy import ndimage as ndi_
            empty = ndi_.binary_erosion(img.min(-1) > 0.92, iterations=3)
            hm = ndi_.gaussian_filter((hm > 0.5) & empty, 2).astype(np.float32)
            if hm.mean() > 0.01:
                hole = torch.from_numpy(hm)[None, None].to(dev)
        e = te.encode([(a.prompt or PROMPT).format(part=phrase.strip(), material=material.strip(), desc=a.desc)])
        cond, m = e.embeddings.to(device=dev, dtype=torch.bfloat16), e.mask.to(device=dev)
        slug = re.sub(r"[^a-z0-9]+", "_", re.sub(r"^the\s+", "", phrase.strip().lower())).strip("_")
        if FULL:
            nl = te.null(negs[0].replace(";", ", ") if negs and negs[0].strip() else "")
            unc, um = nl.embeddings.to(device=dev, dtype=torch.bfloat16), nl.mask.to(device=dev)
            for s in a.start_sigma:
                g = torch.Generator(device=dev).manual_seed(a.seed)
                noise = torch.randn(1, model.cfg.in_channels, 1024, 1024, generator=g, device=dev)
                x = (1 - s) * x0 + s * noise
                for w in a.cfg:
                    res = run_full(cond, m, unc, um, x.clone(), s, w)
                    p = out / f"{slug}_full_s{s:g}_cfg{w:g}.png"
                    save_image(res.float(), p, normalize=True, value_range=(-1, 1))
                    print(p, f"{time.time() - t0:.0f}s", flush=True)
            continue
        for s in a.start_sigma:
            k = min(range(len(SIG)), key=lambda i: abs(SIG[i] - s))
            g = torch.Generator(device=dev).manual_seed(a.seed)
            noise = torch.randn(1, model.cfg.in_channels, 1024, 1024, generator=g, device=dev)
            x = (1 - SIG[k]) * x0 + SIG[k] * noise
            if hole is not None and s > 0.9:
                k = 0                                  # the strongest level redraws the hole from pure noise
                x = noise.clone()
            for w in (a.neg_scale if neg is not None else [0.0]):
                res = run(cond, m, x.clone(), SIG[k:], neg, w, hole, x0, noise)
                p = out / (f"{slug}_s{SIG[k]:g}" + (f"_n{w:g}" if neg is not None else "") + ".png")
                save_image(res.float(), p, normalize=True, value_range=(-1, 1))
                print(p, f"{time.time() - t0:.0f}s", flush=True)
