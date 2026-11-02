"""Iris-3B Turbo (speridlabs, Apache-2.0) locally, sized for a 24 GB Apple Silicon Mac:
DiT weights in bf16 instead of upstream's fp32 (6 GB instead of 12 GB) + Qwen3-VL-4B text encoder (8 GB).

    python scripts/iris_local.py "prompt" out.png [--seed 7] [--size 1024x1024] [--device mps]
Run with the Iris venv (third_party/iris-3b/.venv) and, on the Mac, under the GPU lock (shared.sh).
"""
import argparse, sys, time
from pathlib import Path
import torch

IRIS = Path(__file__).resolve().parents[1] / "third_party/iris-3b"

from omegaconf import OmegaConf
from iris3b.config import inference_config
from iris3b.sampling import load_for_inference
from iris3b.models.dit import IrisDiT
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import generate_few_step

ap = argparse.ArgumentParser(); ap.add_argument("prompt", help="prompt, or @file with one prompt per line"); ap.add_argument("out", help="output .png, or a folder for @file")
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--size", default="1024x1024")
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
ap.add_argument("--checkpoint", default=str(IRIS / "weights/turbo"))
a = ap.parse_args()
W, H = map(int, a.size.lower().split("x"))
dev = torch.device(a.device)
t0 = time.time()
raw, weights = load_for_inference(a.checkpoint)
cfg = inference_config(raw, [])
sigmas = list(OmegaConf.load(Path(a.checkpoint, "distill.yaml")).dmd.sigmas)
with torch.device("meta"):
    model = IrisDiT(cfg.model)
model.load_state_dict(weights, strict=True, assign=True)
model = model.eval().to(device=dev, dtype=torch.bfloat16)
del weights
import iris3b.text  # noqa: F401
te = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=dev)
t1 = time.time()
from torchvision.utils import save_image
if a.prompt.startswith("@"):
    prompts = [l.strip() for l in Path(a.prompt[1:]).read_text().splitlines() if l.strip()]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    outs = [str(Path(a.out) / f"{i:02d}.png") for i in range(len(prompts))]
else:
    prompts, outs = [a.prompt], [a.out]
for pr, o in zip(prompts, outs):
    gen = torch.Generator(device=dev).manual_seed(a.seed)     # same seed for every prompt
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
        imgs = generate_few_step(model, te, [pr], height=H, width=W, sigmas=sigmas, generator=gen, device=dev,
                                 num_train_timesteps=cfg.flow.num_train_timesteps, prediction=cfg.flow.prediction)
    save_image(imgs.float(), o, normalize=True, value_range=(-1, 1))
    print(o, "|", pr[-80:], flush=True)
mem = torch.mps.driver_allocated_memory() / 2**30 if dev.type == "mps" else torch.cuda.max_memory_allocated() / 2**30
print(f"{a.out}  load {t1 - t0:.0f}s  generate {time.time() - t1:.1f}s  gpu mem {mem:.1f} GB")
