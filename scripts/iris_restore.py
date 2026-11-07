"""Iris-3B image restoration (speridlabs/iris-3b/upscaler, Apache-2.0) on the Mac, in bf16 so it fits next to everything
else: the export is FP32 (12 GB), and loading it as Restorer does would hold the CPU copy and the MPS copy at once on
unified memory (~24 GB). Here each tensor is read from the safetensors file and cast to bf16 on its own (~6 GB).

    python scripts/iris_restore.py IN.png OUT.png [--scale 2] [--no-color-fix] [--device mps]
Run with the Iris venv under the GPU lock (shared.sh). The input is downscaled first like upscale.py's budget
(short side <= 512, long side <= 1024), the output is scale x that.
"""
import argparse, sys, time
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
try:                                 # the configured Iris checkout (tools/iris-3b on Windows), else third_party
    from splat2mesh.config import cfg
    IRIS = Path(cfg("iris.dir")).expanduser()
except Exception:
    IRIS = ROOT / "third_party/iris-3b"
sys.path.insert(0, str(IRIS / "src"))
from iris3b.downstream import restoration as R
from iris3b.models.dit import IrisDiT


class Restorer16(R.Restorer):
    def __init__(self, source, device):
        from omegaconf import OmegaConf
        from safetensors import safe_open
        from safetensors.torch import load_file
        from iris3b.config import inference_config
        from huggingface_hub import snapshot_download
        owner, repo, sub = source.split("/", 2)
        path = Path(snapshot_download(f"{owner}/{repo}", allow_patterns=f"{sub}/*")) / sub
        raw = OmegaConf.to_container(OmegaConf.load(path / "config.yaml")); settings = raw.pop("task")
        cfg = inference_config(raw, [])
        self.device = torch.device(device)
        with torch.device("meta"):
            model = IrisDiT(cfg.model)
        w = {}
        with safe_open(str(path / "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                w[k] = f.get_tensor(k).to(torch.bfloat16)
        model.load_state_dict(w, strict=True, assign=True); del w
        self.model = model.eval().requires_grad_(False).to(self.device)
        prompt = load_file(path / "empty_prompt.safetensors")
        self.embeddings = prompt["embeddings"].to(self.device, torch.bfloat16)
        self.mask = prompt["mask"].to(self.device, torch.bool)
        self.patch = cfg.model.patch_size; self.tile = settings["tile"]; self.sigma = settings["sigma"]
        self.time = settings["sigma"] * cfg.flow.num_train_timesteps

    def restore_tile(self, x):
        t = torch.full((x.shape[0],), self.time, device=x.device)
        y = self.embeddings.expand(len(x), *self.embeddings.shape[1:])
        v = self.model(x.to(torch.bfloat16), t, y, y_mask=self.mask.expand(len(x), -1)).x
        return x.float() - self.sigma * v.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=2.0); ap.add_argument("--no-color-fix", action="store_true")
    ap.add_argument("--no-budget", action="store_true", help="keep the input's size (no downscale first)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    a = ap.parse_args()
    t0 = time.time()
    r = Restorer16("speridlabs/iris-3b/upscaler", a.device)
    im = Image.open(a.inp).convert("RGB")
    if not a.no_budget:
        im = R.fit_budget(im) if hasattr(R, "fit_budget") else im
    r(im, scale=a.scale, color_fix=not a.no_color_fix).save(a.out)
    print(f"{a.out} {round(time.time() - t0, 1)} s", flush=True)


if __name__ == "__main__":
    main()
