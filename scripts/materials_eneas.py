"""Semi-automatic materials: a short description of the image's materials → ENEAS masks (speridlabs, on the
CUDA host's WSL) → each colour region gets the material of the mask that covers it → PBR maps in the USD.

    python scripts/materials_eneas.py ASSET_DIR INPUT_IMAGE "gold: the bell, the coin; red lacquer: the collar; porcelain: the body"
ASSET_DIR/mat/regions.json is created (colour regions) if missing.
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from splat2mesh.materials import prepare, parse_description, from_masks, apply

HOST = os.environ.get("SPLAT2MESH_IDOL_HOST", "ixwin-tb")
asset, image, desc = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
spec = asset / "mat" / "regions.json"
if not spec.exists():
    prepare(asset, image, spec, prompt=desc)
items = parse_description(desc)
phrases = [t for _, t in items]
tag = asset.resolve().as_posix().replace("/", "_")[-60:]
win_dir = f"C:/splat2mesh/eneas_jobs/{tag}"
subprocess.run(["ssh", HOST, f'mkdir "{win_dir}" 2>nul & exit 0'], check=True)
subprocess.run(["scp", "-q", str(image), str(ROOT / "scripts/eneas_seg.py"), f"{HOST}:{win_dir}/"], check=True)
wsl_dir = "/mnt/c/" + win_dir[3:]
quoted = " ".join("'" + p.replace("'", "") + "'" for p in phrases)
cmd = (f'wsl -d Ubuntu-24.04 -- bash -lc "cd ~/splat2mesh/eneas && ~/.local/bin/uv run python {wsl_dir}/eneas_seg.py '
       f'{wsl_dir}/{image.name} {wsl_dir}/out {quoted}"')
print(subprocess.run(["ssh", HOST, cmd], capture_output=True, text=True).stdout.replace("\x00", "")[-2000:])
mdir = asset / "mat" / "eneas"; mdir.mkdir(parents=True, exist_ok=True)
subprocess.run(["scp", "-q", "-r", f"{HOST}:{win_dir}/out/*.png", str(mdir)], check=False)
H, W = np.asarray(Image.open(image)).shape[:2]
masks = {}
for cls, ph in items:
    f = mdir / (ph.replace(" ", "_") + ".png")
    if f.exists():
        m = np.asarray(Image.open(f).convert("L").resize((W, H))) > 127
        masks[f"{cls}|{ph}"] = m
print("masks:", {k: round(float(v.mean()), 4) for k, v in masks.items()})
print(from_masks(spec, masks))
print(apply(spec))
