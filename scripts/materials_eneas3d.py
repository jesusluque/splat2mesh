"""Semi-automatic 3D materials: description "material: part, part; ..." → ENEAS on a 24-view orbit of the textured
mesh (tracked across views) → masks back-projected to texels and voted → roughness/metallic maps in the USD.

    python scripts/materials_eneas3d.py ASSET_DIR INPUT_IMAGE "gold: the gold coin, the gold bell; porcelain: the body"
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from splat2mesh.materials import prepare, parse_description, render_orbit, from_view_masks, apply_texel_classes

HOST = os.environ.get("SPLAT2MESH_IDOL_HOST", "ixwin-tb")
asset, image, desc = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
spec = asset / "mat" / "regions.json"
if not spec.exists():
    prepare(asset, image, spec, prompt=desc)
orbit = render_orbit(asset, asset / "mat" / "orbit")
# annotate on the orbit frame that looks most like the input image (the side the generator really saw)
ref = np.asarray(Image.open(image).convert("L").resize((128, 128)), np.float32)
ref = np.where(ref < 8, 255, ref)                                   # Idol input: black bg → white like the orbit
frames = sorted((orbit / "frames").glob("*.jpg"))
score = [-np.abs(np.asarray(Image.open(f).convert("L").resize((128, 128)), np.float32) - ref).mean() for f in frames]
ann = frames[int(np.argmax(score))].name
items = parse_description(desc)
tag = asset.resolve().as_posix().replace("/", "_")[-60:]
win = f"C:/splat2mesh/eneas_jobs/{tag}"; wsl = "/mnt/c/" + win[3:]
subprocess.run(["ssh", HOST, f'rmdir /s /q "{win}" 2>nul & mkdir "{win}\\frames"'], check=False)
subprocess.run(["scp", "-q", *map(str, frames), f"{HOST}:{win}/frames/"], check=True)
subprocess.run(["scp", "-q", str(ROOT / "scripts/eneas_seq.py"), f"{HOST}:{win}/"], check=True)
q = " ".join("'" + t.replace("'", "") + "'" for _, t in items)
cmd = f'wsl -d Ubuntu-24.04 -- bash -lc "cd ~/splat2mesh/eneas && ~/.local/bin/uv run python {wsl}/eneas_seq.py {wsl}/frames {wsl}/masks.npz {ann} {q}"'
print(subprocess.run(["ssh", HOST, cmd], capture_output=True, text=True).stdout.replace("\x00", "")[-1500:])
subprocess.run(["scp", "-q", f"{HOST}:{win}/masks.npz", str(asset / "mat" / "masks.npz")], check=True)
z = np.load(asset / "mat" / "masks.npz")
masks = {f"{cls}|{ph}": z[ph] for cls, ph in items if ph in z.files}
print("annotation frame", ann, "| phrases", list(masks))
print(from_view_masks(spec, orbit, masks))
print(apply_texel_classes(spec))
