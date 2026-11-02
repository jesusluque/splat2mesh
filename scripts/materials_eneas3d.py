"""Semi-automatic 3D materials: description "material: part, part; ..." → ENEAS on a 24-view orbit of the textured
mesh (tracked across views) → masks back-projected to texels and voted → roughness/metallic maps in the USD.

    python scripts/materials_eneas3d.py ASSET_DIR INPUT_IMAGE "gold: the gold coin, the gold bell; porcelain: the body"
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from splat2mesh.materials import prepare, parse_description, render_orbit, from_view_masks, apply_texel_classes, export_parts

HOST = os.environ.get("SPLAT2MESH_ENEAS_HOST", "ixwin-tb")
LINUX = os.environ.get("SPLAT2MESH_ENEAS_LINUX", "1" if HOST != "ixwin-tb" else "0") == "1"   # plain Linux vs Windows+WSL
asset, image, desc = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
spec = asset / "mat" / "regions.json"
if not spec.exists():
    prepare(asset, image, spec, prompt=desc)
orbit = render_orbit(asset, asset / "mat" / "orbit")
# annotate on the orbit frame that looks most like the input image (the side the generator really saw)
def obj_crop(path, bg_dark):
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    fg = a.max(-1) > 10 if bg_dark else a.min(-1) < 245
    ys, xs = np.nonzero(fg)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((96, 96)), np.float32)
ref = obj_crop(image, bg_dark=True)                                 # Idol input: object on black
frames = sorted((orbit / "frames").glob("*.jpg"))
score = [-np.abs(obj_crop(f, bg_dark=False) - ref).mean() for f in frames]
ann = frames[int(np.argmax(score))].name
print("annotation frame scores (best 3):", sorted(zip(score, [f.name for f in frames]))[-3:])
items = [(c, t.replace("'", "").replace('"', '').strip()) for c, t in parse_description(desc)]
tag = asset.resolve().as_posix().replace("/", "_")[-60:]
if LINUX:
    win = f"splat2mesh_eneas/jobs/{tag}"; wsl = f"$HOME/{win}"
else:
    win = f"C:/splat2mesh/eneas_jobs/{tag}"; wsl = "/mnt/c/" + win[3:]
reuse = "--reuse-masks" in sys.argv and (asset / "mat" / "masks.npz").exists()
if not reuse:
  if LINUX:
    subprocess.run(["ssh", HOST, f"rm -rf ~/{win}; mkdir -p ~/{win}/frames"], check=False)
  else:
    subprocess.run(["ssh", HOST, f'rmdir /s /q "{win}" 2>nul & mkdir "{win}\\frames"'], check=False)
  subprocess.run(["scp", "-q", *map(str, frames), f"{HOST}:{win}/frames/"], check=True)
  subprocess.run(["scp", "-q", str(ROOT / "scripts/eneas_seq.py"), f"{HOST}:{win}/"], check=True)
  q = " ".join("'" + t.replace("'", "") + "'" for _, t in items)
  inner = f"~/.local/bin/uv run python {wsl}/eneas_seq.py {wsl}/frames {wsl}/masks.npz {ann} {q}"
  cmd = (f"cd ~/splat2mesh_eneas/eneas && {inner}" if LINUX
         else f'wsl -d Ubuntu-24.04 -- bash -lc "cd ~/splat2mesh/eneas && {inner}"')
  print(subprocess.run(["ssh", HOST, cmd], capture_output=True, text=True).stdout.replace("\x00", "")[-1500:])
  subprocess.run(["scp", "-q", f"{HOST}:{win}/masks.npz", str(asset / "mat" / "masks.npz")], check=True)
z = np.load(asset / "mat" / "masks.npz")
masks = {f"{cls}|{ph}": z[ph] for cls, ph in items if ph in z.files}
print("annotation frame", ann, "| phrases", list(masks))
if not masks:          # ENEAS found nothing: whole object = the description's main material
    import json as _j
    sp = _j.loads(spec.read_text())
    main = max({c for c, _ in items}, key=[c for c, _ in items].count) if items else "glazed ceramic"
    for r in sp["regions"]:
        r.update(material=main, metallic=0.0, roughness=0.5)
    from splat2mesh.materials import PRESET
    for r in sp["regions"]:
        r["metallic"], r["roughness"] = PRESET.get(main, (0.0, 0.5))
    spec.write_text(_j.dumps(sp, indent=1))
    from splat2mesh.materials import apply
    print({"seen_texels": 0, "fallback": main}); print(apply(spec)); sys.exit(0)
print(from_view_masks(spec, orbit, masks))
print(apply_texel_classes(spec))
print('parts', export_parts(asset), export_parts(asset, 'asset_materials.usda', group='material'))
