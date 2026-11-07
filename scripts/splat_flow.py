"""Picture (or a sentence's picture) → a relightable 3DGS with PBR materials, with no geometry at any step.

  1. Idol: picture → splat, with its prepared input and its estimate of the input camera;
  2. DeepSeek lists the parts by material (the generation prompt, if given, corrects single parts' materials);
  3. athenea renders the splat on a 16-view orbit (colour + depth); ENEAS tracks each part; DeepSeek checks the
     masks and settles overlaps; every gaussian votes for the part its pixel shows (scripts/materials_eneas3d.py
     --gs --splat-only);
  4. per gaussian: library PBR values, normals from the splat's own depth renders, floaters, highlight-free
     albedo, glass tint (splat2mesh/gs_finish.py);
  5. athenea USD: the relit splat, glass as surfels on the glass gaussians' own implicit surface
     (splat2mesh/athenea_splat.py); optionally a path-traced turntable.

    python scripts/splat_flow.py PICTURE OUT_DIR [--prompt "..."] [--seed 42] [--gaussians 262144]
    python scripts/splat_flow.py SPLAT.ply OUT_DIR --input IDOL_INPUT.png --camera IDOL_CAMERA.json [--prompt ...]
        [--reuse-masks] [--turntable N] [--technique rt|raster]
OUT_DIR/src/{splat.ply, idol_input.png, idol_camera.json}, OUT_DIR/materials.txt, OUT_DIR/mat/splat_materials.ply
(+ mat_names.json, finish.json), OUT_DIR/splat.usdc and OUT_DIR/turntable.mp4.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source"); ap.add_argument("out")
    ap.add_argument("--prompt", default=""); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gaussians", type=int, default=262144)
    ap.add_argument("--input", help="Idol's prepared input picture (when SOURCE is a .ply)")
    ap.add_argument("--camera", help="Idol's camera estimate for it (when SOURCE is a .ply)")
    ap.add_argument("--reuse-masks", action="store_true"); ap.add_argument("--turntable", type=int, default=0)
    ap.add_argument("--technique", default="rt", choices=["rt", "raster"])
    a = ap.parse_args()
    out = Path(a.out); src = out / "src"; src.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:6.0f}s]", *m, flush=True)

    ply = src / "splat.ply"
    if Path(a.source).suffix.lower() == ".ply":
        if not (a.input and a.camera):
            sys.exit("a .ply needs --input and --camera (Idol's prepared input and camera estimate)")
        shutil.copy(a.source, ply); shutil.copy(a.input, src / "idol_input.png"); shutil.copy(a.camera, src / "idol_camera.json")
    elif not ply.exists():
        from splat2mesh.remote import idol_sample
        got = idol_sample(a.source, src / "idol", a.gaussians, a.seed)
        shutil.copy(got, ply)
        for f in ("idol_input.png", "idol_camera.json"):
            shutil.copy(src / "idol" / f, src / f)
        log("Idol:", ply)

    desc_f = out / "materials.txt"
    if not desc_f.exists():
        from splat2mesh.describe import complete_parts, describe_ds, part_materials_from_prompt
        # the list, then a coverage check: large visible parts it left out (hands, a cloak) are added
        desc = complete_parts(describe_ds(src / "idol_input.png", a.prompt), src / "idol_input.png")
        if a.prompt:
            desc = part_materials_from_prompt(desc, a.prompt.lower())
        desc_f.write_text(desc)
    desc = desc_f.read_text().strip()
    log("parts:", desc)

    cmd = [sys.executable, str(ROOT / "scripts/materials_eneas3d.py"), str(out), str(src / "idol_input.png"), desc,
           "--gs", str(ply), "--splat-only"] + (["--reuse-masks"] if a.reuse_masks else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    (out / "materials.log").write_text(p.stdout + p.stderr)
    if p.returncode:
        sys.exit("materials failed, see " + str(out / "materials.log") + "\n" + p.stderr[-1500:])
    log("materials:", [l for l in p.stdout.splitlines() if l.startswith(("splat materials", "splat finish"))])

    from splat2mesh.athenea_splat import write_splat_usd
    log("USD:", write_splat_usd(out / "mat" / "splat_materials.ply", out / "splat.usdc", y_down=True))
    if a.turntable:
        from splat2mesh.athenea import turntable
        import json
        import math
        c = json.loads((src / "idol_camera.json").read_text())
        start = (180.0 - math.degrees(c["yaw"])) % 360                 # the side the picture showed
        turntable(out / "splat.usdc", out / "turntable.mp4", frames=a.turntable, technique=a.technique,
                  start_deg=start)
        log("turntable:", out / "turntable.mp4")


if __name__ == "__main__":
    main()
