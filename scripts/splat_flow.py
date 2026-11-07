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
    ap.add_argument("--max-parts", type=int, default=20, help="parts DeepSeek may list")
    ap.add_argument("--no-coverage", action="store_true", help="skip DeepSeek's check for large parts the list missed")
    ap.add_argument("--no-overlap", action="store_true", help="keep ENEAS's masks as they are (no check, no overlap resolution)")
    ap.add_argument("--glass", default="path", choices=["path", "browser"],
                    help="glass surfels: fully covering (refracts when path traced) or at the raster cover (seen through in a browser)")
    ap.add_argument("--glass-cell", type=float, default=1.0, help="glass surfel spacing in grid steps (2: a quarter as many)")
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
        desc = describe_ds(src / "idol_input.png", a.prompt, max_parts=a.max_parts)
        if not a.no_coverage:
            desc = complete_parts(desc, src / "idol_input.png")
        if a.prompt:
            desc = part_materials_from_prompt(desc, a.prompt.lower())
        desc_f.write_text(desc)
    desc = desc_f.read_text().strip()
    log("parts:", desc)

    cmd = [sys.executable, str(ROOT / "scripts/materials_eneas3d.py"), str(out), str(src / "idol_input.png"), desc,
           "--gs", str(ply), "--splat-only"] + (["--reuse-masks"] if a.reuse_masks else []) \
          + (["--no-overlap"] if a.no_overlap else [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    lines = []
    with open(out / "materials.log", "w", encoding="utf-8") as f:
        for line in p.stdout:                      # live: the app moves its progress bar on these lines
            lines.append(line); f.write(line); f.flush(); print("  " + line.rstrip(), flush=True)
    if p.wait():
        sys.exit("materials failed, see " + str(out / "materials.log") + "\n" + "".join(lines)[-1500:])
    log("materials:", [l for l in lines if l.startswith(("splat materials", "splat finish"))])

    from splat2mesh.athenea_splat import write_splat_usd
    log("USD:", write_splat_usd(out / "mat" / "splat_materials.ply", out / "splat.usdc", y_down=True,
                                raster_glass=a.glass == "browser", glass_cell=a.glass_cell))
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
