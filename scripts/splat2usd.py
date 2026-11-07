"""splat2usd: one command from an image or a 3DGS .ply to a UV-textured USD, with a QA report.

    python scripts/splat2usd.py INPUT [INPUT ...] [--out out/auto] [--direct] [--faces 80000] [--tex 4096]
                                [--y-down | --y-up] [--gaussians 262144] [--notify]

INPUT can be:
  - an image (.png/.jpg/.webp): Idol (speridlabs, image → 3DGS) generates the splat first, on the CUDA host
    named by SPLAT2MESH_IDOL_HOST (docs/setup.md);
  - a 3DGS .ply: used as is;
  - a folder: every image/.ply inside it (batch).

Steps per input: [Idol] → --direct: mesh from the splat's opacity field; otherwise cameras framing the splat →
athenea `render --splats` (one job under the GPU lock)
→ TSDF + visual hull → clean, decimate, cap → xatlas UV → robust multi-view bake → USD → QA
(mesh stats + re-render PSNR + compare.png) → report.json; a summary table for the batch at the end.

Several views (like Tripo's "Multiview to 3D"):
  --views LEFT BACK RIGHT   more photos of the one input object; Idol is sampled from all of them (idol_mv.py)
  --multiview gen           one photo: the hidden sides are rendered from the first splat, cleaned by Iris, checked
                            (outline kept, deepseek-flash sees the same object) and given back to Idol (gen_views.py)
  Both are experimental: in the measurements so far they did not beat one photo (docs/multiview.md).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

IMG = {".png", ".jpg", ".jpeg", ".webp"}
PY = sys.executable
SHARED = Path(os.environ.get("SPLAT2MESH_GPU_LOCK", Path.home() / "luc/athenea-gpu/shared.sh"))
TG = Path(os.environ.get("SPLAT2MESH_NOTIFY", Path.home() / "bin/tg"))
IXWIN = os.environ.get("SPLAT2MESH_IDOL_HOST", "ixwin-tb")
IX_PY = r"C:\splat2mesh\venv\Scripts\python"
IX_DIR = "C:/splat2mesh/auto"


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}):\n{(r.stdout + r.stderr)[-2000:]}")
    return r.stdout


def idol_linux(image: Path, work: Path, gaussians: int) -> Path:
    """Idol on a plain Linux CUDA host (~/splat2mesh_eneas/idol-sdk)."""
    d = f"splat2mesh_eneas/idol_jobs/{work.name}"
    sh(["ssh", IXWIN, f"mkdir -p ~/{d}"])
    sh(["scp", "-q", str(image), f"{IXWIN}:{d}/{image.name}"])
    mode = "real" if image.suffix.lower() in {".jpg", ".jpeg"} else "synthetic"
    sh(["ssh", IXWIN, f"cd ~/splat2mesh_eneas/idol-sdk && ATTN_BACKEND=sdpa .venv/bin/python -m idol.sample "
                      f"--image ~/{d}/{image.name} --mode {mode} --out ~/{d} --num_gaussians {gaussians}"])
    ply = work / f"{image.stem}.ply"
    sh(["scp", "-q", f"{IXWIN}:{d}/{image.stem}.ply", str(ply)])
    sh(["scp", "-q", f"{IXWIN}:{d}/{image.stem}_input.png", str(work / "idol_input.png")])
    return ply


def idol(image: Path, work: Path, gaussians: int) -> Path:
    """Image → 3DGS .ply with Idol on ixwin (synthetic mode for renders/generated images, real for photos)."""
    if os.environ.get("SPLAT2MESH_IDOL_LINUX") == "1":
        return idol_linux(image, work, gaussians)
    sh(["ssh", IXWIN, f'mkdir "{IX_DIR}" 2>nul & exit 0'])
    sh(["scp", "-q", str(image), f"{IXWIN}:{IX_DIR}/{image.name}"])
    mode = "real" if image.suffix.lower() in {".jpg", ".jpeg"} else "synthetic"
    sh(["ssh", IXWIN, rf'cd /d C:\splat2mesh\idol-sdk && set ATTN_BACKEND=sdpa&& {IX_PY} -m idol.sample '
                      rf'--image {IX_DIR}/{image.name} --mode {mode} --out {IX_DIR} --no_optimize --num_gaussians {gaussians}'])
    ply = work / f"{image.stem}.ply"
    sh(["scp", "-q", f"{IXWIN}:{IX_DIR}/{image.stem}.ply", str(ply)])
    sh(["scp", "-q", f"{IXWIN}:{IX_DIR}/{image.stem}_input.png", str(work / "idol_input.png")])
    # Idol's estimate of the input camera (yaw/pitch around the object): places parts generated from one picture
    subprocess.run(["scp", "-q", f"{IXWIN}:{IX_DIR}/{image.stem}_camera.json", str(work / "idol_camera.json")])
    return ply


def process(inp: Path, out_root: Path, a) -> dict:
    t0 = time.time()
    name = inp.stem
    work = out_root / name
    work.mkdir(parents=True, exist_ok=True)
    rep = {"input": str(inp), "name": name}
    try:
        if inp.suffix.lower() in IMG:
            shutil.copy(inp, work / f"source{inp.suffix.lower()}")
            if a.upscale:              # Iris restorer on the object crop: crisper detail in, cleaner splat out
                up = work / f"{inp.stem}_up.png"
                sh(["bash", str(SHARED), str(ROOT / "third_party/iris-3b/.venv/bin/python"),
                    str(ROOT / "scripts/upscale_detail.py"), str(inp), "", str(work / "upscale"),
                    "--restore-only", str(up)], stdin=subprocess.DEVNULL)
                rep["upscaled"] = True
                ply = idol(up, work, a.gaussians)
                (work / f"{up.stem}.ply").rename(work / f"{inp.stem}.ply"); ply = work / f"{inp.stem}.ply"
            elif a.views:              # several real photos of the object: Idol from all of them (idol_mv.py)
                from gen_views import run_mv
                mode = "real" if inp.suffix.lower() in {".jpg", ".jpeg"} else "synthetic"
                from gen_views import SIDES
                offs = list(SIDES.values())[:len(a.views)]          # in the order left, back, right
                ply = run_mv(inp, [Path(v) for v in a.views], {}, work, a.gaussians, mode, yaw_offsets=offs)
                shutil.copy(work / f"{inp.stem}_mv_input.png", work / "idol_input.png")
                shutil.copy(work / f"{inp.stem}_mv_camera.json", work / "idol_camera.json")
                rep["views"] = {"photos": a.views}
            else:
                ply = idol(inp, work, a.gaussians)
            if a.multiview == "gen" and not a.views:   # hidden sides generated (Iris) and given back to Idol
                sh([PY, str(ROOT / "scripts/gen_views.py"), str(inp), str(ply), str(work / "idol_camera.json"),
                    str(work), "--gaussians", str(a.gaussians)])
                g = json.loads((work / "gen_views.json").read_text())
                rep["views"] = {"generated": g.get("kept", []), "gates": g.get("views"), "desc": g.get("desc")}
                mv = work / f"{inp.stem}_mv.ply"
                if mv.exists():
                    ply = mv
                    shutil.copy(work / f"{inp.stem}_mv_camera.json", work / "idol_camera.json")
            y_down = not a.y_up            # Idol writes Y-down (COLMAP-style)
            antialias = False
            rep["t_idol_s"] = round(time.time() - t0, 1)
        else:
            ply = inp
            y_down = a.y_down
            antialias = not a.no_antialias
        if a.direct:                     # no render: opacity field → mesh, colour from the gaussians
            from splat2mesh.direct import run_direct
            rep["mesh"] = run_direct(ply, work / "usd", faces=a.faces, tex_res=a.tex, y_down=y_down,
                                     smooth=a.smooth)
            m = rep["mesh"]
            if m.get("folds", 0) <= a.max_folds:
                rep["ok"] = m["boundary_edges"] == 0 and m["non_manifold_edges"] <= 200
                rep["t_total_s"] = round(time.time() - t0, 1)
                (work / "report.json").write_text(json.dumps(rep, indent=2))
                return rep
            # crumpled surface (flat walls the opacity field cannot hold, e.g. black lacquer boxes): render path
            print(f"    direct mode: {m['folds']} folds > {a.max_folds} → falling back to the render path", flush=True)
            rep["direct_rejected"] = {k: m[k] for k in ("folds", "uv_charts", "components")}
        views_dir = work / "render"
        cams = [PY, str(ROOT / "scripts/athenea_cams.py"), str(ply), str(views_dir)] + (["--rotate-x-180"] if y_down else [])
        sh(cams)
        flags = (["--rotate-x", "180"] if y_down else []) + ([] if antialias else ["--no-antialias"])
        t1 = time.time()
        out = sh(["bash", str(SHARED), "bash", str(ROOT / "scripts/render_athenea.sh"), str(ply), str(views_dir)] + flags)
        if "failed" in out:
            rep["render_warnings"] = [l for l in out.splitlines() if "failed" in l]
        rep["t_render_s"] = round(time.time() - t1, 1)

        from splat2mesh.cli import run
        from splat2mesh.rerender import qa
        from splat2mesh.views import load_athenea
        views = load_athenea(views_dir)
        asset = work / "usd"
        # render path bakes 87 views per texel: 4096² is ~4× slower for little gain on these objects
        rep["mesh"] = run(views, asset, faces=a.faces, tex_res=min(a.tex, a.render_tex), voxel=None, smooth=a.smooth)
        rep["qa"] = qa(asset, load_athenea(views_dir, erode_px=0))
        if a.glass and a.smooth == 0:
            print("    note: --glass without --smooth; glass shows fusion bumps (try --smooth 3)", flush=True)
        if a.glass:
            from splat2mesh.glass import estimate, load_background
            from splat2mesh.evaluate import usd_to_trimesh
            from splat2mesh.usd_export import set_glass
            gviews = Path(a.glass_views) if a.glass_views else views_dir
            mesh = usd_to_trimesh(str(asset / "asset.usda"), "/Asset/Geom")
            mesh.merge_vertices()
            g = estimate(mesh, load_athenea(gviews, erode_px=1), load_background(a.glass))
            (asset / "glass.json").write_text(json.dumps(g, indent=2))
            set_glass(asset / "asset.usda", g["ior"], g["transmission_per_unit"])
            rep["glass"] = {k: g[k] for k in ("ior", "transmission_per_unit", "loss", "rays_lost")}
        m = rep["mesh"]
        rep["ok"] = (m["boundary_edges"] == 0 and m["non_manifold_edges"] <= 20 and m["texels_covered"] > 0.99
                     and rep["qa"]["psnr_mean"] >= a.min_psnr)
    except Exception as e:  # keep the batch going
        rep["ok"] = False
        rep["error"] = str(e)[-1500:]
    rep["t_total_s"] = round(time.time() - t0, 1)
    (work / "report.json").write_text(json.dumps(rep, indent=2))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "out/auto"))
    ap.add_argument("--faces", type=int, default=80000)
    ap.add_argument("--tex", type=int, default=4096)
    ap.add_argument("--smooth", type=float, default=0.0, help="Taubin smoothing factor X (0 off, 1 = 10 it.; glass: 3+)")
    ap.add_argument("--direct", action="store_true",
                    help="no render: mesh from the splat's opacity field (~1 min; good on Idol splats, poor on thin surface splats)")
    ap.add_argument("--max-folds", type=int, default=300,
                    help="--direct: above this many folded edges (dihedral > 120°) use the render path instead")
    ap.add_argument("--render-tex", type=int, default=2048, help="texture size cap on the render path")
    ap.add_argument("--skip-done", action="store_true", help="skip inputs whose report.json says ok")
    ap.add_argument("--gaussians", type=int, default=262144)
    ap.add_argument("--upscale", action="store_true",
                    help="pass the object crop through the Iris-3B restorer before Idol (Mac, ~4 min)")
    ap.add_argument("--views", nargs="+", default=None,
                    help="more photos of the same (single) input object, e.g. left back right: Idol uses them all")
    ap.add_argument("--multiview", choices=["off", "gen"], default="off",
                    help="gen: the hidden sides are generated from the one photo (Iris) and given back to Idol")
    ap.add_argument("--y-down", action="store_true", help=".ply input is Y-down (COLMAP/Idol)")
    ap.add_argument("--y-up", action="store_true", help="override: Idol output already Y-up")
    ap.add_argument("--no-antialias", action="store_true", help="plain 3DGS .ply trained without Mip-Splatting")
    ap.add_argument("--min-psnr", type=float, default=22.0)
    ap.add_argument("--glass", default=None, help="BACKGROUND.json: estimate a glass material (IOR, tint) and apply it")
    ap.add_argument("--glass-views", default=None, help="views of the object in front of that background (default: the splat renders)")
    ap.add_argument("--notify", action="store_true", help="send the summary and compare images through SPLAT2MESH_NOTIFY at the end")
    a = ap.parse_args()

    items = []
    for s in a.inputs:
        p = Path(s)
        if p.is_dir():
            items += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG | {".ply"})
        else:
            items.append(p)
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    reps = []
    for i, p in enumerate(items, 1):
        done = out_root / p.stem / "report.json"
        if a.skip_done and done.exists() and json.loads(done.read_text()).get("ok"):
            reps.append(json.loads(done.read_text())); print(f"[{i}/{len(items)}] {p.name} done, skipped", flush=True)
            continue
        print(f"[{i}/{len(items)}] {p.name} ...", flush=True)
        r = process(p, out_root, a)
        reps.append(r)
        q = r.get("qa", {})
        m = r.get("mesh", {})
        print(f"    {'OK ' if r['ok'] else 'FAIL'} psnr {q.get('psnr_mean', float('nan')):.1f} faces {m.get('faces', '-')} "
              f"open {m.get('boundary_edges', '-')} nonmanifold {m.get('non_manifold_edges', '-')} "
              f"{r['t_total_s']:.0f}s {r.get('error', '')[:200]}", flush=True)
    summary = out_root / "summary.json"
    summary.write_text(json.dumps(reps, indent=2))
    ok = sum(r["ok"] for r in reps)
    line = f"splat2usd: {ok}/{len(reps)} OK → {out_root}"
    print(line)
    if a.notify and TG.exists():
        rows = [f"{'✔' if r['ok'] else '✘'} {r['name']}: PSNR {r.get('qa', {}).get('psnr_mean', 0):.1f}" for r in reps]
        subprocess.run([str(TG), "--from", "splat2mesh", line + "\n" + "\n".join(rows)])
        for r in reps[:5]:
            img = out_root / r["name"] / "usd" / "compare.png"
            if img.exists():
                subprocess.run([str(TG), "-s", "--from", "splat2mesh", "-f", str(img), r["name"]])


if __name__ == "__main__":
    main()
