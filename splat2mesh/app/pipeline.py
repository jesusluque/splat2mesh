"""One job of the app: from a sentence, a picture or a splat to the outputs asked for, with progress.

    python -m splat2mesh.app.pipeline JOB.json            (the app writes JOB.json and reads the progress lines)

JOB.json (every key optional except source and its input):
  {"name": "dragon", "out": "<folder>",
   "source": "prompt" | "image" | "splat",
   "prompt": "...", "image": "path.png", "splat": "path.ply", "camera": "idol_camera.json", "input": "idol_input.png",
   "text2image": {"seed": 7, "size": "1024x1024", "mode": "auto" | "gpu" | "cpu-encoder" | "space", "fill": true},
   "detail": "all" | "faces" | "off",           # Iris-3B restorer's detail at the picture's own size, before Idol
   "idol": {"seeds": 1, "gaussians": 262144, "mode": "auto" | "synthetic" | "real", "steps": 50, "cfg": 3.0,
            "rescale_t": 3.0, "dinov3_dtype": "bf16" | "fp32"},
   "outputs": {"splat": true, "mesh": false, "materials": true},
   "materials": {"max_parts": 20, "coverage": true, "overlap": true, "deepseek_model": ""},
   "glass": {"mode": "path" | "browser", "cell": 1.0},
   "mesh": {"path": "direct" | "render", "faces": 80000, "tex": 4096, "smooth": 0.0, "max_folds": 300,
            "render_tex": 2048},
   "parts": "",                                  # "material: part, part; ..." to skip the automatic part list
   "render": {"turntable": true, "frames": 48, "size": 900, "technique": "raster", "spp": 96, "elev": 15,
              "dist": 0, "grey": 0.8, "hdri": ""}}

Progress lines:  PROGRESS <0-100> <stage>: <message>      Outputs:  <out>/results.json
"""
from __future__ import annotations

import json
import os
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..config import ROOT, cfg, iris_python

PY = sys.executable

# Windows pipes default to the ANSI code page: text with "→" or accents would stop a step. UTF-8 here and in every
# process started from here
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class Progress:
    """Stages with weights; each reports its own 0..1 and the job's 0..100 is printed."""

    def __init__(self, stages: list[tuple[str, float]]):
        tot = sum(w for _, w in stages) or 1
        self.start, acc = {}, 0.0
        for n, w in stages:
            self.start[n] = (acc / tot * 100, w / tot * 100); acc += w

    def __call__(self, stage: str, frac: float, msg: str):
        a, w = self.start.get(stage, (0, 0))
        print(f"PROGRESS {int(a + w * max(0.0, min(1.0, frac)))} {stage}: {msg}", flush=True)


def sh(cmd, log: Path, **kw):
    """Runs a step, appending its output to the job's log; raises with the tail on failure."""
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n"); f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, **kw)
    if r.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"{Path(str(cmd[1] if len(cmd) > 1 else cmd[0])).name} failed\n{tail}")


FILL = ("the object fills the entire frame edge to edge, tightly framed, sharp focus on every detail, "
        "studio product photo, soft even light, plain pure white background")


def framed(prompt: str) -> str:
    """The prompt with the framing Idol does best with, unless it already asks for it."""
    low = prompt.lower()
    if "fills the" in low or "edge to edge" in low:
        return prompt
    return prompt.rstrip(" .,") + ", " + FILL


def text2image(job, out: Path, log: Path, prog) -> Path:
    t = job.get("text2image", {})
    if t.get("fill", True):
        job = {**job, "prompt": framed(job["prompt"])}
        (out / "prompt_used.txt").write_text(job["prompt"])
    mode = t.get("mode", "auto")
    if mode == "auto":
        mode = cfg("iris.mode") or "gpu"
        if mode != "space" and not (Path(cfg("iris.dir")) / "weights" / "turbo").exists():
            mode = "space"                      # Iris-3B not installed here: the online Space
    png = out / "input.png"
    prog("picture", 0.05, f"Iris-3B Turbo ({mode})")
    if mode == "space":
        sh([PY, ROOT / "scripts/text2image.py", job["prompt"], png, "--seed", t.get("seed", 7)], log)
    else:
        cmd = [iris_python(), ROOT / "scripts/iris_local.py", job["prompt"], png, "--seed", t.get("seed", 7),
               "--size", t.get("size", "1024x1024"), "--checkpoint", Path(cfg("iris.dir")) / "weights" / "turbo"]
        if mode == "cpu-encoder":
            cmd += ["--te-device", "cpu"]
        from ..config import gpu_lock
        with gpu_lock():
            sh(cmd, log)
    prog("picture", 1, "picture ready")
    return png


def detail(job, image: Path, out: Path, log: Path, prog) -> Path:
    """The Iris-3B restorer's detail at the picture's own size (scripts/detail_pass.py): the whole picture and the
    faces, or the faces only. Needs Iris installed locally; otherwise the picture goes on as it is."""
    mode = job.get("detail", "all")
    from ..config import iris_python
    if mode == "off" or cfg("iris.mode") == "space" or not Path(iris_python()).exists():
        if mode != "off":
            prog("detail", 1, "no local Iris-3B: picture used as it is")
        return image
    prog("detail", 0.05, "Iris-3B restorer: " + ("whole picture, then the faces" if mode == "all" else "the faces"))
    dst = out / ("detail" + image.suffix)
    try:
        sh([PY, ROOT / "scripts/detail_pass.py", image, dst] + (["--all"] if mode == "all" else []), log)
    except Exception as e:                        # no DeepSeek for the faces, etc.: keep what there is
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"detail pass skipped: {e}\n")
        prog("detail", 1, "detail pass skipped (see the log)")
        return image
    prog("detail", 1, "detail added")
    return dst


def idol(job, image: Path, out: Path, log: Path, prog) -> tuple[Path, Path, Path]:
    """Picture → splat; with several seeds each is scored from four sides and the best is kept."""
    from ..remote import idol_sample
    from ..config import gpu_lock
    o = job.get("idol", {}); n = max(1, int(o.get("seeds", 1))); g = int(o.get("gaussians", 262144))
    seeds = [42, 1, 2, 3, 4, 5, 6, 7][:n]
    scores = {}
    for i, s in enumerate(seeds):
        prog("splat", i / n, f"Idol, seed {s} ({i + 1}/{n})")
        d = out / "seeds" / f"s{s}"
        with gpu_lock():
            ply = idol_sample(image, d, g, seed=s, mode=o.get("mode"), steps=o.get("steps"),
                              cfg_strength=o.get("cfg"), rescale_t=o.get("rescale_t"),
                              dinov3_dtype=o.get("dinov3_dtype"))
        if n > 1:
            prog("splat", (i + 0.7) / n, f"scoring seed {s} from four sides (athenea + DeepSeek)")
            try:
                from ..seeds import score
                scores[s] = score(ply, d / "idol_camera.json", image, d / "check")["score"]
            except Exception as e:
                with open(log, "a") as f:
                    f.write(f"seed {s} not scored: {e}\n")
                scores[s] = None
    best = seeds[0] if n == 1 else max(scores, key=lambda k: -1 if scores[k] is None else scores[k])
    d = out / "seeds" / f"s{best}"
    ply = next(d.glob("*.ply"))
    (out / "seeds.json").write_text(json.dumps({"scores": scores, "best": best}, indent=1))
    prog("splat", 1, f"splat ready (seed {best}{', score ' + str(scores[best]) if scores.get(best) else ''})")
    return ply, d / "idol_camera.json", d / "idol_input.png"


def splat_materials(job, ply, cam, inp, out: Path, log: Path, prog) -> Path:
    """The relightable splat with materials, no geometry (scripts/splat_flow.py)."""
    d = out / "splat"
    d.mkdir(parents=True, exist_ok=True)
    if job.get("parts"):
        (d / "materials.txt").write_text(job["parts"])
    prog("materials", 0.02, "parts (DeepSeek), orbit renders, ENEAS, votes per gaussian")
    cmd = [PY, ROOT / "scripts/splat_flow.py", ply, d, "--input", inp, "--camera", cam]
    if job.get("prompt"):
        cmd += ["--prompt", job["prompt"]]
    m, g = job.get("materials", {}), job.get("glass", {})
    cmd += ["--max-parts", m.get("max_parts", 20), "--glass", g.get("mode", "path"), "--glass-cell", g.get("cell", 1.0)]
    cmd += (["--no-coverage"] if m.get("coverage") is False else []) + (["--no-overlap"] if m.get("overlap") is False else [])
    env = dict(os.environ)
    if m.get("deepseek_model"):
        env["SPLAT2MESH_DEEPSEEK_MODEL"] = m["deepseek_model"]
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    # the stages of scripts/splat_flow.py and materials_eneas3d.py as they print them; ENEAS (the long part)
    # moves the bar phrase by phrase
    marks = [("parts:", 0.05, "parts listed"), ("annotation frame", 0.12, "orbit rendered with athenea"),
             ("ENEAS phrases:", 0.13, "ENEAS: loading"), ("mask check:", 0.78, "masks checked (DeepSeek)"),
             ("overlap resolved", 0.82, "overlaps resolved"), ("gaussians voted directly", 0.86, "votes per gaussian"),
             ("normals:", 0.9, "normals and floaters from the depth renders"),
             ("splat finish", 0.94, "albedo and glass finished"), ("USD:", 1.0, "relightable splat written")]
    n_ph, done_ph = 0, 0
    with open(log, "a", encoding="utf-8") as f:
        for line in p.stdout:
            f.write(line); f.flush()
            for k, fr, msg in marks:
                if k in line:
                    prog("materials", fr, msg)
            if "ENEAS phrases:" in line:
                n_ph = int(line.rsplit(":", 1)[1] or 0)
            elif n_ph and (" -> (" in line or " ERROR " in line):
                done_ph += 1
                prog("materials", 0.13 + 0.65 * done_ph / n_ph,
                     f"ENEAS {done_ph}/{n_ph}: {line.strip().lstrip('-').split(' -> ')[0].split(' ERROR ')[0][:60]}")
    if p.wait():
        raise RuntimeError("materials on the splat failed (see the log)")
    return d / "splat.usdc"


def mesh(job, ply, cam, inp, out: Path, log: Path, prog, reuse_from: Path | None) -> Path:
    """Closed mesh, UVs, texture; with materials the --gs flow (its masks reused from the splat when there)."""
    src, dst = out / "mesh_src", out / "mesh"
    (src / "00").mkdir(parents=True, exist_ok=True)
    shutil.copy(ply, src / "00" / "00.ply"); shutil.copy(inp, src / "00" / "idol_input.png")
    shutil.copy(cam, src / "00" / "idol_camera.json")
    (out / "prompts.txt").write_text((job.get("prompt") or "") + "\n")
    m = job.get("mesh", {})
    prog("mesh", 0.05, ("direct mode" if m.get("path", "direct") == "direct" else "render path") + ": solid, UVs, texture")
    cmd = [PY, ROOT / "scripts/splat2usd.py", src / "00" / "00.ply", "--y-down", "--out", dst,
           "--faces", m.get("faces", 80000), "--tex", m.get("tex", 4096), "--smooth", m.get("smooth", 0.0),
           "--max-folds", m.get("max_folds", 300), "--render-tex", m.get("render_tex", 2048)]
    sh(cmd + (["--direct"] if m.get("path", "direct") == "direct" else []), log)
    if not job.get("outputs", {}).get("materials", True):
        prog("mesh", 1, "mesh ready")
        return next(dst.rglob("asset.usda"))
    prog("mesh", 0.3, "mesh ready; materials per texel from the splat's votes")
    if reuse_from is not None:                       # the splat's own ENEAS masks: no second segmentation
        mat = dst / "00" / "usd" / "mat"; mat.mkdir(parents=True, exist_ok=True)
        for f in ("masks_gs.npz",):
            if (reuse_from / "mat" / f).exists():
                shutil.copy(reuse_from / "mat" / f, mat / f)
        if (reuse_from / "mat" / "gs_orbit").exists():
            shutil.copytree(reuse_from / "mat" / "gs_orbit", mat / "gs_orbit", dirs_exist_ok=True)
        if (reuse_from / "materials.txt").exists():
            shutil.copy(reuse_from / "materials.txt", dst / "00" / "materials.txt")
    m = job.get("materials", {})          # the same mask rules as the splat (no overlap / coverage when turned off)
    sh([PY, ROOT / "scripts/flow_batch.py", src, dst, out / "prompts.txt", "--gs", "--reuse-masks"]
       + (["--no-coverage"] if m.get("coverage") is False else []) + (["--no-overlap"] if m.get("overlap") is False else []), log)
    prog("mesh", 1, "mesh with materials ready")
    return dst / "00" / "usd" / "asset.usda"


def turntable(job, asset: Path, mp4: Path, front_deg: float, log: Path, prog, stage: str):
    r = job.get("render", {})
    from ..athenea import turntable as tt
    prog(stage, 0.05, f"athenea {r.get('technique', 'raster')} turntable, {r.get('frames', 48)} frames")
    kw = {}
    if r.get("hdri"):
        kw["hdri"] = r["hdri"]
    tt(asset, mp4, frames=int(r.get("frames", 48)), size=int(r.get("size", 900)),
       technique=r.get("technique", "raster"), start_deg=front_deg, spp=int(r.get("spp", 96)),
       elev=float(r.get("elev", 15)), dist=float(r.get("dist", 0)), grey=float(r.get("grey", 0.8)), **kw)
    prog(stage, 1, mp4.name)


def run(job: dict) -> dict:
    out = Path(job["out"]); out.mkdir(parents=True, exist_ok=True)
    log = out / "log.txt"
    o = {"splat": True, "mesh": False, "materials": True, **job.get("outputs", {})}
    r = job.get("render", {})
    want_tt = r.get("turntable", True)
    stages = []
    if job["source"] == "prompt":
        stages.append(("picture", 5))
    if job["source"] in ("prompt", "image"):
        if job.get("detail", "all") != "off":
            stages.append(("detail", 5))
        stages.append(("splat", 20 * max(1, int(job.get("idol", {}).get("seeds", 1)))))
    if o["splat"]:
        stages.append(("materials", 45))
    if o["mesh"]:
        stages.append(("mesh", 25 if o["materials"] else 10))
    if want_tt:
        stages += [("render splat", 8)] * bool(o["splat"]) + [("render mesh", 8)] * bool(o["mesh"])
    prog = Progress(stages)
    res = {"name": job.get("name", out.name), "started": time.time(), "files": {}}
    t0 = time.time()
    if job["source"] == "prompt":
        image = text2image(job, out, log, prog)
    elif job["source"] == "image":
        image = out / ("input" + Path(job["image"]).suffix.lower())
        shutil.copy(job["image"], image)
    if job["source"] in ("prompt", "image"):
        image = detail(job, image, out, log, prog)
        res["files"]["detail"] = str(image)
        ply, cam, inp = idol(job, image, out, log, prog)
    else:
        src = out / "src"; src.mkdir(exist_ok=True)          # inside the job: the app can show and serve them
        ply, cam, inp = (Path(shutil.copy(job[k], src / n)) for k, n in
                         (("splat", "splat.ply"), ("camera", "idol_camera.json"), ("input", "idol_input.png")))
    res["files"]["input"] = str(inp); res["files"]["splat_ply"] = str(ply)
    c = json.loads(Path(cam).read_text())
    front = (180.0 - math.degrees(c["yaw"])) % 360               # the side the picture showed
    splat_dir = None
    if o["splat"] and not o["materials"]:            # the splat as captured, for athenea (not relit)
        from ..athenea_splat import write_splat_usd
        (out / "splat").mkdir(parents=True, exist_ok=True)
        write_splat_usd(ply, out / "splat" / "splat.usdc", y_down=True, radiance=True)
        prog("materials", 1, "splat written for athenea (as captured)")
        res["files"]["splat_usd"] = str(out / "splat" / "splat.usdc")
    elif o["splat"]:
        usd = splat_materials(job, ply, cam, inp, out, log, prog)
        splat_dir = usd.parent
        res["files"]["splat_usd"] = str(usd)
        res["files"]["splat_materials"] = str(splat_dir / "mat" / "splat_materials.ply")
        res["parts"] = (splat_dir / "materials.txt").read_text() if (splat_dir / "materials.txt").exists() else ""
    if o["mesh"]:
        asset = mesh(job, ply, cam, inp, out, log, prog, splat_dir if o["materials"] else None)
        res["files"]["mesh_usd"] = str(asset)
    if want_tt and o["splat"]:
        turntable(job, Path(res["files"]["splat_usd"]), out / "splat_turntable.mp4", front, log, prog, "render splat")
        res["files"]["splat_turntable"] = str(out / "splat_turntable.mp4")
    if want_tt and o["mesh"]:
        turntable(job, Path(res["files"]["mesh_usd"]), out / "mesh_turntable.mp4", front, log, prog, "render mesh")
        res["files"]["mesh_turntable"] = str(out / "mesh_turntable.mp4")
    res["seconds"] = round(time.time() - t0)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print("PROGRESS 100 done: finished", flush=True)
    return res


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text())
    try:
        run(job)
    except Exception as e:
        print(f"FAILED {e}", flush=True)
        sys.exit(1)
