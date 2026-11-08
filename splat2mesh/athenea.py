"""athenea (https://github.com/jesusluque/athenea, Apache-2.0) as splat2mesh's renderer, from Python.

The executable is found by splat2mesh.config (third_party/athenea build, configured path or $PATH); GPU jobs run
under the configured lock, if any. EXR outputs are converted here (numpy + OpenEXR), no oiiotool needed.

    render_splats(ply, views, out_dir, rotate_x180=True)      # splat views: views/NAME.exr/.png/_mask.png
    stage(scene, out_dir, turntable=96, eye=..., target=...)   # USD stage, one process for all frames
    stage(scene, out_dir, cameras=[(name, eye, target, up), ...])
    dome_scene(asset_usda, hdri, path)                         # the asset under an HDRI dome light
    write_splat_usd(...)  →  see splat2mesh/athenea_splat.py

CLI:
    python -m splat2mesh.athenea turntable ASSET.usda OUT.mp4 [--frames 96] [--size 1080] [--technique raster|rt]
           [--spp 96] [--elev 15] [--start-deg 0] [--dist 3.3] [--hdri FILE] [--grey 0.8]
    python -m splat2mesh.athenea splat-turntable SPLAT.ply OUT.mp4 [--frames 96] [--elev 15] [--y-down]
    python -m splat2mesh.athenea view ASSET.usda OUT.png MX,MY,MZ [--size 1024] [--spp 96] [--technique rt|raster]
    python -m splat2mesh.athenea render-splats SPLAT.ply CAMERAS.txt OUT_DIR [--y-down] [-- extra athenea flags]
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ROOT, athenea_bin, athenea_env, cfg, gpu_prefix

HDRI = Path(cfg("render.hdri")).expanduser()
FOCAL, APERTURE = 50.0, 24.576          # athenea's default camera: 50 mm on a 24.576 mm sensor


def _bin() -> str:
    b = athenea_bin()
    if b is None:
        raise RuntimeError("athenea not found: build third_party/athenea (docs/setup.md) or set athenea.bin")
    return str(b)


def run_script(cmds: list, lock: bool = True, log: Path | None = None) -> None:
    """Run commands as ONE job (one GPU lock for all of them). Each command is an argument list, or
    (cmd, "tolerate") to go on when it fails; plain shell strings are still accepted on Mac/Linux.
    On Windows the commands run one after another under splat2mesh's own GPU lock (no bash)."""
    if os.name == "nt":
        from .config import gpu_lock
        with gpu_lock(), (open(log, "w") if log else open(os.devnull, "w")) as lf:
            for c in cmds:
                tol = isinstance(c, tuple)
                argv = c[0] if tol else c
                r = subprocess.run(argv, env=athenea_env(), stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
                if r.returncode and not tol:
                    raise subprocess.CalledProcessError(r.returncode, argv)
                if r.returncode:
                    lf.write(f"failed ({r.returncode}): {argv[-1]}\n")
        return
    lines = []
    for c in cmds:
        if isinstance(c, str):
            lines.append(c)
        elif isinstance(c, tuple):
            lines.append(" ".join(_q(x) for x in c[0]) + " > /dev/null 2>&1 || echo 'failed: " + str(c[0][-1]) + "'")
        else:
            lines.append(" ".join(_q(x) for x in c))
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write("set -e\n" + "\n".join(lines) + "\n")
    cmd = (gpu_prefix() if lock else []) + ["bash", f.name]
    with open(log, "w") if log else open(os.devnull, "w") as lf:
        subprocess.run(cmd, check=True, env=athenea_env(), stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
    os.unlink(f.name)


def _q(x) -> str:
    return "'" + str(x).replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------------------------------- EXR → PNG
def read_exr(path) -> dict:
    import OpenEXR
    return {k: np.asarray(v.pixels, np.float32) for k, v in OpenEXR.File(str(path)).channels().items()}


def _srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def exr_to_png(exr, png, background=(0.5, 0.5, 0.5), mask_png=None, by_depth: bool = False, soft: float = 0.7):
    """Linear EXR → sRGB PNG over a flat background. Coverage from alpha, or from depth (by_depth: a stage under a
    dome counts the dome as covered, so the object is where depth > 0), softened over ~one pixel."""
    from scipy import ndimage as ndi
    ch = read_exr(exr)
    if "RGBA" in ch:                                   # OpenEXR 3 groups R,G,B,A as one channel
        ch["A"] = ch["RGBA"][..., 3]; rgb = ch["RGBA"][..., :3]
    elif "RGB" in ch:
        rgb = ch["RGB"]
    else:
        rgb = np.stack([ch["R"], ch["G"], ch["B"]], -1)
    if by_depth:
        z = ch.get("Z")
        obj = np.isfinite(z) & (z > 0)
        a = ndi.gaussian_filter(obj.astype(np.float32), soft) if soft else obj.astype(np.float32)
        col = _srgb(rgb)
    else:
        a = np.clip(ch.get("A", np.ones(rgb.shape[:2], np.float32)), 0, 1)
        col = _srgb(rgb / np.maximum(a, 1e-6)[..., None])          # athenea's splat RGB is premultiplied
        obj = a > 0.5
    out = col * a[..., None] + np.asarray(background, np.float32) * (1 - a[..., None])
    Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8)).save(png)
    if mask_png:
        Image.fromarray((np.clip(a * 2, 0, 1) * 255).astype(np.uint8)).save(mask_png)
    return obj


# ---------------------------------------------------------------------------------------------------- splats
def render_splats(ply, views, out_dir, size: int = 1024, rotate_x180: bool = False, extra=(), background=(0.5,) * 3,
                  log: Path | None = None) -> list[Path]:
    """views: [(name, eye, target, up)]. Writes out_dir/views/NAME.exr (+ .png on the background, _mask.png).
    All views in one locked job."""
    d = Path(out_dir) / "views"; d.mkdir(parents=True, exist_ok=True)
    b = _bin()
    rot = ["--rotate-x", "180"] if rotate_x180 else []
    lines = []
    for name, eye, tgt, up in views:
        args = ["render", "--splats", str(ply), "--eye", *map(str, eye), "--target", *map(str, tgt), "--up",
                *map(str, up), "--focal", str(FOCAL), "--size", f"{size}x{size}", *rot, *extra, "-o", str(d / f"{name}.exr")]
        lines.append(([str(b), *args], "tolerate"))
    run_script(lines, log=log)
    out = []
    for name, *_ in views:
        e = d / f"{name}.exr"
        if e.exists():
            exr_to_png(e, d / f"{name}.png", background, mask_png=d / f"{name}_mask.png")
            out.append(d / f"{name}.png")
    return out


def read_cameras_txt(path):
    views = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].split()
        if len(line) >= 7:
            v = list(map(float, line[1:]))
            views.append((line[0], v[0:3], v[3:6], v[6:9] if len(v) >= 9 else [0, 1, 0]))
    return views


def splat_frame(ply, y_down: bool, margin: float = 1.05):
    """Centre and camera distance that frame a splat (opaque gaussians, 0.5-99.5 percentile box)."""
    from .ply import read_vertex
    v = read_vertex(ply)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1) * (np.array([1, -1, -1]) if y_down else 1)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    p = xyz[op > 0.05]; lo, hi = np.percentile(p, 0.5, 0), np.percentile(p, 99.5, 0)
    C = (lo + hi) / 2; r = float(np.linalg.norm(hi - lo) / 2)
    return C, r, r / math.sin(math.atan(APERTURE / 2 / FOCAL)) * margin


def orbit(C, D, n: int, elev_deg: float = 15.0, start_deg: float = 0.0):
    """n views on a circle (eye = C + D·(sin a cos e, sin e, cos a cos e)), first at start_deg."""
    e = math.radians(elev_deg)
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n + math.radians(start_deg)
        eye = np.asarray(C) + D * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
        out.append((f"f_{i:04d}", eye.tolist(), list(map(float, C)), [0.0, 1.0, 0.0]))
    return out


# ---------------------------------------------------------------------------------------------------- stages
def asset_bounds(usd):
    from pxr import Usd, UsdGeom
    s = Usd.Stage.Open(str(usd))
    b = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(s.GetPseudoRoot()).ComputeAlignedRange()
    c = (b.GetMin() + b.GetMax()) / 2
    return np.array([c[0], c[1], c[2]]), float((b.GetMax() - b.GetMin()).GetLength() / 2)


def fit_distance(usd, elev_deg: float = 15.0, margin: float = 1.06) -> float:
    """The camera distance from the box centre at which the asset fills the frame at every turn of a turntable:
    the vertical extent seen at this elevation and the horizontal reach both fit the 50 mm lens, with the near
    side of the object (closer, so larger) taken into account."""
    from pxr import Usd, UsdGeom
    s = Usd.Stage.Open(str(usd))
    b = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(s.GetPseudoRoot()).ComputeAlignedRange()
    ex, ey, ez = ((b.GetMax() - b.GetMin()) / 2)
    rh, e = math.hypot(ex, ez), math.radians(elev_deg)
    t = APERTURE / 2 / FOCAL                                 # tan of the half field of view
    need = max(ey * math.cos(e) + rh * math.sin(e), rh) * margin
    return need / t + rh * math.cos(e)


def dome_scene(asset, hdri=HDRI, out_dir=None, intensity: float = 1.0) -> Path:
    """OUT_DIR/scene.usda: a stage referencing the asset under an HDRI dome light."""
    a = Path(asset).resolve()
    path = Path(out_dir or tempfile.mkdtemp()) / "scene.usda"
    path.write_text(f'''#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)
def Xform "World"
{{
    def "Asset" (prepend references = @{a}@) {{}}
    def DomeLight "Env"
    {{
        asset inputs:texture:file = @{Path(hdri).resolve()}@
        float inputs:intensity = {intensity}
    }}
}}
''')
    return path


def stage(scene, out_dir, *, turntable: int = 0, cameras=None, eye=None, target=None, up=(0, 1, 0),
          size: int = 1080, technique: str = "raster", spp: int = 96, bounces: int = 12, denoise: bool = True,
          log: Path | None = None) -> list[Path]:
    """Render a USD stage: turntable N views from eye about target (one process, one scene load), or a list of
    cameras [(name, eye, target, up)]. Returns the EXR paths in order."""
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    tech = ["--technique", technique] + (["--path-bounces", str(bounces), "--path-total", str(spp)]
                                         + (["--denoise"] if denoise else []) if technique == "rt" else [])
    args = [_bin(), "stage", str(scene), "--focal", str(FOCAL), "--size", f"{size}x{size}", *tech]
    if (turntable or cameras) and not _supports("--turntable"):
        # an athenea without --turntable/--cameras (the 0.3.0 release): one run a view, the scene loaded each time
        if turntable:
            c, e = np.asarray(target, float), np.asarray(eye, float) - np.asarray(target, float)
            cameras = []
            for i in range(turntable):                  # about +Y, as athenea's own --turntable turns
                a = 2 * math.pi * i / turntable; ca, sa = math.cos(a), math.sin(a)
                r = np.array([ca * e[0] + sa * e[2], e[1], -sa * e[0] + ca * e[2]])
                cameras.append((f"f_{i:04d}", (c + r).tolist(), list(target), list(up)))
        run_script([[str(x) for x in args] + ["--eye", *map(str, ce), "--target", *map(str, ct), "--up", *map(str, cu),
                                             "-o", str(d / f"{n}.exr")] for n, ce, ct, cu in cameras], log=log)
        return [d / f"{n}.exr" for n, *_ in cameras]
    if turntable:
        args += ["--eye", *map(str, eye), "--target", *map(str, target), "--up", *map(str, up),
                 "--turntable", str(turntable), "-o", str(d) + "/"]
        names = [f"f_{i:04d}" for i in range(turntable)]
    elif cameras:
        cf = d / "cameras.txt"
        cf.write_text("".join(f"{n} {' '.join(map(str, e))} {' '.join(map(str, t))} {' '.join(map(str, u))}\n"
                              for n, e, t, u in cameras))
        args += ["--cameras", str(cf), "-o", str(d) + "/"]
        names = [c[0] for c in cameras]
    else:
        args += ["--eye", *map(str, eye), "--target", *map(str, target), "--up", *map(str, up), "-o", str(d / "view.exr")]
        names = ["view"]
    run_script([[str(x) for x in args]], log=log)
    return [d / f"{n}.exr" for n in names]


def ffmpeg() -> str:
    """ffmpeg on PATH, else the one the imageio-ffmpeg package ships (Windows has none)."""
    w = shutil.which("ffmpeg")
    if w:
        return w
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


_HELP: dict = {}


def _supports(flag: str) -> bool:
    """Whether this athenea's stage command knows a flag (older releases lack --turntable / --cameras)."""
    if "stage" not in _HELP:
        try:
            _HELP["stage"] = subprocess.run([str(_bin()), "stage", "--help"], capture_output=True, text=True,
                                            env=athenea_env(), timeout=60).stdout
        except Exception:
            _HELP["stage"] = ""
    return flag in _HELP["stage"]


def encode(frames_glob_pattern: str, out_mp4, fps: int = 24):
    subprocess.run([ffmpeg(), "-loglevel", "error", "-y", "-framerate", str(fps), "-i", frames_glob_pattern,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out_mp4)],
                   check=True)


def turntable(asset, out_mp4, frames: int = 96, size: int = 1080, technique: str = "raster", spp: int = 96,
              elev: float = 15.0, start_deg: float = 0.0, dist: float = 0.0, hdri=HDRI, grey: float = 0.8):
    """Asset under the HDRI dome, turned about its bounding-box centre, composited on grey; PNG frames + MP4.
    dist: camera distance in bounding-sphere radii; 0 fits the asset to the frame (fit_distance)."""
    out_mp4 = Path(out_mp4); fr = out_mp4.with_name(out_mp4.stem + "_frames"); fr.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    C, R = asset_bounds(asset)
    eye = orbit(C, R * dist if dist > 0 else fit_distance(asset, elev), 1, elev, start_deg)[0][1]
    exrs = stage(dome_scene(asset, hdri, tmp), tmp / "exr", turntable=frames, eye=eye, target=C, size=size,
                 technique=technique, spp=spp, log=tmp / "athenea.log")
    for i, e in enumerate(exrs):
        exr_to_png(e, fr / f"f_{i:04d}.png", (grey,) * 3, by_depth=True)
    encode(str(fr / "f_%04d.png"), out_mp4)
    return fr


def view(asset, out_png, eye_in_radii=(1.3, 0.9, 3.3), size: int = 1024, technique: str = "rt", spp: int = 96,
         hdri=HDRI, grey: float = 0.8):
    """One view of an asset under the HDRI dome; the eye is given in units of the bounding-sphere radius from its
    centre. Composited on grey by depth."""
    tmp = Path(tempfile.mkdtemp())
    C, R = asset_bounds(asset)
    eye = (C + R * np.asarray(eye_in_radii, float)).tolist()
    e = stage(dome_scene(asset, hdri, tmp), tmp, eye=eye, target=C.tolist(), size=size, technique=technique, spp=spp,
              log=tmp / "athenea.log")[0]
    exr_to_png(e, out_png, (grey,) * 3, by_depth=True)
    return out_png


def splat_turntable(ply, out_mp4, frames: int = 96, elev: float = 15.0, y_down: bool = False, start_deg: float = 0.0,
                    size: int = 1024):
    out_mp4 = Path(out_mp4); fr = out_mp4.with_name(out_mp4.stem + "_frames")
    C, r, D = splat_frame(ply, y_down)
    render_splats(ply, orbit(C, D, frames, elev, start_deg), fr, size=size, rotate_x180=y_down)
    encode(str(fr / "views" / "f_%04d.png"), out_mp4)
    return fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("turntable"); t.add_argument("asset"); t.add_argument("out")
    for k, v in (("frames", 96), ("size", 1080), ("spp", 96)):
        t.add_argument(f"--{k}", type=int, default=v)
    t.add_argument("--technique", default="raster", choices=["raster", "rt"])
    for k, v in (("elev", 15.0), ("start-deg", 0.0), ("dist", 0.0), ("grey", 0.8)):
        t.add_argument(f"--{k}", type=float, default=v)
    t.add_argument("--hdri", default=str(HDRI))
    s = sp.add_parser("splat-turntable"); s.add_argument("ply"); s.add_argument("out")
    s.add_argument("--frames", type=int, default=96); s.add_argument("--elev", type=float, default=15.0)
    s.add_argument("--start-deg", type=float, default=0.0); s.add_argument("--y-down", action="store_true")
    v = sp.add_parser("view"); v.add_argument("asset"); v.add_argument("out")
    v.add_argument("eye", help="eye in bounding radii from the centre: X,Y,Z (commas: negative values are fine)")
    v.add_argument("--size", type=int, default=1024); v.add_argument("--spp", type=int, default=96)
    v.add_argument("--technique", default="rt", choices=["raster", "rt"]); v.add_argument("--hdri", default=str(HDRI))
    r = sp.add_parser("render-splats"); r.add_argument("ply"); r.add_argument("cameras"); r.add_argument("out")
    r.add_argument("--y-down", action="store_true"); r.add_argument("--size", type=int, default=1024)
    a, extra = ap.parse_known_args()
    if a.cmd == "turntable":
        print(turntable(a.asset, a.out, a.frames, a.size, a.technique, a.spp, a.elev, a.start_deg, a.dist, a.hdri, a.grey))
    elif a.cmd == "view":
        print(view(a.asset, a.out, [float(x) for x in a.eye.split(",")], a.size, a.technique, a.spp, a.hdri))
    elif a.cmd == "splat-turntable":
        print(splat_turntable(a.ply, a.out, a.frames, a.elev, a.y_down, a.start_deg))
    else:
        print(len(render_splats(a.ply, read_cameras_txt(a.cameras), a.out, a.size, a.y_down,
                                [x for x in extra if x != "--"])), "views")


if __name__ == "__main__":
    main()
