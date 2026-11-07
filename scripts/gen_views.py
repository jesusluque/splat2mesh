"""The object's hidden sides from one photo, with the models we already run (like Tripo's "image to multiview"),
then Idol again from the photo plus those views (scripts/idol_mv.py).

1. The single-view splat (Idol) is rendered by athenea from Idol's own camera turned by +90, +180 and +270 degrees of
   yaw (same pitch): left, back, right. These renders are blurry where Idol guessed.
   Their colours are set to the photo's (Lab mean and spread of the object's pixels): guessed sides come out pale.
2. Iris-3B Turbo img2img (SDEdit) turns each render into a clean product photo of that side, prompted with the
   object's description (deepseek-flash, from the photo) and the side; outside the render's mask the picture is
   white.
3. Gates per view: the silhouette must stay the render's (IoU >= SIL_MIN: Iris may not change the shape), and
   deepseek-flash must see the same object as the photo from that side (score >= SAME_MIN). The best start level
   that passes is kept; a side with none is dropped.
4. idol_mv.py on the CUDA host: the photo (weight 1, free camera) and the kept views (weight VIEW_W, camera
   direction known: the one they were rendered from).

    python scripts/gen_views.py PHOTO SPLAT.ply IDOL_CAMERA.json WORK_DIR [--gaussians N]
Writes WORK_DIR/views/{left,back,right}.png, WORK_DIR/gen_views.json and WORK_DIR/<photo stem>_mv.ply.
"""
import argparse, base64, io, json, math, shutil, subprocess, sys, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / ".claude/skills/research-deepseek/scripts"))
SHARED = Path.home() / "luc/athenea-gpu/shared.sh"
IRIS_PY = ROOT / "third_party/iris-3b/.venv/bin/python"
IXWIN, IX_PY = "ixwin-tb", r"C:\splat2mesh\venv\Scripts\python"
IX_DIR = "C:/splat2mesh/mv"
# Idol yaw offsets of the object's sides: USD azimuth = 180 deg - Idol yaw, so the camera that walks to the viewer's
# right (azimuth +90, the object's left side) is Idol yaw -90
SIDES = {"left": -90.0, "back": -180.0, "right": -270.0}
SIGMAS = ["0.625", "0.833"]
SIL_MIN, SAME_MIN, VIEW_W = 0.9, 3, 0.5
# the turned views are not 'left / back / right' of the object when the photo is three-quarter: no side is named
PROMPT = ("Studio product photo of {desc}, the whole object centred, isolated on a plain white background, soft even "
          "studio light, sharp focus")


def ds(content):
    import dsr
    body = {"model": "deepseek-flash", "temperature": 0.1, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(dsr.API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + dsr.key()})
    return json.loads(json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"])


def img_url(im):
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def describe(photo):
    j = ds([{"type": "image_url", "image_url": {"url": img_url(Image.open(photo))}},
            {"type": "text", "text": "Describe the one object in this photo for an image generator, in one English noun "
                                     "phrase of at most 25 words: what it is, its shape, materials and colours "
                                     "(no background, no camera words). Answer JSON: {\"desc\": ...}"}])
    return str(j.get("desc", "the object")).strip().rstrip(".")


def same_object(photo, cand, side):
    S = 512; sheet = Image.new("RGB", (2 * S, S), "white")
    for i, p in enumerate([photo, cand]):
        t = Image.open(p).convert("RGB"); t.thumbnail((S, S)); sheet.paste(t, (i * S + (S - t.width) // 2, (S - t.height) // 2))
    j = ds([{"type": "image_url", "image_url": {"url": img_url(sheet)}},
            {"type": "text", "text": f"Left: a photo of an object. Right: a picture meant to show THE SAME object seen from "
                                     f"another angle (turned {side} degrees around it). Score 1-5: 5 = clearly the same "
                                     f"object (shape, proportions, materials, colours, parts) from another angle; 3 = "
                                     f"same object, small differences; 1 = another object, or broken. Answer JSON: "
                                     f"{{\"score\": n, \"why\": short}}"}])
    return float(j.get("score", 0)), j.get("why")


def cameras(ply, cam_json, out):
    """athenea cameras (USD frame, splat turned X 180 like the pipeline) on Idol's camera turned in yaw."""
    from splat2mesh.ply import read_vertex
    v = read_vertex(ply)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1) * np.array([1, -1, -1])
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    p = xyz[op > 0.05]; lo, hi = np.percentile(p, 0.5, 0), np.percentile(p, 99.5, 0)
    C = (lo + hi) / 2; r = float(np.linalg.norm(hi - lo) / 2)
    D = r / math.sin(math.atan(24.576 / 2 / 50.0)) * 1.08
    c = json.loads(Path(cam_json).read_text())
    lines, known = [], {}
    for side, dy in [("front", 0.0)] + list(SIDES.items()):
        y, pt = c["yaw"] + math.radians(dy), c["pitch"]
        dx, dyy, dz = math.cos(pt) * math.sin(y), math.cos(pt) * math.cos(y), math.sin(pt)
        eye = C + D * np.array([dx, dz, -dyy])                     # Idol direction → USD (components.idol_view)
        lines.append(f"{side} {eye[0]} {eye[1]} {eye[2]} {C[0]} {C[1]} {C[2]} 0 1 0")
        known[f"{side}.png"] = {"yaw": y, "pitch": pt}
    out.mkdir(parents=True, exist_ok=True); (out / "views").mkdir(exist_ok=True)
    (out / "cameras.txt").write_text("\n".join(lines) + "\n")
    return known


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def sil(im):
    """Object on a pure white background (pale leather or porcelain is not background: only near 255)."""
    a = np.asarray(Image.open(im).convert("RGB"), np.float32)
    return ndi.binary_fill_holes(ndi.binary_opening(a.min(-1) < 250, iterations=2))


def match_colour(rgb, mask, ref, ref_mask):
    """Mean and spread of each Lab channel of the object's pixels set to the photo's: a guessed side of the splat is
    often washed out, and the hidden side is the same material as the seen one."""
    from skimage import color
    lab = color.rgb2lab(np.asarray(rgb, np.float32) / 255); rl = color.rgb2lab(np.asarray(ref, np.float32) / 255)
    a, b = lab[mask], rl[ref_mask]
    lab[mask] = (a - a.mean(0)) / np.maximum(a.std(0), 1e-3) * b.std(0) + b.mean(0)
    return Image.fromarray((np.clip(color.lab2rgb(lab), 0, 1) * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo"); ap.add_argument("ply"); ap.add_argument("camera"); ap.add_argument("work")
    ap.add_argument("--gaussians", type=int, default=262144)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--no-idol", action="store_true", help="stop after the views (Idol later, e.g. when the GPU is busy)")
    a = ap.parse_args()
    photo, work = Path(a.photo), Path(a.work); rd = work / "views_render"
    known = cameras(a.ply, a.camera, rd)
    subprocess.run(["bash", str(SHARED), "bash", str(ROOT / "scripts/render_athenea.sh"), a.ply, str(rd),
                    "--rotate-x", "180", "--no-antialias"], check=True, capture_output=True, stdin=subprocess.DEVNULL)
    rep = {"known": known, "views": {}}
    # check: the render from Idol's own camera is the photo's view (same outline after cropping)
    def crop_sil(m):
        ys, xs = np.nonzero(m); b = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return np.asarray(Image.fromarray(b.astype(np.uint8) * 255).resize((256, 256))) > 127
    fm = np.asarray(Image.open(rd / "views/front_mask.png").convert("L")) > 127
    inp = work / "idol_input.png"
    if inp.exists():
        im = np.asarray(Image.open(inp).convert("RGB")); rep["front_check_iou"] = float(iou(crop_sil(fm), crop_sil(im.max(-1) > 10)))
        print("front check IoU", round(rep["front_check_iou"], 3), flush=True)
    # renders on white for Iris
    src = work / "views_src"; src.mkdir(parents=True, exist_ok=True)
    masks = {}
    # the photo's object pixels, as Idol cut them out (black background)
    ref = Image.open(inp if inp.exists() else photo).convert("RGB")
    ref_m = ndi.binary_erosion(np.asarray(ref).max(-1) > 10, iterations=3)
    for side in SIDES:
        rgb = Image.open(rd / f"views/{side}.png").convert("RGB")
        m = np.asarray(Image.open(rd / f"views/{side}_mask.png").convert("L")) > 127
        masks[side] = m
        rgb = match_colour(rgb, m, ref, ref_m)
        w = Image.new("RGB", rgb.size, "white"); w.paste(rgb, mask=Image.fromarray(m.astype(np.uint8) * 255))
        w.save(src / f"{side}.png")
    desc = describe(photo); rep["desc"] = desc; print("desc:", desc, flush=True)
    gen = work / "views_iris"
    subprocess.run(["bash", str(SHARED), str(IRIS_PY), str(ROOT / "scripts/iris_parts.py"), str(gen),
                    *[f"{s}=-={src / (s + '.png')}" for s in SIDES], "--start-sigma", *SIGMAS,
                    "--prompt", PROMPT, "--desc", desc], check=True, stdin=subprocess.DEVNULL)
    vd = work / "views"; vd.mkdir(exist_ok=True)
    kept = []
    for side in SIDES:
        best = None
        for c in sorted(gen.glob(f"{side}_s*.png")):
            im = Image.open(c).convert("RGB").resize(masks[side].shape[::-1])
            m = ndi.binary_dilation(masks[side], iterations=3)
            out = Image.new("RGB", im.size, "white"); out.paste(im, mask=Image.fromarray(m.astype(np.uint8) * 255))
            tmp = gen / f"{c.stem}_masked.png"; out.save(tmp)
            j = float(iou(sil(tmp), masks[side]))
            sc, why = same_object(photo, tmp, f"{abs(SIDES[side]):g}") if j >= SIL_MIN else (0.0, "silhouette changed")
            rep["views"].setdefault(side, []).append({"file": c.name, "sil_iou": round(j, 3), "same": sc, "why": why})
            print(side, c.name, round(j, 3), sc, why, flush=True)
            if j >= SIL_MIN and sc >= SAME_MIN and (best is None or (sc, j) > best[:2]):
                best = (sc, j, tmp)
        if best:
            # RGBA: Idol uses this alpha instead of BiRefNet's
            rgba = Image.open(best[2]).convert("RGBA"); rgba.putalpha(Image.fromarray((sil(best[2])).astype(np.uint8) * 255))
            rgba.save(vd / f"{side}.png"); kept.append(side)
    rep["kept"] = kept
    (work / "gen_views.json").write_text(json.dumps(rep, indent=1))
    if not kept:
        print("no generated view passed: single-view result kept", flush=True); return
    if a.no_idol:
        return
    run_mv(photo, [vd / f"{s}.png" for s in kept], {f"{s}.png": known[f"{s}.png"] for s in kept}, work, a.gaussians,
           a.mode or ("real" if photo.suffix.lower() in (".jpg", ".jpeg") else "synthetic"))


def run_mv(front, views, known, work, gaussians, mode, weights=None, yaw_offsets=None):
    """idol_mv.py on ixwin: front picture (free camera) + views (known cameras when given)."""
    job = f"{IX_DIR}/{work.name}"
    subprocess.run(["ssh", IXWIN, f'mkdir "{job}" 2>nul & exit 0'], check=True)
    subprocess.run(["scp", "-q", str(ROOT / "scripts/idol_mv.py"), f"{IXWIN}:C:/splat2mesh/idol-sdk/idol_mv.py"], check=True)
    subprocess.run(["scp", "-q", str(front), *map(str, views), f"{IXWIN}:{job}/"], check=True)
    cam = work / "mv_known.json"; cam.write_text(json.dumps(known))
    subprocess.run(["scp", "-q", str(cam), f"{IXWIN}:{job}/mv_known.json"], check=True)
    w = weights or [1.0] + [VIEW_W if known else 1.0] * len(views)
    imgs = " ".join(f"{job}/{Path(p).name}" for p in [front, *views])
    r = subprocess.run(["ssh", IXWIN, rf"cd /d C:\splat2mesh\idol-sdk && set ATTN_BACKEND=sdpa&& {IX_PY} idol_mv.py --images {imgs} "
                        rf"--weights {' '.join(map(str, w))} --cameras {job}/mv_known.json --out {job}/out --mode {mode} "
                        rf"--num_gaussians {gaussians}" + (f" --yaw-offsets {' '.join(map(str, yaw_offsets))}" if yaw_offsets else "")],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout[-1500:] + r.stderr[-1500:])
    st = Path(front).stem
    for f in (f"{st}.ply", f"{st}_input.png", f"{st}_camera.json", f"{st}_cameras.json", f"{st}_mv.json"):
        subprocess.run(["scp", "-q", f"{IXWIN}:{job}/out/{f}", str(work / f.replace(st, st + '_mv', 1))], check=True)
    return work / f"{st}_mv.ply"


if __name__ == "__main__":
    main()
