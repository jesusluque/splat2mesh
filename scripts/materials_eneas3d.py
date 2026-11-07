"""Semi-automatic 3D materials: description "material: part, part; ..." → ENEAS on a 16-view orbit of the textured
mesh (tracked across views) → masks back-projected to texels and voted → roughness/metallic maps in the USD.

    python scripts/materials_eneas3d.py ASSET_DIR INPUT_IMAGE "gold: the gold coin, the gold bell; porcelain: the body"
"""
import json, os, re, subprocess, sys
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
# --gs SPLAT.ply: ENEAS on athenea renders of the splat and votes per gaussian (splat2mesh/gs_materials.py); the
# texels then inherit the votes of their nearest gaussians. Masks are kept apart (other frames).
GS = Path(sys.argv[sys.argv.index("--gs") + 1]) if "--gs" in sys.argv else None
MASKS = "masks_gs.npz" if GS else "masks.npz"
# annotate on the orbit frame that looks most like the input image (the side the generator really saw)
def obj_crop(path, bg_dark):
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    fg = a.max(-1) > 10 if bg_dark else a.min(-1) < 245
    ys, xs = np.nonzero(fg)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((96, 96)), np.float32)
ref = obj_crop(image, bg_dark=True)                                 # Idol input: object on black
frames = sorted((orbit / "frames").glob("*.jpg"))
if GS:
    from splat2mesh.gs_materials import render_orbit_gs
    frames = render_orbit_gs(GS, asset / "mat" / "gs_orbit")
score = [-np.abs(obj_crop(f, bg_dark=False) - ref).mean() for f in frames]
# the splat orbit has its cameras: the frame nearest to the camera Idol predicted for the input picture is the
# side that was really seen (colour alone can pick the back of a figure whose silhouette looks the same)
_cam = image.parent / "idol_camera.json"
if GS and _cam.exists():
    import math as _m
    _c = json.loads(_cam.read_text()); _y, _p = _c["yaw"], _c["pitch"]
    _d = np.array([_m.cos(_p) * _m.sin(_y), _m.sin(_p), -_m.cos(_p) * _m.cos(_y)])     # Idol → USD (components.idol_view)
    _meta = json.loads((asset / "mat" / "gs_orbit" / "cameras.json").read_text())
    _eye = np.array([v["eye"] for v in _meta["views"]]) - np.array(_meta["centre"])
    _eye /= np.linalg.norm(_eye, axis=1, keepdims=True)
    score = list(_eye @ _d)
    print("annotation frame from Idol's camera", flush=True)
ann = frames[int(np.argmax(score))].name
ann2 = frames[(int(np.argmax(score)) + len(frames) // 2) % len(frames)].name   # opposite side, tracked too
print("annotation frame scores (best 3):", sorted(zip(score, [f.name for f in frames]))[-3:])
items = [(c, t.replace("'", "").replace('"', '').strip()) for c, t in parse_description(desc)]
# a transparent material is also searched by its own name ("the glass"): the description may not list every pane
# (a car's rear window), and the glass-openings step (splat2mesh/openings.py) uses the union of all glass masks
from splat2mesh.materials import _lib as _libm
SAME_MASK_IOU = 0.9       # two phrases of different materials with masks this alike: ENEAS did not separate them
GENERIC_MAX_COVER = 0.3   # the material-name phrase is for missed panes, not for most of the object
auto_items = []
for c in sorted({c for c, _ in items if _libm(c).get("transmission", 0) > 0}):
    if f"the {c}" not in [t for _, t in items]:
        items.append((c, f"the {c}")); auto_items.append((c, f"the {c}"))
tag = asset.resolve().as_posix().replace("/", "_")[-60:]
if LINUX:
    win = f"splat2mesh_eneas/jobs/{tag}"; wsl = f"$HOME/{win}"
else:
    win = f"C:/splat2mesh/eneas_jobs/{tag}"; wsl = "/mnt/c/" + win[3:]
reuse = "--reuse-masks" in sys.argv and (asset / "mat" / MASKS).exists()
run_items = items
if reuse:                     # only the phrases the saved masks do not have yet
    have = set(np.load(asset / "mat" / MASKS).files)
    run_items = [(c, t) for c, t in items if t not in have]
    if run_items:
        print("ENEAS for new phrases only:", [t for _, t in run_items])
import contextlib, time as _time
@contextlib.contextmanager
def _gpu_lock(path=Path.home() / ".cache/splat2mesh_eneas.lock"):
    """One ENEAS job at a time on the remote GPU (two on an 8 GB card thrash)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            path.mkdir(); break
        except FileExistsError:
            if _time.time() - path.stat().st_mtime > 3600:   # stale lock
                path.rmdir()
            _time.sleep(10)
    try:
        yield
    finally:
        path.rmdir()
if not reuse or run_items:
 with _gpu_lock():
   if LINUX:
     subprocess.run(["ssh", HOST, f"rm -rf ~/{win}; mkdir -p ~/{win}/frames"], check=False)
   else:
     subprocess.run(["ssh", HOST, f'rmdir /s /q "{win}" 2>nul & mkdir "{win}\\frames"'], check=False)
   subprocess.run(["scp", "-q", *map(str, frames), f"{HOST}:{win}/frames/"], check=True)
   subprocess.run(["scp", "-q", str(ROOT / "scripts/eneas_seq.py"), f"{HOST}:{win}/"], check=True)
   q = " ".join("'" + t.replace("'", "") + "'" for _, t in run_items)
   inner = f"~/.local/bin/uv run python {wsl}/eneas_seq.py {wsl}/frames {wsl}/masks.npz {ann},{ann2} {q}"
   cmd = (f"cd ~/splat2mesh_eneas/eneas && {inner}" if LINUX
          else f'wsl -d Ubuntu-24.04 -- bash -lc "cd ~/splat2mesh/eneas && {inner}"')
   print(subprocess.run(["ssh", HOST, cmd], capture_output=True, text=True, timeout=5400).stdout.replace("\x00", "")[-1500:])
   if reuse:                 # merge the new phrases into the saved masks
       subprocess.run(["scp", "-q", f"{HOST}:{win}/masks.npz", str(asset / "mat" / "masks_new.npz")], check=True)
       merged = {**dict(np.load(asset / "mat" / MASKS)), **dict(np.load(asset / "mat" / "masks_new.npz"))}
       np.savez_compressed(asset / "mat" / MASKS, **merged); (asset / "mat" / "masks_new.npz").unlink()
   else:
       subprocess.run(["scp", "-q", f"{HOST}:{win}/masks.npz", str(asset / "mat" / MASKS)], check=True)
z = np.load(asset / "mat" / MASKS)
def _fit(m, n):            # ENEAS may return fewer frames than sent: pad with empty views
    if m.shape[0] >= n:
        return m[:n]
    return np.concatenate([m, np.zeros((n - m.shape[0],) + m.shape[1:], bool)])
masks = {f"{cls}|{ph}": _fit(z[ph], len(frames)) for cls, ph in items if ph in z.files}
# overlapping masks (a phrase grounded on too much: "the body" = the whole figurine, "the crystal" = dragon and
# crystal): resolved per pixel by reliable specific masks, then colour clusters named by DeepSeek (splat2mesh/overlap)
if "--no-overlap" not in sys.argv and len(masks) > 1:
    try:
        from splat2mesh.overlap import resolve, verify
        sys.path.insert(0, str(ROOT / "scripts"))
        from gen_views import ds as _ds, img_url as _img_url
        _ask = lambda sheet, q: _ds([{"type": "image_url", "image_url": {"url": _img_url(sheet)}}, {"type": "text", "text": q}])
        # the material-name phrases ("the crystal") are only for panes the description missed: they take part
        # after the listed parts are settled (rule below), not in the vote, where they grab look-alike colours
        _auto = {f"{c}|{t}" for c, t in auto_items}
        _listed = {k: v for k, v in masks.items() if k not in _auto}
        _ai = frames.index(next(f for f in frames if f.name == ann))
        _parts = {k: k.split("|", 1)[1] for k in _listed}
        # each mask checked first: a phrase ENEAS grounded on another part is emptied (its class stays listed)
        _listed, _vans, _vsheet = verify(_listed, frames, _ai, image, _parts, _ask)
        print("mask check:", _vans, flush=True)
        if _vsheet is not None:
            _vsheet.save(asset / "mat" / "mask_check.jpg", quality=85)
        _res, _orep, _osheet = resolve(_listed, frames, _ai, image, _parts, _ask)
        masks = {k: (_res[k] if k in _res else v) for k, v in masks.items()}
        print("overlap resolved:", _orep, flush=True)
        if _osheet is not None:
            _osheet.save(asset / "mat" / "overlap_clusters.jpg", quality=85)
    except Exception as e:                     # DeepSeek down etc.: keep ENEAS masks and the drop rules below
        print("overlap resolution skipped:", e, flush=True)
# grounding sanity: a secondary part (not the first, main body) that covers most of the object is a mislocated
# phrase ("the side inlay" → the whole box) — drop it
obj = np.stack([np.asarray(Image.open(f).convert("L"), np.float32) < 250 for f in frames])
first = f"{items[0][0]}|{items[0][1]}" if items else None
for k in list(masks):
    if k == first:
        continue
    cover = (masks[k] & obj).sum() / max(obj.sum(), 1)
    words = set(re.findall(r"[a-z]+", k.split("|", 1)[1].lower()))     # whole words: "chainmail" is not a chain
    thin = bool(words & {"rim", "edge", "line", "wire", "trim", "hinge", "knob", "filament",
                         "band", "stripe", "cap", "cork", "wick", "chain", "eyelet"})
    if cover > (0.15 if thin else 0.75):
        print("dropped (covers", round(float(cover), 2), "of the object):", k)
        masks[k] = np.zeros_like(masks[k])      # keep the material in the list so colour priors can still claim it
# ENEAS could not tell two phrases of different materials apart (near-identical masks: "the case" and "the
# crystal" of a watch): the description lists the main parts first, so the later one is dropped (colour priors
# can still claim its class)
order = [f"{c}|{t}" for c, t in items]
nz = [k for k in order if k in masks and masks[k].any()]
for ia, a_ in enumerate(nz):
    for b_ in nz[ia + 1:]:
        if a_.split("|")[0] == b_.split("|")[0] or not masks[b_].any() or not masks[a_].any():
            continue
        iou = (masks[a_] & masks[b_]).sum() / max((masks[a_] | masks[b_]).sum(), 1)
        if iou > SAME_MASK_IOU:
            print(f"dropped (same mask as {a_}, IoU {iou:.2f}):", b_)
            masks[b_] = np.zeros_like(masks[b_])
# the material's own name ("the glass") only looks for panes the description did not list: it fills what no
# described part claims, and is dropped when it spreads over much of the object (a reflection-wide guess)
for c, t in auto_items:
    k = f"{c}|{t}"
    if k not in masks:
        continue
    others = np.zeros_like(masks[k])
    for o in masks:
        if o != k:
            others |= masks[o]
    cover = (masks[k] & obj).sum() / max(obj.sum(), 1)
    if cover > GENERIC_MAX_COVER:
        print(f"dropped (generic phrase covers {cover:.2f} of the object):", k)
        masks[k] = np.zeros_like(masks[k])
    else:
        masks[k] = masks[k] & ~others
# the masks as decided here (after the drops) are what later steps (glass openings) must use, not the raw ones
np.savez_compressed(asset / "mat" / ("masks_used_gs.npz" if GS else "masks_used.npz"), **{k.split("|", 1)[1]: v for k, v in masks.items()})
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
if GS:
    from splat2mesh.gs_materials import vote_gaussians, texel_votes, write_ply
    votes_g, _ = vote_gaussians(GS, asset / "mat" / "gs_orbit", masks)
    R = Image.open(asset / "basecolor.png").size[0]
    print("splat materials:", write_ply(GS, asset / "mat" / "splat_materials.ply", votes_g, list(masks)))
    print(from_view_masks(spec, orbit, masks, votes=texel_votes(asset, GS, votes_g, R)))
else:
    print(from_view_masks(spec, orbit, masks))
print(apply_texel_classes(spec))
print('parts', export_parts(asset), export_parts(asset, 'asset_materials.usda', group='material'))
