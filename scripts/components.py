"""Per-component pipeline: one reference picture of the whole object + the list of its components → one closed mesh
per component, each generated on its own (Idol) and placed where it is in the whole object.

    python scripts/components.py REFERENCE.png WHOLE_ASSET_DIR OUT_DIR "the bulb=glass" "the screw base=brass" \
           "the base=brass: the thread; black glass: the insulator"

Each component is "phrase=materials": one library material (the component is all of it) or a material description
"material: part, part; material: part" (the component gets its own material stage: ENEAS on its orbit, per-texel
maps, like any asset).

WHOLE_ASSET_DIR is the object made the usual way (splat2usd + materials_eneas3d): it gives the layout (where each
part is, at what scale) through its per-texel material votes. Steps, each cached in OUT_DIR:
 1. ENEAS segments every component phrase on the reference picture itself (ixwin).
 2. Each component is cut out of the picture (its mask on white) → component image.
 3. Idol + direct meshing per component image (splat2usd --direct) → a closed mesh with its own UVs and texture.
 4. The component mesh is placed in its box from the reference mask (the whole mesh faces the picture), then ICP
    with scale (open3d). Components ENEAS / Idol could not produce keep the whole mesh's faces for that phrase.
 5. One USD: /Asset/Components/<part> (mesh + material from the library: metal F0, glass transmission, ...).
"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from splat2mesh.materials import parse_description, _lib

HOST = "ixwin-tb"
MIN_MASK = 0.004          # component mask below this share of the picture: too small for its own generation
PAD = 0.12                # margin around the component cut-out


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower().replace("the ", "")).strip("_")


def eneas_on_reference(ref: Path, phrases, out_npz: Path):
    """ENEAS on a one-frame 'sequence' (the reference picture), one mask per phrase."""
    if out_npz.exists() and set(np.load(out_npz).files) >= set(phrases):
        return
    win = "C:/splat2mesh/eneas_jobs/components_" + ref.stem; wsl = "/mnt/c/" + win[3:]
    fr = out_npz.parent / "ref_frames"; fr.mkdir(exist_ok=True)
    Image.open(ref).convert("RGB").save(fr / "frame_0000.jpg", quality=95)
    subprocess.run(["ssh", HOST, f'rmdir /s /q "{win}" 2>nul & mkdir "{win}\\frames"'], check=False)
    subprocess.run(["scp", "-q", str(fr / "frame_0000.jpg"), f"{HOST}:{win}/frames/"], check=True)
    subprocess.run(["scp", "-q", str(ROOT / "scripts/eneas_seq.py"), f"{HOST}:{win}/"], check=True)
    q = " ".join("'" + p.replace("'", "") + "'" for p in phrases)
    inner = f"~/.local/bin/uv run python {wsl}/eneas_seq.py {wsl}/frames {wsl}/masks.npz frame_0000.jpg {q}"
    print(subprocess.run(["ssh", HOST, f'wsl -d Ubuntu-24.04 -- bash -lc "cd ~/splat2mesh/eneas && {inner}"'],
                         capture_output=True, text=True, timeout=3600).stdout.replace("\x00", "")[-800:])
    subprocess.run(["scp", "-q", f"{HOST}:{win}/masks.npz", str(out_npz)], check=True)


def cut_out(ref: Path, mask: np.ndarray, out_png: Path):
    """The component on white, square, with a margin (Idol's own matting takes it from there)."""
    img = np.asarray(Image.open(ref).convert("RGB")).copy()
    if mask.shape != img.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(img.shape[1::-1])) > 127
    from scipy import ndimage as ndi
    m = ndi.binary_dilation(mask, iterations=2)
    img[~m] = 255
    ys, xs = np.nonzero(m)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    h = max(np.ptp(ys), np.ptp(xs)) * (1 + 2 * PAD) / 2
    canvas = Image.new("RGB", (int(2 * h), int(2 * h)), (255, 255, 255))
    canvas.paste(Image.fromarray(img), (int(h - cx), int(h - cy)))
    canvas.resize((1024, 1024), Image.LANCZOS).save(out_png)


def face_labels(asset: Path):
    """Per face of the whole asset: the index of the phrase its texels voted for (majority at the face's UV centre)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(asset / "asset.usda"))
    m = UsdGeom.Mesh(st.GetPrimAtPath("/Asset/Geom"))
    V = np.asarray(m.GetPointsAttr().Get(), np.float64)
    F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
    uv = np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar("st").Get(), np.float64)
    cls = np.load(asset / "mat" / "texel_class.npy")
    R = cls.shape[0]
    c = uv[F].mean(1)
    px = np.clip((c[:, 0] * R).astype(int), 0, R - 1); py = np.clip(((1 - c[:, 1]) * R).astype(int), 0, R - 1)
    lab = cls[py, px]
    if (lab < 0).mean() > 0.5:          # texel_class stored bottom-up
        lab = cls[R - 1 - py, px]
    return V, F, lab


USE_IRIS = __import__("os").environ.get("COMP_IRIS", "1") == "1"
IRIS_SIGMAS = ["0.938", "0.833", "0.625"]


def iris_picture(ref, phrase, material, cut, others, out):
    """The part alone, complete, on white: Iris-3B Turbo img2img from the cut-out at three redraw levels, and
    deepseek-flash picks the one that shows only that part with the reference's shape and material (or none)."""
    d = out / "iris"; d.mkdir(exist_ok=True)
    slug_ = re.sub(r"[^a-z0-9]+", "_", re.sub(r"^the\s+", "", phrase.strip().lower())).strip("_")
    cands = [d / f"{slug_}_s{s}.png" for s in IRIS_SIGMAS]
    if not all(c.exists() for c in cands):
        item = f"{phrase}={material}={cut}"
        subprocess.run(["bash", str(Path.home() / "luc/athenea-gpu/shared.sh"), str(ROOT / "third_party/iris-3b/.venv/bin/python"),
                        str(ROOT / "scripts/iris_parts.py"), str(d), item, "--start-sigma", *IRIS_SIGMAS], check=False)
    cands = [c for c in cands if c.exists()]
    if not cands:
        return None
    sys.path.insert(0, str(ROOT / "scripts"))
    import pick_parts
    try:
        r = pick_parts.pick(ref, [(phrase, cut, cands)])[phrase]
    except Exception as e:
        print("pick failed", e, flush=True); return None
    return r["chosen"]


REVOLUTION = set()
LATHE_H, LATHE_T, LATHE_SEG = 64, 72, 128


def lathe(V, F, UV, tex, tex_out):
    """A body of revolution is round all the way, also where the picture hid it (a wheel behind the body): the
    generated mesh becomes a turned surface. Axis: the principal direction whose spread differs from the other two.
    Profile r(h): per height band, the median over angle sectors of the outermost radius (hidden or invented sectors
    are outvoted). Colour: per (height, angle) texel, the generated texture at the nearest original point."""
    from scipy.spatial import cKDTree
    c = V.mean(0); X = V - c
    w, E = np.linalg.eigh(np.cov(X.T))                       # ascending
    ax = E[:, 0] if (w[1] - w[0]) > (w[2] - w[1]) else E[:, 2]
    u = np.cross(ax, [1, 0, 0] if abs(ax[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); v = np.cross(ax, u)
    h = X @ ax; r = np.hypot(X @ u, X @ v); th = np.arctan2(X @ v, X @ u)
    hb = np.clip(((h - h.min()) / max(np.ptp(h), 1e-9) * LATHE_H).astype(int), 0, LATHE_H - 1)
    tb = np.clip(((th + np.pi) / (2 * np.pi) * LATHE_T).astype(int), 0, LATHE_T - 1)
    rmax = np.full((LATHE_H, LATHE_T), np.nan)
    np.fmax.at(rmax, (hb, tb), r)
    prof = np.nanmedian(rmax, 1)
    prof = np.interp(np.arange(LATHE_H), np.nonzero(~np.isnan(prof))[0], prof[~np.isnan(prof)])
    prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, "valid")
    hs = np.linspace(h.min(), h.max(), LATHE_H)
    ang = np.linspace(-np.pi, np.pi, LATHE_SEG, endpoint=False)
    ring = lambda i: c + hs[i] * ax + prof[i] * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    P = np.concatenate([ring(i) for i in range(LATHE_H)] + [[c + hs[0] * ax], [c + hs[-1] * ax]])
    Fs = []
    for i in range(LATHE_H - 1):
        for j in range(LATHE_SEG):
            a, b = i * LATHE_SEG + j, i * LATHE_SEG + (j + 1) % LATHE_SEG
            Fs += [[a, b, b + LATHE_SEG], [a, b + LATHE_SEG, a + LATHE_SEG]]
    c0, c1 = len(P) - 2, len(P) - 1; top = (LATHE_H - 1) * LATHE_SEG
    for j in range(LATHE_SEG):
        Fs += [[c0, (j + 1) % LATHE_SEG, j], [c1, top + j, top + (j + 1) % LATHE_SEG]]
    Fn = np.array(Fs)
    # orient outwards
    import trimesh
    tm = trimesh.Trimesh(P, Fn, process=False)
    if tm.volume < 0:
        Fn = Fn[:, ::-1]
    # texture: (angle, height) grid sampled from the generated texture at the nearest original vertex
    uvn = np.r_[np.c_[np.tile(np.arange(LATHE_SEG) / LATHE_SEG, LATHE_H), np.repeat(hs - hs.min(), LATHE_SEG) / max(np.ptp(hs), 1e-9)],
                [[0.5, 0.0], [0.5, 1.0]]]
    tree = cKDTree(V)
    R = 512
    gy, gx = np.mgrid[0:R, 0:R]
    gh = h.min() + (1 - (gy + 0.5) / R) * np.ptp(h); ga = -np.pi + (gx + 0.5) / R * 2 * np.pi
    gi = np.interp(gh, hs, prof)
    pts = c + gh[..., None] * ax + gi[..., None] * (np.cos(ga)[..., None] * u + np.sin(ga)[..., None] * v)
    _, nn = tree.query(pts.reshape(-1, 3))
    src = np.asarray(Image.open(tex).convert("RGB")); S_ = src.shape[0]
    suv = UV[nn] if len(UV) == len(V) else np.zeros((len(nn), 2))
    px = np.clip((suv[:, 0] * S_).astype(int), 0, S_ - 1); py = np.clip(((1 - suv[:, 1]) * S_).astype(int), 0, S_ - 1)
    Image.fromarray(src[py, px].reshape(R, R, 3)).save(tex_out)
    return P, Fn, uvn, tex_out


MAIN_PICTURE = {}
SAME_MASK_IOU = 0.9        # two phrases ENEAS answered with the same mask: one component
MAIN_COVER = 0.6           # the main part's mask covers less than this of what is left: it takes the rest


def silhouette(ref: Path, shape):
    return silhouette_arr(np.asarray(Image.open(ref).convert("RGB").resize(shape[::-1])))


def silhouette_arr(img):
    img = np.asarray(img).astype(np.float32)
    bg = np.median(np.r_[img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3), img[:, :4].reshape(-1, 3),
                         img[:, -4:].reshape(-1, 3)], 0)
    sil = ndi.binary_opening(np.abs(img - bg).max(-1) > 18, iterations=2)
    lab_, n = ndi.label(sil)
    return lab_ == (np.argmax(ndi.sum(sil, lab_, range(1, n + 1))) + 1) if n else sil


def merge_and_complete(ref, comps, z):
    """1. Phrases with the same mask are one component (ENEAS finds the same wheel for 'the front tire' and 'the
    front wheel rim'): the first keeps it; differing materials make it a multi-material component.
    2. The first part (the biggest, the main body) takes the object's silhouette minus the other parts when its own
    mask covers only a piece of that (ENEAS found the bonnet for 'the car body')."""
    keep = []
    for ph, mats in comps:
        if ph not in z:
            keep.append([ph, mats]); continue
        a = z[ph][0]
        for k in keep:
            if k[0] in z:
                b = z[k[0]][0]
                if (a & b).sum() > SAME_MASK_IOU * max((a | b).sum(), 1):
                    m0 = k[1].split(":")[0].strip() if ":" not in k[1] else None
                    mine = mats.replace("@rev", "")
                    if slug(k[0]) in REVOLUTION:
                        pass                  # a turned part stays one material: its texture keeps the rest
                    elif m0 is not None and mine != m0.replace("@rev", ""):
                        k[1] = f"{m0.replace('@rev', '')}: {k[0]}; {mine}: {ph}"
                    elif m0 is None and mine not in k[1]:
                        k[1] += f"; {mine}: {ph}"
                    print("same mask:", ph, "→", k[0], flush=True)
                    break
        else:
            keep.append([ph, mats])
    comps = [tuple(k) for k in keep]
    if comps and comps[0][0] in z:
        main = comps[0][0]; m = z[main][0]
        rest = silhouette(ref, m.shape)
        for ph, _ in comps[1:]:
            if ph in z:
                rest &= ~z[ph][0]
        if (m & rest).sum() < MAIN_COVER * rest.sum():
            z[main] = (m | rest)[None]
            # its picture is the whole silhouette: holes where other parts sit would become white discs and walls
            MAIN_PICTURE[slug(main)] = silhouette(ref, m.shape)
            print("main part takes the rest of the silhouette:", main, round(float(m.mean()), 3), "→",
                  round(float((m | rest).mean()), 3), flush=True)
    return comps, z


INSIDE_MARGIN = 0.03
ORBIT_N, ORBIT_E, ORBIT_SIZE = 16, 15.0, 768      # the cameras of materials.render_orbit


def orbit_cam(V, i):
    lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; rad = np.linalg.norm(hi - lo) / 2
    f = ORBIT_SIZE / 2 / np.tan(np.radians(15)); dist = rad / np.sin(np.radians(15)) * 1.1
    a, e = 2 * np.pi * i / ORBIT_N, np.radians(ORBIT_E)
    eye = c + dist * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    fw = c - eye; fw /= np.linalg.norm(fw); rt = np.cross(fw, [0, 1, 0]); rt /= np.linalg.norm(rt); dn = np.cross(fw, rt)
    return eye, np.stack([rt, dn, fw]), f


def view_hits(V, F, i):
    """Per pixel of orbit view i: the 3D surface point hit (NaN where the ray misses)."""
    import open3d as o3d
    eye, M, f = orbit_cam(V, i)
    u, v = np.meshgrid(np.arange(ORBIT_SIZE) + .5, np.arange(ORBIT_SIZE) + .5)
    d = np.stack([(u - ORBIT_SIZE / 2) / f, (v - ORBIT_SIZE / 2) / f, np.ones_like(u)], -1) @ M
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(V.astype(np.float32), F.astype(np.uint32))
    r = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eye, d.shape), d], -1).reshape(-1, 6).astype(np.float32)))
    t = r["t_hit"].numpy().reshape(ORBIT_SIZE, ORBIT_SIZE)
    P = eye + d * t[..., None]; P[~np.isfinite(t)] = np.nan
    return P


def fg_box(a, white_bg):
    a = np.asarray(a, np.float32)
    fg = a.min(-1) < 245 if white_bg else a.max(-1) > 10
    ys, xs = np.nonzero(fg)
    return ys.min(), ys.max(), xs.min(), xs.max()


def mask_to_view(mask, ref_img, frame_img):
    """The reference mask carried into the orbit frame that looks like the reference: foreground boxes matched."""
    sil = silhouette_arr(np.asarray(ref_img)); ys, xs = np.nonzero(sil)   # photo backgrounds are not white
    ry0, ry1, rx0, rx1 = ys.min(), ys.max(), xs.min(), xs.max()
    fy0, fy1, fx0, fx1 = fg_box(frame_img, True)
    my, mx = np.nonzero(mask)
    sy = (fy1 - fy0) / max(ry1 - ry0, 1); sx = (fx1 - fx0) / max(rx1 - rx0, 1)
    py = np.clip(((my - ry0) * sy + fy0).astype(int), 0, ORBIT_SIZE - 1)
    px = np.clip(((mx - rx0) * sx + fx0).astype(int), 0, ORBIT_SIZE - 1)
    out = np.zeros((ORBIT_SIZE, ORBIT_SIZE), bool); out[py, px] = True
    return ndi.binary_closing(out, iterations=2)


def best_view(orbit_dir, image):
    """Orbit frame most like the picture: both cropped to their foreground (white or light backgrounds)."""
    def crop(a, box):
        y0, y1, x0, x1 = box
        return np.asarray(Image.fromarray(np.asarray(a)[y0:y1 + 1, x0:x1 + 1]).resize((96, 96)), np.float32)
    im = Image.open(image).convert("RGB")
    sil = silhouette(Path(image), im.size[::-1]); ys, xs = np.nonzero(sil)
    ref = crop(im, (ys.min(), ys.max(), xs.min(), xs.max()))
    frames = sorted((Path(orbit_dir) / "frames").glob("*.jpg"))
    sc = []
    for f in frames:
        a = np.asarray(Image.open(f).convert("RGB")); sc.append(-np.abs(crop(a, fg_box(a, True)) - ref).mean())
    return int(np.argmax(sc))


def place_view(PV, PF, part_dir, part_img, V, F, mask, ref, whole):
    """Similarity transform of a generated part into the whole object, through the views: the reference picture is
    the whole's orbit view that looks most like it; the part's picture is the part's orbit view that looks most like
    it. Both are the same camera, so the rotation is the one between those two orbit cameras; scale and position
    match the part's visible points to the whole's surface under the reference mask (box + front depth), then ICP."""
    from splat2mesh.materials import render_orbit
    import open3d as o3d
    wo = whole / "mat" / "orbit"
    vw = best_view(wo, ref)
    frame = Image.open(sorted((wo / "frames").glob("*.jpg"))[vw]).convert("RGB")
    refi = Image.open(ref).convert("RGB").resize(mask.shape[::-1])
    mv = mask_to_view(mask, refi, frame)
    Pw = view_hits(V, F, vw)[mv]; Pw = Pw[np.isfinite(Pw).all(1)]
    po = part_dir / "mat" / "orbit"
    if not (po / "frames").exists():
        render_orbit(part_dir, po)
    best = None
    _, Mw, _ = orbit_cam(V, vw)
    cw = Pw @ Mw.T
    ew = np.percentile(cw[:, :2], 98, 0) - np.percentile(cw[:, :2], 2, 0)
    mid_w = (np.percentile(cw[:, :2], 98, 0) + np.percentile(cw[:, :2], 2, 0)) / 2
    diag = float(np.linalg.norm(ew))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw))
    vlike = best_view(po, part_img)
    # every orbit view of the part is a candidate rotation (the picture-likeness of thin parts is ambiguous): the
    # one whose visible points fit the whole's surface under the mask best wins; ties go to the most alike view
    for vp in range(ORBIT_N):
        Pp = view_hits(PV, PF, vp).reshape(-1, 3); Pp = Pp[np.isfinite(Pp).all(1)]
        if len(Pw) < 100 or len(Pp) < 100:
            continue
        Pp = Pp[np.random.default_rng(0).permutation(len(Pp))[:20000]]
        _, Mp, _ = orbit_cam(PV, vp)
        R = Mw.T @ Mp
        cp = (Pp @ R.T) @ Mw.T
        ep = np.percentile(cp[:, :2], 98, 0) - np.percentile(cp[:, :2], 2, 0)
        sc = float(np.mean(ew / np.maximum(ep, 1e-9)))
        mid_p = (np.percentile(cp[:, :2], 98, 0) + np.percentile(cp[:, :2], 2, 0)) / 2 * sc
        dz = np.percentile(cw[:, 2], 5) - np.percentile(cp[:, 2], 5) * sc      # front surfaces meet
        T0 = np.eye(4); T0[:3, :3] = sc * R; T0[:3, 3] = Mw.T @ np.r_[mid_w - mid_p, dz]
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pp))
        reg = o3d.pipelines.registration.registration_icp(
            src, dst, 0.1 * diag, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        T = reg.transformation
        if abs(np.cbrt(abs(np.linalg.det(T[:3, :3]))) / sc - 1) > BOX_SCALE_TOL:
            T = T0
        ev = o3d.pipelines.registration.evaluate_registration(src, dst, 0.1 * diag, T)
        # two-way: the part's points on the whole surface and the whole's mask points covered by the part
        back = o3d.pipelines.registration.evaluate_registration(dst, src.transform(T), 0.1 * diag, np.eye(4))
        # the part lies inside the whole object's volume (with a margin): a rotation that sticks out is wrong
        W = (np.c_[PV[::7], np.ones(len(PV[::7]))] @ T.T)[:, :3]
        lo_, hi_ = V.min(0), V.max(0); mg = INSIDE_MARGIN * np.linalg.norm(hi_ - lo_)
        inside = np.all((W > lo_ - mg) & (W < hi_ + mg), 1).mean()
        score = ev.fitness * back.fitness * inside ** 4 + (0.01 if vp == vlike else 0)
        if best is None or score > best[0]:
            best = (score, T, ev.fitness * back.fitness, ev.inlier_rmse, vp)
    if best is None:
        return None
    _, T, fit, rmse, vp = best
    return T, float(fit), float(rmse) / max(diag, 1e-9), {"whole_view": int(vw), "part_view": int(vp)}


def ref_box(ref: Path, mask: np.ndarray, V: np.ndarray):
    """Where the component is in the whole object, from the reference picture: the whole mesh is made from that
    picture seen from the front (x right, y up), so the mask's extent inside the object's silhouette maps linearly
    onto the mesh's x and y extent; depth keeps the whole object's range."""
    img = np.asarray(Image.open(ref).convert("RGB")).astype(np.float32)
    bg = np.median(np.r_[img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3), img[:, :4].reshape(-1, 3),
                         img[:, -4:].reshape(-1, 3)], 0)
    sil = np.abs(img - bg).max(-1) > 18
    sil = ndi.binary_opening(sil, iterations=2)
    lab_, n = ndi.label(sil)
    if n:
        sil = lab_ == (np.argmax(ndi.sum(sil, lab_, range(1, n + 1))) + 1)
    if mask.shape != sil.shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(sil.shape[::-1])) > 127
    ys, xs = np.nonzero(sil); my, mx = np.nonzero(mask)
    lo, hi = V.min(0), V.max(0)
    fx = lambda x: lo[0] + (x - xs.min()) / max(np.ptp(xs), 1) * (hi[0] - lo[0])
    fy = lambda y: hi[1] - (y - ys.min()) / max(np.ptp(ys), 1) * (hi[1] - lo[1])
    blo = np.array([fx(np.percentile(mx, 1)), fy(np.percentile(my, 99)), lo[2]])
    bhi = np.array([fx(np.percentile(mx, 99)), fy(np.percentile(my, 1)), hi[2]])
    return blo, bhi


ENVELOPE_SHARE = 0.5      # of another component's mask inside this one's: this one is a hollow envelope
GLASS_SMOOTH_ITERS = 40   # Taubin iterations on transmissive components
BOX_SCALE_TOL = 0.25       # ICP may change the box-fitted scale by at most this much


def place(part_V, part_F, target_pts, box):
    """Similarity transform of the component mesh into its box from the reference picture (x/y extent fitted, depth
    centred), refined by scaled ICP on the whole mesh's surface inside the box. Fitness: share of the component's
    points that land on that surface."""
    import open3d as o3d
    blo, bhi = box
    lo, hi = part_V.min(0), part_V.max(0)
    s = np.mean((bhi[:2] - blo[:2]) / np.maximum(hi[:2] - lo[:2], 1e-9))
    T0 = np.eye(4); T0[:3, :3] *= s; T0[:3, 3] = (blo + bhi) / 2 - s * (lo + hi) / 2
    diag = np.linalg.norm(bhi - blo)
    src = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(part_V), o3d.utility.Vector3iVector(part_F))
    src_pc = src.sample_points_uniformly(20000)
    pad = 0.08 * diag
    inside = np.all((target_pts > blo - pad) & (target_pts < bhi + pad), 1)
    tp = target_pts[inside] if inside.sum() > 200 else target_pts
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tp))
    thr = 0.1 * diag
    reg = o3d.pipelines.registration.registration_icp(
        src_pc, dst, thr, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T = reg.transformation
    s1 = np.cbrt(abs(np.linalg.det(T[:3, :3])))
    if abs(s1 / s - 1) > BOX_SCALE_TOL:                       # ICP drifted away from the picture: keep the box fit
        T = T0
    ev = o3d.pipelines.registration.evaluate_registration(src_pc, dst, thr, T)
    return T, float(ev.fitness), float(ev.inlier_rmse) / max(diag, 1e-9)


CONTACT_DILATE = 6         # px: reference masks this close touch, so the parts must touch in 3D
CONTACT_OVERLAP = 0.01     # of the object size: the closed joint overlaps a little, no visible seam


def close_contacts(parts, geo, report):
    """Generated components are placed one by one and can float apart. Two components whose masks touch in the
    reference picture touch in the object: the smaller one moves along the line between their picture centres
    (x right, y up) until its extent meets the bigger one's."""
    names = sorted(geo, key=lambda n: -geo[n][1].sum())
    shift = {n: np.zeros(3) for n in names}
    allp = np.concatenate([geo[n][0] for n in names]) if names else np.zeros((1, 3))
    size = np.linalg.norm(np.ptp(allp, 0))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma, mb = geo[a][1], geo[b][1]
            if not (ndi.binary_dilation(ma, iterations=CONTACT_DILATE) & mb).any():
                continue
            if (ma & mb).sum() > ENVELOPE_SHARE * min(ma.sum(), mb.sum()):
                continue                      # one seen through the other (inside it), not side by side
            ca = np.argwhere(ma).mean(0); cb = np.argwhere(mb).mean(0)
            d = np.array([cb[1] - ca[1], ca[0] - cb[0], 0.0]); d /= max(np.linalg.norm(d), 1e-9)   # a → b
            pa = (geo[a][0] + shift[a]) @ d; pb = (geo[b][0] + shift[b]) @ d
            gap = np.percentile(pb, 1) - np.percentile(pa, 99)         # > 0: b floats beyond a
            if gap <= 0:
                continue                      # already touching or overlapping: placed parts are not pushed out
            mv = -(gap + CONTACT_OVERLAP * size) * d
            shift[b] = shift[b] + mv
            report[b]["contact_shift"] = round(float(-(gap + CONTACT_OVERLAP * size) / size), 4)
    outp = []
    for name, cls, V, F, uv, tex in parts:
        t = shift.get(name)
        if t is None or not t.any():
            outp.append((name, cls, V, F, uv, tex)); continue
        if cls is None:                       # referenced component: V is its 4x4 transform
            T = V.copy(); T[:3, 3] += t; outp.append((name, cls, T, F, uv, tex))
        else:
            outp.append((name, cls, V + t, F, uv, tex))
    return outp


def write_usd(out: Path, parts, thin=()):
    """parts: [(name, material class, V, F, uv or None, texture path or None)]"""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    st = Usd.Stage.CreateNew(str(out / "asset.usda"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Asset"); st.SetDefaultPrim(root.GetPrim())
    for name, cls, V, F, uv, tex in parts:
        xf = f"/Asset/Components/{name}"
        if cls is None:                    # own material stage: reference its asset, placed by the ICP transform
            T, src = V, F
            x = UsdGeom.Xform.Define(st, xf)
            x.AddTransformOp().Set(Gf.Matrix4d(T.T.tolist()))      # USD row vectors
            x.GetPrim().GetReferences().AddReference("./" + str((Path(src) / "asset_materials.usda"
                                                                  if (Path(src) / "asset_materials.usda").exists()
                                                                  else Path(src) / "asset.usda").relative_to(out)))
            continue
        L = _lib(cls)
        UsdGeom.Xform.Define(st, xf)
        m = UsdGeom.Mesh.Define(st, xf + "/Geom")
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(F.astype(np.int32).ravel()))
        m.GetSubdivisionSchemeAttr().Set("none")
        import trimesh
        nrm = trimesh.Trimesh(V, F, process=False).vertex_normals
        m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(nrm, np.float32))); m.SetNormalsInterpolation("vertex")
        mat = UsdShade.Material.Define(st, xf + "/Mat")
        sh = UsdShade.Shader.Define(st, xf + "/Mat/PBR"); sh.CreateIdAttr("UsdPreviewSurface")
        metal = float(L.get("metallic", 0)); rough = float(L.get("roughness", 0.5)); trans = float(L.get("transmission", 0))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        if L.get("ior"):
            sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(L["ior"]))
        if trans > 0:
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.25)
        if metal >= 0.5 and L.get("f0"):            # metal: the library's F0, not the baked reflections
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, L["f0"])))
        elif trans > 0:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.95, 0.95))
        elif tex is not None and uv is not None:
            pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
            rd = UsdShade.Shader.Define(st, xf + "/Mat/stReader"); rd.CreateIdAttr("UsdPrimvarReader_float2")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(st, xf + "/Mat/BaseColor"); tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./" + str(Path(tex).relative_to(out)))
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        # MaterialX standard_surface for path tracers (glass transmission, metal F0)
        ss = UsdShade.Shader.Define(st, xf + "/Mat/StandardSurface"); ss.CreateIdAttr("ND_standard_surface_surfaceshader")
        ss.CreateInput("metalness", Sdf.ValueTypeNames.Float).Set(metal)
        ss.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(rough)
        ss.CreateInput("transmission", Sdf.ValueTypeNames.Float).Set(trans)
        if L.get("ior"):
            ss.CreateInput("specular_IOR", Sdf.ValueTypeNames.Float).Set(float(L["ior"]))
        if metal >= 0.5 and L.get("f0"):
            ss.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, L["f0"])))
        elif trans > 0:
            ss.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
        elif tex is not None and uv is not None:
            # dielectric with its generated texture: the path tracer reads base_color, not the preview diffuse
            im = UsdShade.Shader.Define(st, xf + "/Mat/MtlxBaseColor"); im.CreateIdAttr("ND_image_color3")
            fi = im.CreateInput("file", Sdf.ValueTypeNames.Asset); fi.Set("./" + str(Path(tex).relative_to(out)))
            fi.GetAttr().SetColorSpace("srgb_texture")
            im.CreateOutput("out", Sdf.ValueTypeNames.Color3f)
            ss.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(im.ConnectableAPI(), "out")
            ss.CreateInput("base", Sdf.ValueTypeNames.Float).Set(1.0)
        if trans > 0 and name in thin:
            # an envelope with parts inside (a bulb, a jar): a thin shell, not a solid lump of glass
            ss.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
        mat.CreateSurfaceOutput("mtlx").ConnectToSource(ss.ConnectableAPI(), "out")
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    st.GetRootLayer().Save()


def main():
    ref, whole, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    # components: "phrase=materials"
    comps = []
    for a in sys.argv[4:]:
        ph, mats = a.split("=", 1)
        if mats.strip().endswith("@rev"):          # a body of revolution (wheel, bottle, cup): turned, see lathe()
            REVOLUTION.add(slug(ph.strip())); mats = mats.strip()[:-4]
        comps.append((ph.strip(), mats.strip()))
    items_c = [(m.split(":")[0].strip() if ":" in m else m, ph) for ph, m in comps]
    # 1. components on the reference picture
    phrases = [ph for ph, _ in comps]
    eneas_on_reference(ref, phrases, out / "masks_ref.npz")
    z = {k: v for k, v in np.load(out / "masks_ref.npz").items()}
    comps, z = merge_and_complete(ref, comps, z)
    items_c = [(m.split(":")[0].strip() if ":" in m else m, ph) for ph, m in comps]
    # layout: the whole object's faces per phrase (its material votes)
    names_whole = json.loads((whole / "mat" / "regions.json").read_text()).get("names")
    V, F, lab = face_labels(whole)
    wnames = names_whole or [f"{c}|{t}" for c, t in parse_description((whole.parent / "materials.txt").read_text())]
    parts, report, geo = [], {}, {}
    for (ph, mats), (cls, _) in zip(comps, items_c):
        name = slug(ph)
        multi = ":" in mats
        sub = parse_description(mats) if multi else []
        m = z[ph][0] if ph in z else None
        share = float(m.mean()) if m is not None else 0.0
        # where the component is in the whole object: faces the whole asset voted for this phrase (or for any of
        # its sub-parts' phrases)
        ks = [i for i, n in enumerate(wnames) if n.split("|", 1)[1] in [ph] + [t for _, t in sub]]
        tgt_faces = np.nonzero(np.isin(lab, ks))[0] if ks else np.array([], int)
        rep = {"material": cls, "mask_share": round(share, 4), "whole_faces": int(len(tgt_faces))}
        placed = False
        if m is not None and share >= MIN_MASK:              # its place comes from the picture mask
            img = out / "parts" / f"{name}.png"; img.parent.mkdir(exist_ok=True)
            if not img.exists():
                cut = out / "parts" / f"{name}_cut.png"
                cut_out(ref, MAIN_PICTURE.get(name, m), cut)
                chosen = iris_picture(ref, ph, cls, cut, [p for p, _ in comps if p != ph], out) if USE_IRIS else None
                Image.open(chosen or cut).save(img)
                rep["picture"] = "iris " + Path(chosen).stem if chosen else "cut-out"
            run = out / "runs"
            usd_dir = run / name / "usd"
            if not (usd_dir / "asset.usda").exists():
                subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/splat2usd.py"), str(img), "--out", str(run),
                                "--direct", "--faces", "40000", "--tex", "2048"], check=False)
            if (usd_dir / "asset.usda").exists():
                from pxr import Usd, UsdGeom
                pst = Usd.Stage.Open(str(usd_dir / "asset.usda"))          # keep the stage alive while reading
                pm = UsdGeom.Mesh(pst.GetPrimAtPath("/Asset/Geom"))
                PV = np.asarray(pm.GetPointsAttr().Get(), np.float64)
                PF = np.asarray(pm.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
                PUV = np.asarray(UsdGeom.PrimvarsAPI(pm).GetPrimvar("st").Get(), np.float64)
                pv = None
                if (whole / "mat" / "orbit" / "frames").exists():
                    try:
                        pv = place_view(PV, PF, usd_dir, img, V, F, m, ref, whole)
                    except Exception as e:
                        print("view placement failed", name, e, flush=True)
                if pv is not None:
                    T, fit, rmse, views = pv; rep.update(views)
                else:
                    T, fit, rmse = place(PV, PF, V[F].mean(1), ref_box(ref, m, V))
                rep.update(icp_fitness=round(fit, 3), icp_rmse=round(rmse, 4))
                if fit > 0.3:
                    PV2 = (np.c_[PV, np.ones(len(PV))] @ T.T)[:, :3]
                    if multi:
                        # several materials: the component's own material stage, then it is referenced placed
                        if not (usd_dir / "parts.json").exists():
                            subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/materials_eneas3d.py"),
                                            str(usd_dir), str(img), mats], check=False)
                        parts.append((name, None, T, usd_dir, None, None))
                    else:
                        if float(_lib(cls).get("transmission", 0)) > 0:
                            # glass is smooth: Idol's lumps refract like stones, so the surface is relaxed
                            import trimesh
                            # smoothed on the welded surface: UV seams duplicate vertices, which would split apart
                            key = np.round(PV2 / (1e-6 * np.linalg.norm(np.ptp(PV2, 0)))).astype(np.int64)
                            _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
                            inv = inv.ravel()
                            tm = trimesh.Trimesh(PV2[first].copy(), inv[PF], process=False)
                            trimesh.smoothing.filter_taubin(tm, iterations=GLASS_SMOOTH_ITERS)
                            PV2 = np.asarray(tm.vertices)[inv]
                        tex = usd_dir / "basecolor.png"
                        if name in REVOLUTION:
                            PV2, PF, PUV, tex = lathe(PV2, PF, PUV, tex, out / "parts" / f"{name}_lathe.png")
                            rep["revolution"] = True
                        parts.append((name, cls, PV2, PF, PUV, tex))
                    placed = True
                    geo[name] = (PV2, m)
                    rep["source"] = "generated" + (" (own materials)" if multi else "")
        if not placed and len(tgt_faces):
            # keep the whole mesh's faces for this component
            used = np.unique(F[tgt_faces]); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
            parts.append((name, cls, V[used], remap[F[tgt_faces]], None, None)); rep["source"] = "whole mesh"
        report[name] = rep
        print(name, rep, flush=True)
    parts = close_contacts(parts, geo, report)
    # envelopes: a component whose picture mask holds a good share of another component's mask (the filament seen
    # through the bulb) is hollow
    thin = set()
    for ph, _ in comps:
        if ph not in z:
            continue
        a = z[ph][0]
        for q, _ in comps:
            if q != ph and q in z and z[q][0].sum() < a.sum() and (a & z[q][0]).sum() > ENVELOPE_SHARE * z[q][0].sum():
                thin.add(slug(ph)); report[slug(ph)]["envelope"] = True
    write_usd(out, parts, thin)
    (out / "components.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
