"""Two-colour overlay of a 3D model on the photo it was made from: photo in red, model in green, both in yellow.

The camera is found automatically: the model's silhouette (ray-cast, no path tracing) is compared with the photo's
over a grid of azimuths and elevations around the bounding sphere, both cropped to their bounding boxes; the best one
is path traced with athenea and fitted onto the photo by its silhouette box.

    python scripts/overlay.py PHOTO OUT.jpg [--mask MASK.npz|MASK.png] name=asset.usda [name=asset.usda ...]
--mask: the object's mask in the photo (an ENEAS mask of the whole object): without it the silhouette is the
difference from the background, which takes cast shadows in.
Each asset gives one row: silhouettes (red photo, green model) | pictures (photo luminance in red, model in green).
The camera is searched on the first asset and reused for the others, so every row is seen from the same place.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from components import silhouette_arr

FOCAL, FILM = 50.0, 36.0               # athenea_view.sh: --focal 50 on a 36 mm film
DIST = 4.5                             # eye distance in bounding-sphere radii (the whole object in frame)
N = 192


def load(asset):
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(asset)); cache = UsdGeom.XformCache(0)
    Vs, Fs, n = [], [], 0
    for p in Usd.PrimRange(st.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(p); P = np.asarray(m.GetPointsAttr().Get(), np.float64)
        c = np.asarray(m.GetFaceVertexCountsAttr().Get()); I = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        if not len(P) or not (c == 3).all():
            continue
        M = np.asarray(cache.GetLocalToWorldTransform(p), np.float64)
        Vs.append(np.c_[P, np.ones(len(P))] @ M); Fs.append(I.reshape(-1, 3) + n); n += len(P)
    V = np.concatenate(Vs)[:, :3]; F = np.concatenate(Fs)
    lo, hi = V.min(0), V.max(0)
    return V, F, (lo + hi) / 2, np.linalg.norm(hi - lo) / 2


def eye_dir(az, el):
    return np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])


def sil(scene, c, r, az, el, size=N):
    import open3d as o3d
    eye = c + DIST * r * eye_dir(az, el)
    fw = c - eye; fw /= np.linalg.norm(fw); rt = np.cross(fw, [0, 1, 0]); rt /= np.linalg.norm(rt); up = np.cross(rt, fw)
    f = size * FOCAL / FILM
    u, v = np.meshgrid(np.arange(size) + .5, np.arange(size) + .5)
    d = ((u - size / 2) / f)[..., None] * rt - ((v - size / 2) / f)[..., None] * up + fw
    rays = np.concatenate([np.broadcast_to(eye, d.shape), d], -1).reshape(-1, 6).astype(np.float32)
    return np.isfinite(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()).reshape(size, size)


def crop(m, out=64):
    ys, xs = np.nonzero(m)
    if not len(ys):
        return np.zeros((out, out), bool), 1.0
    b = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.asarray(Image.fromarray(b.astype(np.uint8) * 255).resize((out, out))) > 127, b.shape[1] / b.shape[0]


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def find_camera(V, F, c, r, photo_sil):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(V.astype(np.float32), F.astype(np.uint32))
    ref, ref_ar = crop(photo_sil)
    def score(az, el):
        s, ar = crop(sil(sc, c, r, az, el))
        return iou(s, ref) - 0.5 * abs(np.log(ar / ref_ar))       # shape and aspect ratio of the boxes
    best = max(((score(a, e), a, e) for a in np.radians(np.arange(0, 360, 10)) for e in np.radians(np.arange(-10, 61, 10))))
    for step in (np.radians(4), np.radians(1.5)):
        _, a0, e0 = best
        best = max([best] + [(score(a0 + i * step, e0 + j * step), a0 + i * step, e0 + j * step)
                             for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)])
    return best


def render(asset, az, el, png, size=768, spp=48):
    x, y, z = DIST * eye_dir(az, el)
    subprocess.run(["env", "VIEW_HDRI=brown_photostudio_02", f"OVER_MASK={Path(png).with_suffix('.mask.png')}", "bash", str(ROOT / "scripts/athenea_view.sh"), str(asset),
                    str(png), str(x), str(y), str(z), str(size), str(spp)], capture_output=True, check=True)


def fit(img, photo_shape, photo_sil, s):
    """Render scaled and moved so that its silhouette box lies on the photo's (height matched, centres matched);
    s: the render's object mask (from the depth channel)."""
    a = np.asarray(img.convert("RGB"))
    ys, xs = np.nonzero(s); py, px = np.nonzero(photo_sil)
    k = (py.max() - py.min()) / max(ys.max() - ys.min(), 1)
    big = img.convert("RGB").resize((round(a.shape[1] * k), round(a.shape[0] * k)), Image.LANCZOS)
    sb = Image.fromarray(s.astype(np.uint8) * 255).resize(big.size, Image.NEAREST)
    dy = round((py.min() + py.max()) / 2 - k * (ys.min() + ys.max()) / 2)
    dx = round((px.min() + px.max()) / 2 - k * (xs.min() + xs.max()) / 2)
    H, W = photo_shape
    canvas = Image.new("RGB", (W, H), tuple(int(v) for v in a[2, 2])); canvas.paste(big, (dx, dy))
    mask = Image.new("L", (W, H), 0); mask.paste(sb, (dx, dy))
    return np.asarray(canvas), np.asarray(mask) > 127


def panels(photo, psil, rimg, rsil):
    S = np.zeros(photo.shape, np.uint8)
    S[psil & ~rsil] = (230, 40, 40); S[rsil & ~psil] = (40, 200, 40); S[psil & rsil] = (235, 215, 60)
    S[~psil & ~rsil] = (25, 25, 25)
    g = lambda a: np.asarray(Image.fromarray(a).convert("L"), np.float32)
    lp, lr = (255 - g(photo)) * psil, (255 - g(rimg)) * rsil     # dark = strong; nothing outside the object
    L = np.zeros(photo.shape, np.uint8); L[..., 0] = np.clip(lp * 1.3, 0, 255); L[..., 1] = np.clip(lr * 1.3, 0, 255)
    return S, L


def main():
    photo_p, out = Path(sys.argv[1]), Path(sys.argv[2])
    args = sys.argv[3:]; mask = None
    if args[0] == "--mask":
        mask, args = Path(args[1]), args[2:]
    items = [a.split("=", 1) for a in args]
    photo = np.asarray(Image.open(photo_p).convert("RGB").resize((768, 768)))
    if mask is None:
        psil = silhouette_arr(photo)
    else:
        m = np.load(mask)[np.load(mask).files[0]].reshape(-1, *np.load(mask)[np.load(mask).files[0]].shape[-2:])[0] \
            if mask.suffix == ".npz" else np.asarray(Image.open(mask).convert("L")) > 127
        psil = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((768, 768))) > 127
    V, F, c, r = load(items[0][1])
    sc, az, el = find_camera(V, F, c, r, psil)
    print(f"camera az {np.degrees(az):.1f} el {np.degrees(el):.1f} score {sc:.3f}", flush=True)
    rows = []
    with tempfile.TemporaryDirectory() as t:
        for name, asset in items:
            png = Path(t) / f"{name}.png"; render(asset, az, el, png)
            rimg, rsil = fit(Image.open(png), photo.shape[:2], psil,
                             np.asarray(Image.open(png.with_suffix(".mask.png"))) > 127)
            S, L = panels(photo, psil, rimg, rsil)
            rows.append((name, iou(psil, rsil), S, L))
    W = 768
    sheet = Image.new("RGB", (2 * W, W * len(rows)), "black"); d = ImageDraw.Draw(sheet)
    for i, (name, j, S, L) in enumerate(rows):
        sheet.paste(Image.fromarray(S), (0, i * W)); sheet.paste(Image.fromarray(L), (W, i * W))
        d.text((10, i * W + 8), f"{name}: silhouettes, red photo / green model / yellow both, IoU {j:.2f}", fill="white")
        d.text((W + 10, i * W + 8), f"{name}: pictures, red photo / green model", fill="white")
        print(name, f"IoU {j:.3f}", flush=True)
    sheet.save(out, quality=90)
    print(out)


if __name__ == "__main__":
    main()
