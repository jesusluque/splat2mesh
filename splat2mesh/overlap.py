"""Overlapping part masks resolved by appearance, with DeepSeek as the judge.

ENEAS sometimes grounds a phrase on too much: "the dragon's body" on the whole figurine, "the crystal" on the dragon
and the crystal. The drop rules then throw the smaller part away. Here the object's pixels on the annotation frame are
split into colour clusters (Lab k-means); DeepSeek sees the photo and the numbered clusters and names the listed part
each cluster belongs to. Then, in every frame, a pixel claimed by several masks goes to the claiming part its colour
cluster was given to (other pixels keep their masks), so nested or swollen masks become disjoint parts.
"""
import numpy as np
from PIL import Image, ImageDraw

K = 8
SWOLLEN = 0.5      # IoU with a bigger mask above which a part mask is not trusted on its own


def _lab(rgb):
    from skimage.color import rgb2lab
    return rgb2lab(rgb.astype(np.float32) / 255.0).reshape(-1, 3)


def _kmeans(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return C


def _assign(X, C):
    return ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)


def verify(masks, frames, ann_index, photo, parts, ask):
    """DeepSeek checks every phrase's mask on the annotation frame: a sheet with the photo and one tile per phrase
    (its mask in red). Masks it says show something else ("no") are emptied; "partly" (the part plus more) is kept
    for resolve() to trim. Returns (masks, answers, sheet). ENEAS sometimes grounds a phrase on another part: "the
    shield face" on the base, "the face" on helmet and beard."""
    keys = [k for k in masks if masks[k][ann_index].any()]
    if not keys:
        return masks, {}, None
    F, H, W = next(iter(masks.values())).shape
    im = np.asarray(Image.open(frames[ann_index]).convert("RGB").resize((W, H)), np.float32)
    T = 256
    obj = np.zeros((H, W), bool)
    for k in keys:
        obj |= masks[k][ann_index]
    ys, xs = np.nonzero(im.min(-1) < 245)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else (0, 0, W, H)
    sheet = Image.new("RGB", ((len(keys) + 1) * T, T + 24), "white"); d = ImageDraw.Draw(sheet)
    p = Image.open(photo).convert("RGB"); p.thumbnail((T, T)); sheet.paste(p, (0, 24)); d.text((6, 4), "PHOTO", fill="black")
    for j, k in enumerate(keys):
        m = masks[k][ann_index]
        t = im * 0.45 + 255 * 0.55
        t[m] = im[m] * 0.35 + np.array([255, 30, 30]) * 0.65
        tile = Image.fromarray(t.astype(np.uint8)).crop(box); tile.thumbnail((T, T))
        sheet.paste(tile, ((j + 1) * T, 24)); d.text(((j + 1) * T + 6, 4), f"tile {j}", fill="black")
    q = ("First picture: a photo of an object. The other tiles show a render of its 3D model with one region "
         "highlighted in red. For each tile, does the red region show the named part? Answer \"yes\" (it is that "
         "part, maybe a little more or less), \"partly\" (that part plus large areas of other parts) or \"no\" (it is "
         "mostly another part). Tiles: " + "; ".join(f'tile {j}: "{parts[k]}"' for j, k in enumerate(keys)) +
         ". Answer JSON: {\"tiles\": {\"0\": \"yes|partly|no\", ...}}")
    ans = {keys[int(j)]: str(v).lower() for j, v in ask(sheet, q).get("tiles", {}).items()
           if str(j).isdigit() and int(j) < len(keys)}
    out = {k: (np.zeros_like(v) if ans.get(k) == "no" else v) for k, v in masks.items()}
    return out, ans, sheet


def resolve(masks, frames, ann_index, photo, parts, ask=None):
    """masks: {key: [F, H, W] bool}; frames: orbit image paths; parts: {key: phrase}. Returns (masks, report)."""
    F, H, W = next(iter(masks.values())).shape
    keys = [k for k in masks if masks[k].any()]
    allk = list(masks)                                               # parts without a mask can still own a cluster
    claim = np.stack([masks[k] for k in keys])                       # [P, F, H, W]
    multi = claim.sum(0) > 1
    if len(keys) < 2:
        return masks, {"overlap": 0.0}, None
    im = np.asarray(Image.open(frames[ann_index]).convert("RGB").resize((W, H)))
    obj = claim[:, ann_index].any(0)
    X = _lab(im[obj]); C = _kmeans(X[:: max(1, len(X) // 20000)], K)
    lab_img = np.full((H, W), -1); lab_img[obj] = _assign(X, C)
    # one tile per cluster: the render with only that cluster's pixels shown, the rest faded
    T = 256
    tiles = []
    for j in range(K):
        t = (im.astype(np.float32) * 0.15 + 255 * 0.85)
        m = lab_img == j
        t[m] = im[m]
        ys, xs = np.nonzero(obj)
        tiles.append(Image.fromarray(t.astype(np.uint8)).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    sheet = Image.new("RGB", ((K + 1) * T, T + 24), "white"); d = ImageDraw.Draw(sheet)
    p = Image.open(photo).convert("RGB"); p.thumbnail((T, T)); sheet.paste(p, (0, 24))
    d.text((6, 4), "PHOTO", fill="black")
    for j, t in enumerate(tiles):
        t.thumbnail((T, T)); sheet.paste(t, ((j + 1) * T, 24)); d.text(((j + 1) * T + 6, 4), f"cluster {j}", fill="black")
    names = {k: parts[k] for k in allk}
    q = ("First picture: a photo of an object. The others show a render of its 3D model where only one colour "
         f"cluster's pixels are visible (clusters 0-{K - 1}). Parts of the object: " + "; ".join(f'"{v}"' for v in names.values()) +
         ". For each cluster, name the ONE listed part most of its visible pixels belong to (copy the part text "
         "exactly), or \"mixed\" when it spreads over several parts about equally. Answer JSON: "
         "{\"clusters\": {\"0\": \"<part or mixed>\", ...}}")
    ans = ask(sheet, q).get("clusters", {})
    inv = {v.lower(): k for k, v in names.items()}
    owner = {int(j): inv.get(str(v).lower()) for j, v in ans.items() if str(j).isdigit()}
    # parts that own a cluster but have no mask take part too (ENEAS missed them or verify() emptied them)
    keys = keys + [k for k in sorted({o for o in owner.values() if o}) if k not in keys]
    claim = np.stack([masks[k] for k in keys])
    # every frame: pixels claimed by several masks go to the claimant their colour cluster belongs to
    out = {k: masks[k].copy() for k in masks}
    size = np.array([masks[k].sum() for k in keys], np.float64)
    big = int(size.argmax())
    iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    # swollen: ENEAS grounded the phrase on much of a bigger part too (IoU > SWOLLEN with a bigger mask)
    swollen = [any(size[j] > size[i] and iou(masks[keys[i]], masks[keys[j]]) > SWOLLEN for j in range(len(keys)))
               for i in range(len(keys))]
    reliable = np.array([not swollen[i] and i != big for i in range(len(keys))])
    moved = 0
    # pixels that only unreliable masks claim (one covering the whole figure, "the chainmail") also follow their
    # colour cluster's part, so a part ENEAS missed (the shield) is recovered from its colour
    only_unrel = (claim & ~reliable[:, None, None, None]).any(0) & ~(claim & reliable[:, None, None, None]).any(0)
    for f in range(F):
        mf = multi[f] | only_unrel[f]
        if not mf.any():
            continue
        imf = np.asarray(Image.open(frames[f]).convert("RGB").resize((W, H)))
        cl = _assign(_lab(imf[mf]), C)
        ys, xs = np.nonzero(mf)
        cs = claim[:, f, ys, xs]
        # 1) a reliable specific mask (not swollen over a bigger one, not the biggest) wins: the smallest such claimant
        rel = np.where(cs & reliable[:, None], size[:, None], np.inf)
        has_rel = np.isfinite(rel).any(0)
        wi = np.where(has_rel, rel.argmin(0), -1)
        # 2) else the colour cluster's part: when it claims the pixel, or when only unreliable masks claim it
        vote = np.array([keys.index(owner[c]) if owner.get(c) in keys else -1 for c in range(K)])[cl]
        vok = (vote >= 0) & ~has_rel
        free = only_unrel[f][ys, xs]
        vok[vok] = cs[vote[vok], np.nonzero(vok)[0]] | free[vok]
        wi = np.where(vok, vote, wi)
        # 3) else the smallest claimant
        small = np.where(cs, size[:, None], np.inf).argmin(0)
        wi = np.where(wi < 0, small, wi); ok = np.ones(len(ys), bool)
        for i, k in enumerate(keys):
            lose = ok & (wi != i) & claim[i, f, ys, xs]
            out[k][f, ys[lose], xs[lose]] = False
            gain = ok & (wi == i) & ~claim[i, f, ys, xs]
            out[k][f, ys[gain], xs[gain]] = True
            moved += int(lose.sum())
    rep = {"reliable": [k for k, r in zip(keys, reliable) if r], "overlap": float(multi.mean()), "clusters": {j: names.get(owner.get(j)) for j in range(K)},
           "pixels_moved": moved}
    return out, rep, sheet
