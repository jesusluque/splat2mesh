"""Scene mode: a 3DGS capture of an interior (a room seen from inside) → an open, textured mesh at several LODs.

The object path makes one closed solid seen from outside: orbit cameras, a filled volume, capped holes. A room is the
opposite: a thin shell seen from the air it encloses, and its gaussians are often faint (a wall is a stack of
semi-transparent ones), so the opacity field itself has no clean level set. Here:

  1. floaters and far gaussians are left out by percentiles;
  2. a coarse grid finds the room's air (free cells with gaussians in at least five of the six axis directions) and
     camera stations are spread in it, far from the walls;
  3. the surface is taken where light leaving the room's air along an axis is half absorbed by the gaussians
     (transmittance 0.5): one definition for the whole room, evaluated on a lattice in tiles (CPU), open where the
     air sees no wall, with nothing filled or capped. A splat's rendered depths are not used for this: they are
     means over semi-transparent gaussians and differ from view to view;
  4. small pieces go, and so does what no point of the room's air can see (the back of a sheet, second layers);
     the room's large planes are made plane (what stands out of them stays), flat holes with nothing of the splat
     behind them are closed (a patch no camera saw; a window, with its bars and the street behind, stays open),
     and the mesh is decimated into the LODs;
  5. athenea renders the splat from each station as a cube of wide views, without its view-dependent colour (the
     spherical harmonics above degree 0 set to zero): views of the whole splat do not agree with one another,
     the same wall can be far brighter from one place than from another, and no texture can match them all;
  6. each LOD gets UVs by planar projection onto the room's planes (islands packed over several UDIM tiles when
     the density asks), a base colour that is, for every texel, the mean of the stations that see it, and the
     detail the decimation lost as normal, height and occlusion maps baked from the finest mesh. One station is
     kept out of the bake: the mesh is compared from it with the splat's views, the ones without view-dependent
     colour (what a texture can reach) and the whole splat's (PSNR, SSIM).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh

from .ply import read_vertex

SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------- gaussians
def load(path, min_opacity: float = 0.03, up: str = "y", crop_pct: float = 1.0, margin: float = 0.06,
         bounds=None) -> dict:
    """Gaussians as float32 arrays, rotated so `up` becomes +Y, cropped to the scene's box.
    The box is the crop_pct … 100-crop_pct percentiles of the centres widened by margin × its largest side, or
    `bounds` = (lo, hi)."""
    v = read_vertex(path)
    n_all = len(v)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    alpha = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    keep = alpha > min_opacity
    if bounds is None:
        lo, hi = np.percentile(xyz, crop_pct, 0), np.percentile(xyz, 100 - crop_pct, 0)
        m = margin * (hi - lo).max()
        lo, hi = lo - m, hi + m
    else:
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    keep &= np.all((xyz > lo) & (xyz < hi), 1)
    v, xyz, alpha = v[keep], xyz[keep], alpha[keep]
    scale = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float64)
    scale = np.maximum(scale, 0.01 * scale.max(1, keepdims=True))   # flat (2D) gaussians: a scale of 1e-19, or 0, has no inverse
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rgb = np.clip(0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1), 0, 1).astype(np.float32)
    del v
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    U = up_matrix(up)
    xyz = xyz @ U.T
    R = np.einsum("ij,njk->nik", U, R)
    cov = np.einsum("nij,nj,nkj->nik", R, scale ** 2, R)
    inv = np.einsum("nij,nj,nkj->nik", R, 1 / scale ** 2, R)
    c = np.stack([lo, hi]) @ U.T
    return {"xyz": xyz.astype(np.float32), "alpha": alpha.astype(np.float32), "rgb": rgb,
            "inv6": np.stack([inv[:, 0, 0], inv[:, 1, 1], inv[:, 2, 2], inv[:, 0, 1], inv[:, 0, 2], inv[:, 1, 2]],
                             1).astype(np.float32),
            "sig": np.sqrt(np.stack([cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2]], 1)).astype(np.float32),
            "lo": c.min(0), "hi": c.max(0), "read": n_all}


def up_matrix(up: str) -> np.ndarray:
    """Rotation that takes the file's `up` axis (x, y, z, -x, -y, -z) to +Y."""
    return {"y": np.eye(3), "-y": np.diag([1.0, -1.0, -1.0]),
            "z": np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]), "-z": np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]),
            "x": np.array([[0, -1.0, 0], [1, 0, 0], [0, 0, 1]]), "-x": np.array([[0, 1.0, 0], [-1, 0, 0], [0, 0, 1]])}[up]


def base_colour_copy(ply, dst) -> Path:
    """A copy of a splat's .ply with its view-dependent colour removed (the f_rest_* harmonics set to zero): every
    gaussian keeps its mean colour. Views of it agree with one another, which views of the whole splat do not
    (the same wall can be far brighter from one place than from another), so a texture can be baked from them."""
    raw = Path(ply).read_bytes()
    i = raw.index(b"end_header\n") + 11
    head = raw[:i].decode("ascii", "replace")
    if "binary_little_endian" not in head:
        raise ValueError(f"{ply}: only binary little-endian .ply")
    props = [l.split() for l in head.splitlines() if l.startswith("property")]
    if any(w[1] != "float" for w in props):
        raise ValueError(f"{ply}: a vertex property that is not float")
    a = np.frombuffer(raw[i:], np.float32).reshape(-1, len(props)).copy()
    a[:, [k for k, w in enumerate(props) if w[2].startswith("f_rest")]] = 0
    Path(dst).write_bytes(raw[:i] + a.tobytes())
    return Path(dst)


# ---------------------------------------------------------------- the room's air
def room_air(g: dict, step: float, solid_alpha: float = 0.5, need: int = 5):
    """Coarse grid: solid = cells holding enough opacity; air of the room = free cells with solid somewhere along
    at least `need` of the six axis directions, largest connected piece. Returns (mask, origin, step)."""
    from scipy import ndimage as ndi
    lo, hi = g["lo"], g["hi"]
    n = np.ceil((hi - lo) / step).astype(int) + 1
    idx = np.clip(((g["xyz"] - lo) / step).astype(int), 0, n - 1)
    den = np.bincount(np.ravel_multi_index(idx.T, n), weights=g["alpha"], minlength=int(n.prod())).reshape(n)
    solid = den > solid_alpha
    blocked = np.zeros(n, np.int8)
    for ax in range(3):
        fwd = np.maximum.accumulate(solid, axis=ax)
        back = np.flip(np.maximum.accumulate(np.flip(solid, ax), axis=ax), ax)
        blocked += fwd.astype(np.int8) + back.astype(np.int8)
    air = ~ndi.binary_dilation(solid) & (blocked >= need)
    lab, k = ndi.label(air)
    if k == 0:
        raise RuntimeError("scene: no enclosed air found (is this an interior? try a larger --coarse)")
    sizes = ndi.sum(air, lab, range(1, k + 1))
    return lab == (1 + int(np.argmax(sizes))), lo.copy(), step


# ---------------------------------------------------------------- stations and views
def stations(air, lo, step, n: int = 6, inner: float = 0.45) -> np.ndarray:
    """n points in the room's air: the first where the walls are farthest, each next one the cell that is far from
    the walls and from the stations already chosen. Only cells at least `inner` × the largest distance to a wall
    are taken (a splat seen from close to a wall falls apart)."""
    from scipy import ndimage as ndi
    dist = ndi.distance_transform_edt(air)
    cells = np.argwhere(dist >= inner * dist.max())
    dw = dist[tuple(cells.T)]
    pts = [cells[int(np.argmax(dw))]]
    for _ in range(n - 1):
        ds = np.min([np.linalg.norm(cells - p, axis=1) for p in pts], 0)
        pts.append(cells[int(np.argmax(np.minimum(ds, 4 * dw)))])
    return lo + (np.array(pts) + 0.5) * step


CUBE = [((1, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, 1, 0)), ((0, 0, 1), (0, 1, 0)), ((0, 0, -1), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, 1))]


def cube_views(points, size: int, fov_deg: float = 100.0) -> dict:
    """Six views a station (the cube's faces, widened to fov_deg so they overlap), as athenea's cameras.json."""
    from .athenea import APERTURE
    focal = APERTURE / 2 / np.tan(np.radians(fov_deg) / 2)
    fx = focal / APERTURE * size
    views = []
    for i, p in enumerate(points):
        for j, (f, up) in enumerate(CUBE):
            f, up = np.array(f, float), np.array(up, float)
            r = np.cross(f, up); d = np.cross(f, r)
            R = np.stack([r, d, f])
            views.append({"name": f"s{i:02d}_{j}", "eye": list(map(float, p)), "target": list(map(float, p + f)),
                          "up": up.tolist(), "R_world_to_cam": R.tolist(), "t_world_to_cam": (-R @ p).tolist()})
    return {"convention": "opencv", "focal_mm": float(focal),
            "intrinsics": {"width": size, "height": size, "fx": fx, "fy": fx, "cx": size / 2, "cy": size / 2},
            "views": views}


def _compact(exr: Path) -> None:
    """A rendered view kept as NAME.rgba.png (8-bit sRGB, colour not premultiplied, alpha), a sixth of the EXR,
    which is deleted: the bake reads 8-bit colour anyway."""
    from PIL import Image
    from .athenea import read_exr, _srgb
    ch = read_exr(exr)
    if "RGBA" in ch:
        rgb, a = ch["RGBA"][..., :3], ch["RGBA"][..., 3]
    else:
        rgb = ch["RGB"] if "RGB" in ch else np.stack([ch["R"], ch["G"], ch["B"]], -1)
        a = ch.get("A", np.ones(rgb.shape[:2], np.float32))
    col = _srgb(rgb / np.maximum(a, 1e-6)[..., None])
    img = (np.clip(np.concatenate([col, a[..., None]], -1), 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(img, "RGBA").save(exr.with_suffix(".rgba.png"))
    exr.unlink()


def compact_views(out: Path) -> int:
    """Every EXR view left in out/views (an earlier run's) made compact; the previews go. Returns how many."""
    d, n = Path(out) / "views", 0
    for e in sorted(d.glob("*.exr")):
        try:
            _compact(e)
        except Exception:                                      # half written (another job is on it): left for later
            continue
        n += 1
        (d / f"{e.stem}.png").unlink(missing_ok=True)
    return n


def render(ply, cams: dict, out: Path, batch: int = 18, log=print) -> None:
    """The views of cams rendered by athenea into out/views (NAME.rgba.png), a GPU job every `batch`."""
    from . import athenea
    d = Path(out) / "views"; d.mkdir(parents=True, exist_ok=True)
    compact_views(out)
    size = cams["intrinsics"]["width"]
    todo = [v for v in cams["views"] if not (d / f"{v['name']}.rgba.png").exists()]
    for a in range(0, len(todo), batch):
        cmds = []
        for v in todo[a:a + batch]:
            cmds.append(([athenea._bin(), "render", "--splats", str(ply), "--eye", *map(str, v["eye"]), "--target",
                          *map(str, v["target"]), "--up", *map(str, v["up"]),
                          "--focal", str(v.get("focal_mm", cams.get("focal_mm"))),
                          *(["--near", str(v["near"])] if "near" in v else []),
                          "--size", f"{size}x{size}", "-o", str(d / f"{v['name']}.exr")], "tolerate"))
        athenea.run_script(cmds, log=Path(out) / f"render_{a:03d}.log")
        compact_views(out)
        log(f"    rendered {min(a + batch, len(todo))}/{len(todo)} views")


def load_views(out: Path, cams: dict, min_alpha: float = 0.3) -> list[dict]:
    """The rendered views grouped by station: [{"eye", "faces": [(rgb uint8 HxWx3 sRGB un-premultiplied, covered
    HxW bool, R world→camera)] in CUBE's order, "K"}]. A station with a view missing is left out."""
    from PIL import Image
    it = cams["intrinsics"]
    K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1.0]])
    out_st = []
    for i in range(len(cams["views"]) // 6):
        faces = []
        for v in cams["views"][6 * i:6 * i + 6]:
            f = Path(out) / "views" / f"{v['name']}.rgba.png"
            if not f.exists():
                break
            im = np.asarray(Image.open(f))
            faces.append((np.ascontiguousarray(im[..., :3]), im[..., 3] > min_alpha * 255,
                          np.asarray(v["R_world_to_cam"], np.float64)))
        if len(faces) == 6:
            out_st.append({"eye": np.asarray(cams["views"][6 * i]["eye"], np.float64), "faces": faces, "K": K})
    return out_st


def _sample_station(st: dict, P):
    """Colour a station sees at the points P (n×3): the cube face each direction falls in, bilinear.
    Returns (rgb n×3 float 0..1, covered n bool)."""
    d = P - st["eye"]
    ax = np.argmax(np.abs(d), 1)
    face = np.where(ax == 0, 0, np.where(ax == 2, 2, 4)) + (d[np.arange(len(d)), ax] < 0)      # CUBE's order
    K = st["K"]
    col = np.zeros((len(P), 3), np.float32)
    ok = np.zeros(len(P), bool)
    for j, (img, cov, R) in enumerate(st["faces"]):
        m = np.nonzero(face == j)[0]
        if len(m) == 0:
            continue
        c = d[m] @ R.T
        u = K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2] - 0.5
        v = K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2] - 0.5
        h, w = cov.shape
        x0 = np.clip(np.floor(u).astype(int), 0, w - 2); y0 = np.clip(np.floor(v).astype(int), 0, h - 2)
        fx, fy = np.clip(u - x0, 0, 1)[:, None], np.clip(v - y0, 0, 1)[:, None]
        col[m] = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
                  + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy) / 255.0
        ok[m] = cov[y0, x0] & cov[y0 + 1, x0 + 1]
    return col, ok


# ---------------------------------------------------------------- the skin: where the air stops seeing
_G: dict = {}                      # arrays shared with the tile workers (fork: copy on write)
_RADII = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24])
LN2 = float(np.log(2.0))


def _density(sel, origin, shape, step):
    """Optical depth a voxel along each axis: for axis d, Σ τ_i G_i(x) sqrt(dᵀΣ⁻¹d / 2π) · step, with
    τ_i = -log(1 - α_i): summed along d through a gaussian's centre it gives back that gaussian's own τ_i."""
    xyz, inv6, tau, rad, fac = _G["xyz"][sel], _G["inv6"][sel], _G["tau"][sel], _G["rad"][sel], _G["fac"][sel]
    n = int(np.prod(shape))
    acc = np.zeros((3, n), np.float64)
    p = (xyz - origin) / step
    c = np.rint(p).astype(np.int32)
    key = rad[:, 0] * 10000 + rad[:, 1] * 100 + rad[:, 2]
    shape = np.asarray(shape)
    for k in np.unique(key):
        grp = np.nonzero(key == k)[0]
        r = rad[grp[0]]
        off = np.stack(np.meshgrid(*[np.arange(-a, a + 1) for a in r], indexing="ij"), -1).reshape(-1, 3).astype(np.int32)
        for ch in np.array_split(grp, max(1, len(grp) * len(off) // 3_000_000 + 1)):
            vox = c[ch, None, :] + off[None]
            d = (vox - p[ch, None, :]) * step
            a = inv6[ch]
            m = (a[:, None, 0] * d[..., 0] ** 2 + a[:, None, 1] * d[..., 1] ** 2 + a[:, None, 2] * d[..., 2] ** 2
                 + 2 * (a[:, None, 3] * d[..., 0] * d[..., 1] + a[:, None, 4] * d[..., 0] * d[..., 2]
                        + a[:, None, 5] * d[..., 1] * d[..., 2]))
            ok = (m < 9.0) & np.all((vox >= 0) & (vox < shape), -1)
            w = (tau[ch, None] * np.exp(-0.5 * np.maximum(m, 0.0)))[ok]
            flat = (vox[..., 0][ok].astype(np.int64) * shape[1] + vox[..., 1][ok]) * shape[2] + vox[..., 2][ok]
            row = np.nonzero(ok)[0]
            for ax in range(3):
                acc[ax] += np.bincount(flat, weights=w * (fac[ch][row, ax] * np.sqrt(a[row, ax] / (2 * np.pi)) * step), minlength=n)
    return acc.reshape(3, *shape).astype(np.float32)


def _tile(job):
    """One tile of the lattice: the optical depth from the room's air along the six axis directions, and the
    surface where the least of them reaches ln 2 (the air's light is half gone). Vertices in lattice units."""
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes
    i0, core, halo = job
    step, lo, blur = _G["step"], _G["lo"], _G["blur"]
    a0 = i0 - halo
    shape = core + 2 * halo
    reach = (_G["rad"] + 1) * step
    bmin, bmax = lo + a0 * step, lo + (a0 + shape) * step
    sel = np.nonzero(np.all((_G["xyz"] + reach > bmin) & (_G["xyz"] - reach < bmax), 1))[0]
    if len(sel) == 0:
        return None
    cm, clo, cstep = _G["air"], _G["air_lo"], _G["air_step"]
    ax = [np.clip(((lo[k] + (a0[k] + np.arange(shape[k])) * step - clo[k]) / cstep).astype(int), 0, cm.shape[k] - 1)
          for k in range(3)]
    air = cm[np.ix_(*ax)]
    if not air.any():
        return None
    S = _density(sel, lo + a0 * step, shape, step)
    cap = np.float32(4 * LN2)
    phi = np.full(shape, cap, np.float32)
    for k in range(3):
        for flip in (False, True):
            s_, a_ = (np.flip(S[k], k), np.flip(air, k)) if flip else (S[k], air)
            c = np.cumsum(s_, axis=k, dtype=np.float32)
            base = np.maximum.accumulate(np.where(a_, c, -np.inf), axis=k)     # the sum at the last air voxel behind
            t = np.minimum(c - base, cap)
            phi = np.minimum(phi, np.flip(t, k) if flip else t)
    del S
    phi[air] = 0
    if blur > 0:
        phi = ndi.gaussian_filter(phi, blur)
    if phi.min() >= LN2 or phi.max() < LN2:
        return None
    v, f, _, _ = marching_cubes(phi, LN2)
    cen = v[f].mean(1)
    keep = np.all((cen >= halo) & (cen < halo + core), 1)
    f, cen = f[keep], cen[keep]
    if len(f) == 0:
        return None
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    ahead = ndi.map_coordinates(phi, (cen + nrm).T, order=1, mode="nearest")
    behind = ndi.map_coordinates(phi, (cen - nrm).T, order=1, mode="nearest")
    if (ahead > behind).mean() > 0.5:                          # normals towards the air
        f = f[:, ::-1]
    used, f = np.unique(f, return_inverse=True)
    return (v[used] + a0).astype(np.float32), f.reshape(-1, 3).astype(np.int32)


def skin(g: dict, air, air_lo, air_step, voxel: float, blur: float = 0.5, tile: int = 128, halo: int = 48,
         procs: int = 6, log=print) -> trimesh.Trimesh:
    """The room's inner surface on a lattice of `voxel`: the points where light from the room's air, travelling
    along an axis, is half absorbed by the gaussians (transmittance 0.5, the median depth of a view straight at
    the wall). One definition for the whole room, so there is no view to disagree with another; open where the air
    sees no wall (a doorway). Tiles run in `procs` processes; the mesh is welded."""
    import multiprocessing as mp
    from scipy import ndimage as ndi
    lo, hi = g["lo"], g["hi"]
    # A gaussian thinner than the lattice falls between its points. Each one is widened by half a voxel, Σ' = Σ + (voxel/2)² I,
    # and its optical depth scaled per axis so that what it absorbs over its whole cross-section stays the same.
    a6 = g["inv6"].astype(np.float64)
    inv = np.stack([np.stack([a6[:, 0], a6[:, 3], a6[:, 4]], 1), np.stack([a6[:, 3], a6[:, 1], a6[:, 5]], 1),
                    np.stack([a6[:, 4], a6[:, 5], a6[:, 2]], 1)], 1)
    cov = np.linalg.inv(inv)
    cov2 = cov + np.eye(3) * (0.5 * voxel) ** 2
    inv2 = np.linalg.inv(cov2)
    dd, dd2 = np.diagonal(inv, axis1=1, axis2=2), np.diagonal(inv2, axis1=1, axis2=2)
    fac = np.sqrt(np.linalg.det(cov)[:, None] * dd / (np.linalg.det(cov2)[:, None] * dd2)).astype(np.float32)
    g = dict(g, inv6=np.stack([inv2[:, 0, 0], inv2[:, 1, 1], inv2[:, 2, 2], inv2[:, 0, 1], inv2[:, 0, 2], inv2[:, 1, 2]],
                              1).astype(np.float32),
             sig=np.sqrt(np.diagonal(cov2, axis1=1, axis2=2)).astype(np.float32))
    del a6, inv, cov, cov2, inv2, dd, dd2
    rad = np.ceil(3.0 * g["sig"] / voxel).astype(int)
    rad = _RADII[np.minimum(np.searchsorted(_RADII, rad), len(_RADII) - 1)].astype(np.int32)
    _G.update(fac=fac, xyz=g["xyz"], inv6=g["inv6"], tau=-np.log1p(-np.minimum(g["alpha"], 0.99)), rad=rad, step=voxel,
              lo=lo.astype(np.float32), blur=blur, air=air, air_lo=air_lo, air_step=air_step)
    n = np.ceil((hi - lo) / voxel).astype(int) + 1
    tn = np.ceil(n / tile).astype(int)
    occ = np.zeros(tn, bool)
    occ[tuple(np.clip(((g["xyz"] - lo) / (voxel * tile)).astype(int), 0, tn - 1).T)] = True
    occ = ndi.binary_dilation(occ)
    jobs = [(np.array(t) * tile, np.array([tile] * 3), halo) for t in np.argwhere(occ)]
    log(f"    lattice {n.tolist()} at {voxel:.4f}: {len(jobs)} tiles of {tile}³ (+{halo}), {procs} processes")
    t0 = time.time()
    V, F, nv = [], [], 0
    with mp.get_context("fork").Pool(procs) as pool:
        for k, r in enumerate(pool.imap_unordered(_tile, jobs, chunksize=1)):
            if r is not None:
                V.append(r[0]); F.append(r[1] + nv); nv += len(r[0])
            if (k + 1) % 50 == 0:
                log(f"    tile {k + 1}/{len(jobs)}  {time.time() - t0:.0f} s  {sum(map(len, F))} faces")
    if not F:
        raise RuntimeError("scene: no surface found")
    V, F = np.concatenate(V), np.concatenate(F)
    key = np.rint(V * 256).astype(np.int64)                    # the tiles share lattice edges: weld
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V, F = V[first], inv.reshape(-1)[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return trimesh.Trimesh(V.astype(np.float64) * voxel + lo, F, process=False)


def drop_small(tm: trimesh.Trimesh, min_area_frac: float = 0.0005) -> trimesh.Trimesh:
    """Connected pieces under min_area_frac of the whole area go (specks in the air, flakes behind the walls)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adj = tm.face_adjacency
    k, lab = connected_components(coo_matrix((np.ones(len(adj), bool), adj.T), shape=(len(tm.faces),) * 2), directed=False)
    area = np.bincount(lab, weights=tm.area_faces, minlength=k)
    keep = area[lab] >= min_area_frac * area.sum()
    out = tm.submesh([np.nonzero(keep)[0]], append=True, repair=False)
    return trimesh.Trimesh(out.vertices, out.faces, process=False)


def seen(tm: trimesh.Trimesh, probes, off: float, grow: int = 2, log=print) -> trimesh.Trimesh:
    """Only what can be seen from the room. The skin also wraps what lies behind the walls (the back of a sheet,
    a second layer, pockets): faces no point of the room's air sees from their front. They take atlas, break the
    UV islands and are never coloured. A face stays when a ray from its centre (lifted `off` along the normal)
    reaches one of the `probes` free, or when it is within `grow` faces of one that does."""
    import open3d as o3d
    from scipy.sparse import coo_matrix
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(tm.vertices.astype(np.float32), tm.faces.astype(np.uint32))
    fn = tm.face_normals
    org = tm.triangles_center + fn * off
    vis = np.zeros(len(tm.faces), bool)
    for p in probes:
        idx = np.nonzero(~vis)[0]
        to = p - org[idx]
        ln = np.linalg.norm(to, axis=1)
        m = (to * fn[idx]).sum(1) > 0.02 * ln
        idx, to, ln = idx[m], to[m], ln[m]
        rays = np.concatenate([org[idx], to / ln[:, None]], 1).astype(np.float32)
        vis[idx[sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() >= ln]] = True
    direct = float(tm.area_faces[vis].sum() / tm.area)
    adj = tm.face_adjacency
    A = coo_matrix((np.ones(2 * len(adj), np.float32), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
                   shape=(len(vis),) * 2).tocsr()
    keep = vis.astype(np.float32)
    for _ in range(grow):
        keep = np.minimum(keep + A @ keep, 1)
    out = tm.submesh([np.nonzero(keep > 0)[0]], append=True, repair=False)
    log(f"    seen from the room: {direct:.1%} of the area, kept {tm.area_faces[keep > 0].sum() / tm.area:.1%}")
    return trimesh.Trimesh(out.vertices, out.faces, process=False)


def _ears(P):
    """Ear clipping of a simple polygon P (n×2, counter-clockwise). Returns triangles as rows of indices into P."""
    idx = list(range(len(P)))
    tris = []

    def cr(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    guard = 0
    while len(idx) > 3 and guard < 4 * len(P):
        n = len(idx)
        for i in range(n):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % n]
            if cr(P[a], P[b], P[c]) <= 0:
                continue
            rest = np.array([j for j in idx if j not in (a, b, c)])
            if len(rest):
                Q = P[rest]
                ins = ((cr(P[a], P[b], Q.T) > 0) & (cr(P[b], P[c], Q.T) > 0) & (cr(P[c], P[a], Q.T) > 0))
                if ins.any():
                    continue
            tris.append((a, b, c)); idx.pop(i)
            break
        else:
            return None                                        # not simple: left open
        guard += 1
    tris.append(tuple(idx))
    return np.array(tris)


def fill_holes(tm: trimesh.Trimesh, max_size: float, flat: float, g: dict | None = None, depth: float = 0.0,
               opaque: float = 0.5, pore: float = 0.0, max_loop: int = 3000, log=print):
    """Flat holes closed. Where the splat has nothing (a patch no camera saw) the skin has a hole, and through it
    the back of the room shows. A boundary loop is closed when it is no wider than max_size, lies within `flat` of
    its own plane (a hole in a wall, not the edge of a reveal or of the shell) and has nothing behind it. With the
    gaussians `g`, a hole is an opening of the room, and stays, when the splat goes on behind it: the gaussians up
    to `depth` behind the hole's plane cover at least `opaque` of it (a window's bars, what is outside). A hole
    no wider than `pore` is closed whatever lies behind: a wall is thick, and its own gaussians are behind its pores.
    The loop is triangulated in its plane; its vertices stay where they are.
    Returns (mesh, {"closed", "openings", "other"})."""
    vfn = None
    if g is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(g["xyz"])
        vfn = tm.vertex_normals
    openings = 0
    e = tm.edges_sorted
    be = tm.edges[trimesh.grouping.group_rows(e, require_count=1)]         # directed as their face has them
    nxt = {}
    multi = set()
    for a, b in be:
        if a in nxt:
            multi.add(a)
        nxt[int(a)] = int(b)
    add, closed, left = [], 0, 0
    done = set()
    for a0 in list(nxt):
        if a0 in done:
            continue
        loop, a = [], a0
        while a not in done and a in nxt and len(loop) <= max_loop:
            done.add(a); loop.append(a); a = nxt[a]
        if a != a0 or len(loop) < 3 or len(loop) > max_loop or multi & set(loop):
            left += 1
            continue
        P = tm.vertices[loop]
        c = P.mean(0)
        _, sv, vt = np.linalg.svd(P - c, full_matrices=False)
        if np.ptp((P - c) @ vt[0]) > max_size or np.abs((P - c) @ vt[2]).max() > flat:
            left += 1
            continue
        p2 = (P - c) @ vt[:2].T
        area2 = float((p2[:, 0] * np.roll(p2[:, 1], -1) - np.roll(p2[:, 0], -1) * p2[:, 1]).sum())
        if g is not None and np.ptp((P - c) @ vt[0]) > pore:
            out_ = -vfn[loop].mean(0)                              # away from the room
            n = vt[2] * (1.0 if vt[2] @ out_ > 0 else -1.0)
            rad = float(np.linalg.norm(p2, axis=1).max())
            near = np.asarray(tree.query_ball_point(c + n * depth / 2, np.hypot(rad, depth / 2)), int)
            if len(near):
                rel = g["xyz"][near] - c
                h = rel @ n
                q = rel @ vt[:2].T
                m = (h > flat) & (h < depth) & (np.linalg.norm(q, axis=1) < rad)
                var = g["sig"][near][m].astype(np.float64) ** 2
                cross = 2 * np.pi * np.sqrt((var @ vt[0] ** 2) * (var @ vt[1] ** 2))
                if float((g["alpha"][near][m] * cross).sum()) > opaque * abs(area2) / 2:
                    openings += 1
                    continue
        order = np.arange(len(loop)) if area2 > 0 else np.arange(len(loop))[::-1]
        t = _ears(p2[order])
        if t is None:
            left += 1
            continue
        t = np.asarray(loop)[order][t]
        if area2 > 0:                                          # the boundary runs with its faces: the patch against it
            t = t[:, ::-1]
        add.append(t); closed += 1
    log(f"    holes: {closed} closed, {openings} kept as openings (the splat goes on behind), {left} other boundary loops")
    info = {"closed": closed, "openings": openings, "other": left}
    if not add:
        return tm, info
    return trimesh.Trimesh(tm.vertices, np.concatenate([tm.faces] + add), process=False), info


def flatten(tm: trimesh.Trimesh, gd, noise: float = 2.5, min_share: float = 0.02, rings: int = 6,
            small: float = 0.01, smooth: int = 12, thin: float = 0.0, minor: float = 0.0, reach: float = 0.0, log=print):
    """The room's large planes made plane. A splat's wall is a noisy sheet; a built wall is flat. Each large
    connected patch of one of the guide's planes gets a robust quadratic fit of its height over the patch (so a
    wall that leans or bows a little keeps doing so); the vertices within `noise` robust deviations of the fit move
    onto it, the ones beyond twice that stay where they are (a window, a niche, a hole, a jamb), with a smooth
    step between. Two refinements: a patch also takes the vertices `rings` edges around it, so at a corner a vertex
    that faces both walls is put on both and the corner comes out a straight line; and what stands out but covers less than
    `small` of the patch (a lump of gaussians, not a feature of the room) is flattened too. A patch that lies on
    a plane already fitted (a wall cut in two by a niche) takes that plane's fit. What stands out in a strip
    narrower than `thin` (scene units: a ridge, a thread of gaussians) is flattened as well. Patches down to
    `minor` of the area (a jamb, a sill) are made plane too, when they are as flat as the large ones. `reach`
    (scene units) widens `rings` on a fine mesh: a lump wider than the rings is not reached to its middle.
    Returns (mesh, [what was done to each patch])."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    fn, nf, nv = tm.face_normals, len(tm.faces), len(tm.vertices)
    lab = gd[1][gd[0].query(tm.triangles_center, workers=-1)[1]]
    face_on = fn[np.arange(nf), lab // 2] * (1 - 2 * (lab % 2)) > 0.5
    adj = tm.face_adjacency
    same = (lab[adj[:, 0]] == lab[adj[:, 1]]) & face_on[adj[:, 0]] & face_on[adj[:, 1]]
    k, comp = connected_components(coo_matrix((np.ones(int(same.sum()), bool), adj[same].T), shape=(nf, nf)), directed=False)
    area = np.bincount(comp, weights=tm.area_faces * face_on, minlength=k)
    e = tm.edges_unique
    VV = coo_matrix((np.ones(2 * len(e), np.float32), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nv, nv)).tocsr()
    varea = np.bincount(tm.faces.ravel(), weights=np.repeat(tm.area_faces / 3, 3), minlength=nv)
    V0, V = tm.vertices, tm.vertices.copy()
    vn = np.asarray(tm.vertex_normals, np.float32).copy()     # smoothed: a lump's flank must still count as wall
    for _ in range(smooth):
        vn = vn + VV @ vn
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    done, fits = [], []
    edge = max(float(np.median(tm.edges_unique_length)), 1e-9)
    k_thin = int(np.ceil(thin / edge / 2)) if thin > 0 else 0
    rings = max(rings, int(np.ceil(reach / edge)))
    for c in np.argsort(-area):
        if area[c] < min(min_share, minor or min_share) * tm.area:
            break
        is_minor = area[c] < min_share * tm.area
        fsel = np.nonzero((comp == c) & face_on)[0]
        core = np.zeros(nv, np.float32); core[np.unique(tm.faces[fsel])] = 1
        axis = int(lab[fsel[0]] // 2)
        vid = np.nonzero(core)[0]
        uv = np.delete(V0[vid], axis, 1)
        h = V0[vid, axis]
        mid, half = uv.mean(0), np.maximum(np.ptp(uv, 0) / 2, 1e-9)

        def design(pts):
            x, y = ((np.delete(pts, axis, 1) - mid) / half).T
            return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)

        for l_, design_, coef_, sig_ in fits:                 # a patch of a plane already fitted: the same plane
            if l_ == lab[fsel[0]] and np.median(np.abs(h - design_(V0[vid]) @ coef_)) < 1.5 * sig_:
                design, coef, sig = design_, coef_, sig_
                break
        else:
            A = design(V0[vid])
            w = np.ones(len(h))
            for _ in range(8):                                 # Tukey biweight
                coef = np.linalg.lstsq(A * w[:, None], h * w, rcond=None)[0]
                r = h - A @ coef
                sig = 1.4826 * np.median(np.abs(r[w > 0.1])) + 1e-12
                w = np.clip(1 - (r / (4.685 * sig)) ** 2, 0, 1)
            if is_minor and fits and sig > 2 * np.median([f[3] for f in fits]):
                continue                                       # not a plane: left as it is
            fits.append((lab[fsel[0]], design, coef, sig))
        grown = core
        for _ in range(rings):                                 # the patch and the vertices a few edges around it
            grown = np.minimum(grown + VV @ grown, 1)
        vid = np.nonzero(grown)[0]
        r = V0[vid, axis] - design(V0[vid]) @ coef
        t = np.clip(np.abs(r) / (noise * sig) - 1, 0, 1)       # 0 inside the noise, 1 at twice it
        pull = 1 - t * t * (3 - 2 * t)
        # only what faces this plane: the next wall's vertices are near the plane at the corner, but not of it
        f_ = np.clip((np.abs(vn[vid, axis]) - 0.35) / 0.4, 0, 1)
        pull *= f_ * f_ * (3 - 2 * f_)
        # what stands out in small pieces is noise as well
        outl = np.zeros(nv, bool); outl[vid[pull < 0.999]] = True
        if k_thin:                                             # and in narrow strips: opened away, where it faces the plane
            own = np.zeros(nv, bool); own[vid] = True
            wide = outl | ~own
            for _ in range(k_thin):
                wide &= (VV @ (~wide).astype(np.float32)) == 0
            for _ in range(k_thin + 1):
                wide = (wide | ((VV @ wide.astype(np.float32)) > 0)) & (outl | ~own)
            strip = outl & ~wide & (np.abs(vn[:, axis]) > 0.6)
            pull = np.where(strip[vid], 1.0, pull)
            outl &= ~strip
        ee = e[outl[e[:, 0]] & outl[e[:, 1]]]
        if len(ee):
            _, cl = connected_components(coo_matrix((np.ones(len(ee), bool), ee.T), shape=(nv, nv)), directed=False)
            ca = np.bincount(cl, weights=varea * outl, minlength=cl.max() + 1)
            pull = np.where(ca[cl[vid]] < small * area[c], 1.0, pull)
        V[vid, axis] = V0[vid, axis] - pull * r
        done.append({"axis": "xyz"[axis], "area": round(float(area[c]), 1), "noise": round(float(sig), 4),
                     "moved": round(float((pull > 0.5).mean()), 3)})
        log(f"    plane {'xyz'[axis]}: area {area[c]:.0f}, noise {sig:.3f}, {(pull > 0.5).mean():.0%} of its vertices flattened")
    return trimesh.Trimesh(V, tm.faces, process=False), done


def decimate(tm: trimesh.Trimesh, faces: int, fine: int = 1_500_000) -> trimesh.Trimesh:
    """Quadric decimation. Above `fine` faces: fast-simplification (quick). Below: Open3D's, which keeps the open
    borders in place (boundary weight) and, being error-driven, spends the faces on creases and corners."""
    if len(tm.faces) <= faces:
        return tm
    if len(tm.faces) > fine:
        import fast_simplification
        v, f = fast_simplification.simplify(tm.vertices.astype(np.float32), tm.faces.astype(np.int32),
                                            target_reduction=1.0 - max(faces, fine) / len(tm.faces), agg=5)
        tm = trimesh.Trimesh(v.astype(np.float64), f, process=False)
        if len(tm.faces) <= faces:
            return tm
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices), o3d.utility.Vector3iVector(tm.faces))
    q = om.simplify_quadric_decimation(int(faces), boundary_weight=4.0)
    q.remove_degenerate_triangles(); q.remove_unreferenced_vertices()
    return trimesh.Trimesh(np.asarray(q.vertices), np.asarray(q.triangles), process=False)


def _dominant(n):
    ax = np.argmax(np.abs(n), 1)
    return ax * 2 + (n[np.arange(len(n)), ax] < 0)


def guide(tm: trimesh.Trimesh, iters: int = 12):
    """A coarse mesh's faces labelled by the axis their smoothed normal is nearest: the room's planes (a wall, the
    floor, the ceiling). Every LOD takes its islands from this one guide, so they agree between LODs."""
    n = tm.face_normals * tm.area_faces[:, None]
    adj = tm.face_adjacency
    for _ in range(iters):
        m = n.copy()
        np.add.at(m, adj[:, 0], n[adj[:, 1]]); np.add.at(m, adj[:, 1], n[adj[:, 0]])
        n = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-30) * tm.area_faces[:, None]
    from scipy.spatial import cKDTree
    return cKDTree(tm.triangles_center), _dominant(n)


# ---------------------------------------------------------------- UVs: planar charts on shelves
def box_unwrap(tm: trimesh.Trimesh, res: int, guide=None, pad: int | None = None, min_chart: int = 24,
               votes: int = 3, steep: float = 0.25, density: float | None = None, max_tiles: int = 1):
    """Each face is projected along the axis its normal is nearest (six directions); neighbours vote so the labels
    come in patches; connected patches are charts, the small ones given to a neighbour. Charts are packed
    (guillotine) at one texel density: `density` texels a unit when given (lowered only if the islands do not fit
    max_tiles tiles), else the highest that fits. A tile is res² texels; tile t holds u in [t, t+1) (UDIM 1001+t).
    Returns (vmap, faces, uvs, islands, texels a unit, each face's tile): vertices = tm.vertices[vmap]."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    fn = tm.face_normals
    own = _dominant(fn)
    nf = len(tm.faces)
    pad = max(3, res // 512) if pad is None else pad

    def facing(lab):                                           # cos between the face and its projection axis
        return fn[np.arange(nf), lab // 2] * (1 - 2 * (lab % 2))

    if guide is not None:                                      # the plane the guide gives, unless the face turns away
        lab = guide[1][guide[0].query(tm.triangles_center, workers=-1)[1]]
        lab = np.where(facing(lab) > steep, lab, own)
    else:
        lab = own
    adj = tm.face_adjacency
    for _ in range(votes if guide is None else 0):
        cnt = np.zeros((nf, 6), np.int32)
        np.add.at(cnt, (adj[:, 0], lab[adj[:, 1]]), 1)
        np.add.at(cnt, (adj[:, 1], lab[adj[:, 0]]), 1)
        cnt[np.arange(nf), lab] += 1                           # ties stay
        lab = np.argmax(cnt, 1)

    def charts_of(lab):
        same = lab[adj[:, 0]] == lab[adj[:, 1]]
        k, ch = connected_components(coo_matrix((np.ones(int(same.sum()), bool), adj[same].T), shape=(nf, nf)),
                                     directed=False)
        return k, ch

    for _ in range(3):                                         # small charts take a neighbour's label
        k, ch = charts_of(lab)
        size = np.bincount(ch, minlength=k)
        small = size[ch] < min_chart
        if not small.any():
            break
        cnt = np.zeros((nf, 6), np.int32)
        for a, b in ((0, 1), (1, 0)):
            e = small[adj[:, a]] & (ch[adj[:, a]] != ch[adj[:, b]])
            np.add.at(cnt, (adj[e, a], lab[adj[e, b]]), 1)
        # every face of a small chart takes the chart's most voted neighbour label
        tot = np.zeros((k, 6), np.int64)
        np.add.at(tot, ch[small], cnt[small])
        has = tot.sum(1) > 0
        new = np.argmax(tot, 1)
        ch_small = small & has[ch]
        cand = np.where(ch_small, new[ch], lab)
        lab = np.where(facing(cand) > 0.05, cand, lab)         # never onto a plane the face turns its back to
    k, ch = charts_of(lab)

    # vertices split by chart
    fv = tm.faces.reshape(-1)
    fc = np.repeat(ch, 3)
    pair, inv = np.unique(np.stack([fc, fv], 1), axis=0, return_inverse=True)
    vmap, vchart = pair[:, 1], pair[:, 0]
    faces = inv.reshape(-1, 3)
    chart_lab = np.zeros(k, np.int64); chart_lab[ch] = lab
    axis = chart_lab[vchart] // 2
    P = tm.vertices[vmap]
    uu = np.where(axis == 0, P[:, 1], P[:, 0])                 # the two axes left after dropping `axis`
    vv = np.where(axis == 2, P[:, 1], P[:, 2])
    order = np.argsort(vchart, kind="stable")
    start = np.searchsorted(vchart[order], np.arange(k))
    umin, umax = np.minimum.reduceat(uu[order], start), np.maximum.reduceat(uu[order], start)
    vmin, vmax = np.minimum.reduceat(vv[order], start), np.maximum.reduceat(vv[order], start)
    w, h = umax - umin, vmax - vmin
    rot = h > w                                                # lie every chart down: wider than tall
    w, h = np.where(rot, h, w), np.where(rot, w, h)

    def pack(tpu):
        """Guillotine packing, the largest islands first, each into the free rectangle it fits best (the least
        left over on its short side); the rectangle is cut in two along its longer leftover. More tiles are opened
        as needed (None past max_tiles)."""
        W = np.ceil(w * tpu).astype(np.int64) + 2 * pad + 1
        H = np.ceil(h * tpu).astype(np.int64) + 2 * pad + 1
        if W.max() > res or H.max() > res:
            return None
        x = np.zeros(k, np.int64); y = np.zeros(k, np.int64); til = np.zeros(k, np.int64)
        free = np.array([[0, 0, 0, res, res]], np.int64)       # tile, x, y, width, height
        tiles = 1
        for i in np.argsort(-(W * H), kind="stable"):
            fit = (free[:, 3] >= W[i]) & (free[:, 4] >= H[i])
            if not fit.any():
                if tiles >= max_tiles:
                    return None
                free = np.vstack([free, [tiles, 0, 0, res, res]]); tiles += 1
                fit = (free[:, 3] >= W[i]) & (free[:, 4] >= H[i])
            cand = np.nonzero(fit)[0]
            j = cand[np.argmin(np.minimum(free[cand, 3] - W[i], free[cand, 4] - H[i]))]
            t, fx, fy, fw, fh = free[j]
            til[i], x[i], y[i] = t, fx, fy
            dw, dh = fw - W[i], fh - H[i]
            if dw > dh:                                        # cut so the larger piece stays whole
                new = [[t, fx + W[i], fy, dw, fh], [t, fx, fy + H[i], W[i], dh]]
            else:
                new = [[t, fx, fy + H[i], fw, dh], [t, fx + W[i], fy, dw, H[i]]]
            new = [r for r in new if r[3] > 2 * pad + 1 and r[4] > 2 * pad + 1]
            free = np.vstack([np.delete(free, j, 0)] + ([new] if new else []))
        return x, y, til

    cap = (res - 2 * pad - 2) / max(w.max(), 1e-9)             # the largest island must fit a tile
    best = None
    if density is not None and pack(min(density, cap)) is not None:
        best = (min(density, cap), pack(min(density, cap)))
    else:
        lo_t, hi_t = 0.0, min(density, cap) if density is not None else cap
        for _ in range(14):
            mid = 0.5 * (lo_t + hi_t)
            r = pack(mid)
            if r is None:
                hi_t = mid
            else:
                best, lo_t = (mid, r), mid
    if best is None:
        raise RuntimeError("box_unwrap: the islands do not fit the atlas")
    tpu, (x, y, til) = best
    cu = np.where(rot[vchart], vv - vmin[vchart], uu - umin[vchart])
    cv = np.where(rot[vchart], uu - umin[vchart], vv - vmin[vchart])
    uvs = np.stack([(x[vchart] + pad + 0.5 + cu * tpu) / res + til[vchart], (y[vchart] + pad + 0.5 + cv * tpu) / res], 1)
    return vmap, faces, uvs.astype(np.float32), int(k), float(tpu), til[ch]


def rasterize_uv(faces, uvs, res: int):
    """Which face covers every texel, and where (texture.rasterize_uv, vectorised over faces grouped by the size of
    their texel boxes: scene meshes have millions of faces). Returns (fid HxW int32, -1 = none; bary HxWx3)."""
    fid = np.full((res, res), -1, np.int32)
    bary = np.zeros((res, res, 3), np.float32)
    px = (uvs * res - 0.5).astype(np.float64)
    tri = px[faces]                                            # F × 3 × 2
    mn = np.floor(tri.min(1)).astype(np.int64)
    mx = np.ceil(tri.max(1)).astype(np.int64)
    size = (mx - mn + 1).max(1)
    v0, v1 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    valid = np.abs(den) > 1e-12
    cls = np.ceil(np.log2(np.maximum(size, 2))).astype(int)
    for c in np.unique(cls[valid]):
        S = 2 ** c
        grp = np.nonzero(valid & (cls == c))[0]
        oy, ox = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
        ox, oy = ox.ravel(), oy.ravel()
        for ch in np.array_split(grp, max(1, len(grp) * S * S // 4_000_000 + 1)):
            X = mn[ch, 0, None] + ox[None]
            Y = mn[ch, 1, None] + oy[None]
            dx = X - tri[ch, 0, 0, None]
            dy = Y - tri[ch, 0, 1, None]
            d = den[ch, None]
            w1 = (dx * v1[ch, 1, None] - v1[ch, 0, None] * dy) / d
            w2 = (v0[ch, 0, None] * dy - dx * v0[ch, 1, None]) / d
            w0 = 1 - w1 - w2
            eps = -0.5 / np.sqrt(np.maximum(np.abs(d), 1.0))   # conservative: the edge texels too
            ok = (w0 >= eps) & (w1 >= eps) & (w2 >= eps) & (X >= 0) & (X < res) & (Y >= 0) & (Y < res)
            fi, ti = np.nonzero(ok)
            yy, xx = Y[fi, ti], X[fi, ti]
            fid[yy, xx] = ch[fi]
            bary[yy, xx] = np.stack([w0[fi, ti], w1[fi, ti], w2[fi, ti]], 1)
    return fid, bary


def interp(attr, faces, fid, bary, unit: bool = False):
    """A vertex attribute at every covered texel (HxWxC, zeros elsewhere)."""
    mask = fid >= 0
    out = np.zeros(fid.shape + attr.shape[1:], np.float32)
    f = faces[fid[mask]]
    out[mask] = np.einsum("nk,nk...->n...", bary[mask], attr[f].astype(np.float32))
    if unit:
        out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
    return out


# ---------------------------------------------------------------- tangents, detail maps
def tangents(verts, faces, uvs, normals):
    """Per-vertex tangents the MikkTSpace way: each face's tangent (dP/du) summed at its corners weighted by the
    corner's angle, made orthogonal to the vertex normal; the bitangent's sign is the face's UV orientation. The
    vertices are already split by UV chart, which is how MikkTSpace groups them. Returns (T Vx3, sign V)."""
    P, st = verts[faces], uvs[faces].astype(np.float64)
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    d1, d2 = st[:, 1] - st[:, 0], st[:, 2] - st[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ft = (e1 * d2[:, 1, None] - e2 * d1[:, 1, None]) * np.sign(det)[:, None]
    ft /= np.maximum(np.linalg.norm(ft, axis=1, keepdims=True), 1e-30)
    T = np.zeros_like(verts, dtype=np.float64)
    sg = np.zeros(len(verts))
    for k in range(3):
        a, b = P[:, (k + 1) % 3] - P[:, k], P[:, (k + 2) % 3] - P[:, k]
        ang = np.arccos(np.clip((a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30), -1, 1))
        np.add.at(T, faces[:, k], ft * ang[:, None])
        np.add.at(sg, faces[:, k], np.sign(det) * ang)
    T -= normals * (T * normals).sum(1, keepdims=True)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-30)
    return T.astype(np.float32), np.where(sg < 0, -1.0, 1.0).astype(np.float32)


def detail_maps(pos, N, T, sign, mask, hi: trimesh.Trimesh, reach: float, ao_reach: float, ao_rays: int = 24,
                chunk: int = 1_000_000, seed: int = 0):
    """What the fine mesh has that this LOD lost, for every texel: a ray along the LOD's normal, both ways, finds
    the fine surface within `reach`. Returns (normal map HxWx3 in tangent space, -1..1; height HxW, units along the
    normal; ao HxW, the share of the hemisphere open within ao_reach). Basis: T, sign·N×T, N as interpolated."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(hi.vertices.astype(np.float32), hi.faces.astype(np.uint32))
    hn = hi.vertex_normals.astype(np.float32)
    P, Nn, Tn, sg = pos[mask], N[mask], T[mask], sign[mask]
    nmap = np.zeros((len(P), 3), np.float32); nmap[:, 2] = 1
    height = np.zeros(len(P), np.float32)
    ao = np.ones(len(P), np.float32)
    rng = np.random.default_rng(seed)
    for a in range(0, len(P), chunk):
        p, n, t, s = P[a:a + chunk], Nn[a:a + chunk], Tn[a:a + chunk], sg[a:a + chunk]
        best = np.full(len(p), np.inf, np.float32); hgt = np.zeros(len(p), np.float32)
        wn = n.copy(); hit_p = p.copy()
        for d in (1.0, -1.0):
            h = sc.cast_rays(o3d.core.Tensor(np.concatenate([p, n * d], 1).astype(np.float32)))
            th = h["t_hit"].numpy()
            ok = (th < reach) & (th < best)
            pid, b = h["primitive_ids"].numpy()[ok], h["primitive_uvs"].numpy()[ok]
            f = hi.faces[pid]
            nn = hn[f[:, 0]] * (1 - b[:, 0] - b[:, 1])[:, None] + hn[f[:, 1]] * b[:, 0, None] + hn[f[:, 2]] * b[:, 1, None]
            wn[ok] = nn; best[ok] = th[ok]; hgt[ok] = d * th[ok]
            hit_p[ok] = p[ok] + n[ok] * (d * th[ok])[:, None]
        wn /= np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-12)
        wn = np.where((wn * n).sum(1, keepdims=True) < 0, -wn, wn)
        B = np.cross(n, t) * s[:, None]
        nmap[a:a + chunk] = np.stack([(wn * t).sum(1), (wn * B).sum(1), (wn * n).sum(1)], 1)
        height[a:a + chunk] = hgt
        if ao_rays:
            # cosine-weighted directions about the fine normal, from just off the fine surface
            u1, u2 = rng.random((len(p), ao_rays), np.float32), rng.random((len(p), ao_rays), np.float32)
            r, ph = np.sqrt(u1), 2 * np.pi * u2
            tb = t - wn * (t * wn).sum(1, keepdims=True)
            tb /= np.maximum(np.linalg.norm(tb, axis=1, keepdims=True), 1e-12)
            bb = np.cross(wn, tb)
            dirs = (tb[:, None] * (r * np.cos(ph))[..., None] + bb[:, None] * (r * np.sin(ph))[..., None]
                    + wn[:, None] * np.sqrt(1 - u1)[..., None])
            org = np.broadcast_to((hit_p + wn * (0.02 * ao_reach))[:, None], dirs.shape)
            rays = np.concatenate([org, dirs], -1).reshape(-1, 6).astype(np.float32)
            occ = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(len(p), ao_rays) < ao_reach
            ao[a:a + chunk] = 1 - occ.mean(1)
    out_n = np.zeros(pos.shape, np.float32); out_n[..., 2] = 1
    out_h = np.zeros(mask.shape, np.float32); out_a = np.ones(mask.shape, np.float32)
    out_n[mask] = nmap; out_h[mask] = height; out_a[mask] = ao
    return out_n, out_h, out_a


def uv_metrics(verts, faces, uvs, res: int, n_islands: int, boundary_len: float) -> dict:
    """How good the UVs are. Per face, the map 3D → texels has two stretches (singular values) Γ ≥ γ, in texels a
    unit. density = sqrt(Γγ); angle distortion = Γ/γ (1 is conformal); L2 stretch (Sander) = sqrt((Γ²+γ²)/2) over
    the mean density, area-weighted (1 is isometric); Linf = the largest Γ over the mean density."""
    P, st = verts[faces], uvs[faces].astype(np.float64) * res
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    l1 = np.linalg.norm(e1, axis=1)
    x2 = (e1 * e2).sum(1) / np.maximum(l1, 1e-30)
    y2 = np.linalg.norm(e2 - e1 * (x2 / np.maximum(l1, 1e-30))[:, None], axis=1)
    A = 0.5 * l1 * y2
    ok = A > 1e-12 * max(A.max(), 1e-30)
    d1, d2 = st[:, 1] - st[:, 0], st[:, 2] - st[:, 0]
    # Jacobian columns: d(st)/dx, d(st)/dy in the face's own frame
    jx = d1 / np.maximum(l1, 1e-30)[:, None]
    jy = (d2 - jx * x2[:, None]) / np.maximum(y2, 1e-30)[:, None]
    a, b, c = (jx * jx).sum(1), (jx * jy).sum(1), (jy * jy).sum(1)
    root = np.sqrt(np.maximum((a - c) ** 2 + 4 * b * b, 0))
    G, g = np.sqrt(np.maximum((a + c + root) / 2, 0)), np.sqrt(np.maximum((a + c - root) / 2, 0))
    w = A[ok] / A[ok].sum()
    dens = np.sqrt(G[ok] * g[ok])
    mean = float((w * dens).sum())
    e = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), 1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    cut = ue[cnt == 1]
    seam = float(np.linalg.norm(verts[cut[:, 0]] - verts[cut[:, 1]], axis=1).sum()) - boundary_len
    ang = G[ok] / np.maximum(g[ok], 1e-12)
    o = np.argsort(ang); cw = np.cumsum(w[o])
    return {"islands": int(n_islands), "texels_per_unit": round(mean, 3),
            "density_cv": round(float(np.sqrt((w * (dens - mean) ** 2).sum()) / mean), 4),
            "stretch_l2": round(float(np.sqrt((w * (G[ok] ** 2 + g[ok] ** 2) / 2).sum()) / mean), 4),
            "stretch_linf_p99": round(float(np.percentile(G[ok], 99)) / mean, 3),
            "angle_distortion_mean": round(float((w * ang).sum()), 4),
            "angle_distortion_p95": round(float(ang[o][np.searchsorted(cw, 0.95)]), 3),
            "seam_length": round(max(seam, 0.0) / 2, 2),                # a seam is cut on both of its sides
            "degenerate_uv_faces": int((ok & (g < 1e-6 * mean)).sum())}


def checker(res: int, tpu: float, square: float) -> np.ndarray:
    """A checker whose squares are `square` world units at this atlas's density, tinted by position in the atlas."""
    c = max(1, int(round(tpu * square)))
    y, x = np.mgrid[0:res, 0:res]
    k = ((x // c + y // c) % 2).astype(np.float32)
    img = np.stack([0.35 + 0.5 * k, 0.35 + 0.5 * k * (x / res), 0.35 + 0.5 * k * (y / res)], -1)
    return (img * 255).astype(np.uint8)


def bake(pos, nrm, mask, sts: list[dict], occluder, off: float, keep: int | None = None, tries: int | None = None,
         cos_power: float = 1.0, dist_power: float = 0.0, sharp: float = 1.0, chunk: int = 1_500_000):
    """Texel colour: the mean of the stations that see it, weighted by the cosine of the view angle. A splat's fine
    detail is not the same from two places (its gaussians are not on the surface), so the colour one station sees
    predicts another station's view worse than the mean of all does (compare them with a station kept out:
    --mix best against --mix mean).
    Each station scores cos(angle)^cos_power / distance^dist_power; the stations are tried from the best down, a ray
    to the station tells whether the mesh (`occluder`, an Open3D ray casting scene) is in the way, and the first
    `keep` that see the texel (default: all) are mixed by score^sharp. keep=2, tries=4, cos_power=4, dist_power=2,
    sharp=3 is the best single view with a short blend. Returns (texture HxWx3, covered HxW)."""
    import open3d as o3d
    P_all, N_all = pos[mask].astype(np.float64), nrm[mask].astype(np.float64)
    cols = np.zeros((len(P_all), 3), np.float32)
    good = np.zeros(len(P_all), bool)
    eyes = np.stack([s["eye"] for s in sts])
    tries = min(tries or len(sts), len(sts))
    keep = keep or len(sts)
    for a in range(0, len(P_all), chunk):
        P, N = P_all[a:a + chunk], N_all[a:a + chunk]
        d = eyes[None] - P[:, None]                            # n × S × 3
        dist = np.linalg.norm(d, axis=2)
        cos = (d * N[:, None]).sum(2) / np.maximum(dist, 1e-9)
        score = np.where(cos > 0.05, np.maximum(cos, 0) ** cos_power / np.maximum(dist, 1e-9) ** dist_power, 0.0)
        order = np.argsort(-score, axis=1)[:, :tries]
        acc = np.zeros((len(P), 3)); ws = np.zeros(len(P)); got = np.zeros(len(P), np.int32)
        org = P + N * off
        for r in range(tries):
            sidx = order[:, r]
            sc = score[np.arange(len(P)), sidx]
            todo = np.nonzero((got < keep) & (sc > 0))[0]
            if len(todo) == 0:
                break
            to = eyes[sidx[todo]] - org[todo]
            ln = np.linalg.norm(to, axis=1)
            rays = np.concatenate([org[todo], to / ln[:, None]], 1).astype(np.float32)
            free = occluder.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() >= ln
            todo, sid = todo[free], sidx[todo][free]
            for s_ in np.unique(sid):
                m = todo[sid == s_]
                col, ok = _sample_station(sts[s_], P[m])
                m, col = m[ok], col[ok]
                wgt = score[m, s_] ** sharp
                acc[m] += col * wgt[:, None]; ws[m] += wgt; got[m] += 1
        ok = ws > 0
        cols[a:a + chunk] = np.where(ok[:, None], acc / np.maximum(ws, 1e-300)[:, None], 0)
        good[a:a + chunk] = ok
    tex = np.zeros(pos.shape, np.float32)
    cov = np.zeros(mask.shape, bool)
    tex[mask] = cols; cov[mask] = good
    return tex, cov


def station_agreement(tm: trimesh.Trimesh, sts: list[dict], occluder, off: float, spacing: float, n: int = 120_000):
    """How well each station's picture of the surface agrees with the others'. A splat looks clean only from near
    where it was captured; from elsewhere it shows specks that no other view shares. At n points of the surface
    each station that sees the point gives the brightness steps to four neighbours `spacing` away on the surface
    (the local detail); two stations agree by the correlation of those steps over the points both see. Returns
    (score per station: its mean agreement with the others, the S×S matrix, NaN where two share too little)."""
    import open3d as o3d
    rng = np.random.default_rng(0)
    P, fi = trimesh.sample.sample_surface(tm, n, seed=0)
    N = tm.face_normals[fi]
    a = np.cross(N, rng.normal(size=N.shape)); a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = np.cross(N, a)
    pts = [P, P + a * spacing, P - a * spacing, P + b * spacing, P - b * spacing]
    S = len(sts)
    G = np.full((S, n, 4), np.nan, np.float32)
    org = P + N * off
    for k, st in enumerate(sts):
        to = st["eye"] - org
        ln = np.linalg.norm(to, axis=1)
        rays = np.concatenate([org, to / ln[:, None]], 1).astype(np.float32)
        m = np.nonzero(((to * N).sum(1) / ln > 0.2) & (occluder.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() >= ln))[0]
        ok = np.ones(len(m), bool)
        lum = []
        for q in pts:
            c, o = _sample_station(st, q[m])
            lum.append(c.mean(1)); ok &= o
        G[k, m[ok]] = np.stack([lum[i] - lum[0] for i in range(1, 5)], 1)[ok]
    C = np.full((S, S), np.nan)
    for i in range(S):
        for j in range(i + 1, S):
            m = ~np.isnan(G[i, :, 0]) & ~np.isnan(G[j, :, 0])
            if m.sum() > n // 100:
                x, y = G[i, m].ravel().astype(np.float64), G[j, m].ravel().astype(np.float64)
                C[i, j] = C[j, i] = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))
    with np.errstate(all="ignore"):
        score = np.nan_to_num(np.nanmean(C, 1), nan=0.0)
    return score, C


def bake_pick(pos, nrm, mask, sts: list[dict], occluder, off: float, sigma: float, chunk: int = 1_500_000):
    """Texel colour from one station, the one that agrees best with the rest around the texel. A splat is clean only
    from near where it was captured: elsewhere it shows specks, and the mean of all stations is a blur of them. A
    station's error at a texel is its squared distance to the cosine-weighted mean of the stations that see the
    texel; smoothed over `sigma` texels of the atlas (so the choice holds over a patch, not a texel) and divided by
    the cosine of its view angle, the least wins and gives the texel its colour unmixed.
    Returns (texture HxWx3, covered HxW)."""
    import open3d as o3d
    from scipy.ndimage import gaussian_filter
    P_all, N_all = pos[mask].astype(np.float64), nrm[mask].astype(np.float64)
    n, S = len(P_all), len(sts)
    C = np.zeros((S, n, 3), np.uint8)
    W = np.zeros((S, n), np.float16)                           # the cosine where the station sees the texel, else 0
    for a in range(0, n, chunk):
        P, N = P_all[a:a + chunk], N_all[a:a + chunk]
        org = P + N * off
        for k, st in enumerate(sts):
            to = st["eye"] - org
            ln = np.linalg.norm(to, axis=1)
            cos = ((st["eye"] - P) * N).sum(1) / np.maximum(np.linalg.norm(st["eye"] - P, axis=1), 1e-9)
            m = np.nonzero(cos > 0.05)[0]
            if len(m) == 0:
                continue
            rays = np.concatenate([org[m], to[m] / ln[m, None]], 1).astype(np.float32)
            m = m[occluder.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() >= ln[m]]
            col, ok = _sample_station(st, P[m])
            m = m[ok]
            C[k, a + m] = (col[ok] * 255).round().astype(np.uint8)
            W[k, a + m] = cos[m]
    acc = np.zeros((n, 3), np.float32); ws = np.zeros(n, np.float32)
    for k in range(S):
        w = W[k].astype(np.float32)
        acc += C[k] * w[:, None]; ws += w
    good = ws > 0
    mean = acc / np.maximum(ws, 1e-9)[:, None]
    del acc
    best = np.full(n, np.inf, np.float32); pick = np.zeros(n, np.int16)
    img = np.zeros(mask.shape, np.float32); den = np.zeros(mask.shape, np.float32)
    for k in range(S):
        vis = W[k] > 0
        if not vis.any():
            continue
        img[:] = 0; den[:] = 0
        e = ((C[k].astype(np.float32) - mean) ** 2).sum(1)
        img[mask] = np.where(vis, e, 0); den[mask] = vis
        cost = (gaussian_filter(img, sigma)[mask] / np.maximum(gaussian_filter(den, sigma)[mask], 1e-3)
                / np.maximum(W[k].astype(np.float32), 0.05))
        cost[~vis] = np.inf
        better = cost < best
        best[better] = cost[better]; pick[better] = k
    cols = C[pick, np.arange(n)].astype(np.float32) / 255.0
    tex = np.zeros(pos.shape, np.float32)
    cov = np.zeros(mask.shape, bool)
    tex[mask] = np.where(good[:, None], cols, 0); cov[mask] = good
    return tex, cov


# ---------------------------------------------------------------- front views: every plane seen along its normal
def front_views(tm: trimesh.Trimesh, pixel: float, clear: float, size: int = 2048, far: float = 20.0,
                min_area: float = 0.004, overlap: float = 0.08, facing: float = 0.9, max_views: int = 96,
                eyes=None, occluder=None, off: float = 0.0) -> dict:
    """Cameras that look at the scene's planes head on. A splat's gaussians lie a little before and behind the
    surface, so a view from the side shifts their colour along the surface by that depth times the tangent of the
    view angle, and the mean of many such views is a blur. Along the normal the depth shifts nothing.
    For each of the six axis directions the faces that look that way are grouped in slabs (their offsets along the
    axis in bins of `clear`, the bins that hold a plane's worth of area, neighbours joined); a slab of at least `min_area` of the surface gets a mosaic of cameras, each
    `far` times its own span away (the rays are parallel to within 1/(2·far)), with the near plane `clear` in front
    of the slab: what stands between is cut away, the opposite wall first. A pixel is `pixel` scene units, made
    larger when the mosaics would pass `max_views`. With `eyes` (the stations) and `occluder`, a slab most of which
    no station sees gets none: it is the far side of a wall. Returns cameras as cube_views does, each view with its own
    focal_mm and near, and n (the slab's normal)."""
    from .athenea import APERTURE
    fn, fa, fc = tm.face_normals, tm.area_faces, tm.triangles_center
    total = fa.sum()
    slabs = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        for sgn in (1.0, -1.0):
            n = np.zeros(3); n[ax] = sgn
            sel = np.nonzero(fn[:, ax] * sgn > facing)[0]
            if len(sel) == 0:
                continue
            h = fc[sel, ax] * sgn                               # towards the air
            b = np.floor((h - h.min()) / clear).astype(int)
            area = np.bincount(b, fa[sel])
            occ = area >= min_area * total                      # a plane stacks its area in a bin; stray faces do not
            start = np.nonzero(occ & ~np.r_[False, occ[:-1]])[0]
            stop = np.nonzero(occ & ~np.r_[occ[1:], False])[0]
            for b0, b1 in zip(start, stop):
                m = sel[(b >= b0) & (b <= b1)]
                if eyes is not None and occluder is not None:
                    import open3d as o3d
                    q = m[np.random.default_rng(0).choice(len(m), min(len(m), 400), replace=False)]
                    org = fc[q] + fn[q] * off
                    vis = np.zeros(len(q), bool)
                    for e in np.asarray(eyes, np.float64):
                        to = e - org; ln = np.linalg.norm(to, axis=1)
                        rays = np.concatenate([org, to / ln[:, None]], 1).astype(np.float32)
                        vis |= ((to * fn[q]).sum(1) > 0) & (occluder.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() >= ln)
                    if vis.mean() < 0.5:
                        continue
                v = tm.vertices[tm.faces[m]].reshape(-1, 3)
                lo, hi = np.percentile(v[:, o], 0.5, 0), np.percentile(v[:, o], 99.5, 0)
                slabs.append((n, ax, o, float(h.min() + (b1 + 1) * clear), lo, hi))

    def mosaic(px):
        span = size * px
        return [(max(1, int(np.ceil((hi[0] - lo[0]) / (span * (1 - overlap))))),
                 max(1, int(np.ceil((hi[1] - lo[1]) / (span * (1 - overlap)))))) for *_, lo, hi in slabs]

    count = sum(a * b for a, b in mosaic(pixel))
    if count > max_views:
        pixel *= float(np.sqrt(count / max_views)) * 1.05
    span = size * pixel
    D = far * span
    focal = APERTURE * D / span
    views = []
    for k, ((n, ax, o, top, lo, hi), (nx, ny)) in enumerate(zip(slabs, mosaic(pixel))):
        up = np.zeros(3); up[o[1]] = 1.0
        f = -n
        r = np.cross(f, up); d = np.cross(f, r)
        R = np.stack([r, d, f])
        for i in range(nx):
            for j in range(ny):
                c = np.zeros(3); c[ax] = top * n[ax]
                c[o[0]] = lo[0] + (hi[0] - lo[0]) * (i + 0.5) / nx
                c[o[1]] = lo[1] + (hi[1] - lo[1]) * (j + 0.5) / ny
                eye = c + n * D
                views.append({"name": f"f{k:02d}_{i}_{j}", "eye": eye.tolist(), "target": c.tolist(), "up": up.tolist(),
                              "R_world_to_cam": R.tolist(), "t_world_to_cam": (-R @ eye).tolist(),
                              "focal_mm": float(focal), "near": float(D - clear), "n": n.tolist()})
    fx = focal / APERTURE * size
    return {"convention": "opencv", "pixel": float(pixel), "overlap": overlap,
            "intrinsics": {"width": size, "height": size, "fx": fx, "fy": fx, "cx": size / 2, "cy": size / 2},
            "views": views}


def load_front(out: Path, cams: dict, min_alpha: float = 0.6, median: int = 3) -> list[dict]:
    """The front views that were rendered: [{"eye", "R", "n", "near", "img", "cov"}]. Head on, a splat without its
    view-dependent colour shows stray gaussians as specks: a median of `median` pixels takes them out and keeps the
    edges, and where the splat is thin (alpha under min_alpha) the view does not count."""
    from PIL import Image
    from scipy.ndimage import median_filter
    res = []
    for v in cams["views"]:
        f = Path(out) / "views" / f"{v['name']}.rgba.png"
        if not f.exists():
            continue
        im = np.asarray(Image.open(f))
        if median > 1:
            im = np.dstack([median_filter(im[..., :3], size=(median, median, 1)), im[..., 3]])
        res.append({"eye": np.asarray(v["eye"], np.float64), "R": np.asarray(v["R_world_to_cam"], np.float64),
                    "n": np.asarray(v["n"], np.float64), "near": float(v["near"]),
                    "img": np.ascontiguousarray(im[..., :3]), "cov": im[..., 3] > min_alpha * 255})
    return res


def bake_front(pos, nrm, mask, fviews: list[dict], cams: dict, occluder, off: float, facing: float = 0.8,
               chunk: int = 4_000_000):
    """Texel colour from the front views (front_views): a texel whose normal is within acos(facing) of a view's
    axis, behind that view's near plane and with nothing of the mesh between it and that plane, takes the view's
    colour. Where two views of a mosaic overlap they are cross-faded. Returns (texture HxWx3, covered HxW)."""
    import open3d as o3d
    it = cams["intrinsics"]
    fx, cx, cy, w, h = it["fx"], it["cx"], it["cy"], it["width"], it["height"]
    feather = max(2.0, 0.5 * cams.get("overlap", 0.08) * w)
    P_all, N_all = pos[mask].astype(np.float64), nrm[mask].astype(np.float64)
    cols = np.zeros((len(P_all), 3), np.float32)
    good = np.zeros(len(P_all), bool)
    for a in range(0, len(P_all), chunk):
        P, N = P_all[a:a + chunk], N_all[a:a + chunk]
        acc = np.zeros((len(P), 3)); ws = np.zeros(len(P))
        for fv in fviews:
            cosn = N @ fv["n"]
            m = np.nonzero(cosn > facing)[0]
            if len(m) == 0:
                continue
            c = (P[m] - fv["eye"]) @ fv["R"].T
            u = fx * c[:, 0] / c[:, 2] + cx - 0.5
            v = fx * c[:, 1] / c[:, 2] + cy - 0.5
            edge = np.minimum(np.minimum(u, w - 1 - u), np.minimum(v, h - 1 - v))
            k = (edge > 1) & (c[:, 2] > fv["near"])
            m, c, u, v, edge = m[k], c[k], u[k], v[k], edge[k]
            if len(m) == 0:
                continue
            org = P[m] + N[m] * off
            to = fv["eye"] - org
            ln = np.linalg.norm(to, axis=1)
            zo = (org - fv["eye"]) @ fv["R"][2]
            reach = ln * (1 - fv["near"] / np.maximum(zo, 1e-9))        # as far as the near plane
            rays = np.concatenate([org, to / ln[:, None]], 1).astype(np.float32)
            k = (zo > fv["near"]) & (occluder.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() >= reach)
            m, u, v, edge = m[k], u[k], v[k], edge[k]
            x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int)
            qx, qy = (u - x0)[:, None], (v - y0)[:, None]
            img, cov = fv["img"], fv["cov"]
            col = (img[y0, x0] * (1 - qx) * (1 - qy) + img[y0, x0 + 1] * qx * (1 - qy)
                   + img[y0 + 1, x0] * (1 - qx) * qy + img[y0 + 1, x0 + 1] * qx * qy) / 255.0
            k = cov[y0, x0] & cov[y0 + 1, x0 + 1] & cov[y0, x0 + 1] & cov[y0 + 1, x0]
            wgt = (np.clip(edge / feather, 0, 1) * cosn[m] ** 4)[k]
            acc[m[k]] += col[k] * wgt[:, None]; ws[m[k]] += wgt
        ok = ws > 0
        cols[a:a + chunk] = np.where(ok[:, None], acc / np.maximum(ws, 1e-300)[:, None], 0)
        good[a:a + chunk] = ok
    tex = np.zeros(pos.shape, np.float32)
    cov = np.zeros(mask.shape, bool)
    tex[mask] = cols; cov[mask] = good
    return tex, cov


def render_mesh(verts, faces, uvs, tiles: list, st: dict):
    """The mesh with its base colour (no light) from a station's six cameras: what the splat's views are compared
    with. tiles: the atlas's tiles as stored (HxWx3 uint8, v up from row 0). Returns [(rgb HxWx3 uint8, hit HxW)]."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(np.ascontiguousarray(verts, np.float32), np.ascontiguousarray(faces, np.uint32))
    K = st["K"]
    out = []
    for img, cov, R in st["faces"]:
        h, w = cov.shape
        u, y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        d = np.stack([(u - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ R
        r = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(st["eye"], d.shape), d], -1).astype(np.float32)))
        hit = np.isfinite(r["t_hit"].numpy())
        fi = faces[np.where(hit, r["primitive_ids"].numpy(), 0)]
        b = r["primitive_uvs"].numpy()
        wgt = np.stack([1 - b[..., 0] - b[..., 1], b[..., 0], b[..., 1]], -1)
        stc = (wgt[..., None] * uvs[fi]).sum(-2)
        t = np.clip(np.floor(stc[..., 0]).astype(int), 0, len(tiles) - 1)
        col = np.zeros((h, w, 3), np.float32)
        for k, tex in enumerate(tiles):
            m = hit & (t == k)
            if not m.any():
                continue
            res = tex.shape[0]
            x = np.clip((stc[m][:, 0] - k) * res - 0.5, 0, res - 1.001); yy = np.clip(stc[m][:, 1] * res - 0.5, 0, res - 1.001)
            x0, y0 = np.floor(x).astype(int), np.floor(yy).astype(int)
            fx, fy = (x - x0)[:, None], (yy - y0)[:, None]
            col[m] = (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x0 + 1] * fx * (1 - fy)
                      + tex[y0 + 1, x0] * (1 - fx) * fy + tex[y0 + 1, x0 + 1] * fx * fy)
        out.append((col.round().clip(0, 255).astype(np.uint8), hit))
    return out


def fidelity(verts, faces, uvs, tiles: list, st: dict, png=None) -> dict:
    """How much of the splat the textured mesh keeps: PSNR and SSIM of the mesh's base colour against the splat's
    own view, over the pixels both cover, for each of a station's six cameras, and `sharp`: the variance of the
    mesh picture's Laplacian over that of the splat's (1 is as sharp; PSNR and SSIM alone reward a blur). With png,
    the camera with the most detail side by side at full size: the splat on the left, the mesh on the right."""
    from scipy.ndimage import laplace, binary_erosion
    from skimage.metrics import structural_similarity
    from PIL import Image
    rows, best = [], None
    for j, ((ref, cov, _), (img, hit)) in enumerate(zip(st["faces"], render_mesh(verts, faces, uvs, tiles, st))):
        m = cov & hit
        if m.mean() < 0.2:
            continue
        a, b = ref.astype(np.float64) / 255, img.astype(np.float64) / 255
        mse = float(((a - b) ** 2)[m].mean())
        ssim_map = structural_similarity(a, b, channel_axis=2, data_range=1.0, full=True)[1].mean(2)
        g = np.asarray(Image.fromarray(ref).convert("L"), np.float64)
        detail = float(np.abs(np.diff(g, axis=0))[m[1:]].mean() + np.abs(np.diff(g, axis=1))[m[:, 1:]].mean())
        inner = binary_erosion(m, iterations=2)
        gm = np.asarray(Image.fromarray(img).convert("L"), np.float64)
        sharp = float(laplace(gm)[inner].var() / max(laplace(g)[inner].var(), 1e-12)) if inner.any() else None
        rows.append({"view": j, "psnr": round(10 * np.log10(1 / max(mse, 1e-12)), 2), "ssim": round(float(ssim_map[m].mean()), 4),
                     "shared": round(float(m.mean()), 3), "detail": round(detail, 3),
                     "sharp": None if sharp is None else round(sharp, 3)})
        if best is None or detail > best[0]:
            best = (detail, np.where(cov[..., None], ref, 0).astype(np.uint8), img)
    if png is not None and best is not None:
        Image.fromarray(np.concatenate([best[1], best[2]], 1)).save(png)
    return {"psnr_mean": round(float(np.mean([r["psnr"] for r in rows])), 2) if rows else None,
            "ssim_mean": round(float(np.mean([r["ssim"] for r in rows])), 4) if rows else None,
            "sharp_mean": round(float(np.mean([r["sharp"] for r in rows if r["sharp"] is not None])), 3) if rows else None,
            "views": rows}


def stats(tm: trimesh.Trimesh) -> dict:
    e = np.sort(tm.edges, 1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return {"faces": int(len(tm.faces)), "vertices": int(len(tm.vertices)), "area": float(tm.area),
            "boundary_edges": int((cnt == 1).sum()), "non_manifold_edges": int((cnt > 2).sum())}


# ---------------------------------------------------------------- USD, preview
def write_lod(path: Path, verts, faces, uvs, normals, tang, sign, maps: dict, height=None, roughness: float = 0.8,
              name: str = "Scene", meters_per_unit: float = 1.0, displace: bool = False) -> None:
    """One LOD: a mesh with st, normals and tangents (primvars:tangents, primvars:bitangentSign), and one material
    with a UsdPreviewSurface and a MaterialX standard_surface reading the same files. maps: basecolor, and when
    present normal, occlusion, height, roughness, metallic (file names beside the USD). height = (scale, bias) of
    the height file, scene units. The height's texture and displacement nodes are always written; they are
    connected to the material only with `displace` (a renderer that displaces for real would dent what was made
    plane)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, f"/{name}/Geom")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(verts, np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces, np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)                           # an open shell: seen from the room, harmless behind
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, verts.min(0))), Gf.Vec3f(*map(float, verts.max(0)))])
    pv = UsdGeom.PrimvarsAPI(mesh)
    pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uvs, np.float32)))
    pv.CreatePrimvar("tangents", Sdf.ValueTypeNames.Vector3fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(tang, np.float32)))
    pv.CreatePrimvar("bitangentSign", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
        Vt.FloatArray.FromNumpy(np.ascontiguousarray(sign, np.float32)))

    mp = Sdf.Path(f"/{name}/Materials/Baked")
    mat = UsdShade.Material.Define(stage, mp)
    pbr = UsdShade.Shader.Define(stage, mp.AppendChild("PreviewSurface")); pbr.CreateIdAttr("UsdPreviewSurface")
    rd = UsdShade.Shader.Define(stage, mp.AppendChild("stReader")); rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")

    def tex(node, file, space, scale=None, bias=None):
        t = UsdShade.Shader.Define(stage, mp.AppendChild(node)); t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{file}")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(space)
        for w in ("wrapS", "wrapT"):
            t.CreateInput(w, Sdf.ValueTypeNames.Token).Set("clamp")
        if scale is not None:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        return t

    t = tex("BaseColor", maps["basecolor"], "sRGB"); t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    pbr.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    for key, inp in (("roughness", "roughness"), ("metallic", "metallic"), ("occlusion", "occlusion")):
        if key in maps:
            t = tex(key.capitalize(), maps[key], "raw"); t.CreateOutput("r", Sdf.ValueTypeNames.Float)
            pbr.CreateInput(inp, Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "r")
    if "roughness" not in maps:
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if "metallic" not in maps:
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if "normal" in maps:
        t = tex("Normal", maps["normal"], "raw", (2, 2, 2, 1), (-1, -1, -1, 0)); t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    if "height" in maps and height is not None:
        t = tex("Height", maps["height"], "raw", (height[0],) * 3 + (1,), (height[1],) * 3 + (0,))
        t.CreateOutput("r", Sdf.ValueTypeNames.Float)
        if displace:
            pbr.CreateInput("displacement", Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "r")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

    # MaterialX: standard_surface on the same files
    ng = mp.AppendChild("MtlxGraph")

    def img(node, file, typ):
        sh = UsdShade.Shader.Define(stage, ng.AppendChild(node)); sh.CreateIdAttr(f"ND_image_{typ}")
        fi = sh.CreateInput("file", Sdf.ValueTypeNames.Asset); fi.Set(f"./{file}")
        fi.GetAttr().SetColorSpace("srgb_texture" if typ == "color3" else "raw")
        sh.CreateOutput("out", {"color3": Sdf.ValueTypeNames.Color3f, "vector3": Sdf.ValueTypeNames.Vector3f,
                                "float": Sdf.ValueTypeNames.Float}[typ])
        return sh

    ss = UsdShade.Shader.Define(stage, mp.AppendChild("StandardSurface")); ss.CreateIdAttr("ND_standard_surface_surfaceshader")
    bc = img("base_color", maps["basecolor"], "color3")
    ss.CreateInput("base", Sdf.ValueTypeNames.Float).Set(1.0)
    if "occlusion" in maps:                                    # standard_surface has no AO input: it darkens the base
        mul = UsdShade.Shader.Define(stage, ng.AppendChild("base_ao")); mul.CreateIdAttr("ND_multiply_color3FA")
        mul.CreateInput("in1", Sdf.ValueTypeNames.Color3f).ConnectToSource(bc.ConnectableAPI(), "out")
        mul.CreateInput("in2", Sdf.ValueTypeNames.Float).ConnectToSource(img("occlusion", maps["occlusion"], "float").ConnectableAPI(), "out")
        mul.CreateOutput("out", Sdf.ValueTypeNames.Color3f)
        ss.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(mul.ConnectableAPI(), "out")
    else:
        ss.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(bc.ConnectableAPI(), "out")
    for key, inp in (("roughness", "specular_roughness"), ("metallic", "metalness")):
        if key in maps:
            ss.CreateInput(inp, Sdf.ValueTypeNames.Float).ConnectToSource(img(inp, maps[key], "float").ConnectableAPI(), "out")
    if "roughness" not in maps:
        ss.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if "normal" in maps:
        nm = UsdShade.Shader.Define(stage, ng.AppendChild("normalmap")); nm.CreateIdAttr("ND_normalmap")
        nm.CreateInput("in", Sdf.ValueTypeNames.Vector3f).ConnectToSource(img("normal", maps["normal"], "vector3").ConnectableAPI(), "out")
        nm.CreateOutput("out", Sdf.ValueTypeNames.Vector3f)
        ss.CreateInput("normal", Sdf.ValueTypeNames.Vector3f).ConnectToSource(nm.ConnectableAPI(), "out")
    ss.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mtlx").ConnectToSource(ss.ConnectableAPI(), "out")
    if "height" in maps and height is not None:
        hi = img("height", maps["height"], "float")
        rm = UsdShade.Shader.Define(stage, ng.AppendChild("height_units")); rm.CreateIdAttr("ND_remap_float")
        rm.CreateInput("in", Sdf.ValueTypeNames.Float).ConnectToSource(hi.ConnectableAPI(), "out")
        rm.CreateInput("outlow", Sdf.ValueTypeNames.Float).Set(float(height[1]))
        rm.CreateInput("outhigh", Sdf.ValueTypeNames.Float).Set(float(height[0] + height[1]))
        rm.CreateOutput("out", Sdf.ValueTypeNames.Float)
        dp = UsdShade.Shader.Define(stage, mp.AppendChild("Displacement")); dp.CreateIdAttr("ND_displacement_float")
        dp.CreateInput("displacement", Sdf.ValueTypeNames.Float).ConnectToSource(rm.ConnectableAPI(), "out")
        dp.CreateInput("scale", Sdf.ValueTypeNames.Float).Set(1.0)
        dp.CreateOutput("out", Sdf.ValueTypeNames.Token)
        if displace:
            mat.CreateDisplacementOutput("mtlx").ConnectToSource(dp.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()


def write_lods(out: Path, lods: list[dict], default: int = 1, name: str = "Scene", meters_per_unit: float = 1.0) -> None:
    """out/scene.usda: an Xform with a variantSet "lod"; each variant references that LOD's own file."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateNew(str(Path(out) / "scene.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
    stage.SetDefaultPrim(root)
    vs = root.GetVariantSets().AddVariantSet("lod")
    for l in lods:
        vs.AddVariant(l["name"])
        vs.SetVariantSelection(l["name"])
        with vs.GetVariantEditContext():
            root.GetReferences().AddReference(f"./{l['file']}")
    vs.SetVariantSelection(lods[min(default, len(lods) - 1)]["name"])
    stage.GetRootLayer().Save()


def preview(verts, faces, uvs, tex, eye, png, size: int = 512, light=None, normals=None, nmap=None, tang=None,
            sign=None) -> None:
    """The textured mesh from `eye`, six cube faces in one image (CPU ray casting; back faces in red). tex: the
    atlas, one tile or a list of tiles (HxWx3 uint8, v up from row 0). With `light` (a point), the colour is lit
    from it: by the vertex normals, or by the normal map over them when nmap (a tile or tiles, HxWx3 tangent space,
    -1..1 or as stored in 8 bits), tang and sign are given."""
    import open3d as o3d
    from PIL import Image
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(np.ascontiguousarray(verts, np.float32), np.ascontiguousarray(faces, np.uint32))
    u, v = np.meshgrid((np.arange(size) + .5) / size * 2 - 1, (np.arange(size) + .5) / size * 2 - 1)
    tiles = tex if isinstance(tex, list) else [tex]           # a tile: HxWx3 uint8, v up from row 0
    nmaps = None if nmap is None else (nmap if isinstance(nmap, list) else [nmap])
    out = []
    for f, up in CUBE:
        f, up = np.array(f, float), np.array(up, float)
        r = np.cross(f, up)
        d = f + u[..., None] * r - v[..., None] * up
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        h = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eye, d.shape), d], -1).astype(np.float32)))
        t, pid, b = h["t_hit"].numpy(), h["primitive_ids"].numpy(), h["primitive_uvs"].numpy()
        hit = np.isfinite(t)
        fi = faces[np.where(hit, pid, 0)]
        w = np.stack([1 - b[..., 0] - b[..., 1], b[..., 0], b[..., 1]], -1)
        st = (w[..., None] * uvs[fi]).sum(-2)
        ti = np.clip(np.floor(st[..., 0]).astype(int), 0, len(tiles) - 1)
        col = np.zeros(st.shape[:2] + (3,), np.float32)
        m_ = np.zeros(st.shape[:2] + (3,), np.float32); m_[..., 2] = 1
        for k_, tl in enumerate(tiles):
            sel = ti == k_
            res = tl.shape[0]
            px = np.clip(((st[sel] - [k_, 0]) * res).astype(int), 0, res - 1)
            col[sel] = tl[px[:, 1], px[:, 0]].astype(np.float32) / 255
            if nmaps is not None:
                nm = nmaps[k_][px[:, 1], px[:, 0]]
                m_[sel] = nm / 127.5 - 1 if nm.dtype == np.uint8 else nm
        back = (h["primitive_normals"].numpy() * d).sum(-1) > 0
        if light is not None and normals is not None:
            n = (w[..., None] * normals[fi]).sum(-2)
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
            if nmap is not None:
                tg = (w[..., None] * tang[fi]).sum(-2)
                bt = np.cross(n, tg) * sign[fi[..., 0]][..., None]
                n = tg * m_[..., :1] + bt * m_[..., 1:2] + n * m_[..., 2:]
                n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
            P = eye + d * np.where(hit, t, 0)[..., None]
            L = light - P
            L /= np.maximum(np.linalg.norm(L, axis=-1, keepdims=True), 1e-12)
            col = col * (0.2 + 0.8 * np.clip((n * L).sum(-1), 0, 1))[..., None]
        col = np.where(back[..., None], col * np.array([1, 0.3, 0.3]), col)
        out.append((np.where(hit[..., None], col, 0) * 255).astype(np.uint8))
    Image.fromarray(np.concatenate([np.concatenate(out[:3], 1), np.concatenate(out[3:], 1)], 0)).save(png)
