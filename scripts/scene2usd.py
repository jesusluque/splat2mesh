"""An interior scene's 3DGS .ply (a room captured from inside) → an open, UV-textured USD mesh with LODs.

    python scripts/scene2usd.py SCENE.ply OUT_DIR [--up y|z|-y|-z|x|-x] [--stations 16] [--size 2048]
           [--voxel V] [--faces 1500000,500000,100000,20000] [--tex 8192,4096,4096,2048] [--density 36,18,9,4.5]

The room's inner surface is taken from the gaussians on a lattice (CPU); what the room cannot see is dropped, its
large planes are made plane, flat holes with nothing behind are closed, and it is decimated into the LODs. Cameras are
placed inside the room and athenea renders the splat from them (under the configured GPU lock), by default without
its view-dependent colour (--colour base: OUT_DIR/base_colour.ply, OUT_DIR/render_base). Each LOD gets its own atlas:
base colour as the mean of the stations that see each texel, and normal, height and occlusion maps from the finest
mesh. One more station is rendered and kept out of the bake: the textured mesh is compared from it with the splat
without view-dependent colour and with the whole splat. The surface is kept in OUT_DIR/mesh.npz: a second run with
the same input and surface options only bakes again.

Writes OUT_DIR/scene.usda (variantSet "lod": lod0 … lodN, each a reference to lodK.usdc with its maps:
lodK_basecolor, _normal (tangent space), _occlusion, _height; `.png`, or `.<UDIM>.png` tiles 1001, 1002 … when the
density needs more than one tile), OUT_DIR/render and render_base (the views, NAME.rgba.png: 8-bit sRGB and alpha), fidelity_lodK.jpg (the splat as baked from | the mesh, from the
station kept out), preview_lodK*.png (the mesh from the first station; lit with and without the normal map),
uv_checker_lodK.png (the UVs as a checker of equal world squares) and report.json (mesh, UV and fidelity numbers).
--up names the file's up axis; the USD is written Y-up.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from splat2mesh import scene, texture                     # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("ply"); ap.add_argument("out")
ap.add_argument("--up", default="y", choices=["y", "-y", "z", "-z", "x", "-x"], help="the file's up axis")
ap.add_argument("--stations", type=int, default=16, help="camera stations inside the room (six views each); one more is "
                "rendered and kept out of the bake, to measure the result")
ap.add_argument("--size", type=int, default=2048, help="side of a rendered view, pixels")
ap.add_argument("--voxel", type=float, default=None, help="the lattice's voxel, scene units (default: the box's diagonal / 600)")
ap.add_argument("--blur", type=float, default=2.5, help="smoothing of the surface field, voxels (a splat's walls are fuzzy)")
ap.add_argument("--procs", type=int, default=6, help="processes for the lattice's tiles")
ap.add_argument("--flatten", type=float, default=2.5, help="the room's planes made plane: vertices within this many "
                "robust deviations of a plane's fit move onto it (0: off)")
ap.add_argument("--renders", default=None, help="a directory of views rendered by an earlier run of the same input, "
                "stations and size (its OUT_DIR/render), used in place of rendering")
ap.add_argument("--faces", default="1500000,500000,100000,20000", help="faces of each LOD, the finest first")
ap.add_argument("--tex", default="8192,4096,4096,2048", help="side of an atlas tile, each LOD")
ap.add_argument("--density", default=None, help="texels a scene unit, each LOD (default: what one tile holds); "
                "more tiles (UDIM) are added to reach it, up to --max-tiles")
ap.add_argument("--max-tiles", type=int, default=8)
ap.add_argument("--crop-pct", type=float, default=1.0, help="percentile of the centres that bounds the scene")
ap.add_argument("--min-piece", type=float, default=0.0005, help="pieces under this share of the area are dropped")
ap.add_argument("--probes", type=int, default=128, help="points of the room's air a face must be seen from to stay "
                "(0: the hidden surface is kept)")
ap.add_argument("--fill-holes", type=float, default=0.08, help="flat holes up to this share of the box's diagonal are "
                "closed when the splat has nothing behind them (0: none)")
ap.add_argument("--colour", default="base", choices=["base", "full"], help="base: the texture is baked from views of "
                "the splat without its view-dependent colour (they agree with one another); full: from the splat as it is")
ap.add_argument("--mix", default="mean", choices=["mean", "best"], help="a texel's colour: the mean of the stations "
                "that see it, or the one that sees it best")
ap.add_argument("--cleanup", action="store_true", help="when done, delete what a new run can make again in minutes: "
                "the whole splat's views but the check station's, and base_colour.ply")
ap.add_argument("--fresh", action="store_true", help="do not reuse OUT_DIR/mesh.npz of an earlier run")
ap.add_argument("--default-lod", type=int, default=1)
ap.add_argument("--no-detail", action="store_true", help="no normal, height and occlusion maps")
ap.add_argument("--detail-reach", type=float, default=None, help="how far along the normal the fine surface is sought (default 6 voxels)")
ap.add_argument("--ao-reach", type=float, default=None, help="occlusion distance (default 3 %% of the diagonal)")
ap.add_argument("--ao-rays", type=int, default=24)
ap.add_argument("--checker", type=float, default=None, help="side of the UV checker's squares, scene units")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
faces = [int(x) for x in a.faces.split(",")]; texs = [int(x) for x in a.tex.split(",")]
texs += [texs[-1]] * (len(faces) - len(texs))
dens = [float(x) for x in a.density.split(",")] if a.density else [None] * len(faces)
dens += [None] * (len(faces) - len(dens))
rep = {"input": a.ply, "up": a.up}
t0 = time.time()


def log(*x):
    print(*x, flush=True)


def coarse(m):
    m = scene.decimate(m, min(faces))
    return trimesh.Trimesh(m.vertices, m.faces, process=True)


src = Path(a.ply)
key = {"ply": str(src.resolve()), "size": src.stat().st_size, "mtime": src.stat().st_mtime, "voxel": a.voxel, "blur": a.blur,
       "flatten": a.flatten, "stations": a.stations, "crop_pct": a.crop_pct, "min_piece": a.min_piece, "probes": a.probes,
       "fill_holes": a.fill_holes}
cache = out / "mesh.npz"
if cache.exists() and not a.fresh and json.loads(str(np.load(cache)["key"])) == key:
    z = np.load(cache)                                               # the surface of an earlier run: only the bake is redone
    raw, pts, voxel, diag = trimesh.Trimesh(z["v"], z["f"], process=False), z["pts"], float(z["voxel"]), float(z["diag"])
    rep.update(json.loads(str(z["rep"])))
    log(f"surface from {cache}: {scene.stats(raw)}")
else:
    g = scene.load(a.ply, min_opacity=0.005, crop_pct=a.crop_pct)  # in the file's own frame: athenea renders the file
    diag = float(np.linalg.norm(g["hi"] - g["lo"]))
    air, alo, astep = scene.room_air(g, (g["hi"] - g["lo"]).max() / 160)
    pts = scene.stations(air, alo, astep, a.stations + 1)          # the last one: kept out of the bake
    log(f"scene: {len(g['xyz'])} of {g['read']} gaussians, box {(g['hi'] - g['lo']).round(1).tolist()}, "
        f"air {air.mean():.0%} of it, {a.stations} stations + 1 to check")
    voxel = a.voxel or diag / 600
    raw = scene.skin(g, air, alo, astep, voxel, blur=a.blur, procs=a.procs, log=log)
    raw = scene.drop_small(raw, a.min_piece)
    rep.update(gaussians=int(len(g["xyz"])), stations=pts.round(3).tolist(), voxel=voxel, blur=a.blur, raw=scene.stats(raw),
               t_skin_s=round(time.time() - t0, 1))
    log(f"skin at voxel {voxel:.4f}: {rep['raw']}  ({rep['t_skin_s']} s)")
    if a.probes > 0:                                               # what the room cannot see goes
        raw = scene.seen(raw, scene.stations(air, alo, astep, a.probes, inner=0.15), 2 * voxel, log=log)
        raw = scene.drop_small(raw, a.min_piece)
        rep["seen"] = scene.stats(raw)
    gd = scene.guide(coarse(raw))      # the coarsest LOD's planes: they give every LOD its islands
    if a.flatten > 0:
        raw, rep["planes"] = scene.flatten(raw, gd, a.flatten, thin=0.008 * diag, minor=0.002, reach=0.03 * diag, log=log)
    if a.fill_holes > 0:
        raw, rep["holes"] = scene.fill_holes(raw, a.fill_holes * diag, 4 * voxel, g, 0.15 * diag, pore=0.02 * diag, log=log)
    del g
    rep["surface"] = scene.stats(raw)
    np.savez(cache, v=raw.vertices.astype(np.float32), f=raw.faces.astype(np.int32), pts=pts, voxel=voxel, diag=diag,
             key=json.dumps(key), rep=json.dumps(rep))
gd = scene.guide(coarse(raw))

t1 = time.time()
cams = scene.cube_views(pts, a.size)
rdir = Path(a.renders) if a.renders else out / "render"
rdir.mkdir(exist_ok=True)
cj = rdir / "cameras.json"
if a.renders:                                                        # the views as they were rendered, not as recomputed
    cams = json.loads(cj.read_text())
    pts = np.array([v["eye"] for v in cams["views"][::6]])
elif cj.exists() and json.loads(cj.read_text()) != json.loads(json.dumps(cams)):
    for d_ in (rdir, out / "render_base"):
        for f in (d_ / "views").glob("*"):                           # views of other cameras: stale
            f.unlink()
if not a.renders:
    cj.write_text(json.dumps(cams, indent=1))
n_all = len(cams["views"]) // 6
if a.colour == "base":                 # the bake's views: the splat's mean colour; the whole splat only from the check station
    bdir = (rdir.parent if a.renders else out) / "render_base"      # beside the whole splat's views
    bdir.mkdir(exist_ok=True)
    base_ply = bdir.parent / "base_colour.ply"
    if not base_ply.exists():
        scene.base_colour_copy(a.ply, base_ply)
    scene.render(base_ply, cams, bdir, log=log)
    sts = scene.load_views(bdir, cams)
    full_cams = dict(cams, views=cams["views"][-6:])
    scene.render(a.ply, full_cams, rdir, log=log)
    full = scene.load_views(rdir, full_cams)
else:
    scene.render(a.ply, cams, rdir, log=log)
    sts = scene.load_views(rdir, cams)
    full = sts[-1:]
check, sts = (sts[-1], sts[:-1]) if len(sts) == n_all else (None, sts)
check_full = full[0] if check is not None and full else None
rep.update(colour=a.colour, stations_baked=len(sts), check_station=check is not None, t_render_s=round(time.time() - t1, 1))
if check is None:
    log("    some views failed to render: no station is kept out, fidelity is measured from a baked one")

import open3d as o3d                                                 # noqa: E402
occ = o3d.t.geometry.RaycastingScene()
occ.add_triangles(raw.vertices.astype(np.float32), raw.faces.astype(np.uint32))
U = scene.up_matrix(a.up)
reach, ao_reach = a.detail_reach or 6 * voxel, a.ao_reach or 0.03 * diag
square = a.checker or diag / 60
light = pts[0] + 0.2 * (pts[1] - pts[0])


def save(img, path):
    Image.fromarray(img[::-1]).save(path)                              # rows top-down; st has v up


lods, tm = [], raw
for k, (nf, res, den) in enumerate(zip(faces, texs, dens)):
    t2 = time.time()
    tm = scene.decimate(tm, nf)
    tm = trimesh.Trimesh(tm.vertices, tm.faces, process=True)
    st = scene.stats(tm)
    be = tm.edges_sorted[trimesh.grouping.group_rows(tm.edges_sorted, require_count=1)]
    blen = float(np.linalg.norm(tm.vertices[be[:, 0]] - tm.vertices[be[:, 1]], axis=1).sum())
    vmap, uvf, uvs, charts, tpu, ftile = scene.box_unwrap(tm, res, guide=gd, density=den, max_tiles=a.max_tiles)
    nt = int(ftile.max()) + 1
    verts, normals = tm.vertices[vmap], tm.vertex_normals[vmap]
    tang, sign = scene.tangents(verts, uvf, uvs, normals)
    name = (lambda m, t: f"lod{k}_{m}.{1001 + t}.png") if nt > 1 else (lambda m, t: f"lod{k}_{m}.png")
    maps = {"basecolor": f"lod{k}_basecolor.<UDIM>.png" if nt > 1 else f"lod{k}_basecolor.png"}
    ctiles, ntiles, hraw = [], [], []
    used = cov_n = tex_n = 0
    for t in range(nt):
        fs = np.nonzero(ftile == t)[0]
        fid, bary = scene.rasterize_uv(uvf[fs], uvs - [t, 0], res)
        mask = fid >= 0
        pos = scene.interp(verts, uvf[fs], fid, bary)
        nrm = scene.interp(normals, uvf[fs], fid, bary, unit=True)
        col, cov = scene.bake(pos, nrm, mask, sts, occ, off=2 * voxel,
                              **({} if a.mix == "mean" else dict(keep=2, tries=4, cos_power=4.0, dist_power=2.0, sharp=3.0)))
        used += int(mask.sum()); cov_n += int(cov.sum()); tex_n += mask.size
        img = (texture.inpaint(col, cov).clip(0, 1) * 255).round().astype(np.uint8)
        del col
        save(img, out / name("basecolor", t)); ctiles.append(img)
        if not a.no_detail:
            tg = scene.interp(tang, uvf[fs], fid, bary, unit=True)
            sg = np.zeros(mask.shape, np.float32); sg[mask] = sign[uvf[fs][fid[mask], 0]]
            nmap, hmap, ao = scene.detail_maps(pos, nrm, tg, sg, mask, raw, reach, ao_reach, ao_rays=a.ao_rays)
            nimg = (texture.inpaint(nmap * 0.5 + 0.5, mask).clip(0, 1) * 255).round().astype(np.uint8)
            save(nimg, out / name("normal", t)); ntiles.append(nimg)
            aimg = texture.inpaint(np.repeat(ao[..., None], 3, -1), mask)[..., 0]
            save((aimg.clip(0, 1) * 255).round().astype(np.uint8), out / name("occlusion", t))
            hraw.append((hmap, mask))
            del tg, nmap, ao, nimg, aimg
        del pos, nrm, fid, bary
    height = None
    if not a.no_detail:
        maps["normal"] = maps["basecolor"].replace("basecolor", "normal")
        maps["occlusion"] = maps["basecolor"].replace("basecolor", "occlusion")
        hv = np.concatenate([h[m] for h, m in hraw])
        h0, h1 = float(np.percentile(hv, 0.1)), float(np.percentile(hv, 99.9))
        if h1 - h0 > 1e-9:                                     # 16 bit, low..high over the whole LOD
            for t, (h, m) in enumerate(hraw):
                himg = texture.inpaint(np.repeat(((h - h0) / (h1 - h0))[..., None], 3, -1), m)[..., 0]
                Image.fromarray((himg[::-1].clip(0, 1) * 65535).round().astype(np.uint16)).save(out / name("height", t))
            maps["height"] = maps["basecolor"].replace("basecolor", "height"); height = (h1 - h0, h0)
        del hraw
        scene.preview(verts, uvf, uvs, ctiles, pts[0], out / f"preview_lod{k}_lit_normalmap.png", light=light,
                      normals=normals, nmap=ntiles, tang=tang, sign=sign)
        scene.preview(verts, uvf, uvs, ctiles, pts[0], out / f"preview_lod{k}_lit_plain.png", light=light, normals=normals)
        del ntiles
    scene.write_lod(out / f"lod{k}.usdc", verts @ U.T, uvf, uvs, normals @ U.T, tang @ U.T, sign, maps, height)
    scene.preview(verts, uvf, uvs, ctiles, pts[0], out / f"preview_lod{k}.png")
    scene.preview(verts, uvf, uvs, [scene.checker(res, tpu, square)] * nt, pts[0], out / f"uv_checker_lod{k}.png")
    fid_check = scene.fidelity(verts, uvf, uvs, ctiles, check or sts[0], out / f"fidelity_lod{k}.jpg")
    fid_baked = scene.fidelity(verts, uvf, uvs, ctiles, sts[0]) if check is not None else None
    fid_full = scene.fidelity(verts, uvf, uvs, ctiles, check_full) if check_full is not None and a.colour == "base" else None
    del ctiles
    st.update(name=f"lod{k}", file=f"lod{k}.usdc", atlas=res, tiles=nt, maps=maps, height_scale_bias=height,
              uv=scene.uv_metrics(verts, uvf, uvs, res, charts, blen), atlas_used=used / tex_n,
              texels_covered=cov_n / max(used, 1), fidelity_check_station=fid_check, fidelity_baked_station=fid_baked,
              fidelity_check_station_whole_splat=fid_full,
              t_s=round(time.time() - t2, 1))
    lods.append(st)
    log(f"lod{k}: {st['faces']} faces, {charts} islands, {tpu:.2f} texels a unit on {nt} tile(s) of {res}, atlas used "
        f"{used / tex_n:.0%}, covered {cov_n / max(used, 1):.1%}, PSNR {fid_check['psnr_mean']} SSIM {fid_check['ssim_mean']} "
        f"from the station kept out"
        + (f" (against the whole splat, view-dependent colour and all: {fid_full['psnr_mean']} / {fid_full['ssim_mean']})" if fid_full else "")
        + f"  ({st['t_s']} s)")
rep["tangent_space"] = ("per-vertex tangents computed the MikkTSpace way (angle-weighted, split by UV island) and written "
                        "as primvars:tangents / primvars:bitangentSign; normal maps baked in that basis, +Y up (OpenGL)")
scene.write_lods(out, lods, a.default_lod)
rep.update(lods=lods, t_total_s=round(time.time() - t0, 1))
(out / "report.json").write_text(json.dumps(rep, indent=2))
if a.cleanup:
    keep = {v["name"] for v in cams["views"][-6:]}
    if a.colour == "base":
        for f in (rdir / "views").glob("*.rgba.png"):
            if f.name[:-9] not in keep:
                f.unlink()
        (bdir.parent / "base_colour.ply").unlink(missing_ok=True)
log(f"wrote {out / 'scene.usda'} in {rep['t_total_s']} s")
