"""An interior scene's 3DGS .ply (a room captured from inside) → an open, UV-textured USD mesh with LODs.

    python scripts/scene2usd.py SCENE.ply OUT_DIR [--up y|z|-y|-z|x|-x] [--stations 6] [--size 1536]
           [--voxel V] [--faces 3000000,500000,100000,20000] [--tex 8192,8192,4096,2048]

The room's inner surface is taken from the gaussians on a lattice (CPU) and decimated into the LODs; cameras are
placed inside the room and athenea renders the splat's colour from them (under the configured GPU lock); each LOD
gets its own atlas: base colour from those views, and normal, height and occlusion maps from the finest mesh.
Writes OUT_DIR/scene.usda (variantSet "lod": lod0 … lodN, each a reference to lodK.usdc with its maps:
lodK_basecolor.png, _normal.png (tangent space), _occlusion.png, _height.png), OUT_DIR/render (the views),
preview_lodK*.png (the textured mesh from the first station; lit with and without the normal map),
uv_checker_lodK.png (the UVs as a checker of equal world squares) and report.json (mesh and UV metrics a LOD).
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
from splat2mesh.usd_export import write_usd               # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("ply"); ap.add_argument("out")
ap.add_argument("--up", default="y", choices=["y", "-y", "z", "-z", "x", "-x"], help="the file's up axis")
ap.add_argument("--stations", type=int, default=6, help="camera stations inside the room (six views each)")
ap.add_argument("--size", type=int, default=1536, help="side of a rendered view, pixels")
ap.add_argument("--voxel", type=float, default=None, help="the lattice's voxel, scene units (default: the box's diagonal / 600)")
ap.add_argument("--blur", type=float, default=2.5, help="smoothing of the surface field, voxels (a splat's walls are fuzzy)")
ap.add_argument("--procs", type=int, default=6, help="processes for the lattice's tiles")
ap.add_argument("--faces", default="3000000,500000,100000,20000", help="faces of each LOD, the finest first")
ap.add_argument("--tex", default="8192,8192,4096,2048", help="atlas side of each LOD")
ap.add_argument("--crop-pct", type=float, default=1.0, help="percentile of the centres that bounds the scene")
ap.add_argument("--min-piece", type=float, default=0.0005, help="pieces under this share of the area are dropped")
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
rep = {"input": a.ply, "up": a.up}
t0 = time.time()


def log(*x):
    print(*x, flush=True)


g = scene.load(a.ply, min_opacity=0.005, crop_pct=a.crop_pct)      # in the file's own frame: athenea renders the file
diag = float(np.linalg.norm(g["hi"] - g["lo"]))
air, alo, astep = scene.room_air(g, (g["hi"] - g["lo"]).max() / 160)
pts = scene.stations(air, alo, astep, a.stations)
log(f"scene: {len(g['xyz'])} of {g['read']} gaussians, box {(g['hi'] - g['lo']).round(1).tolist()}, "
    f"air {air.mean():.0%} of it, {len(pts)} stations")
voxel = a.voxel or diag / 600
raw = scene.skin(g, air, alo, astep, voxel, blur=a.blur, procs=a.procs, log=log)
raw = scene.drop_small(raw, a.min_piece)
rep.update(gaussians=int(len(g["xyz"])), stations=pts.round(3).tolist(), voxel=voxel, blur=a.blur, raw=scene.stats(raw),
           t_skin_s=round(time.time() - t0, 1))
log(f"skin at voxel {voxel:.4f}: {rep['raw']}  ({rep['t_skin_s']} s)")
del g

t1 = time.time()
cams = scene.cube_views(pts, a.size)
(out / "render").mkdir(exist_ok=True)
(out / "render/cameras.json").write_text(json.dumps(cams, indent=1))
scene.render(a.ply, cams, out / "render", log=log)
views, names = scene.load_views(out / "render", cams)
scene.mesh_depth(views, raw)
rep.update(views=len(views), t_render_s=round(time.time() - t1, 1))
if len(views) < len(cams["views"]):
    log(f"    {len(cams['views']) - len(views)} views failed to render")

U = scene.up_matrix(a.up)
# the guide: the coarsest LOD's planes give every LOD its islands
gd = scene.guide(trimesh.Trimesh(*(lambda m: (m.vertices, m.faces))(scene.decimate(raw, min(faces))), process=True))
hi = trimesh.Trimesh(raw.vertices, raw.faces, process=False)          # what the detail maps are baked from
reach, ao_reach = a.detail_reach or 6 * voxel, a.ao_reach or 0.03 * diag
square = a.checker or diag / 60


def save(img, path):
    Image.fromarray(img[::-1]).save(path)                              # rows top-down; st has v up


lods, tm = [], raw
for k, (nf, res) in enumerate(zip(faces, texs)):
    t2 = time.time()
    tm = scene.decimate(tm, nf)
    tm = trimesh.Trimesh(tm.vertices, tm.faces, process=True)
    st = scene.stats(tm)
    be = tm.edges_sorted[trimesh.grouping.group_rows(tm.edges_sorted, require_count=1)]
    blen = float(np.linalg.norm(tm.vertices[be[:, 0]] - tm.vertices[be[:, 1]], axis=1).sum())
    vmap, uvf, uvs, charts, tpu = scene.box_unwrap(tm, res, guide=gd)
    verts, normals = tm.vertices[vmap], tm.vertex_normals[vmap]
    tang, sign = scene.tangents(verts, uvf, uvs, normals)
    fid, bary = scene.rasterize_uv(uvf, uvs, res)
    mask = fid >= 0
    pos = scene.interp(verts, uvf, fid, bary)
    nrm = scene.interp(normals, uvf, fid, bary, unit=True)
    col, cov = scene.bake(pos, nrm, mask, views, depth_tol=4 * voxel)
    covered = float(cov.sum() / max(mask.sum(), 1))
    img = (texture.inpaint(col, cov)[...].clip(0, 1) * 255).astype(np.uint8)
    del col
    maps = {"basecolor": f"lod{k}_basecolor.png"}
    save(img, out / maps["basecolor"])
    height = None
    if not a.no_detail:
        tg = scene.interp(tang, uvf, fid, bary, unit=True)
        sg = np.zeros(mask.shape, np.float32); sg[mask] = sign[uvf[fid[mask], 0]]
        nmap, hmap, ao = scene.detail_maps(pos, nrm, tg, sg, mask, hi, reach, ao_reach, ao_rays=a.ao_rays)
        nimg = texture.inpaint(nmap * 0.5 + 0.5, mask)
        maps["normal"] = f"lod{k}_normal.png"; save((nimg.clip(0, 1) * 255).round().astype(np.uint8), out / maps["normal"])
        aimg = texture.inpaint(np.repeat(ao[..., None], 3, -1), mask)[..., 0]
        maps["occlusion"] = f"lod{k}_occlusion.png"; save((aimg.clip(0, 1) * 255).round().astype(np.uint8), out / maps["occlusion"])
        h0, h1 = float(np.percentile(hmap[mask], 0.1)), float(np.percentile(hmap[mask], 99.9))
        if h1 - h0 > 1e-9:
            himg = texture.inpaint(np.repeat(((hmap - h0) / (h1 - h0))[..., None], 3, -1), mask)[..., 0]
            maps["height"] = f"lod{k}_height.png"; height = (h1 - h0, h0)
            Image.fromarray((himg[::-1].clip(0, 1) * 65535).round().astype(np.uint16)).save(out / maps["height"])
        scene.preview(verts, uvf, uvs, img[::-1], pts[0], out / f"preview_lod{k}_lit_normalmap.png", light=pts[0] + 0.2 * (pts[-1] - pts[0]),
                      normals=normals, nmap=(nimg * 2 - 1)[::-1], tang=tang, sign=sign)
        scene.preview(verts, uvf, uvs, img[::-1], pts[0], out / f"preview_lod{k}_lit_plain.png", light=pts[0] + 0.2 * (pts[-1] - pts[0]),
                      normals=normals)
        del tg, nmap, hmap, ao
    del pos, nrm
    scene.write_lod(out / f"lod{k}.usdc", verts @ U.T, uvf, uvs, normals @ U.T, tang @ U.T, sign, maps, height)
    scene.preview(verts, uvf, uvs, img[::-1], pts[0], out / f"preview_lod{k}.png")
    scene.preview(verts, uvf, uvs, scene.checker(res, tpu, square)[::-1], pts[0], out / f"uv_checker_lod{k}.png")
    st.update(name=f"lod{k}", file=f"lod{k}.usdc", atlas=res, maps=maps, height_scale_bias=height,
              uv=scene.uv_metrics(verts, uvf, uvs, res, charts, blen), atlas_used=float(mask.mean()),
              texels_covered=covered, t_s=round(time.time() - t2, 1))
    lods.append(st)
    log(f"lod{k}: {st['faces']} faces, {charts} islands, {tpu:.2f} texels a unit, atlas used {mask.mean():.0%}, "
        f"covered {covered:.1%}  ({st['t_s']} s)")
rep["tangent_space"] = ("per-vertex tangents computed the MikkTSpace way (angle-weighted, split by UV island) and written "
                        "as primvars:tangents / primvars:bitangentSign; normal maps baked in that basis, +Y up (OpenGL)")
scene.write_lods(out, lods, a.default_lod)
rep.update(lods=lods, t_total_s=round(time.time() - t0, 1))
(out / "report.json").write_text(json.dumps(rep, indent=2))
log(f"wrote {out / 'scene.usda'} in {rep['t_total_s']} s")
