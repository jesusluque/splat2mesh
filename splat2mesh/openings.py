"""Glass that the splat left as an opening (car windows, lantern panes, display cases).

Glass is nearly invisible to a gaussian splat, so the opacity surface goes *through* the window into the cabin.
The segmenter still finds the window on the orbit views ("the front windshield"), but its pixels back-project
onto whatever lies behind — the seats — and the glass part ends up empty.

Generic fix, only when a described glass part got (almost) no texels:
  1. the object's envelope: its solid closed with a large radius; envelope faces away from the real surface
     are the caps of its openings (windows, but also wheel arches, the underside…);
  2. the orbit cameras are re-cast against mesh + caps; a cap face that is the first hit inside the glass
     phrases' masks votes glass, any other first hit votes not-glass;
  3. connected cap pieces that are mostly glass become /Asset/Glass, a thin-walled glass sheet.
"""
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def envelope_caps(V, F, res: int = 320, radius: float = 0.07, gap_vox: float = 3.0):
    """Faces of the closed envelope that are more than gap_vox voxels from the solid (the openings' caps)."""
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes
    lo, hi = V.min(0), V.max(0)
    step = (hi - lo).max() / (res - 1)
    pad = int(np.ceil(radius * np.linalg.norm(hi - lo) / step)) + 4
    tm = trimesh.Trimesh(V, F, process=False)
    n = np.ceil((hi - lo) / step).astype(int) + 1
    # solid by ray parity along z (robust for a closed shell, cheap)
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(V.astype(np.float32), F.astype(np.uint32))
    gx, gy, gz = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing="ij")
    P = (np.stack([gx, gy, gz], -1).reshape(-1, 3) * step + lo).astype(np.float32)
    solid = np.zeros(len(P), bool)
    for a in range(0, len(P), 4_000_000):
        q = sc.compute_occupancy(o3d.core.Tensor(P[a:a + 4_000_000])).numpy()
        solid[a:a + 4_000_000] = q > 0.5
    solid = np.pad(solid.reshape(n), pad)
    r = radius * np.linalg.norm(hi - lo) / step
    dil = ndi.distance_transform_edt(~solid) <= r
    env = (ndi.distance_transform_edt(dil) > r) | solid
    env = ndi.binary_fill_holes(env)
    dist = ndi.distance_transform_edt(~solid)
    v, f, _, _ = marching_cubes(ndi.gaussian_filter(env.astype(np.float32), 1.0), 0.5)
    v = (v - pad) * step + lo
    em = trimesh.Trimesh(v, f[:, ::-1], process=True)
    vox = np.clip(np.round((em.triangles_center - lo) / step + pad).astype(int), 0, np.array(solid.shape) - 1)
    cap = dist[vox[:, 0], vox[:, 1], vox[:, 2]] > gap_vox
    if not cap.any():
        return None
    return em.submesh([np.nonzero(cap)[0]], append=True)


def orbit_votes(V, F, caps, masks, n: int = 16, elevation: float = 15.0, size: int = 768):
    """Per cap face: (glass votes, views where it is the first hit). Same cameras as materials.render_orbit."""
    import open3d as o3d
    lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; rad = np.linalg.norm(hi - lo) / 2
    Vc, Fc = np.asarray(caps.vertices, np.float32), np.asarray(caps.faces, np.uint32)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(V.astype(np.float32), F.astype(np.uint32))
    cap_id = sc.add_triangles(Vc, Fc)
    f = size / 2 / np.tan(np.radians(15)); dist = rad / np.sin(np.radians(15)) * 1.1
    u, v = np.meshgrid(np.arange(size) + .5, np.arange(size) + .5)
    gv = np.zeros((n, len(Fc))); seen = np.zeros((n, len(Fc)))
    for i in range(n):
        a, e = 2 * np.pi * i / n, np.radians(elevation)
        eye = c + dist * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        fw = c - eye; fw /= np.linalg.norm(fw); rt = np.cross(fw, [0, 1, 0]); rt /= np.linalg.norm(rt); dn = np.cross(fw, rt)
        d = np.stack([(u - size / 2) / f, (v - size / 2) / f, np.ones_like(u)], -1) @ np.stack([rt, dn, fw])
        res = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eye, d.shape), d], -1).reshape(-1, 6).astype(np.float32)))
        gid = res["geometry_ids"].numpy().reshape(size, size); pid = res["primitive_ids"].numpy().reshape(size, size)
        on_cap = gid == cap_id
        np.add.at(seen[i], pid[on_cap], 1)
        g = on_cap & masks[i]
        np.add.at(gv[i], pid[g], 1)
    return gv, seen                                   # per view × cap face


def surface_glass_faces(V, F, masks, min_glass: float = 0.7, min_faces: int = 300, n: int = 16,
                        elevation: float = 15.0, size: int = 768) -> np.ndarray:
    """Faces of the real surface that are glass the generator painted opaque (a grey rear window): seen in front
    on some orbit view with most of their pixels inside the glass masks, kept only as large connected pieces."""
    import open3d as o3d
    lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; rad = np.linalg.norm(hi - lo) / 2
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(V.astype(np.float32), F.astype(np.uint32))
    f = size / 2 / np.tan(np.radians(15)); dist = rad / np.sin(np.radians(15)) * 1.1
    u, v = np.meshgrid(np.arange(size) + .5, np.arange(size) + .5)
    best = np.zeros(len(F), np.float32)
    for i in range(n):
        a, e = 2 * np.pi * i / n, np.radians(elevation)
        eye = c + dist * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        fw = c - eye; fw /= np.linalg.norm(fw); rt = np.cross(fw, [0, 1, 0]); rt /= np.linalg.norm(rt); dn = np.cross(fw, rt)
        d = np.stack([(u - size / 2) / f, (v - size / 2) / f, np.ones_like(u)], -1) @ np.stack([rt, dn, fw])
        res = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eye, d.shape), d], -1).reshape(-1, 6).astype(np.float32)))
        pid = res["primitive_ids"].numpy().reshape(size, size)
        hit = pid != o3d.t.geometry.RaycastingScene.INVALID_ID
        px = np.bincount(pid[hit], minlength=len(F)); gl = np.bincount(pid[hit & masks[i]], minlength=len(F))
        ok = px >= 1
        best[ok] = np.maximum(best[ok], gl[ok] / px[ok])
    cand = best > min_glass
    if not cand.any():
        return cand
    tm = trimesh.Trimesh(V, F, process=False)
    adj = tm.face_adjacency
    adj = adj[cand[adj[:, 0]] & cand[adj[:, 1]]]
    lab = trimesh.graph.connected_component_labels(adj, node_count=len(F))
    cnt = np.bincount(lab[cand], minlength=lab.max() + 1)
    return cand & (cnt[lab] >= min_faces)


def surface_glass_texels(asset_dir, R: int) -> np.ndarray:
    """Texels on the faces surface_glass_faces picked (stored with the panes)."""
    from scipy.spatial import cKDTree
    from .materials import _raster
    from .rerender import load_usd_textured
    asset = Path(asset_dir)
    out = np.zeros((R, R), bool)
    f = asset / "glass_openings.npz"
    if not f.exists() or "surface_faces" not in np.load(f).files:
        return out
    fi = np.load(f)["surface_faces"]
    if not len(fi):
        return out
    V, F, _, _ = load_usd_textured(str(asset / "asset.usda"))
    cen = V[F[fi]].mean(1)
    edge = float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
    pos, _, msk = _raster(asset, R)
    P = pos[msk]
    d, _ = cKDTree(cen).query(P, k=1, workers=-1, distance_upper_bound=1.2 * edge)
    out[msk] = np.isfinite(d)
    return out


def near_pane_texels(asset_dir, R: int, near: float = 0.035) -> np.ndarray:
    """Texels of the main surface within near × object size of a glass pane: the faint glass gaussians that
    crossed the opacity level and hang in the opening as lumps. They are glass too."""
    from scipy.spatial import cKDTree
    from .materials import _raster
    asset = Path(asset_dir)
    f = asset / "glass_openings.npz"
    out = np.zeros((R, R), bool)
    if not f.exists():
        return out
    z = np.load(f)
    pos, _, msk = _raster(asset, R)
    P = pos[msk]
    size = float(np.linalg.norm(P.max(0) - P.min(0)))
    tri = z["V"][z["F"]]                                          # dense samples of the pane
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    w = np.random.default_rng(0).dirichlet((1, 1, 1), (len(tri), 6))
    S = np.einsum("fkj,fji->fki", w, tri).reshape(-1, 3); N = np.repeat(fn, 6, 0)
    d, j = cKDTree(S).query(P, k=1, workers=-1, distance_upper_bound=near * size)
    ok = np.isfinite(d)
    behind = np.zeros(len(P), bool)                               # inside the opening, behind the pane (the
    depth = ((P[ok] - S[j[ok]]) * N[j[ok]]).sum(1)                # frame around it is level with it or outside)
    behind[ok] = depth < -0.004 * size
    out[msk] = behind
    return out


def glass_openings(asset_dir, names, share: dict, min_found: float = 0.01, min_glass: float = 0.7) -> dict:
    asset = Path(asset_dir)
    from .materials import _lib
    glass_parts = [n for n in names if _lib(n.split("|")[0]).get("transmission", 0) > 0]
    out = {"glass_parts": len(glass_parts)}
    cache = asset / "glass_openings.npz"
    if not glass_parts or sum(share.get(n, 0) for n in glass_parts) > min_found:
        cache.unlink(missing_ok=True)
        return out
    if cache.exists() and cache.stat().st_mtime > (asset / "mat" / "masks.npz").stat().st_mtime:
        add_to_usd(asset / "asset.usda", asset)                  # same masks, same mesh: reuse the panes
        out["cached"] = True
        return out
    cache.unlink(missing_ok=True)
    z = np.load(asset / "mat" / "masks.npz")
    keys = [n.split("|", 1)[1] if "|" in n else n for n in glass_parts]
    keys = [k for k in keys if k in z.files]
    if not keys:
        return out
    masks = np.any(np.stack([z[k] for k in keys]), 0)
    from .rerender import load_usd_textured
    V, F, _, _ = load_usd_textured(str(asset / "asset.usda"))
    caps = envelope_caps(V, F)
    if caps is None:
        return out
    gv, seen = orbit_votes(V, F, caps, masks)
    keep = np.zeros(len(caps.faces), bool)
    labels = trimesh.graph.connected_component_labels(caps.face_adjacency, node_count=len(caps.faces))
    for l in np.unique(labels):
        sel = labels == l
        px = seen[:, sel].sum(1); gl = gv[:, sel].sum(1)           # per view: visible pixels, glass pixels
        views = np.nonzero((px >= 200) & (px >= 0.25 * px.max()))[0]
        if not len(views):
            continue
        # the segmenter tracks a window on some views of the orbit and loses it on others: one view that sees
        # the piece well and calls most of it glass is enough (a wheel arch is never called glass)
        if (gl[views] / px[views]).max() > min_glass:
            keep |= sel
    out.update(cap_faces=int(len(caps.faces)), glass_faces=int(keep.sum()))
    if keep.sum() < 20:
        return out
    g = caps.submesh([np.nonzero(keep)[0]], append=True)
    trimesh.smoothing.filter_taubin(g, lamb=0.5, nu=0.53, iterations=30)
    # what the glass looked like: the generator painted what is seen through it (a dark cabin, reflections);
    # its median colour inside the glass masks becomes the pane's transmission tint, so the pane shows what the
    # picture showed instead of the invented interior behind it
    frames = sorted((asset / "mat" / "orbit" / "frames").glob("*.jpg"))
    px = np.concatenate([np.asarray(Image.open(f).convert("RGB"), np.float32)[masks[i]] / 255
                         for i, f in enumerate(frames[:len(masks)]) if masks[i].any()])
    srgb = np.median(px, 0) if len(px) else np.array([0.9, 0.92, 0.95])
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    tint = np.clip(lin, 0.03, 1.0)
    out["tint"] = [round(float(x), 3) for x in tint]
    sf = np.nonzero(surface_glass_faces(V, F, masks))[0]
    out["surface_glass_faces"] = int(len(sf))
    np.savez_compressed(asset / "glass_openings.npz", V=np.asarray(g.vertices, np.float32),
                        F=np.asarray(g.faces, np.int32), N=np.asarray(g.vertex_normals, np.float32),
                        tint=tint.astype(np.float32), surface_faces=sf.astype(np.int32))
    add_to_usd(asset / "asset.usda", asset)
    return out


def define_glass(stage, path: str, asset: Path, mat_path: str, ior: float = 1.5):
    """Glass sheet mesh + thin-walled glass material (UsdPreviewSurface + MaterialX). None if there is none."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    f = asset / "glass_openings.npz"
    if not f.exists():
        return None
    z = np.load(f)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(z["V"]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(z["F"]), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(z["F"].ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(z["N"])); mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)
    lo, hi = z["V"].min(0), z["V"].max(0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    tint = tuple(float(x) for x in z["tint"]) if "tint" in z.files else (0.9, 0.93, 0.95)
    mat = UsdShade.Material.Define(stage, mat_path)
    pbr = UsdShade.Shader.Define(stage, mat_path + "/PreviewSurface"); pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tint)
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.03)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(np.clip(1 - np.mean(tint), 0.15, 0.9)))
    pbr.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    ss = UsdShade.Shader.Define(stage, mat_path + "/StandardSurface"); ss.CreateIdAttr("ND_standard_surface_surfaceshader")
    ss.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
    ss.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(0.03)
    ss.CreateInput("specular_IOR", Sdf.ValueTypeNames.Float).Set(ior)
    ss.CreateInput("transmission", Sdf.ValueTypeNames.Float).Set(1.0)
    ss.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).Set(tint)
    ss.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    ss.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mtlx").ConnectToSource(ss.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    mesh.GetPrim().SetCustomDataByKey("splat2mesh:material", "glass")
    return mesh


def add_to_usd(usd_path: Path, asset: Path):
    from pxr import Usd
    st = Usd.Stage.Open(str(usd_path))
    st.RemovePrim("/Asset/Glass"); st.RemovePrim("/Asset/Materials/M_glass_openings")
    define_glass(st, "/Asset/Glass", asset, "/Asset/Materials/M_glass_openings")
    st.GetRootLayer().Save()


if __name__ == "__main__":
    import sys
    a = Path(sys.argv[1])
    spec = json.loads((a / "mat" / "regions.json").read_text())
    names = spec["texel_classes"]
    tc = np.load(a / "mat" / "texel_class.npy")
    _, m = None, None
    share = {n: float((tc == i).sum()) / max((tc >= 0).sum(), 1) for i, n in enumerate(names)}
    print(json.dumps(glass_openings(a, names, share)))
