"""Inner structure of transparent objects (filaments, wicks, liquids, ships in bottles).

Direct mode builds one closed solid, so whatever the generator put *inside* a glass part disappears in it.
The splat still has it: opaque gaussians well inside the shell, under a transmissive surface. They are
meshed on their own (opacity field → marching cubes), unwrapped, baked from the same gaussians and added to
the asset as `/Asset/Interior` with its own texture. Nothing here is object specific: an object without
transmissive parts, or with nothing inside them, is left untouched.

    python -m splat2mesh.interior ASSET_DIR SPLAT.ply [--y-down]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .direct import bake_direct, extract, load_gaussians, opacity_field
from .geometry import decimate
from .rerender import load_usd_textured
from .texture import inpaint, rasterize_uv, unwrap


def find_inside(asset: Path, xyz, alpha, rgb, min_depth: float = 0.08, min_alpha: float = 0.3, min_trans: float = 0.4,
                min_colour_diff: float = 0.0):
    """Gaussians at least `min_depth` (fraction of the object's size) inside the shell whose closest surface
    is transmissive."""
    V, F, st, _ = load_usd_textured(str(asset / "asset.usda"))
    T = np.asarray(Image.open(asset / "transmission.png"), np.float32)[::-1] / 255
    if T.ndim == 3:
        T = T[..., 0]
    R = T.shape[0]
    cen = st[F].mean(1)
    face_t = T[np.clip((cen[:, 1] * R).astype(int), 0, R - 1), np.clip((cen[:, 0] * R).astype(int), 0, R - 1)]
    if (face_t >= min_trans).mean() < 0.02:
        return None, V, F
    mesh = trimesh.Trimesh(V, F, process=False)
    pq = trimesh.proximity.ProximityQuery(mesh)
    size = np.linalg.norm(V.max(0) - V.min(0))
    probe = np.random.default_rng(0).choice(len(xyz), min(3000, len(xyz)), replace=False)
    if np.median(pq.on_surface(xyz[probe])[1]) > 0.02 * size:     # splat and mesh not in the same frame: skip
        return None, V, F
    cand = np.nonzero(alpha > min_alpha)[0]
    _, dist, tri = pq.on_surface(xyz[cand])
    deep = dist > min_depth * size
    cand, tri = cand[deep], tri[deep]
    if len(cand) == 0:
        return cand, V, F
    inside = mesh.contains(xyz[cand]) if mesh.is_watertight else pq.signed_distance(xyz[cand]) > 0
    sel = cand[inside & (face_t[tri] >= min_trans)]
    # the glass seen through the glass (refraction Idol paints into the volume) has the glass's own colour:
    # only what differs from it is inner structure
    if len(sel):
        from .materials import _srgb_to_lab
        base = np.asarray(Image.open(asset / "basecolor_delit.png").convert("RGB"), np.float32)[::-1] / 255
        glass = _srgb_to_lab(np.median(base[T >= min_trans], 0)[None])[0]
        sel = sel[np.linalg.norm(_srgb_to_lab(rgb[sel]) - glass, axis=1) > min_colour_diff]
    return sel, V, F


def extract_interior(asset_dir, ply, y_down: bool = False, res: int = 384, faces: int = 30000,
                     tex_res: int = 1024, min_gaussians: int = 300) -> dict:
    asset = Path(asset_dir)
    xyz, scale, R, inv, alpha, rgb = load_gaussians(ply, y_down=y_down)
    sel, V, F = find_inside(asset, xyz, alpha, rgb)
    out = {"inside_gaussians": 0 if sel is None else int(len(sel))}
    if sel is None or len(sel) < min_gaussians:
        (asset / "interior.npz").unlink(missing_ok=True)
        return out
    occ, lo, step = opacity_field(xyz[sel], scale[sel], inv[sel], alpha[sel], res=res)
    try:
        tm = extract(occ, lo, step, level=0.4, close_iters=0, blur=0.5, fill=False)
    except (ValueError, RuntimeError):          # nothing above the level: no inner structure
        tm = None
    if tm is None or len(tm.faces) < 50:
        (asset / "interior.npz").unlink(missing_ok=True)
        return out
    if len(tm.faces) > faces:
        tm = decimate(tm, faces)
    vmap, fi, uv, _ = unwrap(tm, tex_res)
    Vi, Fi = np.asarray(tm.vertices)[vmap], fi
    Ni = np.asarray(trimesh.Trimesh(Vi, Fi, process=False).vertex_normals)
    pos, nrm, mask = rasterize_uv(Vi, Fi, uv, Ni, tex_res)
    tex, known = bake_direct(pos, nrm, mask, xyz[sel], inv[sel], alpha[sel], rgb[sel])
    tex = inpaint(tex, known)
    Image.fromarray((tex[::-1].clip(0, 1) * 255).astype(np.uint8)).save(asset / "interior_texture.png")
    np.savez_compressed(asset / "interior.npz", V=Vi.astype(np.float32), F=Fi.astype(np.int32),
                        st=uv.astype(np.float32), N=Ni.astype(np.float32))
    out.update(faces=int(len(Fi)), vertices=int(len(Vi)))
    add_to_usd(asset / "asset.usda", asset)
    if (asset / "asset_parts.usda").exists():
        from .materials import export_parts
        export_parts(asset); export_parts(asset, "asset_materials.usda", group="material")
    return out


def define_interior(stage, path: str, asset: Path, mat_path: str):
    """Mesh + simple textured non-metal material for the interior; returns the mesh (None if no interior)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    f = asset / "interior.npz"
    if not f.exists():
        return None
    z = np.load(f)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(z["V"]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(z["F"]), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(z["F"].ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(z["N"])); mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    lo, hi = z["V"].min(0), z["V"].max(0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(z["st"]))
    mat = UsdShade.Material.Define(stage, mat_path)
    pbr = UsdShade.Shader.Define(stage, mat_path + "/PreviewSurface"); pbr.CreateIdAttr("UsdPreviewSurface")
    rd = UsdShade.Shader.Define(stage, mat_path + "/stReader"); rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    t = UsdShade.Shader.Define(stage, mat_path + "/BaseColor"); t.CreateIdAttr("UsdUVTexture")
    t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("interior_texture.png")
    t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    mesh.GetPrim().SetCustomDataByKey("splat2mesh:material", "interior")
    return mesh


def add_to_usd(usd_path: Path, asset: Path):
    from pxr import Usd
    st = Usd.Stage.Open(str(usd_path))
    st.RemovePrim("/Asset/Interior"); st.RemovePrim("/Asset/Materials/M_interior")
    define_interior(st, "/Asset/Interior", asset, "/Asset/Materials/M_interior")
    st.GetRootLayer().Save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("asset"); ap.add_argument("ply"); ap.add_argument("--y-down", action="store_true")
    a = ap.parse_args()
    print(json.dumps(extract_interior(a.asset, a.ply, a.y_down)))
