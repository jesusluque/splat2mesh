"""Glass that refracts, on a splat: the glass of an Idol splat rebuilt as athenea surfels with a light transfer.

Idol leaves glass as a haze of gaussians, and a piece cut out of the object's single mesh is open where other parts
touch it (rays entering there come out black). So, per transparent material of a splat with per-gaussian materials
(gs_materials.write_ply):

  1. its own closed solid: the opacity field of that material's gaussians only (thickened: Idol's glass is faint),
     marching cubes, decimated — watertight, outward normals;
  2. a stage with that mesh and a MaterialX standard_surface: transmission 1, the library ior and roughness, the
     transmission colour from the gaussians' own colour (their hue at full brightness);
  3. athenea mesh2splat --transfer on it: surfels on the surface with normals, cover and the light transfer
     (the reflected field is what a lens reads);
  4. a stage with the relit opaque splat (athenea_splat.write_splat_usd) and the glass clouds.

    python -m splat2mesh.gs_glass SPLAT_MATERIALS.ply OUT_DIR [--y-down] [--max-splats N] [--opaque OPAQUE.usdc]
OUT_DIR/<material>_mesh.usda, <material>_stage.usda, <material>.usdc and scene.usda (the opaque splat + the glass).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .athenea import run_script, _bin, _q
from .materials import _lib

THICKEN = 12.0        # Idol's glass gaussians are faint: their opacity × this (≤ 0.9) builds the solid
FACES = 40000
SMOOTH = 3.0          # Taubin factor (× 10 iterations): marching cubes leaves rippled facets, and each ripple sends a
                      # refracted ray elsewhere in the sky (swirls); real crystal faces are flat


def glass_solid(ply, names, cls, y_down=True, res=256):
    """Closed mesh of the gaussians whose class is cls (trimesh.Trimesh, watertight)."""
    import trimesh
    from .direct import load_gaussians, opacity_field, extract
    from .geometry import decimate
    from .ply import read_vertex
    v = read_vertex(ply)
    mid = np.asarray(v["mat_id"]).astype(int)
    ids = [i for i, n in enumerate(names) if n.split("|")[0] == cls]
    xyz, scale, R, inv, alpha, rgb = load_gaussians(ply, y_down=y_down, min_opacity=0.0)
    sel = np.isin(mid, ids)
    a = np.clip(alpha[sel] * THICKEN, 0, 0.9)
    occ, lo, step = opacity_field(xyz[sel], scale[sel], inv[sel], a, res=res)
    tm = extract(occ, lo, step, level=0.3, blur=1.2)
    parts = sorted(tm.split(only_watertight=False), key=lambda c: -len(c.faces))
    tm = trimesh.util.concatenate([c for c in parts if len(c.faces) > 0.02 * len(parts[0].faces)])
    from .geometry import smooth
    tm = smooth(decimate(tm, FACES), SMOOTH)
    return tm, rgb[sel], alpha[sel]


def glass_stage(tm, rgb, alpha, cls, out_dir) -> Path:
    """<cls>_mesh.usda and <cls>_stage.usda (the mesh with a transmitting MaterialX standard_surface)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    slug = cls.replace(" ", "_").replace("/", "_")
    m = Usd.Stage.CreateNew(str(out / f"{slug}_mesh.usda")); UsdGeom.SetStageUpAxis(m, UsdGeom.Tokens.y)
    g = UsdGeom.Mesh.Define(m, "/Glass/Geom")
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(tm.vertices.astype(np.float32)))
    g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tm.faces), 3, np.int32)))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tm.faces.astype(np.int32).ravel()))
    m.SetDefaultPrim(m.GetPrimAtPath("/Glass")); m.GetRootLayer().Save()
    # transmission colour: the hue of the more opaque half of the glass gaussians (less background in them)
    med = np.median(rgb[alpha > np.percentile(alpha, 50)], 0)
    lin = np.where(med <= 0.04045, med / 12.92, ((med + 0.055) / 1.055) ** 2.4)
    tint = lin / max(lin.max(), 1e-6)
    L = _lib(cls)
    s = Usd.Stage.CreateNew(str(out / f"{slug}_stage.usda"))
    UsdGeom.SetStageUpAxis(s, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(s, 1.0)
    root = UsdGeom.Xform.Define(s, "/World"); s.SetDefaultPrim(root.GetPrim())
    s.DefinePrim("/World/Glass").GetReferences().AddReference(f"./{slug}_mesh.usda", "/Glass")
    mat = UsdShade.Material.Define(s, "/World/Looks/Glass")
    sh = UsdShade.Shader.Define(s, "/World/Looks/Glass/Surface"); sh.CreateIdAttr("ND_standard_surface_surfaceshader")
    for k, val in (("base", 0.0), ("specular", 1.0), ("specular_roughness", float(L.get("roughness", 0.03))),
                   ("specular_IOR", float(L.get("ior") or 1.5)), ("transmission", 1.0), ("metalness", 0.0)):
        sh.CreateInput(k, Sdf.ValueTypeNames.Float).Set(val)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(bool(L.get("thin_walled", False)))
    sh.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, tint)))
    sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateOutput("mtlx:surface", Sdf.ValueTypeNames.Token).ConnectToSource(sh.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(s.OverridePrim("/World/Glass/Geom")).Bind(mat)
    s.GetRootLayer().Save()
    return out / f"{slug}_stage.usda"


def mesh2splat(stage, out_usdc, max_splats: int = 0, log=None) -> Path:
    """athenea mesh2splat --transfer (surfels with normals, cover and the light transfer)."""
    args = [_bin(), "mesh2splat", str(Path(stage).resolve()), "--prim", "/World/Glass", "--transfer",
            "-o", str(Path(out_usdc).resolve())] + (["--max-splats", str(max_splats)] if max_splats else [])
    run_script([[str(a) for a in args]], log=log)
    return Path(out_usdc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply"); ap.add_argument("out"); ap.add_argument("--y-down", action="store_true")
    ap.add_argument("--max-splats", type=int, default=0); ap.add_argument("--opaque", default=None,
                    help="the relit opaque splat (athenea_splat output) to put in scene.usda with the glass")
    a = ap.parse_args()
    from .ply import read_vertex
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = json.loads(Path(a.ply).with_name("mat_names.json").read_text())["mat_id"]
    v = read_vertex(a.ply); mid = np.asarray(v["mat_id"]).astype(int)
    classes = sorted({n.split("|")[0] for i, n in enumerate(names)
                      if _lib(n.split("|")[0]).get("transmission", 0) > 0 and (mid == i).any()})
    refs = []
    for cls in classes:
        tm, rgb, alpha = glass_solid(a.ply, names, cls, a.y_down)
        print(f"{cls}: solid {len(tm.faces)} faces, watertight {tm.is_watertight}", flush=True)
        st = glass_stage(tm, rgb, alpha, cls, out)
        u = mesh2splat(st, out / f"{cls.replace(' ', '_').replace('/', '_')}.usdc", a.max_splats, log=out / "mesh2splat.log")
        refs.append(u.name)
    lines = ['#usda 1.0', '(', '    defaultPrim = "World"', '    metersPerUnit = 1', '    upAxis = "Y"', ')',
             'def Xform "World"', '{']
    if a.opaque:
        lines.append(f'    def "Body" (prepend references = @{Path(a.opaque).resolve()}@</Splat/Opaque>) {{}}')
    for i, r in enumerate(refs):
        lines.append(f'    def "Glass{i}" (prepend references = @./{r}@) {{}}')
    lines.append('}')
    (out / "scene.usda").write_text("\n".join(lines) + "\n")
    print({"glass": classes, "scene": str(out / "scene.usda")})


if __name__ == "__main__":
    main()
