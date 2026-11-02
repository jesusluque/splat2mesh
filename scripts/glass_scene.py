"""Build a path-traced glass test for athenea: a mesh with a dielectric MaterialX material inside an
emissive checker cube (the "3D checker background"), plus background.json for splat2mesh.glass and
truth.json with the material used.

    python scripts/glass_scene.py MESH_USD OUTDIR [--ior 1.5] [--tint 0.92 0.97 0.95] [--half 6]
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

ap = argparse.ArgumentParser(); ap.add_argument("mesh"); ap.add_argument("out")
ap.add_argument("--ior", type=float, default=1.5); ap.add_argument("--tint", type=float, nargs=3, default=[0.92, 0.97, 0.95])
ap.add_argument("--half", type=float, default=6.0); ap.add_argument("--texture", default="data/idol/checker_ref.png")
ap.add_argument("--prim", default="/Asset/Geom")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
shutil.copy(a.texture, out / "checker.png")

(out / "materials.mtlx").write_text(f'''<?xml version="1.0"?>
<materialx version="1.38">
  <standard_surface name="SR_glass" type="surfaceshader">
    <input name="base" type="float" value="0" />
    <input name="specular" type="float" value="1" />
    <input name="specular_roughness" type="float" value="0" />
    <input name="specular_IOR" type="float" value="{a.ior}" />
    <input name="transmission" type="float" value="1" />
    <input name="transmission_color" type="color3" value="{a.tint[0]}, {a.tint[1]}, {a.tint[2]}" />
  </standard_surface>
  <surfacematerial name="M_glass" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_glass" />
  </surfacematerial>
  <nodegraph name="NG_checker">
    <image name="img" type="color3">
      <input name="file" type="filename" value="checker.png" colorspace="srgb_texture" />
    </image>
    <output name="out" type="color3" nodename="img" />
  </nodegraph>
  <standard_surface name="SR_checker" type="surfaceshader">
    <input name="base" type="float" value="0" />
    <input name="specular" type="float" value="0" />
    <input name="emission" type="float" value="1" />
    <input name="emission_color" type="color3" nodegraph="NG_checker" output="out" />
  </standard_surface>
  <surfacematerial name="M_checker" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_checker" />
  </surfacematerial>
</materialx>
''')

src = Usd.Stage.Open(a.mesh)
m = UsdGeom.Mesh(src.GetPrimAtPath(a.prim))
P = np.asarray(m.GetPointsAttr().Get(), np.float32)
idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int32)
cnt = np.asarray(m.GetFaceVertexCountsAttr().Get(), np.int32)

st = Usd.Stage.CreateNew(str(out / "scene.usda"))
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st, 1.0)
world = UsdGeom.Xform.Define(st, "/World"); st.SetDefaultPrim(world.GetPrim())
mats = st.DefinePrim("/World/Materials", "Scope")
mats.GetReferences().AddReference("./materials.mtlx", "/MaterialX/Materials")
g = UsdGeom.Mesh.Define(st, "/World/Glass")
g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P)); g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx))
g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(cnt)); g.CreateSubdivisionSchemeAttr("none")
UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).GetDirectBindingRel().SetTargets([Sdf.Path("/World/Materials/M_glass")])

# checker cube, faces looking inwards; each face = one plane of background.json
h = a.half
faces = {  # name: (origin, u, v) with point = origin + s u + t v, viewed from inside (texture upright)
    "back":   ([-h, -h, -h], [2 * h, 0, 0], [0, 2 * h, 0]),
    "front":  ([h, -h, h], [-2 * h, 0, 0], [0, 2 * h, 0]),
    "left":   ([-h, -h, h], [0, 0, -2 * h], [0, 2 * h, 0]),
    "right":  ([h, -h, -h], [0, 0, 2 * h], [0, 2 * h, 0]),
    "bottom": ([-h, -h, h], [2 * h, 0, 0], [0, 0, -2 * h]),
    "top":    ([-h, h, -h], [2 * h, 0, 0], [0, 0, 2 * h]),
}
planes = []
for name, (o, u, v) in faces.items():
    o, u, v = map(np.array, (o, u, v))
    q = UsdGeom.Mesh.Define(st, f"/World/Checker/{name}")
    pts = [o, o + u, o + u + v, o + v]
    n = np.cross(u, v); inward = -o - (u + v) / 2      # towards the centre
    order = [0, 1, 2, 3] if n @ inward > 0 else [0, 3, 2, 1]
    q.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    q.CreateFaceVertexCountsAttr([4]); q.CreateFaceVertexIndicesAttr(order); q.CreateSubdivisionSchemeAttr("none")
    stp = UsdGeom.PrimvarsAPI(q).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    stp.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    UsdShade.MaterialBindingAPI.Apply(q.GetPrim()).GetDirectBindingRel().SetTargets([Sdf.Path("/World/Materials/M_checker")])
    planes.append({"origin": o.tolist(), "u": u.tolist(), "v": v.tolist(), "texture": "checker.png"})
st.GetRootLayer().Save()
(out / "background.json").write_text(json.dumps({"type": "planes", "planes": planes}, indent=1))
(out / "truth.json").write_text(json.dumps({"ior": a.ior, "transmission_color": a.tint,
                                            "note": "MaterialX standard_surface transmission_color (at depth 0 = surface tint)"}, indent=1))
print("scene:", out / "scene.usda")
