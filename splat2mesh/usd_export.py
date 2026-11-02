"""Write UsdGeomMesh + UsdPreviewSurface (relative texture paths, metersPerUnit=1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def write_usd(path: str | Path, verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray,
              normals: np.ndarray, base_color_png: str, up_axis: str = "Y",
              roughness: float = 0.6, metallic: float = 0.0, name: str = "Asset") -> None:
    path = Path(path)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if up_axis == "Y" else UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, f"/{name}/Geom")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    lo, hi = verts.min(0), verts.max(0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                 UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(uvs.astype(np.float32)))

    mat = UsdShade.Material.Define(stage, f"/{name}/Materials/Baked")
    pbr = UsdShade.Shader.Define(stage, f"/{name}/Materials/Baked/PreviewSurface")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

    reader = UsdShade.Shader.Define(stage, f"/{name}/Materials/Baked/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    tex = UsdShade.Shader.Define(stage, f"/{name}/Materials/Baked/BaseColor")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(base_color_png)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")

    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()


def set_glass(usd_path: str | Path, ior: float, transmission_rgb, opacity: float = 0.05) -> None:
    """Turn the baked material into a dielectric: UsdPreviewSurface ior + low opacity, tint as diffuse.
    The full estimate also goes to customData (splat2mesh:glass) for renderers with real transmission."""
    stage = Usd.Stage.Open(str(usd_path))
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Shader" and UsdShade.Shader(prim).GetIdAttr().Get() == "UsdPreviewSurface":
            s = UsdShade.Shader(prim)
            s.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))
            s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
            s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
            s.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
            dc = s.GetInput("diffuseColor")
            if dc:
                dc.DisconnectSource()
            s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, transmission_rgb)))
            prim.SetCustomDataByKey("splat2mesh:glass", {"ior": float(ior),
                                                         "transmission_per_unit": Vt.Vec3fArray([Gf.Vec3f(*map(float, transmission_rgb))])})
    stage.GetRootLayer().Save()
