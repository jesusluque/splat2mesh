"""UV-island check texture: every UV island (chart) in its own colour with a checker on top, plus a USD next to the
asset that uses it, to render with athenea (scripts/hdri_turntable.sh). Even checker squares = little stretch.

    python scripts/uv_islands.py ASSET_DIR [--res 2048] [--check 32]
Writes ASSET_DIR/uv_islands.png and ASSET_DIR/uv.usda (the asset referenced, /Asset/Geom bound to the texture).
Islands are the mesh's connected pieces in UV space (vertices are split along seams, so shared vertices = same chart).
"""
import argparse, colorsys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

USDA = """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Asset" (
    prepend references = @./asset.usda@</Asset>
)
{
    over "Geom"
    {
        rel material:binding = </Asset/UVIslands>
    }

    def Material "UVIslands"
    {
        token outputs:surface.connect = </Asset/UVIslands/Surface.outputs:surface>

        def Shader "Surface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Asset/UVIslands/Tex.outputs:rgb>
            float inputs:roughness = 0.55
            float inputs:metallic = 0
            token outputs:surface
        }

        def Shader "Reader"
        {
            uniform token info:id = "UsdPrimvarReader_float2"
            string inputs:varname = "st"
            float2 outputs:result
        }

        def Shader "Tex"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./uv_islands.png@
            float2 inputs:st.connect = </Asset/UVIslands/Reader.outputs:result>
            token inputs:sourceColorSpace = "sRGB"
            float3 outputs:rgb
        }
    }
}
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("asset_dir")
    ap.add_argument("--res", type=int, default=2048); ap.add_argument("--check", type=int, default=32)
    a = ap.parse_args(); d = Path(a.asset_dir); R = a.res
    stage = Usd.Stage.Open(str(d / "asset.usda")); m = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Geom"))
    F = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    st = np.array(UsdGeom.PrimvarsAPI(m).GetPrimvar("st").Get())
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    n, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(st),) * 2), directed=False)
    rng = np.random.default_rng(3)
    cols = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(rng.random(), 0.55 + 0.35 * rng.random(),
                                                            0.75 + 0.25 * rng.random())) for _ in range(n)]
    im = Image.new("RGB", (R, R), (40, 40, 40)); dr = ImageDraw.Draw(im)
    for f in F:
        dr.polygon([(st[i, 0] * R, (1 - st[i, 1]) * R) for i in f], fill=cols[lab[f[0]]])
    y, x = np.mgrid[0:R, 0:R]
    chk = (((x // a.check) + (y // a.check)) % 2) * 0.28 + 0.72
    Image.fromarray((np.asarray(im, np.float32) * chk[..., None]).clip(0, 255).astype(np.uint8)).save(d / "uv_islands.png")
    (d / "uv.usda").write_text(USDA)
    print(f"{n} UV islands → {d / 'uv_islands.png'}, {d / 'uv.usda'}")


if __name__ == "__main__":
    main()
