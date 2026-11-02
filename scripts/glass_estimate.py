"""Automatic glass material: mesh + views of the object in front of a known background → IOR & tint → USD.

    python scripts/glass_estimate.py ASSET_DIR VIEWS_DIR BACKGROUND.json [--apply]
ASSET_DIR has asset.usda (+ mesh.ply); VIEWS_DIR is an athenea-layout view set (cameras.json, views/).
Writes ASSET_DIR/glass.json; with --apply the USD material becomes a dielectric with the estimate.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trimesh
from splat2mesh.glass import estimate, load_background
from splat2mesh.views import load_athenea
from splat2mesh.evaluate import usd_to_trimesh
from splat2mesh.usd_export import set_glass

ap = argparse.ArgumentParser(); ap.add_argument("asset"); ap.add_argument("views"); ap.add_argument("background")
ap.add_argument("--apply", action="store_true"); ap.add_argument("--prim", default="/Asset/Geom")
a = ap.parse_args()
asset = Path(a.asset)
src = asset / "asset.usda" if (asset / "asset.usda").exists() else asset
mesh = usd_to_trimesh(str(src), a.prim if src.suffix.startswith(".usd") else None) if str(src).endswith((".usda", ".usd", ".usdc")) else trimesh.load(src)
mesh.merge_vertices(); trimesh.repair.fix_normals(mesh)
r = estimate(mesh, load_athenea(a.views, erode_px=1), load_background(a.background))
out = (asset if asset.is_dir() else asset.parent) / "glass.json"
out.write_text(json.dumps(r, indent=2))
print(json.dumps({k: r[k] for k in ("ior", "transmission_per_unit", "loss", "rays_lost")}, indent=1))
if a.apply and src.suffix == ".usda":
    set_glass(src, r["ior"], r["transmission_per_unit"])
    print("material updated:", src)
