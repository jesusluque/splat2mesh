"""Re-run the per-texel material stage (no ENEAS) on every finished asset: picks up rule changes."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splat2mesh.materials import apply_texel_classes, export_parts
dirs = [Path("out/mat_cat3")] + sorted(Path("out").glob("testset*_v5/0*/usd"))
for d in dirs:
    spec = d / "mat" / "regions.json"
    if not (d / "mat" / "texel_class.npy").exists() or not spec.exists():
        print(d, "skip"); continue
    r = apply_texel_classes(spec)
    export_parts(d); export_parts(d, "asset_materials.usda", group="material")
    print(d, "lifted", r.get("lifted_texels"), flush=True)
