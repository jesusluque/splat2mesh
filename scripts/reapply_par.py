"""Parallel re-apply of the per-texel material stage (no ENEAS) on all finished assets."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def one(d):
    from splat2mesh.materials import apply_texel_classes, export_parts
    d = Path(d); spec = d / "mat" / "regions.json"
    if not (d / "mat" / "texel_class.npy").exists() or not spec.exists():
        return f"{d} skip"
    r = apply_texel_classes(spec)
    export_parts(d); export_parts(d, "asset_materials.usda", group="material")
    return f"{d} lifted {r.get('lifted_texels')} smoothed {r.get('glass_smoothed')}"
if __name__ == "__main__":
    dirs = [str(Path("out/mat_cat3"))] + [str(p) for p in sorted(Path("out").glob("testset*_v5/0*/usd")) if "testset5" not in str(p)]
    with ProcessPoolExecutor(3) as ex:
        for msg in ex.map(one, dirs):
            print(msg, flush=True)
