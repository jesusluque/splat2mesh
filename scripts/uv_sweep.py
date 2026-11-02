"""Sweep xatlas chart options on a mesh: charts, utilisation, area-stretch spread (σ of log texel density)."""
import sys, time
import numpy as np, trimesh, xatlas
m = trimesh.load(sys.argv[1], process=True); m.merge_vertices()
sc = 1 / np.linalg.norm(m.bounds[1] - m.bounds[0])
V = ((m.vertices - m.bounds[0]) * sc).astype(np.float32); F = m.faces.astype(np.uint32); A = m.area * sc * sc

def stretch(vm, f, uv):
    P = V[vm][f]; T = uv[f]
    a3 = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1) / 2
    e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    a2 = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
    ok = a3 > 0
    r = a2[ok] / a3[ok]; r = r / (a2.sum() / a3.sum())
    return float(np.std(np.log(np.maximum(r, 1e-6))))

def run(**co):
    t = time.time(); at = xatlas.Atlas(); at.add_mesh(V, F)
    c = xatlas.ChartOptions()
    for k, v in co.items(): setattr(c, k, v)
    po = xatlas.PackOptions(); po.resolution = 2048; po.padding = 4; po.bilinear = True
    po.texels_per_unit = float(np.sqrt(0.5 * 2048 ** 2 / A))
    at.generate(c, po); vm, f, uv = at[0]
    print(f"{str(co):70s} charts {at.chart_count:4d} atlases {at.atlas_count} util {at.utilization:.2f} "
          f"stretch {stretch(vm, f, uv):.3f} {time.time() - t:.1f}s", flush=True)

run()
for mc in (4.0, 8.0, 16.0, 32.0):
    run(max_cost=mc)
run(max_cost=16.0, normal_deviation_weight=1.0)
run(max_cost=16.0, roundness_weight=0.05, straightness_weight=12.0)
run(max_cost=16.0, max_iterations=4)
