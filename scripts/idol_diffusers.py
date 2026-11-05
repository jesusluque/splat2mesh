"""Idol through Diffusers (OzzyGT/idol_diffusers + OzzyGT/idol_custom_blocks, the same speridlabs/idol weights in
bf16), following asomoza/diffusers-recipes models/idol/scripts/base_example.py. Runs on the CUDA host.

    python idol_diffusers.py --image PREPARED.png --out OUT.ply [--seed 42] [--steps 50] [--gaussians 262144]
PREPARED.png: 1024x1024, the object cropped with a margin and composited on black (what idol.sample writes as
<name>_input.png). OUT.ply: standard 3DGS PLY, y-up like the recipe; OUT_camera.pt keeps the predicted camera latent.
"""
import argparse, json, time
import numpy as np
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image
from plyfile import PlyData, PlyElement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--gaussians", type=int, default=262144)
    a = ap.parse_args()
    t0 = time.time()
    pipe = ModularPipeline.from_pretrained("OzzyGT/idol_custom_blocks", trust_remote_code=True)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to("cuda")
    kw = {"num_inference_steps": a.steps}
    if a.gaussians != 262144:
        kw["num_gaussians"] = a.gaussians
    res = pipe(image=load_image(a.image), generator=torch.Generator("cuda").manual_seed(a.seed),
               output=["gaussians", "camera_latents"], **kw)
    g = res["gaussians"]
    xyz = g["xyz"][0].float().cpu().numpy()
    dc = g["features_dc"][0].float().reshape(-1, 3).cpu().numpy()
    op = torch.logit(g["opacity"][0].float().clamp(1e-6, 1 - 1e-6)).cpu().numpy().reshape(-1, 1)
    sc = torch.log(g["scaling"][0].float().clamp_min(1e-6)).cpu().numpy()
    w, x, y, z = g["rotation"][0].float().cpu().numpy().T
    xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)          # z-up to y-up (recipe)
    rot = np.stack([w - x, x + w, y - z, z + y], axis=1) * np.sqrt(0.5)
    attrs = np.concatenate([xyz, np.zeros_like(xyz), dc, op, sc, rot], axis=1)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.empty(attrs.shape[0], dtype=[(n, "f4") for n in names])
    for i, n in enumerate(names):
        v[n] = attrs[:, i]
    PlyData([PlyElement.describe(v, "vertex")]).write(a.out)
    torch.save(res["camera_latents"], a.out.rsplit(".", 1)[0] + "_camera.pt")
    print(json.dumps({"gaussians": int(attrs.shape[0]), "seconds": round(time.time() - t0, 1),
                      "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)


if __name__ == "__main__":
    main()
