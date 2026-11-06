"""Idol activation probe and steering (runs on the CUDA host next to the Idol SDK, like idol.sample).

Records, for every (image, seed), the residual stream of the 24 joint DiT blocks averaged over the 8192 object-latent
tokens (conditional pass only) at a few sampling steps, next to the usual .ply / _input.png / _camera.json. With
--steer it adds alpha * direction[block] to the latent tokens of the chosen blocks' outputs (both guidance passes), the
inference-time version of a weight edit (a bias on the block output).

    python idol_probe.py --images a.png b.jpg --seeds 42 1 --out DIR [--steer dir.npz --alpha -1 --blocks 8 12
                         --steps-on 0 25]
dir.npz: 'dir' [24, 1024] (unit vectors) and 'scale' [24] (typical activation norm per block); the added vector is
alpha * scale[b] * dir[b].
"""
import argparse, json, os, time
import numpy as np
import torch
from idol.pipeline import IdolPipeline

REC_STEPS = (2, 5, 10, 20, 30, 40)
Q = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True); ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--out", required=True); ap.add_argument("--num_gaussians", type=int, default=262144)
    ap.add_argument("--steer"); ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--blocks", nargs="*", type=int, default=[])
    ap.add_argument("--steps-on", nargs=2, type=int, default=[0, 50])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pipe = IdolPipeline.from_pretrained("speridlabs/idol", optimize=False)
    den = pipe.denoiser
    state = {"call": 0, "rec": {}}
    steer = None
    if a.steer:
        z = np.load(a.steer)
        steer = {b: torch.tensor(a.alpha * float(z["scale"][b]) * z["dir"][b], device=pipe.device) for b in a.blocks}

    def pre(_m, _args):
        state["blk"] = 0

    def hook(i):
        def f(_m, _inp, out):
            k = state["call"]; step, is_cond = k // 2, k % 2 == 0
            if is_cond and step in REC_STEPS:
                state["rec"][(step, i)] = out[0, :Q].float().mean(0).cpu().numpy()
            if steer is not None and i in steer and a.steps_on[0] <= step < a.steps_on[1]:
                out = out.clone(); out[:, :Q] += steer[i].to(out.dtype)
                return out
        return f

    def post(_m, _inp, _out):
        state["call"] += 1

    den.register_forward_pre_hook(pre); den.register_forward_hook(post)
    for i, blk in enumerate(den.blocks):
        blk.register_forward_hook(hook(i))
    for path in a.images:
        name = os.path.splitext(os.path.basename(path))[0]
        mode = "real" if path.lower().endswith((".jpg", ".jpeg")) else "synthetic"
        for seed in a.seeds:
            base = os.path.join(a.out, f"{name}_s{seed}{a.tag}")
            if os.path.exists(base + ".ply"):
                continue
            t0 = time.time(); state.update(call=0, rec={})
            o = pipe(path, mode=mode, seed=seed, num_gaussians=a.num_gaussians)
            o.gaussian.save_ply(base + ".ply"); o.preprocessed_image.save(base + "_input.png")
            json.dump(o.camera, open(base + "_camera.json", "w"), indent=2)
            acts = np.stack([np.stack([state["rec"][(s, i)] for i in range(len(den.blocks))]) for s in REC_STEPS])
            np.save(base + "_acts.npy", acts.astype(np.float32))           # [steps, 24, 1024]
            print(json.dumps({"out": base, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
