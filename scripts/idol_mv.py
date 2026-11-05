"""Idol from several pictures of one object (front, left, back, right), like Tripo's "Multiview to 3D", without
retraining: Idol samples an object latent in its own canonical frame together with a camera token for the picture.
Here every picture has its own camera token and its own conditioning; at each flow step the latent moves by the
weighted mean of the views' velocities, each camera token by its own view's. Runs on the CUDA host next to the
installed idol-sdk, which is imported, not changed.

    python idol_mv.py --images front.png left.png ... --out DIR [--weights 1 0.5 ...] [--cameras known.json]
           [--mode synthetic|real] [--seed 42] [--steps 50] [--cfg_strength 3] [--rescale_t 3] [--num_gaussians N]
           [--dump-token]
--yaw-offsets: for photos whose angle to the front is known (front, left, back, right = 0 -90 -180 -270 in Idol's
yaw, see gen_views.SIDES), their cameras follow the front's at those offsets.
--cameras: {"<image file name>": {"yaw": rad, "pitch": rad}} for pictures whose view direction is known (views
rendered from a known camera): their camera token's direction is held to it at every step (inpainting: the noisy
value t·noise + (1−t)·known); its distance and width stay free, since the crop decides them.
Writes DIR/<front stem>.ply, <front stem>_input.png, <front stem>_camera.json (the front view, same format as
idol.sample), <front stem>_cameras.json (every view) and <front stem>_mv.json (weights, seconds).
--dump-token: also prints the front view's raw camera token next to its decoded camera (checks the encoding).
"""
import argparse, json, math, os, time
import numpy as np
import torch
from tensordict import TensorDict
from PIL import Image

from idol.pipeline import IdolPipeline
from idol.camera import decode_camera


def known_direction(yaw, pitch):
    # inverse of decode_camera: yaw = atan2(dx, dy), pitch = asin(dz)
    return torch.tensor([math.cos(pitch) * math.sin(yaw), math.cos(pitch) * math.cos(yaw), math.sin(pitch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg_strength", type=float, default=3.0); ap.add_argument("--rescale_t", type=float, default=3.0)
    ap.add_argument("--num_gaussians", type=int, default=262144)
    ap.add_argument("--weights_repo", default="speridlabs/idol")
    ap.add_argument("--dump-token", action="store_true")
    ap.add_argument("--yaw-offsets", type=float, nargs="+", default=None,
                    help="degrees, one per picture after the first: its camera is the front's turned by this yaw "
                         "(photos taken around the object; ties the views of symmetric objects)")
    ap.add_argument("--dir-norm", type=float, default=1.0,
                    help="length of the direction part of a camera token (decode_camera ignores it; read it with --dump-token)")
    a = ap.parse_args()
    n = len(a.images); w = torch.tensor(a.weights if a.weights else [1.0] * n, dtype=torch.float32)
    assert len(w) == n, "one weight per image"
    known = json.load(open(a.cameras)) if a.cameras else {}
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    pipe = IdolPipeline.from_pretrained(a.weights_repo, optimize=False)
    dev = pipe.device; model = pipe.denoiser
    torch.manual_seed(a.seed)
    conds, prepared, fixed = [], [], []
    for p in a.images:
        im = pipe.preprocess(p, mode=a.mode); prepared.append(im)
        px = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255.0)[None].to(dev)
        conds.append(pipe.image_encoder(px))
        k = known.get(os.path.basename(p))
        fixed.append(None if k is None else known_direction(k["yaw"], k["pitch"]).to(dev))
    latent = torch.randn(1, model.q_token_length, model.in_channels, device=dev)
    cams = [torch.randn(1, 1, model.cam_channels, device=dev) for _ in range(n)]
    cam_noise = [c.clone() for c in cams]
    t_seq = np.linspace(1, 0, a.steps + 1)
    t_seq = a.rescale_t * t_seq / (1 + (a.rescale_t - 1) * t_seq)

    rel = [None] + ([math.radians(d) for d in a.yaw_offsets] if a.yaw_offsets else [None] * (n - 1))
    assert len(rel) == n, "one yaw offset per picture after the first"
    target = list(fixed)

    def hold(k, t):
        if target[k] is not None:     # the known direction, noised to this step's level
            cams[k][0, 0, :3] = t * cam_noise[k][0, 0, :3] + (1 - t) * target[k] * a.dir_norm

    with torch.no_grad():
        for i in range(a.steps):
            t, tp = float(t_seq[i]), float(t_seq[i + 1])
            vl = torch.zeros_like(latent)
            for k in range(n):
                if k > 0 and rel[k] is not None and fixed[k] is None:
                    # tied to the front view: its predicted clean direction, turned by this picture's yaw offset
                    d0 = decode_camera(clean0[0, 0])
                    target[k] = known_direction(d0["yaw"] + rel[k], d0["pitch"]).to(dev)
                hold(k, t)
                s = TensorDict({"latent": latent, "camera": cams[k]}, batch_size=1, device=dev)
                tt = torch.tensor([1000 * t], device=dev, dtype=torch.float32)
                v = model(s, tt, conds[k])
                if a.cfg_strength > 0:
                    vn = model(s, tt, conds[k].apply(torch.zeros_like))
                    v = TensorDict(
                        {key: (1 + a.cfg_strength) * v[key] - a.cfg_strength * vn[key] for key in ("latent", "camera")},
                        batch_size=1, device=dev)
                vl += w[k] * v["latent"]
                if k == 0:
                    clean0 = cams[0] - t * v["camera"]       # flow matching: x0 = x_t - t v
                cams[k] = cams[k] - v["camera"] * (t - tp)
            latent = latent - vl / w.sum() * (t - tp)
        for k in range(n):
            hold(k, 0.0)
        g = pipe.decode(latent, a.num_gaussians)
    stem = os.path.splitext(os.path.basename(a.images[0]))[0]
    g.save_ply(os.path.join(a.out, f"{stem}.ply"))
    prepared[0].save(os.path.join(a.out, f"{stem}_input.png"))
    dec = [decode_camera(c[0, 0]) for c in cams]
    json.dump(dec[0], open(os.path.join(a.out, f"{stem}_camera.json"), "w"), indent=2)
    json.dump({os.path.basename(p): d for p, d in zip(a.images, dec)}, open(os.path.join(a.out, f"{stem}_cameras.json"), "w"), indent=2)
    for p, im in zip(a.images[1:], prepared[1:]):
        im.save(os.path.join(a.out, f"{stem}_input_{os.path.splitext(os.path.basename(p))[0]}.png"))
    info = {"images": [os.path.basename(p) for p in a.images], "weights": w.tolist(),
            "known": [f is not None for f in fixed], "yaw_offsets": a.yaw_offsets, "seconds": round(time.time() - t0, 1)}
    if a.dump_token:
        info["raw_tokens"] = [c[0, 0].tolist() for c in cams]
    json.dump(info, open(os.path.join(a.out, f"{stem}_mv.json"), "w"), indent=2)
    print(json.dumps({"cameras": dec, **info}), flush=True)


if __name__ == "__main__":
    main()
