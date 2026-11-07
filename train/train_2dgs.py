"""2DGS training with gsplat (Apache-2.0) on posed images with masks; exports median depth,
normals and RGB per view for splat2mesh's fusion + bake.

Losses (re-implemented from the 2DGS paper, Huang et al. 2024 — not their code):
  L = (1-λ)·L1 + λ·(1-SSIM) + λ_n·(1 - <n_render, n_surf>) + λ_d·distortion + λ_m·BCE(alpha, mask)

Input layout = athenea export: cameras.json (intrinsics + views[R_world_to_cam, t_world_to_cam]),
views/vNNN.png, views/vNNN_mask.png. Optional init.ply (xyz[, rgb]) for initial points.

    python train_2dgs.py DATA OUT --iters 15000
"""
from __future__ import annotations

import argparse
import sys
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn
from PIL import Image


def load_data(root: Path, device, downscale: int = 1):
    meta = json.loads((root / "cameras.json").read_text())
    it = meta["intrinsics"]
    K = torch.tensor([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]], dtype=torch.float32)
    K[:2] /= downscale
    imgs, masks, w2cs, names = [], [], [], []
    for v in meta["views"]:
        n = v["name"]
        im = Image.open(root / "views" / f"{n}.png").convert("RGB")
        mk = Image.open(root / "views" / f"{n}_mask.png").convert("L")
        if downscale > 1:
            im = im.resize((im.width // downscale, im.height // downscale), Image.BILINEAR)
            mk = mk.resize((mk.width // downscale, mk.height // downscale), Image.BILINEAR)
        imgs.append(torch.from_numpy(np.asarray(im, np.float32) / 255))
        masks.append(torch.from_numpy(np.asarray(mk, np.float32) / 255))
        E = torch.eye(4)
        E[:3, :3] = torch.tensor(v["R_world_to_cam"])
        E[:3, 3] = torch.tensor(v["t_world_to_cam"])
        w2cs.append(E)
        names.append(n)
    return (torch.stack(imgs).to(device), torch.stack(masks).to(device), torch.stack(w2cs).to(device),
            K.to(device), names)


def ssim(a, b):
    # a, b: B,3,H,W
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = Fnn.avg_pool2d(a, 11, 1, 5)
    mu_b = Fnn.avg_pool2d(b, 11, 1, 5)
    sa = Fnn.avg_pool2d(a * a, 11, 1, 5) - mu_a ** 2
    sb = Fnn.avg_pool2d(b * b, 11, 1, 5) - mu_b ** 2
    sab = Fnn.avg_pool2d(a * b, 11, 1, 5) - mu_a * mu_b
    return (((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))).mean()


def init_points(root: Path, w2cs, K, masks, n: int, device):
    p = root / "init.ply"
    if p.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from splat2mesh.ply import read_vertex
        v = read_vertex(p)
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
        if len(xyz) > n:
            xyz = xyz[np.random.default_rng(0).choice(len(xyz), n, replace=False)]
        return torch.from_numpy(xyz).to(device)
    # fall back: random points inside the visual hull bounding box
    c2w = torch.linalg.inv(w2cs)
    centres = c2w[:, :3, 3]
    r = centres.norm(dim=1).mean() * 0.15
    return (torch.rand(n, 3, device=device) * 2 - 1) * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--init-points", type=int, default=100_000)
    ap.add_argument("--max-gaussians", type=int, default=600_000)
    ap.add_argument("--lambda-normal", type=float, default=0.05)
    ap.add_argument("--lambda-dist", type=float, default=100.0)
    ap.add_argument("--lambda-mask", type=float, default=0.1)
    ap.add_argument("--bg", type=float, nargs=3, default=None, help="background colour; default = per-view mean outside mask")
    a = ap.parse_args()

    from gsplat import rasterization_2dgs
    from gsplat.strategy import DefaultStrategy

    dev = torch.device("cuda")
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs, masks, w2cs, K, names = load_data(root, dev, a.downscale)
    V, H, W = imgs.shape[:3]
    # background = sky colour: estimate per view from pixels outside the mask
    bgs = torch.stack([imgs[i][masks[i] < 0.5].mean(0) for i in range(V)])

    xyz = init_points(root, w2cs, K, masks, a.init_points, dev)
    N = len(xyz)
    d = torch.cdist(xyz[:2048], xyz).topk(4, largest=False).values[:, 1:].mean()
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(xyz),
        "scales": torch.nn.Parameter(torch.full((N, 3), math.log(float(d)), device=dev)),
        "quats": torch.nn.Parameter(Fnn.normalize(torch.randn(N, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.full((N,), math.log(0.1 / 0.9), device=dev)),
        "sh0": torch.nn.Parameter(torch.zeros(N, 1, 3, device=dev)),
        "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device=dev)),
    })
    scene_scale = float((torch.linalg.inv(w2cs)[:, :3, 3]).norm(dim=1).mean()) * 0.3
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    strategy = DefaultStrategy(verbose=False, refine_stop_iter=int(a.iters * 0.6),
                               absgrad=False, key_for_gradient="gradient_2dgs")
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1 / a.iters))

    def render(i, sh_degree):
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        return rasterization_2dgs(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors,
            viewmats=w2cs[i:i + 1], Ks=K[None], width=W, height=H, sh_degree=sh_degree,
            render_mode="RGB+ED", distloss=True, backgrounds=bgs[i:i + 1])

    t0 = time.time()
    rng = np.random.default_rng(0)
    for step in range(a.iters):
        i = int(rng.integers(V))
        sh_deg = min(3, step // 1000)
        (rc, ra, rn, sn, rd, rmed, info) = render(i, sh_deg)
        rgb = rc[..., :3]
        gt = imgs[i:i + 1]
        l1 = (rgb - gt).abs().mean()
        lssim = 1 - ssim(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
        loss = 0.8 * l1 + 0.2 * lssim
        alpha = ra[..., 0].clamp(1e-5, 1 - 1e-5)
        loss = loss + a.lambda_mask * Fnn.binary_cross_entropy(alpha, masks[i:i + 1])
        if step > a.iters // 5:
            nrm = (1 - (rn * sn).sum(-1)) * masks[i:i + 1]
            loss = loss + a.lambda_normal * nrm.mean() + a.lambda_dist * rd.mean()
        strategy.step_pre_backward(params, opts, state, step, info)
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, opts, state, step, info, packed=False)
        if len(params["means"]) > a.max_gaussians and step < strategy.refine_stop_iter:
            strategy.refine_stop_iter = step  # stop growing on small GPUs
        if step % 500 == 0:
            print(f"[{step}] loss {loss.item():.4f} l1 {l1.item():.4f} N {len(params['means'])} "
                  f"{time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)

    # export per-view median depth, normals and RGB (OpenCV z-depth) for fusion on the Mac
    ex = out / "views"
    ex.mkdir(exist_ok=True)
    with torch.no_grad():
        for i, n in enumerate(names):
            (rc, ra, rn, sn, rd, rmed, info) = render(i, 3)
            np.save(ex / f"{n}_depth.npy", (rmed[0, ..., 0] * (ra[0, ..., 0] > 0.5)).cpu().numpy().astype(np.float32))
            np.save(ex / f"{n}_normal.npy", rn[0].cpu().numpy().astype(np.float16))
            Image.fromarray((rc[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(ex / f"{n}.png")
    torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "gaussians.pt")
    (out / "train.json").write_text(json.dumps({"iters": a.iters, "gaussians": len(params["means"]),
                                                "seconds": time.time() - t0}))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
