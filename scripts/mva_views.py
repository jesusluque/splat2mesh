"""Views of an object from one photo with MV-Adapter (Apache-2.0, on SDXL base, OpenRAIL++): all views are generated
together with attention across views, so they agree with each other (Iris retouches each view on its own).
Runs on the CUDA host in its own venv (C:/splat2mesh/venv_mva) next to a clone of github.com/huanngzh/MV-Adapter.

    python mva_views.py --image photo.png --out DIR [--azimuth 0 45 90 180 270 315] [--steps 50] [--cfg 3.0]
           [--seed 42] [--text "high quality"] [--repo C:/splat2mesh/MV-Adapter]
Writes DIR/view_<az>.png (768x768, grey background as MV-Adapter makes them), DIR/reference.png (the cut-out,
centred input) and DIR/mva.json (azimuths, seconds). Azimuths are MV-Adapter's (orthographic cameras, elevation 0).
"""
import argparse, json, os, sys, time, types


def light_mesh_utils(repo):
    # mvadapter.utils.mesh_utils/__init__ pulls in its texturing tools (nvdiffrast, triton, ...), which
    # image-to-views never calls: register the package without running its __init__, then load only the cameras
    for name in ("nvdiffrast", "nvdiffrast.torch"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["nvdiffrast"].torch = sys.modules["nvdiffrast.torch"]
    pkg = types.ModuleType("mvadapter.utils.mesh_utils")
    pkg.__path__ = [os.path.join(repo, "mvadapter", "utils", "mesh_utils")]
    sys.modules[pkg.__name__] = pkg
    import importlib
    cam = importlib.import_module("mvadapter.utils.mesh_utils.camera")
    pkg.get_orthogonal_camera = cam.get_orthogonal_camera; pkg.get_camera = cam.get_camera


def inplace_offload():
    # enable_model_cpu_offload's hook rebuilds every dict argument when moving it to the GPU, so the reference pass
    # fills a copy of MV-Adapter's cache_hidden_states and the cache stays empty: move dict entries in place instead
    import accelerate.hooks as hooks
    orig = hooks.send_to_device

    def send(t, device, *args, **kw):
        if isinstance(t, dict):
            for k in t:
                t[k] = send(t[k], device, *args, **kw)
            return t
        return orig(t, device, *args, **kw)
    hooks.send_to_device = send


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--azimuth", type=int, nargs="+", default=[0, 45, 90, 180, 270, 315])
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--text", default="high quality")
    ap.add_argument("--repo", default="C:/splat2mesh/MV-Adapter")
    a = ap.parse_args()
    sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, "scripts"))
    light_mesh_utils(a.repo)
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    from PIL import Image
    import inference_i2mv_sdxl as mv

    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    img = Image.open(a.image).convert("RGBA")
    if img.getextrema()[3][0] == 255:            # no real alpha: cut the object out with BiRefNet, then free it
        net = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to("cuda").eval()
        tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = mv.remove_bg(img.convert("RGB"), net, tf, "cuda")
        del net; torch.cuda.empty_cache()
    ref = mv.preprocess_image(img, 768, 768)

    n = len(a.azimuth)
    pipe = mv.prepare_pipeline(base_model="stabilityai/stable-diffusion-xl-base-1.0",
                               vae_model="madebyollin/sdxl-vae-fp16-fix", unet_model=None, lora_model=None,
                               adapter_path="huanngzh/mv-adapter", scheduler=None, num_views=n,
                               device="cpu", dtype=torch.float16)
    inplace_offload()
    pipe.enable_model_cpu_offload()              # 8 GB card: one sub-model on the GPU at a time
    pipe.cond_encoder.to("cuda")
    images, _ = mv.run_pipeline(pipe, num_views=n, text=a.text, image=ref, height=768, width=768,
                                num_inference_steps=a.steps, guidance_scale=a.cfg, seed=a.seed,
                                device="cuda", azimuth_deg=a.azimuth)
    ref.save(os.path.join(a.out, "reference.png"))
    for az, im in zip(a.azimuth, images):
        im.save(os.path.join(a.out, f"view_{az:03d}.png"))
    info = {"image": os.path.basename(a.image), "azimuth": a.azimuth, "steps": a.steps, "cfg": a.cfg,
            "seed": a.seed, "seconds": round(time.time() - t0, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    json.dump(info, open(os.path.join(a.out, "mva.json"), "w"), indent=2)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
