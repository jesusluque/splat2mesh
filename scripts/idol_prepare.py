"""Idol's own input preparation (BiRefNet cut-out, square crop with margin, composite on black, 1024 px), as
idol.sample does it, for images that then go to idol_diffusers.py. Runs on the CUDA host next to idol-sdk.

    python idol_prepare.py IN.png OUT.png [--mode synthetic|real]
"""
import argparse
from PIL import Image
from idol.pipeline import IdolPipeline

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
a = ap.parse_args()
IdolPipeline.from_pretrained("speridlabs/idol", optimize=False).preprocess(Image.open(a.inp), mode=a.mode).save(a.out)
print(a.out)
