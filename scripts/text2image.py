"""Text → image with Iris-3B Turbo (speridlabs, Apache-2.0) through its public Hugging Face Space.

    python scripts/text2image.py "a ceramic lucky cat figurine, studio photo, black background" out.png [--seed 7]

The prompt should describe ONE object, centred, whole, on a plain background: that is what Idol expects.
Set HF_TOKEN for a larger ZeroGPU quota. To run Iris-3B locally instead, see github.com/speridlabs/iris-3b.
"""
import argparse, os, shutil
from gradio_client import Client

ap = argparse.ArgumentParser(); ap.add_argument("prompt"); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--size", default="1024×1024")
a = ap.parse_args()
c = Client("speridlabs/iris-3b-turbo", token=os.environ.get("HF_TOKEN"), verbose=False)
img, seed, _ = c.predict(a.prompt, a.size, a.seed, False, api_name="/run")
path = img["path"] if isinstance(img, dict) else img
shutil.copy(path, a.out)
print(a.out, "seed", seed)
