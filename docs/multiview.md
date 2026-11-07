# Several views, seed selection and the validator

Idol makes the whole object from one picture, so the sides it never saw are invented. This page covers what was
built and measured to make those sides better. The working method is **sample several seeds and keep the best one,
judged by a vision model from four sides**. Feeding extra views into Idol, restoring the picture first and editing
Idol's activations did not help in our measurements.

## 1. The validator (`scripts/mv_check.py`)

It answers one question: is this 3D result the photographed object, from every side?

- The splat (or mesh) is rendered from Idol's own predicted camera and from 90°, 180° and 270° around it
  (`scripts/seed_views.sh`, which writes the cameras with `gen_views.cameras` and renders with athenea).
- One sheet goes to a vision model (`deepseek-flash`): the input picture, then the four renders.
- Per view it scores **identity** (same object), **plausible** (a believable hidden side) and **clean** (no holes,
  smears or duplicated details), 1–5. For the whole set it scores **coherent** (the views agree with each other) and
  **overall**.
- `score = 0.5 · mean of the per-view scores + 0.25 · coherent + 0.25 · overall`. The median of 3 runs is used.

```sh
scripts/seed_views.sh SPLAT.ply CAMERA.json PHOTO OUT_DIR       # renders + score → OUT_DIR.check.json
.venv/bin/python scripts/mv_check.py PHOTO 0=front.png 90=left.png 180=back.png 270=right.png --json out.json
```

**How much to trust it.** One run varies by 0.32 (standard deviation over 64 samples), so the median of three
varies by about 0.22. It is strict (a good reference model scored 2.5–3.7). It names concrete faults, for example
"the front gem appears in every view" or "the back is a dark hollow shell". It is good for ranking samples of the
same object, not as an absolute grade.

## 2. Seed selection (what works)

Different seeds of the same picture give different hidden sides. Sampling a few and keeping the best by the
validator is a measurable gain:

| Test | Seed 42 only | Best of the seeds |
|---|---|---|
| A wooden staff with crystals, 6 seeds | 3.79 (seeds 1–5: 2.92–3.21) | 3.79 |
| 16 objects (test sets 3 and 4), 4 seeds each | mean 3.21 | mean 3.67 |

The best-of-4 figure is optimistic: picking the maximum of noisy scores inflates it. Within one picture the seed
spread (0.33) is only a little larger than the validator's noise (0.22), so selection pays most when one seed fails
clearly (a duplicated front, a collapsed back). Each extra seed costs about 2.5 min of Idol on an RTX 3070, plus
about 2 min of renders and checks.

`splat2usd.py --seeds 4` does all of this inside the pipeline and keeps the best sample.

The seed's own camera matters: each Idol sample predicts its own input camera. Renders for the validator must use
that sample's camera (saved by Idol in `_camera.json`, or as a latent in `_camera.pt` by the Diffusers version),
not another seed's.

## 3. Idol through Diffusers (`scripts/idol_diffusers.py`)

The same Idol weights also run as a Diffusers modular pipeline (`OzzyGT/idol_diffusers`, pipeline code in
`OzzyGT/idol_custom_blocks`, loaded with `trust_remote_code`; reviewed). It is useful on hosts where the SDK is
awkward. `guidance_scale` 4.0 in Diffusers equals the SDK's `cfg_strength` 3.0. It peaks at 4.55 GB VRAM in bf16 and
writes a standard 3DGS PLY in the same frame as the SDK (Y-down) plus the camera latent. `--blocks` accepts a
local snapshot folder for offline hosts.

`scripts/idol_prepare.py` reproduces Idol's own input preparation (background removal, square crop, black
composite, 1024 px) for pictures that go to the Diffusers version.

## 4. More than one view into Idol (`scripts/idol_mv.py`, `splat2usd.py --views`)

Idol samples an object latent and a camera token for the input picture together. Several pictures can drive one
object without retraining: at every sampling step the latent's velocity is the weighted mean of each view's
velocity, and each view keeps its own camera token. The relative angle between views can also be fixed by
in-painting the camera token. This is the same idea as TRELLIS's multi-image mode.

Measured on a boot (Chamfer distance to a reference after alignment, lower is better):

| Run | Chamfer |
|---|---|
| one picture | **0.0100** |
| four pictures, free cameras | 0.0119 |
| four pictures, angles tied at −90° steps | 0.0156 |
| four pictures, angles tied at +90° steps | 0.0120 |

The extra views did not help. The reference here is itself an Idol sample of the same picture, which favours the
single-view run, so this test cannot say how much worse fusion really is. It shows no gain, and the option stays
experimental.

`--multiview gen` (`scripts/gen_views.py`) generates the hidden sides from the one picture first. The first splat is
rendered from the other sides, cleaned with Iris-3B Turbo img2img inside the render's outline, and checked
(silhouette IoU ≥ 0.9 and the vision model must see the same object). Those views then go back to Idol with weight
0.5. It inherits the limits of the fusion above.

## 5. Coherent views with MV-Adapter (`scripts/mva_views.py`)

MV-Adapter (Apache-2.0, on SDXL base, OpenRAIL++) generates all views of an object together, with attention across
views, so they agree with each other. On an 8 GB GPU it needs CPU offload (13.6 GB peak, mostly system memory) and
took 64 min for six views. The views were coherent. It is kept as an option; the project's preference is to stay
with the speridlabs models.

## 6. What did not help

- **Restoring the picture with Iris-3B before Idol.** Staff, 6 seeds: mean 2.98 with the restored picture against
  3.23 with the original, and best 3.25 against 3.79. Idol reads the restorer's extra detail as structure.
- **Editing Idol's activations.** Inspired by weight-editing work on language models, 64 samples (16 pictures × 4
  seeds) were recorded with the residual stream of Idol's 24 transformer blocks at 6 sampling steps
  (`scripts/idol_probe.py`). The aim was a direction that separates the better seed of a picture from the worse
  one (`scripts/probe_direction.py`, leave-one-picture-out). With 32 samples a correlation of 0.6 appeared; with 64
  it fell to zero or below at every block and step. Idol's failures differ from seed to seed, and the labels are
  noisy, so no common direction was found. Steering was not attempted.
