# Licences

splat2mesh's own code is **Apache-2.0**. This page lists everything it depends on or calls, so you can check a
use case, especially a commercial one. It is a summary, not legal advice; the licence texts decide.

## Python dependencies (`requirements.txt`)

All permissive:

| Package | Licence |
|---|---|
| numpy, scipy, networkx, scikit-image | BSD |
| trimesh, open3d, xatlas, fast-simplification, rtree, Pillow (MIT-CMU) | MIT or MIT-like |
| manifold3d, gradio_client | Apache-2.0 |
| usd-core | Tomorrow Open Source Technology License 1.0 (Apache-2.0 based) |
| OpenEXR | BSD-3-Clause |

PLY files are read and written by `splat2mesh/ply.py`; the GPL-licensed `plyfile` package is no longer used.

## Models and tools the pipeline calls

They are not shipped with splat2mesh; each is installed separately under its own licence.

| Component | Used for | Licence |
|---|---|---|
| Idol SDK and weights (speridlabs) | picture → splat | Apache-2.0 |
| Idol's image encoder DINOv3 (`facebook/dinov3-vith16plus-pretrain-lvd1689m`) | conditioning | DINOv3 License (Meta, custom) |
| Idol's image encoder FLUX.2 VAE, loaded from `black-forest-labs/FLUX.2-dev` | conditioning | **FLUX.2 [dev] Non-Commercial License** |
| Iris-3B, Turbo and restorer (speridlabs) | text → picture, restoration | Apache-2.0 |
| Qwen3-VL-4B (Iris-3B's text encoder) | material descriptions | Apache-2.0 |
| ENEAS (speridlabs) | text → part masks | Apache-2.0 |
| Idol Diffusers port (`OzzyGT/idol_diffusers`, `OzzyGT/idol_custom_blocks`) | optional Idol runtime | see the repositories |
| MV-Adapter | optional multi-view generation | Apache-2.0; its SDXL base model is CreativeML OpenRAIL++-M |
| CurioMesh | optional quad retopology, run as a separate program | GPL-3.0 |
| gsplat | optional 2DGS training | Apache-2.0 |
| DeepSeek API | optional vision checks | the provider's terms of service |
| three.js | web gallery viewer | MIT |
| Poly Haven HDRIs | lighting of stills and the web viewer | CC0 |

### About the FLUX.2 VAE

As the Idol SDK loads it by default, the VAE comes from the FLUX.2 [dev] repository, which is under a
non-commercial licence. The VAE in `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0) was compared tensor by tensor
with it: all 251 tensors are identical after rounding to bf16, the precision Idol loads it in. Pointing Idol's
`flux2_vae` setting at the klein repository therefore gives the same results under Apache-2.0. Check the current
terms of both repositories before you rely on this.

### About DINOv3

DINOv3 is under Meta's own DINOv3 License, which is not an OSI licence. Read it, including its trade-control
clauses, before commercial use.
