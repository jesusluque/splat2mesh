# Scripts

Every script in `scripts/`, grouped by purpose. Run them from the repository root with `.venv/bin/python` (or
`bash` for `.sh`). Each one documents its arguments in its first lines; `--help` works on the argparse ones.

Status: **core** is the supported path, **tool** is a useful standalone step, **experiment** was written to
measure something (the result is in [findings](findings.md)), **batch** is a one-off run script tied to the
development machines, kept for the record.

## Main pipeline

| Script | Status | What |
|---|---|---|
| `splat2usd.py` | core | picture or splat → USD with QA; batch over folders ([pipeline](pipeline.md)) |
| `athenea_cams.py` | core | the 87 cameras that frame a splat for the render path |
| `render_athenea.sh` | core | renders a splat from those cameras (colour, mask, depth); run under the GPU lock |
| `text2image.py` | tool | text → picture with Iris-3B Turbo through its public Hugging Face Space |
| `iris_local.py` | tool | Iris-3B Turbo locally on Apple Silicon, in bf16 |

## Materials

| Script | Status | What |
|---|---|---|
| `flow_batch.py` | core | the full materials flow on a folder of assets; `--gs` segments on the splat ([materials](materials.md)) |
| `describe_materials.py` | core | Qwen3-VL-4B writes "material: part, part; …" from the picture and prompt |
| `materials_eneas3d.py` | core | ENEAS on the orbit, overlap resolution, per-texel votes, maps, USD |
| `eneas_seq.py` | core | runs on the ENEAS host: ground each phrase and track it through the frames |
| `reapply_par.py`, `reapply_all.py` | tool | rerun the per-texel rules on finished assets (no ENEAS) |
| `rematerial_all.sh` | tool | rerun the material stage with saved masks, ENEAS only for new phrases |
| `refine_descriptions.py` | tool | apply the prompt-authority rule to existing descriptions |
| `eneas_seg.py` | tool | ENEAS on a single picture |
| `materials_eneas.py` | experiment | first version: ENEAS masks on the picture only, assigned to colour regions |
| `material_vlm.py`, `material_vlm_yesno.py` | experiment | Qwen3-VL labelling colour clusters; yes/no questions worked better than multiple choice |
| `qwen_pbr_probe.py` | experiment | can Qwen3-VL give PBR numbers? (no: stereotyped, unstable) |
| `materials_batch.py` | experiment | earlier batch: direct mode + VLM materials |

## Glass

| Script | Status | What |
|---|---|---|
| `glass_estimate.py` | core | IOR and tint from views of the object in front of a known background |
| `glass_scene.py` | tool | builds a path-traced glass test scene (object inside an emissive checker cube) |
| `glass_test_athenea.py` | experiment | blind estimate on that scene, compared with the hidden truth afterwards |

## Several views, seeds and checks

| Script | Status | What |
|---|---|---|
| `mv_check.py` | tool | vision-model validator of a result from four sides ([multiview](multiview.md)) |
| `seed_views.sh` | tool | renders a splat from its own camera at 0/90/180/270° and runs the validator |
| `idol_diffusers.py` | tool | Idol through Diffusers; `--blocks` for offline hosts |
| `idol_prepare.py` | tool | Idol's input preparation (cut-out, crop, 1024 px) |
| `idol_mv.py` | experiment | Idol from several pictures (velocity averaging); runs on the Idol host |
| `gen_views.py` | experiment | hidden sides generated from one picture, gated, given back to Idol |
| `mva_views.py` | experiment | coherent views with MV-Adapter |
| `mv_eval.py` | tool | Chamfer / p95 distance of a model to a reference after alignment with scale |
| `overlay.py` | tool | red/green overlay of a model on its picture; the camera is found automatically |
| `qa_visual.py` | tool | Qwen3-VL compares renders of published models with their pictures |

## Objects made of components

For objects whose parts Idol fuses into one solid (a bulb's glass and its screw base). Each component is cut out of
the picture, generated on its own with Idol and placed back where it is in the whole object.

| Script | Status | What |
|---|---|---|
| `auto_components.py` | experiment | the vision model writes the object's bill of components (kind, count, material) from the picture |
| `components.py` | experiment | one closed mesh per component (ENEAS cut-out → Idol → direct mode), placed using the whole object's material votes |
| `iris_parts.py` | experiment | clean picture of each component with Iris-3B Turbo img2img from its cut-out |
| `pick_parts.py`, `part_sheet.py` | experiment | the vision model picks the best candidate per component; review sheet |

## Picture quality before Idol

| Script | Status | What |
|---|---|---|
| `iris_restore.py` | tool | Iris-3B restorer in bf16 (~6 GB instead of 12 GB) |
| `upscale_detail.py` | experiment | restorer + ILVR re-invention with Iris Turbo, with coherence numbers |
| `iris_coherent.py` | experiment | variations of one object (lighting) that keep its structure |
| `upscale_trial.sh` | batch | rebuild a few objects with `--upscale` for comparison |

## Idol analysis

| Script | Status | What |
|---|---|---|
| `idol_probe.py` | experiment | records Idol's block activations per sample; optional steering |
| `probe_direction.py` | experiment | leave-one-picture-out search for a quality direction |
| `probe_score.sh` | batch | fetches probe samples from the Idol host and scores them |

## Rendering, video and publishing

| Script | Status | What |
|---|---|---|
| `athenea_view.sh` | tool | one path-traced view of a USD asset |
| `hdri_turntable.sh` | tool | athenea path-traced turntable under a studio HDRI, composited on grey, frames + MP4 (`START_DEG` = first azimuth) |
| `splat_turntable.sh` | tool | turntable of the raw splat (athenea `render --splats`), frames + MP4 |
| `uv_islands.py` | tool | UV-island check texture (one colour per chart, checker on top) and a `uv.usda` that uses it |
| `make_video.py` | tool | social video (1080×1350) from a storyboard of stills, frame sequences and text cards |
| `athenea_turntable.sh` | tool | path-traced turntable → MP4 |
| `blender_turntable.py` | tool | turntable in Blender (EEVEE, HDRI) |
| `gallery_stills.sh`, `exr_over.py`, `compose_gallery.py` | tool | gallery stills under a studio HDRI, composited on grey, into one sheet |
| `export_web.py` | core | static USDZ web gallery ([materials](materials.md#10-web-gallery)) |
| `web_all.sh`, `publish_all.sh` | batch | the gallery's model list and publishing run |

## Post-processing

| Script | Status | What |
|---|---|---|
| `curio_post.py` | tool | quad retopology with CurioMesh (GPL-3.0, run as a separate program), UVs remade, textures carried over |

## Benchmarks and diagnostics

| Script | Status | What |
|---|---|---|
| `run_pawn.py`, `run_pawn_direct.py`, `run_pawn_2dgs.py`, `rerender_pawn.py` | experiment | a chess pawn splat against its original mesh: render path, direct mode, 2DGS |
| `run_splat.py` | tool | views folder → USD, with an optional reference mesh for metrics |
| `uv_sweep.py` | experiment | xatlas chart options against charts, atlas use and stretch |
| `shade_views.py` | tool | grey shaded views of meshes (geometry check) |

## Batch runs

`night_run.sh`, `night_run2.sh`, `night_run3.sh`, `night_set5.sh`, `night_q94.sh`, `night_qix.sh`,
`night_qix_redo.sh`, `night_redo.sh`, `night_videos.sh`, `redo_car.sh`, `redo_render.sh`: queues used to build the
test sets on the development machines (host names and paths are hard-coded). Kept as a record of how the gallery
was made.

## Python modules (`splat2mesh/`)

| Module | What |
|---|---|
| `config.py` | the one configuration: athenea, GPU lock, Idol/ENEAS hosts, Iris, DeepSeek, HDRI, notifier ([setup](setup.md)) |
| `athenea.py` | athenea from Python and its CLI: splat views, stages with `--turntable`/`--cameras`, EXR → PNG |
| `athenea_splat.py` | a splat with per-gaussian materials → relightable USD for athenea |
| `remote.py` | Idol and ENEAS locally, on a Linux host or on a Windows host (ENEAS in WSL) |
| `deepseek.py` | DeepSeek vision client (JSON answers) |
| `direct.py`, `geometry.py`, `texture.py`, `usd_export.py`, `rerender.py` | meshing, UVs, bake, USD, QA ([pipeline](pipeline.md)) |
| `materials.py`, `overlap.py`, `gs_materials.py`, `openings.py`, `interior.py`, `glass.py` | materials ([materials](materials.md)) |
| `ply.py` | PLY vertex I/O (numpy) |

## Other folders

- `tests/synthetic.py`, `tests/glass_selftest.py`: the CPU tests ([setup](setup.md)).
- `train/train_2dgs.py`: 2DGS training with gsplat on posed views (the retraining path tried for the pawn).
