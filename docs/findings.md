# Findings

What was tried while building splat2mesh, what was measured, and what was kept. Ordered by topic. Numbers are from
the test objects named in each item; "cat" is the lucky-cat figurine of the tutorial.

## Geometry

- **Direct mode beats the render path on Idol splats.** On the cat, the first direct version (320³ grid, level
  0.3, solid fill) gave PSNR 26.3 dB against 24.1 for the render path, in 41 s instead of ~9 min. Its topology was
  bad, though: 58 components, 89 non-manifold edges and 1 267 UV charts.
- **Topology-preserving decimation** (Open3D quadric first, fast-simplification as fallback) fixed that: one
  component, watertight, 0 non-manifold edges, 108 charts, 27.05 dB, 39 s.
- **Blurring the solid before marching cubes** removes small folds: on the cat, blur 0.8 → 1.2 took folds from 34 to
  6 and charts from 108 to 65, at PSNR 27.05 → 26.86. Default 1.2.
- **Level 0.15 instead of 0.3**: Idol's back and side gaussians are faint, and at 0.3 they fell under the solid
  threshold and left craters (an owl, a lamp). At 0.15 the owl went from 228 folds and 990 charts to 9 and 91.
- **Grid 448** fixed a lamp whose thin parts broke at 320.
- **Crumpled walls fall back to the render path.** A black lacquer box caved in (710 folds) whatever the blur;
  other objects have 4–18 folds. Above 300 folds the render path is used. The metric misses one kind of
  crumpling (a boot at 225 folds looked crumpled).
- **xatlas tuning did not matter.** A sweep of chart options on the direct cat changed charts and stretch little
  (103 → 109 charts, stretch σ 0.096 → 0.081). The large win came from the clean shell. Unwrapping a smoothed proxy
  barely cut charts and made stretch explode.
- **The bake ran out of memory at 4096²** (15 GB) until it was done in chunks.
- **Web copies must not be decimated after the UV unwrap.** fast-simplification collapses the two sides of a UV seam
  independently and tears triangles (a toy car looked shredded). The web copy now quantises the ASCII geometry
  first and decimates only above the size limit.

## Picture → splat

- **Background colour leaks into invented sides.** The cat on black came out with a dark back; on white, clean.
- **Restoring the picture before Idol**: the Iris-3B restorer keeps the object (kettle: PSNR 38.6 dB at 256 px,
  ΔE 0.5, SSIM 0.99, silhouette IoU 0.995, sharpness ×1.19), and the kettle's splat was cleaner (156 folds against
  184). On four detail-heavy objects it was the same or slightly better twice (kettle, leather book) and worse twice
  (owl: flat patches, empty eye; boot: crumpled). On a staff with 6 seeds it was worse on average (2.98 against 3.23
  by the validator). Kept as an option, off by default.
- **Generative re-invention before Idol** (ILVR with Iris Turbo) looked sharper as a picture but Idol turned the
  added grain into holes and stains; at a stronger setting it invented a ghost structure. Not used.
- **Camera-projected texture of the picture** onto the mesh was rejected: the seen side would no longer match the
  invented one.
- **Seed selection works**, extra views into Idol did not, activation editing found nothing: see
  [multiview.md](multiview.md).

## Materials

- **A 4B vision model labels materials poorly with open questions.** Multiple-choice or free-text prompts gave
  "porcelain" for every region. Yes/no questions per material on a zoomed crop work (gold 0.98, red lacquer 0.998,
  white ceramic 0.85).
- **Its PBR numbers are not usable**: `#d4af37` for any gold, `#ffffff` for white, glazed porcelain roughness
  0.7–0.8, and different answers for different crops. Numbers come from the library.
- **Segmentation in 3D**: ENEAS on an orbit of the textured mesh, with masks voted per texel, instead of on the
  picture alone. The annotation frame matters: grounding on the back view made "the bell" the whole cat. The frame
  that matches the picture is chosen by comparing object crops.
- **Stains at material borders** came from masks spilling over neighbours (metallic halos around gold branches).
  Colour regions now only admit classes that hold a real share of their votes.
- **Nested masks.** "The dragon's body" grounded on the whole figurine and "the crystal" on dragon and crystal. With
  the drop rules alone, the smaller part vanished. Overlap resolution (reliable masks first, then colour clusters
  named by a vision model) fixed it. A rule was then needed so that a part also keeps its own colour region (≥ 40 %
  of its votes), or the crystal and the horns were voted away.
- **Watch**: ENEAS returned the same mask for case, rim and crystal, and "the glass" covered half the object, which
  made the case glass. Fixes: near-identical masks of different materials keep the first listed; the generic glass
  phrase only fills unclaimed pixels and is dropped above 30 % cover.
- **Car windows** are not found by ENEAS and the near-black rubber prior takes them. A height-based split would fix
  this car but is object-specific, so it was not adopted. It stays a known limit.
- **Polished metals**: the splat's colour on a metal is frozen reflections, so polished metals use the F0 colour
  times fine detail only (high-pass of log-luminance): hammering and engraving stay, reflections go.
- **Glass tint**: applying the full measured tint on both walls of a thin-walled shell made a blue bottle opaque and
  a bulb pink. The per-wall colour is the tint to the power 1/2. White crack lines on glass were UV seams where the
  transmission map was 0 outside charts; maps are now spread past chart borders.
- **Stained glass**: one median tint made a Tiffany shade clear. Glass with an a\*b\* spread above 12 keeps its colours.
- **Thin-walled glass looked like chrome** (Idol's double walls = four interfaces); specular 0.5 on thin-walled glass.
- **Interior of glass**: gaussians more than 8 % deep under a transmissive surface. At 3 % the back wall seen through
  the front was picked up as blobs.
- **Glossy surfaces**: shading normals from bilateral face-normal filtering keep edges (Taubin rounded a lacquer
  box's lid into terraces).
- **Web materials per part** gave jagged borders (each face takes its centre texel's part), visible as broken glass
  triangles around a dragon's eye. The web viewer now reads the per-texel maps instead.

## Rendering for the gallery

- Polished metals need something to reflect: under a flat grey dome they looked transparent. Stills use a studio
  HDRI and are composited on grey by depth.

## Test sets

Five sets of eight generated pictures (`data/testset*`, not in the repository): porcelain, metals, wood, lacquer,
stone, jade, wax, glass, leather and mixes. 33 finished objects are in the web gallery.
