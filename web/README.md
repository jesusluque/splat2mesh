# Web pages

`splat.html` — one relightable splat (`?model=NAME` reads `models/NAME.athc` and `models/NAME.json`), relit live under
an HDRI with athenea's relighting code in Spark's WebGPU backend, the viewer of sparkwebgpu.lucab.co.uk
(<https://github.com/jesusluque/sparkwebgpu-public>). It needs, next to it, `site.js`, `site.css` and `vendor/` from
that project's published site (Spark, three.js, lil-gui: MIT). Its `site.js` signs the page footer with the
contact address of the 3D sites, jesusluquemuriel@gmail.com (set by the user on 2026-11-08).

Making the files for a model:

```sh
.venv/bin/python -m splat2mesh.athenea_splat out/NN/usd/mat/splat_materials.ply NAME.usdc --y-down   # flow_batch --gs
usd-athc NAME.usdc models/NAME.athc --prim /Splat/Opaque --transfer none --gzip --json NAME_athc.json  # sparkwebgpu tool
```

`models/NAME.json` holds `boundsMin`, `boundsMax` (from the usd-athc JSON), an optional `description` for the page, `clouds` and `frontAzimuthDeg` (180 − Idol's yaw, in
degrees: the side the picture showed). The gallery card is `export_web.py … NAME=page:splat.html?model=NAME|TITLE|…`.
The full-resolution HDRIs come from the sparkwebgpu R2 bucket, whose CORS allows splat2mesh.lucab.co.uk.
