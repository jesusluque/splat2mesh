# Tutorial: de una imagen o un splat a un USD con UV

En 10 minutos: instalar, comprobar que funciona, convertir tu primer objeto y leer el resultado.

## 1. Instalar

```sh
git clone https://github.com/jesusluque/splat2mesh.git && cd splat2mesh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 2. Comprobar que funciona (sin GPU)

```sh
.venv/bin/python tests/synthetic.py
```

Genera un toro y una esfera con colores procedurales, los «fotografía» desde 80 cámaras y los convierte.
Debe terminar en ~40 s con algo así:

```json
"faces": 19998, "boundary_edges": 0, "non_manifold_edges": 0,
"chamfer_mean": 0.0023, "texture_psnr_db": 27.4
```

`chamfer_mean` es el error de forma frente a la geometría real (aquí ≈ 0,25 % del tamaño) y
`texture_psnr_db` lo bien que sale el color. La malla y la textura quedan en `out/synthetic/`.

## 3. Convertir un objeto

### Desde un splat (`.ply` 3DGS)

```sh
.venv/bin/python scripts/splat2usd.py mi_objeto.ply --out out/mio
```

- Si el `.ply` sale boca abajo (convención COLMAP, Y hacia abajo), añade `--y-down`.
- Si es un 3DGS estándar (no Mip-Splatting), añade `--no-antialias`.

### Desde una imagen

```sh
.venv/bin/python scripts/splat2usd.py gato.png --gaussians 524288 --out out/mio
```

La imagen pasa primero por **Idol** (imagen → splat) en una máquina con CUDA
(`SPLAT2MESH_IDOL_HOST`, ver README). Idol inventa los lados que la foto no ve.

### Una carpeta entera

```sh
.venv/bin/python scripts/splat2usd.py carpeta_con_imagenes_y_plys/ --out out/lote --notify
```

Cada objeto se procesa por separado; si uno falla, el lote sigue. `--notify` manda el resumen al terminar.

## 4. Qué obtienes

```
out/mio/gato/
├── report.json          ← medidas y tiempos; "ok": true/false
├── render/              ← las 87 vistas del splat (color, máscara, profundidad, cámaras)
└── usd/
    ├── asset.usda       ← la malla con UV y material (ábrelo en usdview, Blender, Omniverse…)
    ├── basecolor.png    ← la textura
    ├── mesh.ply
    └── compare.png      ← izquierda: el splat; derecha: tu malla texturizada
```

Cómo leer `report.json`:

| Campo | Qué significa | Bueno |
|---|---|---|
| `mesh.boundary_edges` | bordes abiertos (agujeros) | 0 |
| `mesh.non_manifold_edges` | aristas compartidas por más de 2 caras | ≤ 20 |
| `mesh.uv_charts` | islas del mapa UV | cuantas menos, mejor |
| `qa.psnr_mean` | parecido de la malla texturizada al splat, en dB | ≥ 22 (26–30 es muy bueno) |
| `t_total_s` | tiempo total | |

Mira siempre `compare.png`: es la forma más rápida de ver si algo ha ido mal.

## 5. Ajustes útiles

| Opción | Para qué |
|---|---|
| `--faces 50000` | más detalle geométrico (por defecto 20 000) |
| `--tex 4096` | textura más grande (por defecto 2048) |
| `--smooth 3` | suaviza la superficie; recomendado para cristal, empeora un poco la precisión en opacos |
| `--min-psnr 25` | umbral más estricto para marcar un objeto como OK |

## 6. Objetos de cristal

Un splat no sabe que algo es transparente: pinta lo que se ve a través como si fuera color. Si tienes
imágenes del objeto delante de un fondo conocido (p. ej. un damero), la herramienta estima el material:

```sh
.venv/bin/python scripts/glass_estimate.py out/mio/vaso/usd vistas_con_fondo/ fondo.json --apply
```

Calcula el índice de refracción y el tinte trazando rayos a través de la malla hasta el fondo, y cambia
el material del USD a cristal. `fondo.json` describe el fondo como uno o varios planos con textura
(ver `splat2mesh/glass.py`). Compruébalo con `tests/glass_selftest.py`.

## 7. Si algo falla

- **`"ok": false` con `error`**: el mensaje dice en qué paso (Idol, render, malla).
- **Objeto boca abajo en `compare.png`**: repite con `--y-down` (o `--y-up` si venía de Idol ya derecho).
- **PSNR bajo con brillos o reflejos**: normal en materiales muy brillantes; la textura es un color
  fijo y no reproduce reflejos que cambian con el ángulo.
- **Muchas islas UV**: prueba `--smooth 1`; para UV limpias de artista hará falta remallar en quads.
