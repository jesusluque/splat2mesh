#!/bin/bash
# Turntable of a raw 3DGS splat (athenea `render --splats`, unlit SH colour, on grey), as PNG frames + MP4.
# usage: splat_turntable.sh SPLAT.ply OUT.mp4 [FRAMES] [ELEV_DEG] [--y-down]
# --y-down for Idol/COLMAP splats (rendered with --rotate-x 180, like the pipeline). 1024² frames.
set -e
PLY=$1; OUT=$2; N=${3:-96}; ELEV=${4:-15}; YD=${5:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOCK=${SPLAT2MESH_GPU_LOCK:-$HOME/luc/athenea-gpu/shared.sh}
D="${OUT%.mp4}_frames"; mkdir -p "$D/views"
PYTHONPATH="$ROOT" "$ROOT/.venv/bin/python" - "$PLY" "$D" "$N" "$ELEV" "$YD" <<'E'
import sys, math, numpy as np
from pathlib import Path
ply, d, n, elev, yd = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), math.radians(float(sys.argv[4])), sys.argv[5] == "--y-down"
from splat2mesh.ply import read_vertex
v = read_vertex(ply)
xyz = np.stack([v["x"], v["y"], v["z"]], 1) * (np.array([1, -1, -1]) if yd else 1)
op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
p = xyz[op > 0.05]; lo, hi = np.percentile(p, 0.5, 0), np.percentile(p, 99.5, 0)
C = (lo + hi) / 2; r = float(np.linalg.norm(hi - lo) / 2)
D = r / math.sin(math.atan(24.576 / 2 / 50.0)) * 1.05
lines = []
for i in range(n):
    a = 2 * math.pi * i / n
    e = C + D * np.array([math.sin(a) * math.cos(elev), math.sin(elev), math.cos(a) * math.cos(elev)])
    lines.append(f"f_{i:04d} {e[0]} {e[1]} {e[2]} {C[0]} {C[1]} {C[2]} 0 1 0")
(d / "cameras.txt").write_text("\n".join(lines) + "\n")
E
FLAGS=""; [ "$YD" = "--y-down" ] && FLAGS="--rotate-x 180"
bash "$LOCK" bash "$ROOT/scripts/render_athenea.sh" "$PLY" "$D" $FLAGS < /dev/null
ffmpeg -loglevel error -y -framerate ${FPS:-24} -i "$D/views/f_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT"
echo "wrote $OUT"
