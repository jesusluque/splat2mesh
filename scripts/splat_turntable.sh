#!/bin/bash
# Turntable of a raw 3DGS splat (athenea render --splats, unlit SH colour, on grey): PNG frames + MP4.
# usage: splat_turntable.sh SPLAT.ply OUT.mp4 [FRAMES] [ELEV_DEG] [--y-down]     env: START_DEG
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec "$ROOT/.venv/bin/python" -m splat2mesh.athenea splat-turntable "$1" "$2" --frames ${3:-96} --elev ${4:-15} \
  --start-deg ${START_DEG:-0} ${5:+$5}
