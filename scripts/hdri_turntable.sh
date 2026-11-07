#!/bin/bash
# Turntable of a USD asset with athenea under the studio HDRI, composited on grey: PNG frames + MP4.
# usage: hdri_turntable.sh ASSET.usda OUT.mp4 [FRAMES] [SIZE] [SAMPLES] [ELEV_DEG]
# env: TECHNIQUE (raster, default, or rt), START_DEG, DIST (× bounding radius, 3.3), GREY (0.8), HDRI.
# A thin wrapper over: python -m splat2mesh.athenea turntable (one athenea process for all frames).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec "$ROOT/.venv/bin/python" -m splat2mesh.athenea turntable "$1" "$2" --frames ${3:-96} --size ${4:-1080} \
  --spp ${5:-96} --elev ${6:-15} --technique ${TECHNIQUE:-raster} --start-deg ${START_DEG:-0} --dist ${DIST:-3.3} \
  --grey ${GREY:-0.8} ${HDRI:+--hdri "$HDRI"}
