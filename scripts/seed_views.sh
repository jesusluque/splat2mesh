#!/bin/bash
# Raw splat from 4 yaws around its own predicted camera (athenea, GPU lock), then the DeepSeek validator.
#   scripts/seed_views.sh SPLAT.ply CAMERA.json PHOTO OUT_DIR
set -e
cd "$(dirname "$0")/.."
ply=$1; cam=$2; photo=$3; out=$4
.venv/bin/python -c "
import sys;sys.path.insert(0,'scripts')
from pathlib import Path
import gen_views as g
g.SIDES={'left':90.0,'back':180.0,'right':270.0}
g.cameras('$ply','$cam',Path('$out'))"
bash scripts/render_athenea.sh $ply $out --rotate-x 180 < /dev/null > $out.log 2>&1     # takes the GPU lock itself
p=$out/views
.venv/bin/python scripts/mv_check.py $photo 0=$p/front.png 90=$p/left.png 180=$p/back.png 270=$p/right.png --json $out.check.json \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('$out score',d['score'],'coh',d['coherent'],'|',d['worst'][0][:150])"
