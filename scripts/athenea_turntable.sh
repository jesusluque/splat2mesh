#!/bin/bash
# Turntable frames of a USD asset with athenea's path tracer, then an MP4 with ffmpeg.
# usage: athenea_turntable.sh ASSET.usda OUT.mp4 [FRAMES] [SIZE] [SAMPLES] [ELEV_DEG]
# Each frame is one shared.sh job (≤10 min rule), cameras orbit the asset's bounding box centre.
ASSET=$1; OUT=$2; N=${3:-72}; SIZE=${4:-720}; SPP=${5:-32}; ELEV=${6:-12}
B=${ATHENEA_BUILD:-$HOME/luc/athenea-hair/build/hair-run}
D=$(mktemp -d); ROOT=$(cd "$(dirname "$0")/.." && pwd)
read CX CY CZ R < <("$ROOT/.venv/bin/python" -c "
from pxr import Usd, UsdGeom; import sys
s=Usd.Stage.Open('$ASSET'); b=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(s.GetPseudoRoot()).ComputeAlignedRange()
c=(b.GetMin()+b.GetMax())/2; r=(b.GetMax()-b.GetMin()).GetLength()/2; print(c[0],c[1],c[2],r)")
for ((i=0;i<N;i++)); do
  read EX EY EZ < <(python3 -c "import math;a=2*math.pi*$i/$N;e=math.radians($ELEV);d=$R*2.9;print($CX+d*math.sin(a)*math.cos(e),$CY+d*math.sin(e),$CZ+d*math.cos(a)*math.cos(e))")
  bash ~/luc/athenea-gpu/shared.sh bash -c "export ATHENEA_SHADER_DIR=$B/shaders; timeout 300 $B/bin/athenea stage '$ASSET' --eye $EX $EY $EZ --target $CX $CY $CZ --up 0 1 0 --focal 50 --size ${SIZE}x${SIZE} --default-lights --technique rt --path-total $SPP --denoise -o $D/f.exr > /dev/null 2>&1"
  oiiotool $D/f.exr --ch R,G,B --colorconvert linear sRGB -d uint8 -o $D/f_$(printf %04d $i).png 2>/dev/null
done
ffmpeg -loglevel error -y -framerate 12 -i $D/f_%04d.png -vf "minterpolate=fps=24:mi_mode=mci,loop=1:size=$((N*2))" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT"
mkdir -p "${OUT%.mp4}_frames" && cp $D/f_*.png "${OUT%.mp4}_frames/"
cp $D/f_0000.png "${OUT%.mp4}_poster.png"; rm -rf $D; echo "wrote $OUT"
