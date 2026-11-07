#!/bin/bash
# Turntable of a USD asset path traced with athenea under a studio HDRI dome, composited on grey (exr_over.py),
# as PNG frames + MP4. Same lighting as the gallery stills: polished metals need something to reflect.
# usage: hdri_turntable.sh ASSET.usda OUT.mp4 [FRAMES] [SIZE] [SAMPLES] [ELEV_DEG]
# env: HDRI (default out/hdri/brown_photostudio_02.hdr), FPS (24), GREY (0.8), DIST (3.3 × bounding radius),
#      START_DEG (azimuth of the first frame; an Idol asset faces the camera at 180 - yaw in degrees),
#      ATHENEA_BUILD, SPLAT2MESH_GPU_LOCK. Each frame is one job under the GPU lock; frames already rendered are kept.
set -e
ASSET=$1; OUT=$2; N=${3:-96}; SIZE=${4:-1080}; SPP=${5:-128}; ELEV=${6:-15}; FPS=${FPS:-24}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=${ATHENEA_BUILD:-$HOME/luc/athenea-ptaux/build/macos-arm64-release}
LOCK=${SPLAT2MESH_GPU_LOCK:-$HOME/luc/athenea-gpu/shared.sh}
HDRI=${HDRI:-$ROOT/out/hdri/brown_photostudio_02.hdr}
FR="${OUT%.mp4}_frames"; mkdir -p "$FR"; W=$(mktemp -d)
A=$(cd "$(dirname "$ASSET")" && pwd)/$(basename "$ASSET")
printf '#usda 1.0\n(\n defaultPrim = "World"\n upAxis = "Y"\n)\ndef Xform "World"\n{\n def "Asset" (prepend references = @%s@) {}\n def DomeLight "Env"\n {\n  asset inputs:texture:file = @%s@\n  float inputs:intensity = 1.0\n }\n}\n' "$A" "$HDRI" > $W/scene.usda
read CX CY CZ R < <("$ROOT/.venv/bin/python" -c "
from pxr import Usd, UsdGeom
s=Usd.Stage.Open('$A'); b=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(s.GetPseudoRoot()).ComputeAlignedRange()
c=(b.GetMin()+b.GetMax())/2; r=(b.GetMax()-b.GetMin()).GetLength()/2; print(c[0],c[1],c[2],r)")
for ((i=0;i<N;i++)); do
  f=$FR/f_$(printf %04d $i).png; [ -f "$f" ] && continue
  read EX EY EZ < <(python3 -c "import math;a=2*math.pi*$i/$N+math.radians(${START_DEG:-0});e=math.radians($ELEV);d=$R*${DIST:-3.3};print($CX+d*math.sin(a)*math.cos(e),$CY+d*math.sin(e),$CZ+d*math.cos(a)*math.cos(e))")
  bash "$LOCK" bash -c "export ATHENEA_SHADER_DIR=$B/shaders; timeout 300 $B/bin/athenea stage '$W/scene.usda' --eye $EX $EY $EZ --target $CX $CY $CZ --up 0 1 0 --focal 50 --size ${SIZE}x${SIZE} --technique rt --path-bounces 12 --path-total $SPP --denoise -o $W/f.exr > /dev/null 2>&1" < /dev/null
  "$ROOT/.venv/bin/python" "$ROOT/scripts/exr_over.py" $W/f.exr "$f" --grey ${GREY:-0.8} > /dev/null
done
ffmpeg -loglevel error -y -framerate $FPS -i "$FR/f_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT"
rm -rf $W; echo "wrote $OUT ($N frames)"
