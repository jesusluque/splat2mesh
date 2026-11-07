#!/bin/bash
# Turntable of a USD asset path traced with athenea under a studio HDRI dome, composited on grey (exr_over.py),
# as PNG frames + MP4. Same lighting as the gallery stills: polished metals need something to reflect.
# usage: hdri_turntable.sh ASSET.usda OUT.mp4 [FRAMES] [SIZE] [SAMPLES] [ELEV_DEG]
# env: HDRI (default out/hdri/brown_photostudio_02.hdr), FPS (24), GREY (0.8), DIST (3.3 × bounding radius),
#      START_DEG (azimuth of the first frame; an Idol asset faces the camera at 180 - yaw in degrees),
#      TECHNIQUE (rt, default, or raster),
#      ATHENEA_BUILD, SPLAT2MESH_GPU_LOCK. All frames are rendered by one athenea process (stage --turntable), one
#      job under the GPU lock.
set -e
ASSET=$1; OUT=$2; N=${3:-96}; SIZE=${4:-1080}; SPP=${5:-128}; ELEV=${6:-15}; FPS=${FPS:-24}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=${ATHENEA_BUILD:-$HOME/luc/athenea-ptaux/build/macos-arm64-release}
LOCK=${SPLAT2MESH_GPU_LOCK:-$HOME/luc/athenea-gpu/shared.sh}
HDRI=${HDRI:-$ROOT/out/hdri/brown_photostudio_02.hdr}
FR="${OUT%.mp4}_frames"; mkdir -p "$FR"; W=$(mktemp -d)
# TECHNIQUE=raster: athenea's rasteriser with the prefiltered dome; on opaque assets it looks like the path tracer
# (measured: 46 s vs 107 s a 1080² frame, nearly all of it scene loading). Keep rt for glass and translucency.
if [ "${TECHNIQUE:-rt}" = raster ]; then TECHFLAGS="--technique raster"
else TECHFLAGS="--technique rt --path-bounces 12 --path-total $SPP --denoise"; fi
A=$(cd "$(dirname "$ASSET")" && pwd)/$(basename "$ASSET")
printf '#usda 1.0\n(\n defaultPrim = "World"\n upAxis = "Y"\n)\ndef Xform "World"\n{\n def "Asset" (prepend references = @%s@) {}\n def DomeLight "Env"\n {\n  asset inputs:texture:file = @%s@\n  float inputs:intensity = 1.0\n }\n}\n' "$A" "$HDRI" > $W/scene.usda
read CX CY CZ R < <("$ROOT/.venv/bin/python" -c "
from pxr import Usd, UsdGeom
s=Usd.Stage.Open('$A'); b=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(s.GetPseudoRoot()).ComputeAlignedRange()
c=(b.GetMin()+b.GetMax())/2; r=(b.GetMax()-b.GetMin()).GetLength()/2; print(c[0],c[1],c[2],r)")
# one athenea process for all the frames (stage --turntable: the scene is loaded once; frame i turns the eye
# i·360/N degrees about --up through --target, the same sense as below)
read EX EY EZ < <(python3 -c "import math;a=math.radians(${START_DEG:-0});e=math.radians($ELEV);d=$R*${DIST:-3.3};print($CX+d*math.sin(a)*math.cos(e),$CY+d*math.sin(e),$CZ+d*math.cos(a)*math.cos(e))")
mkdir -p $W/exr
bash "$LOCK" bash -c "export ATHENEA_SHADER_DIR=$B/shaders; $B/bin/athenea stage '$W/scene.usda' --eye $EX $EY $EZ --target $CX $CY $CZ --up 0 1 0 --turntable $N --focal 50 --size ${SIZE}x${SIZE} $TECHFLAGS -o $W/exr/ > $W/athenea.log 2>&1" < /dev/null
for ((i=0;i<N;i++)); do
  "$ROOT/.venv/bin/python" "$ROOT/scripts/exr_over.py" $W/exr/f_$(printf %04d $i).exr "$FR/f_$(printf %04d $i).png" --grey ${GREY:-0.8} > /dev/null
done
ffmpeg -loglevel error -y -framerate $FPS -i "$FR/f_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT"
rm -rf $W; echo "wrote $OUT ($N frames)"
