#!/bin/bash
# One athenea path-traced view of a USD asset; the eye is given in units of the bounding-sphere radius.
# usage: athenea_view.sh ASSET.usda OUT.png MX MY MZ [SIZE] [SPP]
ASSET=$1; OUT=$2; MX=$3; MY=$4; MZ=$5; SIZE=${6:-1024}; SPP=${7:-96}
B=${ATHENEA_BUILD:-$HOME/luc/athenea-ptaux/build/macos-arm64-release}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
read CX CY CZ R < <("$ROOT/.venv/bin/python" -c "
from pxr import Usd, UsdGeom
s=Usd.Stage.Open('$ASSET'); b=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(s.GetPseudoRoot()).ComputeAlignedRange()
c=(b.GetMin()+b.GetMax())/2; r=(b.GetMax()-b.GetMin()).GetLength()/2; print(c[0],c[1],c[2],r)" 2>/dev/null)
read EX EY EZ < <(python3 -c "print($CX+$R*$MX, $CY+$R*$MY, $CZ+$R*$MZ)")
T=$(mktemp -d)
bash ~/luc/athenea-gpu/shared.sh bash -c "export ATHENEA_SHADER_DIR=$B/shaders; timeout 550 $B/bin/athenea stage '$ASSET' --eye $EX $EY $EZ --target $CX $CY $CZ --up 0 1 0 --focal 50 --size ${SIZE}x${SIZE} --default-lights --technique rt --path-bounces 12 --path-total $SPP --denoise -o $T/v.exr > /dev/null 2>&1 && oiiotool $T/v.exr --ch R,G,B --colorconvert linear sRGB -d uint8 -o '$OUT'" < /dev/null
rm -rf $T
