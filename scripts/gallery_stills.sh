#!/bin/bash
# One athenea path-traced still per final asset (three-quarter view), for the tutorial gallery and the post.
cd "$(dirname "$0")/.."
B=$HOME/luc/athenea-ptaux/build/macos-arm64-release
mkdir -p out/stills
for a in ${STILL_ONLY:-out/mat_cat3/asset.usda out/testset_v5/0*/usd/asset.usda out/testset2_v5/0*/usd/asset.usda out/testset3_v5/0*/usd/asset.usda out/testset4_v5/0*/usd/asset.usda}; do
  name=$(echo $a | sed 's#out/##; s#/usd/asset.usda##; s#/asset.usda##; s#/#_#g')
  [ -f out/stills/$name.png ] && continue
  read CX CY CZ R < <(.venv/bin/python -c "
from pxr import Usd, UsdGeom
s=Usd.Stage.Open('$a'); b=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(s.GetPseudoRoot()).ComputeAlignedRange()
c=(b.GetMin()+b.GetMax())/2; r=(b.GetMax()-b.GetMin()).GetLength()/2; print(c[0],c[1],c[2],r)" 2>/dev/null)
  read EX EY EZ < <(python3 -c "print($CX+$R*1.3, $CY+$R*0.9, $CZ+$R*3.3)")
  # lit by the studio HDRI (polished metals need something to reflect), composited on grey by depth
  W=$(mktemp -d); A=$(cd "$(dirname $a)" && pwd)/$(basename $a)
  printf '#usda 1.0\n(\n defaultPrim = "World"\n upAxis = "Y"\n)\ndef Xform "World"\n{\n def "Asset" (prepend references = @%s@) {}\n def DomeLight "Env"\n {\n  asset inputs:texture:file = @%s@\n  float inputs:intensity = 1.0\n }\n}\n' "$A" "$PWD/out/hdri/${STILL_HDRI:-brown_photostudio_02}.hdr" > $W/still.usda
  bash ~/luc/athenea-gpu/shared.sh bash -c "export ATHENEA_SHADER_DIR=$B/shaders; timeout 300 $B/bin/athenea stage '$W/still.usda' --eye $EX $EY $EZ --target $CX $CY $CZ --up 0 1 0 --focal 50 --size 640x640 --technique rt --path-bounces 12 --path-total 64 --denoise -o $W/still.exr > /dev/null 2>&1" < /dev/null
  .venv/bin/python scripts/exr_over.py $W/still.exr out/stills/$name.png > /dev/null; rm -rf $W
done
echo "$(date +%H:%M) stills done" >> out/night_progress.log
