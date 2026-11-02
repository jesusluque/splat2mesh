#!/bin/bash
# Render a 3DGS .ply with athenea (`render --splats`, unlit SH colour) from OUTDIR/cameras.txt.
# Always run under shared.sh:  bash ~/luc/athenea-gpu/shared.sh bash scripts/render_athenea.sh IN.ply OUTDIR [extra athenea flags]
B=${ATHENEA_BUILD:-$HOME/luc/athenea-hair/build/hair-run}; A=$B/bin/athenea; export ATHENEA_SHADER_DIR=$B/shaders
IN=$1; O=$2; shift 2; d=$O/views; s=$(date +%s)
while read n ex ey ez tx ty tz ux uy uz; do
  [ -n "${ONLY:-}" ] && [ "$n" != "$ONLY" ] && continue
  timeout 60 $A render --splats "$IN" --eye $ex $ey $ez --target $tx $ty $tz --up $ux $uy $uz --focal 50 --size 1024x1024 "$@" -o $d/$n.exr > /dev/null 2>&1 || echo "$n failed"
  oiiotool $d/$n.exr --ch R,G,B,A --colorconvert linear sRGB --pattern constant:color=0.5,0.5,0.5,1 1024x1024 4 --over -d uint8 -o $d/$n.png 2>/dev/null
  oiiotool $d/$n.exr --ch A --mulc 2 --clamp:min=0:max=1 -d uint8 -o $d/${n}_mask.png 2>/dev/null
done < $O/cameras.txt
echo "done in $(( $(date +%s) - s )) s"
