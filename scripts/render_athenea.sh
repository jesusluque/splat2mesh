#!/bin/bash
# Render a 3DGS .ply with athenea (render --splats, unlit SH colour) from OUTDIR/cameras.txt into OUTDIR/views
# (NAME.exr, NAME.png on grey, NAME_mask.png). The GPU lock (config gpu.lock) is taken inside.
# usage: render_athenea.sh IN.ply OUTDIR [--rotate-x 180] [other athenea render flags]     env: ONLY=NAME
ROOT=$(cd "$(dirname "$0")/.." && pwd); IN=$1; O=$2; shift 2
YD=""; EXTRA=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--rotate-x" ] && [ "$2" = "180" ]; then YD="--y-down"; shift 2; else EXTRA+=("$1"); shift; fi
done
CAMS=$O/cameras.txt
if [ -n "${ONLY:-}" ]; then grep "^$ONLY " "$O/cameras.txt" > "$O/.only.txt"; CAMS=$O/.only.txt; fi
s=$(date +%s)
"$ROOT/.venv/bin/python" -m splat2mesh.athenea render-splats "$IN" "$CAMS" "$O" $YD -- "${EXTRA[@]}"
echo "done in $(( $(date +%s) - s )) s"
