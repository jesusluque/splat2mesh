#!/bin/bash
# Re-run the material stage (saved ENEAS masks, new phrases only) on every finished asset, 2 at a time.
cd "$(dirname "$0")/.."
one() {
  d=$1; src=$(echo $d | sed 's#_v5/\(0[0-9]\)/usd#/\1#')
  img=$src/idol_input.png; [ -f out/mat_cat3/asset.usda ] && [ "$d" = out/mat_cat3 ] && img=out/mat_cat3/idol_input.png
  desc=$(cat $(dirname $d)/materials.txt 2>/dev/null || cat $d/materials.txt 2>/dev/null)
  [ -z "$desc" ] && { echo "$d no description"; return; }
  r=$(.venv/bin/python scripts/materials_eneas3d.py $d $img "$desc" --reuse-masks 2>&1 | grep -E "^\{'seen" | cut -c1-200)
  echo "$(date +%H:%M) $d ${r:-FAILED}"
}
export -f one
{ [ $# -gt 0 ] && printf "%s\n" "$@" || ls -d out/testset{,2,3,4}_v5/0*/usd; } | xargs -P 2 -I{} bash -c 'one {}'
echo "$(date +%H:%M) rematerial done"
