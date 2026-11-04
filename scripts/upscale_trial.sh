#!/bin/bash
# Rebuild a few objects with --upscale (Iris restorer before Idol) next to the current ones, for comparison.
# usage: upscale_trial.sh SET/NN [SET/NN ...]     e.g. testset2/00
cd "$(dirname "$0")/.."
for sn in "$@"; do
  set_=${sn%/*}; nn=${sn#*/}; out=out/upscaled/${set_}_$nn
  [ -f $out/$nn/usd/asset.usda ] || .venv/bin/python scripts/splat2usd.py data/$set_/img/$nn.png --direct --upscale --y-down \
      --gaussians 524288 --out $out > $out.log 2>&1
  desc=$(cat out/${set_}_v5/$nn/materials.txt)
  .venv/bin/python scripts/materials_eneas3d.py $out/$nn/usd $out/$nn/idol_input.png "$desc" 2>&1 | grep -E "^\{'seen" | cut -c1-160
  echo "$(date +%H:%M) $sn done"
done
