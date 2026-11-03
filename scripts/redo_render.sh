#!/bin/bash
# Rebuild crumpled direct meshes with the render path, then re-run materials with the existing ENEAS masks.
cd "$(dirname "$0")/.."
for item in testset:05 testset2:03 testset2:00 testset2:05; do
  s=${item%%:*}; n=${item##*:}; ply=out/$s/$n/$n.ply; dst=out/${s}_v5/$n/usd
  [ -f "$ply" ] || { echo "no ply $ply"; continue; }
  tmp=out/${s}_v5/$n/render_path; rm -rf $tmp
  .venv/bin/python scripts/splat2usd.py "$ply" --y-down --no-antialias --out $tmp > $tmp.log 2>&1 || { echo "render failed $s/$n"; continue; }
  mkdir -p $dst.new && cp -r $tmp/$n/usd/* $dst.new/ && cp -r $dst/mat $dst.new/mat && rm -rf $dst.new/mat/orbit $dst.new/mat/texel_class.npy $dst.new/mat/regions.json
  rm -rf $dst.old && mv $dst $dst.old && mv $dst.new $dst
  SPLAT2MESH_ENEAS_HOST=ixwin-tb .venv/bin/python scripts/materials_eneas3d.py $dst out/$s/$n/idol_input.png "$(cat out/${s}_v5/$n/materials.txt)" --reuse-masks > out/${s}_v5/$n/redo_render_mat.log 2>&1
  rm -rf $dst.old $tmp
  echo "$(date +%H:%M) re-rendered $s/$n" >> out/night_progress.log
done
