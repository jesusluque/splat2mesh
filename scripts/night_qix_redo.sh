#!/bin/bash
cd "$(dirname "$0")/.."
export SPLAT2MESH_ENEAS_HOST=ixwin-tb
while pgrep -f night_qix.sh >/dev/null; do sleep 30; done
R=""; [ -f out/testset3_v5/06/usd/mat/masks.npz ] && R="--reuse-masks"
.venv/bin/python scripts/materials_eneas3d.py out/testset3_v5/06/usd out/testset3/06/idol_input.png "$(cat out/testset3_v5/06/materials.txt)" $R > out/night_redo_silver.log 2>&1
rm -rf out/testset3_v5/07/usd/mat
.venv/bin/python scripts/materials_eneas3d.py out/testset3_v5/07/usd out/testset3/07/idol_input.png "$(cat out/testset3_v5/07/materials.txt)" > out/night_redo_lantern.log 2>&1
echo "$(date +%H:%M) done redo silver teapot + lantern" >> out/night_progress.log
