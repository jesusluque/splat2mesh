#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "splat2usd.py data/testset5" >/dev/null; do sleep 30; done
cp out/testset2/07/idol_input.png /tmp/car_in.png
bash ~/luc/athenea-gpu/shared.sh third_party/iris-3b/.venv/bin/python scripts/describe_materials.py /tmp/car_in.png --prompt "$(sed -n 8p data/testset2/prompts.txt)" --out out/testset2_v5/07/materials.txt > /dev/null 2>&1
grep -q -i "glass" out/testset2_v5/07/materials.txt || echo "; glass: the windows" >> out/testset2_v5/07/materials.txt
rm -f out/testset2_v5/07/usd/mat/masks.npz
SPLAT2MESH_ENEAS_HOST=ixwin-tb .venv/bin/python scripts/materials_eneas3d.py out/testset2_v5/07/usd out/testset2/07/idol_input.png "$(cat out/testset2_v5/07/materials.txt)" > out/redo_car.log 2>&1
echo "$(date +%H:%M) redo car (windows)" >> out/night_progress.log
