#!/bin/bash
# Redo objects whose materials flow failed, after the main runs finish.
cd "$(dirname "$0")/.."
export SPLAT2MESH_ENEAS_HOST=render94
while pgrep -f night_run3.sh >/dev/null; do sleep 60; done
.venv/bin/python scripts/materials_eneas3d.py out/testset_v5/02/usd out/testset/02/idol_input.png "$(cat out/testset_v5/02/materials.txt)" > out/night_redo_owl.log 2>&1
echo "$(date +%H:%M) done redo owl" >> out/night_progress.log
echo "urushi lacquer: the bowl; gold: the gold rim" > out/testset_v5/03/materials.txt
.venv/bin/python scripts/materials_eneas3d.py out/testset_v5/03/usd out/testset/03/idol_input.png "$(cat out/testset_v5/03/materials.txt)" > out/night_redo_bowl.log 2>&1
echo "$(date +%H:%M) done redo bowl" >> out/night_progress.log
