#!/bin/bash
# render94 queue: finish set 1 (already running), then sets 2 and 4, then redos.
cd "$(dirname "$0")/.."
export SPLAT2MESH_ENEAS_HOST=render94
while pgrep -f "flow_batch.py out/testset out/testset_v5" >/dev/null; do sleep 30; done
echo "$(date +%H:%M) done testset" >> out/night_progress.log
.venv/bin/python scripts/flow_batch.py out/testset2 out/testset2_v5 data/testset2/prompts.txt > out/night_flow_testset2.log 2>&1
echo "$(date +%H:%M) done testset2" >> out/night_progress.log
while pgrep -f "splat2usd.py data/testset4" >/dev/null; do sleep 30; done
.venv/bin/python scripts/flow_batch.py out/testset4 out/testset4_v5 data/testset4/prompts.txt > out/night_flow_testset4.log 2>&1
echo "$(date +%H:%M) done testset4" >> out/night_progress.log
.venv/bin/python scripts/materials_eneas3d.py out/testset_v5/02/usd out/testset/02/idol_input.png "$(cat out/testset_v5/02/materials.txt)" > out/night_redo_owl.log 2>&1
echo "urushi lacquer: the bowl; gold: the gold rim" > out/testset_v5/03/materials.txt
.venv/bin/python scripts/materials_eneas3d.py out/testset_v5/03/usd out/testset/03/idol_input.png "$(cat out/testset_v5/03/materials.txt)" > out/night_redo_bowl.log 2>&1
.venv/bin/python scripts/materials_eneas3d.py out/testset_v5/05/usd out/testset/05/idol_input.png "$(cat out/testset_v5/05/materials.txt)" --reuse-masks > out/night_redo_box.log 2>&1
echo "$(date +%H:%M) done redos (owl, bowl, box)" >> out/night_progress.log
.venv/bin/python scripts/materials_eneas3d.py out/testset_v5/07/usd out/testset/07/idol_input.png "$(cat out/testset_v5/07/materials.txt)" --reuse-masks > out/night_redo_vase.log 2>&1
echo "$(date +%H:%M) done redo vase" >> out/night_progress.log
.venv/bin/python scripts/materials_eneas3d.py out/testset2_v5/03/usd out/testset2/03/idol_input.png "$(cat out/testset2_v5/03/materials.txt)" --reuse-masks > out/night_redo_cloisonne.log 2>&1
echo "$(date +%H:%M) done redo cloisonne" >> out/night_progress.log
.venv/bin/python scripts/materials_eneas3d.py out/testset2_v5/05/usd out/testset2/05/idol_input.png "$(cat out/testset2_v5/05/materials.txt)" --reuse-masks > out/night_redo_boot.log 2>&1
echo "$(date +%H:%M) done redo boot" >> out/night_progress.log
.venv/bin/python scripts/materials_eneas3d.py out/testset2_v5/07/usd out/testset2/07/idol_input.png "$(cat out/testset2_v5/07/materials.txt)" --reuse-masks > out/night_redo_car.log 2>&1
echo "$(date +%H:%M) done redo car" >> out/night_progress.log
.venv/bin/python scripts/materials_eneas3d.py out/testset4_v5/00/usd out/testset4/00/idol_input.png "$(cat out/testset4_v5/00/materials.txt)" --reuse-masks > out/night_redo_watch.log 2>&1
echo "$(date +%H:%M) done redo watch" >> out/night_progress.log
.venv/bin/python scripts/materials_eneas3d.py out/testset4_v5/01/usd out/testset4/01/idol_input.png "$(cat out/testset4_v5/01/materials.txt)" --reuse-masks > out/night_redo_tiffany.log 2>&1
echo "$(date +%H:%M) done redo tiffany" >> out/night_progress.log
