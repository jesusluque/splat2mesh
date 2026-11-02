#!/bin/bash
# Night run, part 2: the coherent materials flow with ENEAS on the L4 (render94) while Idol keeps running on ixwin.
cd "$(dirname "$0")/.."
export SPLAT2MESH_ENEAS_HOST=render94
PY=.venv/bin/python
for src in testset testset2 testset3; do
  if [ "$src" = testset3 ]; then   # wait for Idol to finish set 3 on ixwin
    while pgrep -f "splat2usd.py data/testset3" >/dev/null; do sleep 60; done
  fi
  $PY scripts/flow_batch.py out/$src out/${src}_v5 data/$src/prompts.txt > out/night_flow_${src}.log 2>&1
  echo "$(date +%H:%M) done $src" >> out/night_progress.log
done
echo "$(date +%H:%M) ALL DONE" >> out/night_progress.log
