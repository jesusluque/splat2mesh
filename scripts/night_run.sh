#!/bin/bash
# Autonomous run: Idol for test set 3, then the full coherent flow for the three sets (direct HD mesh, render
# fallback, Qwen3-VL part descriptions, ENEAS 3D, PBR with transmission/subsurface/F0, Cryptomatte parts).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/splat2usd.py data/testset3/img --direct --gaussians 524288 --out out/testset3 > out/night_idol3.log 2>&1
for set in testset:testset testset2:testset2 testset3:testset3; do
  src=${set%%:*}; data=${set##*:}
  $PY scripts/flow_batch.py out/$src out/${src}_v5 data/$data/prompts.txt > out/night_flow_${src}.log 2>&1
  echo "$(date +%H:%M) done $src" >> out/night_progress.log
done
echo "$(date +%H:%M) ALL DONE" >> out/night_progress.log
