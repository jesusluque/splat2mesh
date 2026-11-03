#!/bin/bash
# ixwin queue: set 3 (glass, translucent, metals) with ENEAS in ixwin's WSL.
cd "$(dirname "$0")/.."
export SPLAT2MESH_ENEAS_HOST=ixwin-tb
.venv/bin/python scripts/flow_batch.py out/testset3 out/testset3_v5 data/testset3/prompts.txt > out/night_flow_testset3.log 2>&1
echo "$(date +%H:%M) done testset3" >> out/night_progress.log
