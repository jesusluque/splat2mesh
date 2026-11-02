#!/bin/bash
# Set 4: wait for its Idol (render94) and for the flow of sets 1-3, then run the flow on set 4.
cd "$(dirname "$0")/.."
export SPLAT2MESH_ENEAS_HOST=render94
while pgrep -f "splat2usd.py data/testset4" >/dev/null || pgrep -f night_run2.sh >/dev/null; do sleep 60; done
.venv/bin/python scripts/flow_batch.py out/testset4 out/testset4_v5 data/testset4/prompts.txt > out/night_flow_testset4.log 2>&1
echo "$(date +%H:%M) done testset4" >> out/night_progress.log
