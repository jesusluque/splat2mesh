#!/bin/bash
cd "$(dirname "$0")/.."
.venv/bin/python scripts/splat2usd.py data/testset5/img --direct --skip-done --gaussians 524288 --out out/testset5 > out/night_idol5.log 2>&1
SPLAT2MESH_ENEAS_HOST=ixwin-tb .venv/bin/python scripts/flow_batch.py out/testset5 out/testset5_v5 data/testset5/prompts.txt > out/night_flow_testset5.log 2>&1
echo "$(date +%H:%M) done testset5" >> out/night_progress.log
