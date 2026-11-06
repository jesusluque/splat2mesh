#!/bin/bash
# Fetch Idol probe samples from ixwin as they appear and score each with seed_views.sh (4 views + DeepSeek check).
#   scripts/probe_score.sh REMOTE_DIR LOCAL_DIR PHOTO_DIR EXPECTED_COUNT
# PHOTO_DIR holds the inputs by name (t3_00.png ...); a sample "t3_00_s42[tag]" is scored against PHOTO_DIR/t3_00.png.
cd "$(dirname "$0")/.."
remote=$1; local=$2; photos=$3; n=$4
mkdir -p $local
while true; do
  for f in $(ssh ixwin-tb "dir /b $(echo $remote | tr / '\\\\')\\*_acts.npy" 2>/dev/null | tr -d '\r'); do
    b=${f%_acts.npy}
    [ -f $local/$b.check.json ] && continue
    for x in .ply _camera.json _acts.npy; do
      [ -f $local/$b$x ] || scp -q ixwin-tb:$remote/$b$x $local/ || continue 2
    done
    photo=$photos/$(echo $b | sed -E 's/_s[0-9]+.*$//').png
    scripts/seed_views.sh $local/$b.ply $local/${b}_camera.json $photo $local/$b || echo "failed $b"
    rm -rf $local/$b/frames 2>/dev/null
  done
  [ $(ls $local/*.check.json 2>/dev/null | wc -l) -ge $n ] && break
  sleep 120
done
echo done
