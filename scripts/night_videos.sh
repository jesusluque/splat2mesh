#!/bin/bash
# Turntable videos (athenea path tracing, fixed build) of finished models; each one goes to Telegram.
cd "$(dirname "$0")/.."
export ATHENEA_BUILD=$HOME/luc/athenea-ptaux/build/macos-arm64-release
mkdir -p out/video
while pgrep -f "athenea_turntable.sh out/testset3_v5/01" >/dev/null; do sleep 20; done
while read name asset caption; do
  [ -f "$asset" ] || continue
  bash scripts/athenea_turntable.sh "$asset" "out/video/$name.mp4" 24 600 32 > "out/video/$name.log" 2>&1 < /dev/null \
    && /Users/muriel/bin/tg -s --from splat2mesh -f "out/video/$name.mp4" "$caption" < /dev/null
done <<'LIST'
brass_candlestick out/testset_v5/01/usd/asset.usda Candelabro de latón (F0 físico)
mixed_lamp out/testset_v5/04/usd/asset.usda Lámpara: madera, acero cepillado, porcelana
porcelain_teapot out/testset_v5/00/usd/asset.usda Tetera de porcelana
jade_bangle out/testset3_v5/04/usd/asset.usda Brazalete de jade (subsuperficie MaterialX)
wax_candle out/testset3_v5/03/usd/asset.usda Vela de cera (subsuperficie)
perfume_bottle out/testset3_v5/02/usd/asset.usda Frasco de perfume: vidrio + oro
lucky_cat_v2 out/mat_cat3/asset.usda Gato (encuadre y ojos corregidos)
LIST
echo "$(date +%H:%M) videos batch 1 done" >> out/night_progress.log
