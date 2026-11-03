#!/bin/bash
# Build the web gallery from the latest version of every object (one copy each). Titles must not contain ':'.
cd "$(dirname "$0")/.."
ARGS=("lucky_cat_v10=out/mat_cat3:Lucky cat — porcelain, gold, lacquer:data/inbox/maneki_white.png:out/stills/mat_cat3.png")
add() { local set=$1 dir=$2; shift 2; local i=0; for nt in "$@"; do n=${nt%%=*}; t=${nt#*=}
  a=out/$dir/0$i/usd; [ -f $a/asset.usda ] && ARGS+=("${n}_v10=$a:$t:data/$set/img/0$i.png:out/stills/${dir}_0$i.png"); i=$((i+1)); done; }
add testset testset_v5 teapot="Porcelain teapot" brass_candlestick="Brass candlestick" walnut_owl="Walnut owl" \
  red_lacquer_bowl="Red lacquer bowl" mixed_lamp="Lamp — oak, steel, porcelain" lacquer_box="Lacquer box — brass, nacre" \
  alu_robot="Aluminium robot" celadon_vase="Celadon vase with kintsugi"
add testset2 testset2_v5 copper_kettle="Hammered copper kettle" marble_bust="Marble bust" jade_dragon="Jade dragon" \
  cloisonne_vase="Cloisonné enamel vase" chrome_toaster="Chrome toaster" leather_boot="Leather boot" \
  bamboo_basket="Bamboo basket, lacquered lid" tin_toy_car="Tin toy car"
add testset3 testset3_v5 edison_bulb="Edison bulb — glass, brass" wine_glass="Crystal wine glass" perfume_bottle="Perfume bottle — glass, gold" \
  wax_candle="Wax candle (translucent)" jade_bangle="Jade bangle (translucent)" blue_bottle="Blue glass bottle, cork" \
  silver_teapot="Silver teapot, wooden handle" hurricane_lantern="Brass and glass lantern"
add testset4 testset4_v5 pocket_watch="Gold pocket watch" tiffany_lamp="Tiffany lamp — stained glass, bronze" spyglass="Brass spyglass" \
  porcelain_cup="Porcelain cup, gold rim, silver spoon" leather_book="Leather book, gold, brass" marble_knight="Marble knight on lacquer" \
  absinthe_bottle="Absinthe bottle — glass, pewter" clay_teapot="Black clay teapot, bamboo handle"
add testset5 testset5_v5 film_camera="Film camera — leather, chrome, glass" violin="Violin — varnished wood, ebony" \
  silver_candelabra="Silver candelabra, wax candles" diamond_ring="Gold ring with diamond" terrarium="Glass and brass terrarium" \
  coffee_grinder="Coffee grinder — copper, porcelain, wood" mantel_clock="Marble and brass mantel clock" ceramic_mug="Ceramic mug, walnut handle"
rm -rf out/web_new && .venv/bin/python scripts/export_web.py out/web_new "${ARGS[@]}" 2>&1 | grep -v Warning | tail -3 && rm -rf out/web && mv out/web_new out/web
ls out/web/models | grep -c usdz; du -sh out/web
