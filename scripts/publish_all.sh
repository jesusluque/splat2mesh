#!/bin/bash
# Automatic publish: materials re-apply → athenea stills → gallery → web (posters = new stills) → sysop → Telegram.
cd "$(dirname "$0")/.."
while pgrep -f reapply_par.py >/dev/null || pgrep -f redo_render.sh >/dev/null || pgrep -f redo_car.sh >/dev/null; do sleep 20; done
rm -f out/stills/*.png
bash scripts/gallery_stills.sh > out/gallery_stills3.log 2>&1
.venv/bin/python scripts/compose_gallery.py > /dev/null 2>&1
bash scripts/web_all.sh > out/publish_web.log 2>&1
/Users/muriel/bin/cc-send --from splat2mesh sysop "splat2mesh pide a sysop redesplegar https://splat2mesh.lucab.co.uk con /Users/muriel/splat2mesh/out/web, sustituida entera: v8, 33 *_v8.usdz (con *_parts.json), pósters, entradas e hdri/. Rugosidad más realista, acabado suavizado y vidrio y jade físicos en el visor. Gracias." > /dev/null 2>&1
/Users/muriel/bin/tg -s --from splat2mesh -f docs/img/gallery.jpg "Galería final (v8): rugosidad realista, acabado suavizado, metales con F0 + detalle fino. Web en despliegue." < /dev/null
echo "$(date +%H:%M) publish_all done" >> out/night_progress.log
