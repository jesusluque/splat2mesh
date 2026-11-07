#!/bin/bash
# One athenea view of a USD asset under the studio HDRI, on grey; the eye is given in units of the bounding-sphere
# radius from its centre.   usage: athenea_view.sh ASSET.usda OUT.png MX MY MZ [SIZE] [SPP]     env: TECHNIQUE (rt)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec "$ROOT/.venv/bin/python" -m splat2mesh.athenea view "$1" "$2" "$3,$4,$5" --size ${6:-1024} --spp ${7:-96} \
  --technique ${TECHNIQUE:-rt}
