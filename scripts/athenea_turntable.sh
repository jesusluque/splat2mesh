#!/bin/bash
# Path-traced turntable of a USD asset (athenea, studio HDRI, on grey). Same as hdri_turntable.sh with TECHNIQUE=rt.
# usage: athenea_turntable.sh ASSET.usda OUT.mp4 [FRAMES] [SIZE] [SAMPLES] [ELEV_DEG]
TECHNIQUE=rt exec bash "$(dirname "$0")/hdri_turntable.sh" "$@"
