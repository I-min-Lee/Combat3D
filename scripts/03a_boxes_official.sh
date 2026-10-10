#!/bin/bash
# ============================================================================
#  03a_boxes_official.sh  —  STAGE 1.3a label production — OFFICIAL boxes + identity
#  Scene-specific layer. Use the dataset's own boxes/identity when it has them: cheapest and best configured.
#    reads  : $RAW/<take>
#    writes : $B/det_gt2_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 03a_boxes_official.sh <take>}
RAW=${RAW:-$B/data/harmony4d/raw}

$PY "$B/code/label_pipeline/adapters/harmony4d/h4d_boxes.py" \
    --seq-root "$RAW/$TAKE" --out "$B/det_gt2_$TAKE" --views "$V" --tag "$TAKE"
echo "-> $B/det_gt2_$TAKE"
