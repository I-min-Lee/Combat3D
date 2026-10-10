#!/bin/bash
# ============================================================================
#  03b_boxes_self.sh  —  STAGE 1.3b label production — SELF-BUILT detection + identity
#  Scene-specific layer. Only needed when the dataset has no boxes. --border-margin is the position prior.
#    reads  : $B/frames/<ID>/1, $B/calib_gt_<take>
#    writes : $B/det_self2_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 03b_boxes_self.sh <take> <int-seq-id> <last-frame>}
ID=${2:?need integer seq id}
NF=${3:?need last frame index}
DEV=${DEV:-0}

$PY "$B/code/label_pipeline/adapters/harmony4d/pipeline/det_self_final.py" \
    --frames-root "$B/frames/$ID/1" --out "$B/det_self2_$TAKE" \
    --views "$V" --tag "$TAKE" --start 1 --end "$NF" \
    --border-margin 0.15 --ref-view 04 --calib "$B/calib_gt_$TAKE" --device "$DEV"
echo "-> $B/det_self2_$TAKE"
