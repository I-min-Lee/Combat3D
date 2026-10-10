#!/bin/bash
# ============================================================================
#  08_gt_extract.sh  —  STAGE 2.1 ground truth — extract OFFICIAL Harmony4D GT
#  Evaluation must be against official GT, never against our own labels (self-certification).
#    reads  : $RAW/<take>
#    writes : $B/gt_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 08_gt_extract.sh <take>}
RAW=${RAW:-$B/data/harmony4d/raw}

$PY "$B/code/label_pipeline/adapters/harmony4d/h4d_gt.py" \
    --seq-root "$RAW/$TAKE" --out "$B/gt_$TAKE" --views "$V"
echo "-> $B/gt_$TAKE"
