#!/bin/bash
# ============================================================================
#  01_calib.sh  —  STAGE 1.1 label production — calibration
#  Reads COLMAP SfM output, writes extri.yml/intri.yml/h4d_meta.json. World frame = COLMAP frame.
#    reads  : $RAW/<take>
#    writes : $B/calib_gt_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 01_calib.sh <take>   e.g. 016_mma4}
RAW=${RAW:-$B/data/harmony4d/raw}

$PY "$B/code/label_pipeline/adapters/harmony4d/h4d_calib.py" \
    --seq-root "$RAW/$TAKE" --out "$B/calib_gt_$TAKE" --views "$V"
echo "-> $B/calib_gt_$TAKE"
