#!/bin/bash
# ============================================================================
#  14_train_roothead.sh  —  STAGE 4.2 training — absolute root-regression head
#  Trains on all 150 takes / 128,104 frames. Data volume dominates: 104 mm -> 9 mm for 2.4x data.
#    reads  : $B/mb/data_h4d_offtri
#    writes : $B/mb/ckpt/rh_h4d_v04_w0_full.pt
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES="$GPU" $PY "$B/monocular/localization/roothead_h4d_train.py" \
    --out "$B/mb/ckpt/rh_h4d_v04_w0_full.pt"
