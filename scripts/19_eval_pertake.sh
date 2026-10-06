#!/bin/bash
# ============================================================================
#  19_eval_pertake.sh  —  STAGE 5.3 evaluation — per-take breakdown
#  Per take: PA shape residual / post-alignment end-to-end / calibrated R.
#    reads  : ckpt, npz data
#    writes : stdout
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
CK=${1:-$B/mb/ckpt/Combat3D_FULL.pt}
DATA=${2:-$B/mb/data_h4d_offtri}
$PY "$B/scripts/tools/eval_pertake.py" "$CK" "$DATA" --n "${N:-12}"
