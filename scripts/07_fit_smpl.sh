#!/bin/bash
# ============================================================================
#  07_fit_smpl.sh  —  STAGE 1.7 label production — SMPL fitting (OPTIONAL)
#  Only for mesh output / PVE-style comparisons. The reported MPJPE/PA numbers do NOT need this step.
#    reads  : $B/asm_*_<take>, $B/em_*_<take>, $B/calib_gt_<take>, $B/frames/<ID>/4
#    writes : $B/emfit_h4d
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 07_fit_smpl.sh <take> <int-seq-id> <last-frame> <pid> [tag]}
ID=${2:?need integer seq id}
NF=${3:?need last frame index}
PID=${4:?need person id (0 or 1)}
TAG=${5:-self2}
export FIT_K=${FIT_K:-0.024}

$PY "$B/code/adapters/harmony4d/pipeline/fit_h4d.py" "$PID" \
    --start 1 --end "$((NF+1))" \
    --annots "$B/asm_${TAG}_$TAKE/annots" --k3d "$B/em_${TAG}_$TAKE/lam1.0" \
    --calib "$B/calib_gt_$TAKE" --frames "$B/frames/$ID/4" --out "$B/emfit_h4d"
echo "-> $B/emfit_h4d"
