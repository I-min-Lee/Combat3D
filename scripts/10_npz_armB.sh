#!/bin/bash
# ============================================================================
#  10_npz_armB.sh  —  STAGE 3.1 training data — ARM B (our triangulated labels)
#  2D from official asm_off, 3D label from our em_off. 1800 npz / 150 takes -> data_h4d_offtri.
#    reads  : $B/asm_off_<take>, $B/em_off_<take>, $B/gt_<take>
#    writes : $B/mb/data_h4d_offtri
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 10_npz_armB.sh <take>   (run once per take; -out is cumulative)}
$PY "$B/monocular/mb_npz.py" --tag "$TAKE" --views "$V" \
    --root "$B" --out "$B/mb/data_h4d_offtri"
echo "-> $B/mb/data_h4d_offtri"
