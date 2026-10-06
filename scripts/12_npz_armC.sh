#!/bin/bash
# ============================================================================
#  12_npz_armC.sh  —  STAGE 3.3 training data — ARM C (fully self-built, EXPLORATORY)
#  2D from our asm_self2, 3D from our em_self2. Not a fair baseline — see docs/REPRODUCE.md §6.2.
#    reads  : $B/asm_self2_<take>, $B/em_self2_<take>
#    writes : $B/mb/data_h4d_selfnpz
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
$PY "$B/scripts/tools/ae1_mknpz.py"
echo "-> $B/mb/data_h4d_selfnpz   (expect 1476 npz / 123 takes after the leak purge)"
