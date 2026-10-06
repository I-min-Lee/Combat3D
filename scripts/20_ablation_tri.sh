#!/bin/bash
# ============================================================================
#  20_ablation_tri.sh  —  STAGE 6.1 ablation — triangulation layer scope
#  Ours vs naive all-views-equal on the SAME 2D. Identical on clean input; -44% on dirty input.
#    reads  : $B/asm_off_<take>, $B/calib_gt_<take>
#    writes : $B/em_naive_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
bash "$B/scripts/tools/aw2_naive.sh"
echo
echo "REMINDER: AB_NO_ENUM=1 shows NO difference on clean upstream input."
echo "          To verify the switch took effect you must test on DIRTY upstream."
