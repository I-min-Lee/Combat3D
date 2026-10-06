#!/bin/bash
# ============================================================================
#  09_gt_align.sh  —  STAGE 2.2 ground truth — align official GT into our world frame
#  Per-take Umeyama, INVERSE direction. Getting it backwards turns 746 px into 3357 px.
#    reads  : $B/gt_<take>, $B/em_off_<take>
#    writes : $B/gt_<take>_aligned
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
$PY "$B/scripts/tools/ai2_fixalign.py"
echo "-> $B/gt_<take>_aligned/   (verify: re-projection should be ~16.5 px)"
