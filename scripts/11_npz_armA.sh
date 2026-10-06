#!/bin/bash
# ============================================================================
#  11_npz_armA.sh  —  STAGE 3.2 training data — ARM A (official GT labels, controlled)
#  Replaces ONLY k3d in the arm-B npz; k2d/valid/meta stay byte-identical -> clean control.
#    reads  : $B/mb/data_h4d_offtri, $B/gt_<take>_aligned
#    writes : $B/mb/data_h4d_gtalign
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
$PY "$B/scripts/tools/z31_mkab.py"
echo "-> $B/mb/data_h4d_gtalign   (expect 828 npz / 69 takes)"
