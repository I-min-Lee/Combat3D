#!/bin/bash
# ============================================================================
#  25_figs.sh  —  STAGE 8.2 figures — pipeline / triangulation ablation / filtering negative
#  Regenerates fig7, fig8, fig9 straight into the figs directory.
#    reads  : nothing (hard-coded numbers from the ablation logs)
#    writes : figs/fig7..fig9
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
$PY "$B/scripts/tools/bc2_figs2.py"
