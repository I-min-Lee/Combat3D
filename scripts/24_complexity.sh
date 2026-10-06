#!/bin/bash
# ============================================================================
#  24_complexity.sh  —  STAGE 8.1 figures — scene complexity table
#  Motion speed / out-of-frame rate / inter-athlete hip distance, Harmony4D vs Panoptic.
#    reads  : npz data dirs
#    writes : stdout
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
$PY "$B/scripts/tools/z36_complexity.py"
# Key comparison: true frame-to-frame displacement 13.8 mm/frame (H4D) vs 1.4 (Panoptic) = 10x
