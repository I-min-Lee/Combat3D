#!/bin/bash
# ============================================================================
#  04_pose2d.sh  —  STAGE 1.4 label production — 2D pose (ViTPose)
#  Runs ViTPose on the boxes. NEVER run this concurrently with detection — two jobs on one GPU fail silently.
#    reads  : $B/det_*_<take>
#    writes : $B/vp_*_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 04_pose2d.sh <take> <int-seq-id> <last-frame> [det-dir-name]}
ID=${2:?need integer seq id}
NF=${3:?need last frame index}
DET=${4:-det_self2}

VP_ROOT="$B" VP_MATCH="$ID" VP_SEG=1 \
$PY "$B/code/adapters/harmony4d/pipeline/vp_h4d.py" \
    --det "$B/${DET}_$TAKE" --out "$B/vp_${DET#det_}_$TAKE" \
    --views 01 03 04 07 09 14 --start 1 --end "$NF"
echo "-> $B/vp_${DET#det_}_$TAKE"
