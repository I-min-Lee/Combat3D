#!/bin/bash
# ============================================================================
#  22_viz_overlay.sh  —  STAGE 7.2 visualization — re-project 3D nodes onto the original video
#  Fisheye projection, nodes only (no mesh). Container has NO ffmpeg -> mp4v only.
#    reads  : $B/final13_mono_<take>, raw video
#    writes : $B/nodes_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 22_viz_overlay.sh <take> <int-seq-id> [view] [match-frame]}
ID=${2:?need integer seq id}
VIEW=${3:-04}
MATCH=${4:-$ID}

$PY "$B/scripts/tools/v26_node_overlay.py" --take "$TAKE" --view "$VIEW" \
    --match "$MATCH" --f13 "$B/final13_mono_$TAKE" --out "$B/nodes_$TAKE"
