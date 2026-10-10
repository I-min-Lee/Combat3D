#!/bin/bash
# ============================================================================
#  02_frames.sh  —  STAGE 1.2 label production — frame chain
#  Builds the frame index the rest of the pipeline addresses. --match must be an INTEGER.
#    reads  : $RAW/<take>
#    writes : $B/frames/<ID>/1
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 02_frames.sh <take> <int-seq-id>   e.g. 016_mma4 16}
ID=${2:?need the integer sequence id (numeric prefix of the take name, no suffix)}
RAW=${RAW:-$B/data/harmony4d/raw}

$PY "$B/code/label_pipeline/adapters/harmony4d/h4d_frames.py" \
    --seq-root "$RAW/$TAKE" --frames-root "$B/frames" \
    --match "$ID" --seg 1 --views "$V"
echo "-> $B/frames/$ID/1"
