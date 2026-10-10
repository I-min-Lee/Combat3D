#!/bin/bash
# ============================================================================
#  05_assemble.sh  —  STAGE 1.5 label production — assemble
#  The parameter is --raw, NOT --vp/--det. The error message does not tell you this.
#    reads  : $B/vp_*_<take>
#    writes : $B/asm_*_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 05_assemble.sh <take> <last-frame> [2d-tag]}
NF=${2:?need last frame index}
TAG=${3:-self2}

$PY "$B/code/label_pipeline/adapters/harmony4d/pipeline/assemble_h4d.py" \
    --raw "$B/vp_${TAG}_$TAKE" --out "$B/asm_${TAG}_$TAKE" \
    --views "$V" --start 1 --end "$NF"
echo "-> $B/asm_${TAG}_$TAKE"
