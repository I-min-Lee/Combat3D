#!/bin/bash
# ============================================================================
#  23_render3d.sh  —  STAGE 7.3 visualization — 3D node render in world space
#  --bounds/--floor take numbers in the DATASET's units. H4D is METRES; mm collapses the figure.
#    reads  : $B/final13_mono_<take>
#    writes : $B/render3d_mono_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 23_render3d.sh <take> <n-frames> [start]}
N=${2:?need frame count}
START=${3:-0}

$PY "$B/code/label_pipeline/stages/8_render/render3d_5000f13.py" \
    --smpl "$B/final13_mono_$TAKE" --out "$B/render3d_mono_$TAKE" \
    --start "$START" --end "$N" --fps 20 \
    --floor=-1.187,1.643,-1.520,1.100
