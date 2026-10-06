#!/bin/bash
# ============================================================================
#  06_triangulate.sh  —  STAGE 1.6 label production — triangulation (UNIVERSAL layer)
#  Robust multi-view triangulation. Label-production / offline-GT only — never in an inference path.
#    reads  : $B/asm_*_<take>, $B/calib_gt_<take>, $B/det_*_<take>
#    writes : $B/em_*_<take>
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
TAKE=${1:?usage: 06_triangulate.sh <take> <last-frame> [tag] [det-dir]}
NF=${2:?need last frame index}
TAG=${3:-self2}
DET=${4:-det_self2}
W=${W:-3840}; H=${H:-2160}

AB_CAL="$B/calib_gt_$TAKE" AB_VIEWS="$V" AB_ANNOTS="$B/asm_${TAG}_$TAKE/annots" \
AB_W_IMG="$W" AB_H_IMG="$H" AB_DET_DIR="$B/${DET}_$TAKE" \
$PY "$B/code/adapters/harmony4d/pipeline/tri_h4d.py" \
    --out "$B/em_${TAG}_$TAKE" --start 1 --end "$((NF+1))" \
    --min-conf 0.3 --lams 1.0 --order 2
echo "-> $B/em_${TAG}_$TAKE"
