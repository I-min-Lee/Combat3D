#!/bin/bash
# ============================================================================
#  16_ablation_qa.sh  —  STAGE 4.4 ablation — frame-level quality filtering
#  Retrains all three arms with a uniform 10 px filter. All three get WORSE. See finding F3.
#    reads  : qa_data_h4d_*
#    writes : $B/mb/ckpt/Combat3D_*_qa.pt
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
bash "$B/scripts/tools/at5_retrain.sh"
# NOTE: if load_quality() raises FileNotFoundError, it is looking for kendo's
#       _segqa_summary.txt. Workaround:  touch $B/_segqa_summary.txt
