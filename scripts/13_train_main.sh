#!/bin/bash
# ============================================================================
#  13_train_main.sh  —  STAGE 4.1 training — ARM B, the main result
#  KB_FORCE_R_I=1 is MANDATORY (camera-frame target). min_delta=1.0 means <1mm gains are not saved.
#    reads  : $B/mb/data_h4d_offtri
#    writes : $B/mb/ckpt/Combat3D_FULL.pt
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
GPU=${GPU:-0}
cd "$B/mb/kb"
KB_FORCE_R_I=1 CUDA_VISIBLE_DEVICES="$GPU" $PY -u -W ignore kb_train.py \
  --data "$B/mb/data_h4d_offtri" --mode full --init official --no-quality \
  --val-takes train01_hugging --clip-len 121 --stride 40 \
  --epochs 300 --clips-per-epoch 1500 --batch 4 --eval-every 10 \
  --patience 12 --min-delta 1.0 --lr 2e-5 \
  --out "$B/mb/ckpt/Combat3D_FULL.pt"
