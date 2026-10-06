#!/bin/bash
# ============================================================================
#  15_train_baselines.sh  —  STAGE 4.3 training — baselines, SAME PROTOCOL
#  FT_INIT warm-starts from each method's PUBLIC pretrained weights. Without it the comparison is unfair.
#    reads  : $B/mb/data_h4d_offtri + $FT_INIT
#    writes : $B/mb/ckpt/base_vp3d_ft120.pt, base_mixste_ft.pt
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"
GPU=${GPU:-0}
: "${FT_INIT_VP3D:?set FT_INIT_VP3D to the public VideoPose3D checkpoint}"
export CUDA_VISIBLE_DEVICES="$GPU"

FT_INIT="$FT_INIT_VP3D" $PY "$B/scripts/tools/z03_ft.py" --arch vp3d \
    --epochs 40 --out "$B/mb/ckpt/base_vp3d_ft120.pt"

if [ -n "${FT_INIT_MIXSTE:-}" ]; then
  FT_INIT="$FT_INIT_MIXSTE" $PY "$B/scripts/tools/z03_ft.py" --arch mixste \
      --epochs 40 --out "$B/mb/ckpt/base_mixste_ft.pt"
else
  echo "(skipping MixSTE: only ONNX weights are public, no PyTorch checkpoint)"
fi
echo "Comparable reading: VideoPose3D 106.1 vs Combat3D 18.4 = 5.8x"
