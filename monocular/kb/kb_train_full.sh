#!/bin/bash
# 全量微调 MotionBERT（本轮：不用 LoRA，按质量报告筛选+加权高质量标签）
# 吞吐实测 3286 帧/s (batch=8, clip=243, clip 步长81)
#   一个 pass(全库 2.17M 帧 / 步长81 => 2.68万 clip x 243) ≈ 33 分钟
#   60 epoch x 2680 clip ≈ 6 个 pass ≈ 3.3 小时
B=/root/autodl-tmp
P=$B/envs/mb/bin/python
cd $B/mb/kb || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$P kb_train.py \
  --data $B/mb/data \
  --mode full \
  --epochs 60 \
  --clips-per-epoch 2680 \
  --batch 8 \
  --lr 5e-5 \
  --stride 81 \
  --clip-len 243 \
  --bone-w 0.1 \
  --val-frac 0.12 \
  --q-cv 0.18 \
  --q-problems 3 \
  --eval-every 10 \
  --out $B/mb/ckpt/kb_full_v1.pt \
  > $B/mb/_train_full.log 2>&1
echo "EXIT=$?" >> $B/mb/_train_full.log
