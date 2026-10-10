#!/bin/bash
# ★ 单组实验：同一份【官方 2D】+ 两种三角化（我们 vs 朴素 AB_NO_ENUM=1）→ 比 qa
B=/workshop/Lym/combat3d
PY=$B/envs/miniconda3/envs/pose312/bin/python
V=01,03,04,07,09,14
LOG=$B/logs/naive_$(date +%m%d_%H%M).log
mkdir -p $B/logs
cd $B
{
echo "########## 朴素三角化（官方2D）$(date) ##########"
# 取 5 条 test take（覆盖 sword/mma），与我们的 em_off_* 一一对应
for T in 001_sword3 002_sword3 002_mma5 010_sword2 016_mma5; do
  [ -d $B/em_off_$T/lam1.0/pid0 ] || { echo "  $T 无我们的产物，跳过"; continue; }
  [ -d $B/em_naive_$T/lam1.0/pid0 ] && { echo "  $T 已有朴素产物"; continue; }
  CAL=$B/calib_gt_$T
  NF=$($PY -c "
import os,glob
d='$B/asm_off_$T/annots/01'
print(max(int(os.path.basename(p)[:-5]) for p in glob.glob(d+'/*.json')))" 2>/dev/null)
  [ -z "$NF" ] && { echo "  $T 无官方2D"; continue; }
  echo "=== $T NF=$NF"
  AB_CAL=$CAL AB_VIEWS=$V AB_ANNOTS=$B/asm_off_$T/annots \
  AB_W_IMG=3840 AB_H_IMG=2160 AB_DET_DIR=$B/det_gt2_$T \
  AB_NO_ENUM=1 AB_NO_BORDER=1 \
  $PY -W ignore code/label_pipeline/adapters/harmony4d/pipeline/tri_ablate.py \
     --out $B/em_naive_$T --start 1 --end $((NF+1)) --min-conf 0.3 --lams 1.0 --order 2 2>&1 | tail -2
  echo "   -> $(ls $B/em_naive_$T/lam1.0/pid0/keypoints3d 2>/dev/null | wc -l) 帧"
done
echo "########## 完成 $(date) ##########"
} >> $LOG 2>&1
