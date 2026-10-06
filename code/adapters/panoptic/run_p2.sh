#!/bin/bash
# run_p2.sh — 把 CMU Panoptic 的一场跑成「真值框版」与「全自建版」两条结果
#
# 用法: bash run_p2.sh <阶段>
#   calib | frames | gt | gtbox | selfbox | vpgt | vpself | trigt | triself | metrics | all
#
# 约定：帧号用**绝对帧号**（与 body3DScene_%08d.json 一一对应），全链不做重映射。
set -e
B=/workshop/Lym/combat3d
PY=$B/envs/miniconda3/envs/pose312/bin/python
export COMBAT3D_ROOT=$B EMCORE=$B/port/emcore PYTHONPATH=$B/port/emcore TMPDIR=$B/tmp
export EM_MASTER=$B/port/EasyMocap-master EM_PY=$PY VP_ROOT=$B
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

SEQ=${SEQ:-160906_ian2}
TAG=${TAG:-ian2}
NS=${NS:-4256}                 # 起始绝对帧号
NF=${NF:-600}                  # 帧数
END=$((NS + NF - 1))
V=${V:-00_01,00_06,00_13,00_16,00_17}
VS=$(echo "$V" | tr ',' ' ')
DATA=$B/data/panoptic
POSEDIR=$DATA/$SEQ/hdPose3d_stage1_coco19
CAL=$B/calib_p2_$TAG
FR=$B/frames_p2_$TAG
GT=$B/gt_p2_$TAG
STEP=${1:-all}

log(){ echo "[$(date -u '+%F %T')] $*" | tee -a $B/logs/ops.log; }
run(){ case "$STEP" in all|"$1") return 0;; *) return 1;; esac; }

# ---------- ① 标定 ----------
if run calib; then
  echo "=== ① 标定 Panoptic -> intri/extri ==="
  $PY -W ignore $B/code/adapters/panoptic/p2_calib.py \
     --calib $DATA/calibration_$SEQ.json --out $CAL --views $V
fi

# ---------- ② 抽帧 ----------
if run frames; then
  echo "=== ② 抽帧（绝对帧号 $NS..$END）==="
  $PY -W ignore $B/code/adapters/panoptic/p2_frames.py \
     --videos $DATA/videos --out $FR --views $V --start $NS --n $NF --jobs 5
fi

# ---------- ③ GT 3D（COCO19 -> COCO17）----------
if run gt; then
  echo "=== ③ GT 3D ==="
  $PY -W ignore $B/code/adapters/panoptic/p2_gt.py \
     --pose-dir $POSEDIR --out $GT --start $NS --n $NF
fi

# ---------- ④ 真值框（真值框版前端）----------
if run gtbox; then
  echo "=== ④ 真值框 ==="
  $PY -W ignore $B/code/adapters/panoptic/p2_boxes.py \
     --pose-dir $POSEDIR --calib-dir $CAL --views $V --start $NS --n $NF \
     --out $B/det_p2gt_$TAG --bbox-out $B/p2seq_$TAG/processed_data/bbox --tag $TAG
fi

# ---------- ⑤ 全自建前端 ----------
if run selfbox; then
  echo "=== ⑤ 全自建前端（检测+选人+gid稳定+跨视角对齐）==="
  $PY -W ignore $B/code/adapters/harmony4d/pipeline/det_self_final.py \
     --frames-root $FR --out $B/det_p2self_$TAG --views $V --tag $TAG \
     --start $NS --end $END --border-margin 0.15 --calib $CAL \
     --W 1920 --H 1080 2>&1 | tail -12
fi

# ---------- ⑥⑦ ViTPose + 装配 ----------
for fe in gt self; do
  if [ "$STEP" = all ] || [ "$STEP" = vp$fe ]; then
    echo "=== ⑥⑦ ViTPose + 装配 [$fe] ==="
    export VP_MATCH=0 VP_SEG=0 VP_FRAMES_DIR=$FR
    $PY -W ignore $B/code/adapters/harmony4d/pipeline/vp_h4d.py \
       --det $B/det_p2${fe}_$TAG --out $B/vp_p2${fe}_$TAG --views $VS --start $NS --end $END
    $PY -W ignore $B/code/adapters/harmony4d/pipeline/assemble_h4d.py \
       --raw $B/vp_p2${fe}_$TAG --out $B/asm_p2${fe}_$TAG --views $V \
       --start $NS --end $END --H 1080 --W 1920
  fi
done

# ---------- ⑧ 三角化 ----------
for fe in gt self; do
  if [ "$STEP" = all ] || [ "$STEP" = tri$fe ]; then
    echo "=== ⑧ 三角化 [$fe] ==="
    export AB_CAL=$CAL AB_VIEWS=$V AB_ANNOTS=$B/asm_p2${fe}_$TAG/annots \
           AB_W_IMG=1920 AB_H_IMG=1080 AB_DET_DIR=$B/det_p2${fe}_$TAG
    $PY -W ignore $B/code/adapters/harmony4d/pipeline/tri_h4d.py \
       --out $B/em_p2${fe}_$TAG --start $NS --end $((END + 1)) \
       --min-conf 0.3 --lams 1.0 --order 2
  fi
done

# ---------- ⑨ 评估 ----------
if [ "$STEP" = all ] || [ "$STEP" = metrics ]; then
  echo "=== ⑨ 评估 ==="
  $PY -W ignore $B/code/eval/det_vs_gt.py --det $B/det_p2self_$TAG \
     --seq-root $B/p2seq_$TAG --views $V --start $NS --end $((END + 1)) \
     --match center --unexpand 2>&1 | grep -E '整体|2/2 =' || true
  for pa in none seq frame; do
    for fe in gt self; do
      printf 'pa=%-5s %-5s ' $pa $fe
      $PY -W ignore $B/code/eval/h4d_metrics.py --calib $CAL \
         --annots $B/asm_p2${fe}_$TAG/annots --gt $GT/gt3d_colmap \
         --views $V --n 0 --ref x --match-perm --pa $pa \
         --runs "$fe:$B/em_p2${fe}_$TAG/lam1.0" 2>&1 | grep -E "^$fe" || echo "(失败)"
    done
  done
fi
log "run_p2 完成 step=$STEP tag=$TAG"
