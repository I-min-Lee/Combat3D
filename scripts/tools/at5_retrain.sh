#!/bin/bash
# ★ 三臂统一开质量筛选重训（同一 qa_thr=10px，公平）
B=/workshop/Lym/combat3d
PY=$B/envs/miniconda3/envs/pose312/bin/python
cd $B/mb/kb
run() {  # run <数据> <qa目录> <GPU> <输出> <日志>
  CUDA_VISIBLE_DEVICES=$3 nohup $PY -u -W ignore kb_train.py \
     --data $B/mb/$1 --qa-dir $B/mb/$2 --qa-thr 10.0 --qa-min-n 4 \
     --mode full --init official --qa-dir $B/mb/$2 \
     --val-takes train01_hugging --clip-len 121 --stride 40 \
     --epochs 300 --clips-per-epoch 1500 --batch 4 --eval-every 10 \
     --patience 12 --min-delta 1.0 --lr 2e-5 --out $B/mb/ckpt/$4.pt \
     > $B/logs/$5 2>&1 &
  echo "  $4 -> GPU$3 PID $!"
}
echo "=== 启动（三臂并行）==="
run data_h4d_gtalign qa_data_h4d_gtalign 2 Combat3D_GTlabel_qa  train_gtlabel_qa.log
run data_h4d_offtri  qa_data_h4d_offtri  3 Combat3D_OurLabel_qa train_ourlabel_qa.log
run data_h4d_selfv2  qa_data_h4d_selfv2  4 Combat3D_SelfLabel_qa train_selflabel_qa.log
sleep 200
echo
for f in train_gtlabel_qa train_ourlabel_qa train_selflabel_qa; do
  echo "--- $f"; grep -vE "sitecustomize|FutureWarning|hasattr|calib v" $B/logs/$f.log 2>/dev/null | grep -E "qa\]|训练 clip|训练帧|ep [0-9]+/|Error|Traceback" | tail -5
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
