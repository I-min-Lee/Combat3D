#!/bin/bash
# mb_autostop.sh (v2) —— 收敛自动停机看门狗
#
# ★ v1 的 bug：用"连续轮询次数"计数，两次轮询之间可能没有新评测点，
#   同一个评测点被数 3 次就误判收敛、把训练杀了。v2 改为**只在出现新评测点时计数**。
#
# 判据：连续 STOP_AFTER 个**新评测点**(每 5 epoch 一个)相比历史最优改善 < MIN_DELTA 毫米 -> 停
# 停机后**自动补跑测试集评测**（训练被 kill 就不会走到 kb_train.py 末尾的评测）
#
# 用法（服务器上）:
#   LOG=... CKPT=... setsid nohup bash mb_autostop.sh > /root/autodl-tmp/mb/_autostop.log 2>&1 < /dev/null &
set -u

LOG=${LOG:-/root/autodl-tmp/mb/_train_v2.log}
CKPT=${CKPT:-/root/autodl-tmp/mb/ckpt/kb_full_v2.pt}
STOP_AFTER=${STOP_AFTER:-3}      # 连续 N 个【新评测点】三项指标都停滞就停（3 x 5ep = 15 epoch）
MIN_DELTA=${MIN_DELTA:-1.0}      # val_median_mm 改善阈值 mm
D_MED=${D_MED:-1.0}              # 各自阈值
D_CLN=${D_CLN:-0.5}
D_OCC=${D_OCC:-1.0}
POLL=${POLL:-90}                 # 轮询间隔秒
RUN_TEST_EVAL=${RUN_TEST_EVAL:-1}
PY=/root/autodl-tmp/envs/mb/bin/python
KB=/root/autodl-tmp/mb/kb

echo "[$(date +%H:%M:%S)] 看门狗 v2 启动  LOG=$LOG"
echo "  判据: 连续 $STOP_AFTER 个【新评测点】改善<$MIN_DELTA mm；轮询 ${POLL}s"

# ★ v3 起：不能只看中位。实测中位先饱和（55.4→55.8 微升）而 clean/遮挡还在降，
#   只看中位会在难样本还在改善时误判收敛。现在要求**三个指标同时停滞**才算收敛。
best_med=99999; best_cln=99999; best_occ=99999
bad=0
last_neval=-1
echo "  监控: val_median_mm(阈${D_MED}) + clean_mm(阈${D_CLN}) + occ_mm(阈${D_OCC})，三者皆停滞才计数"
while true; do
  sleep $POLL

  PID=$(pgrep -f "kb_trai[n].py" | head -1)
  if [ -z "$PID" ]; then
    echo "[$(date +%H:%M:%S)] 训练进程已不在（自行结束或被停）—— 看门狗退出"
    exit 0
  fi

  # ★ 用 awk 数，避免 `grep -c` 无匹配时输出 "0" 又叠加 `|| echo 0` 变成 "0\n0"
  NEVAL=$(awk '/eval:/{n++} END{print n+0}' "$LOG" 2>/dev/null)
  NEVAL=${NEVAL:-0}
  if [ "$NEVAL" -le "$last_neval" ]; then
    continue                       # ★ 关键修正：没有新评测点就不计数
  fi
  last_neval=$NEVAL

  LAST=$(grep -a "eval:" "$LOG" 2>/dev/null | tail -1)
  CUR=$(echo "$LAST" | sed -n 's/.*"val_median_mm": *\([0-9.]*\).*/\1/p')
  CLN=$(echo "$LAST" | sed -n 's/.*"clean_mm": *\([0-9.]*\).*/\1/p')
  OCC=$(echo "$LAST" | sed -n 's/.*"occ_mm": *\([0-9.]*\).*/\1/p')
  [ -n "$CLN" ] || CLN=99999
  [ -n "$OCC" ] || OCC=99999
  case "$CUR" in ''|*[!0-9.]*) continue ;; esac

  IMP_M=$(awk -v a="$best_med" -v b="$CUR" 'BEGIN{printf "%.3f", a-b}')
  IMP_C=$(awk -v a="$best_cln" -v b="$CLN" 'BEGIN{printf "%.3f", a-b}')
  IMP_O=$(awk -v a="$best_occ" -v b="$OCC" 'BEGIN{printf "%.3f", a-b}')
  GOT_M=0; GOT_C=0; GOT_O=0
  awk -v i="$IMP_M" -v d="$D_MED" 'BEGIN{exit !(i>d)}' && GOT_M=1
  awk -v i="$IMP_C" -v d="$D_CLN" 'BEGIN{exit !(i>d)}' && GOT_C=1
  awk -v i="$IMP_O" -v d="$D_OCC" 'BEGIN{exit !(i>d)}' && GOT_O=1

  # 任一指标仍在改善 -> 不算停滞
  if [ $((GOT_M + GOT_C + GOT_O)) -gt 0 ]; then
    echo "[$(date +%H:%M:%S)] 仍在改善 (第 ${NEVAL} 个评测点): 中位 ${CUR}(Δ${IMP_M}) clean ${CLN}(Δ${IMP_C}) 遮挡 ${OCC}(Δ${IMP_O})  重置计数"
    [ "$GOT_M" = 1 ] && best_med=$CUR
    [ "$GOT_C" = 1 ] && best_cln=$CLN
    [ "$GOT_O" = 1 ] && best_occ=$OCC
    bad=0
  else
    bad=$((bad+1))
    echo "[$(date +%H:%M:%S)] 三项皆停滞 ${bad}/${STOP_AFTER}  中位=${CUR}(best ${best_med})  clean=${CLN}(best ${best_cln})  遮挡=${OCC}(best ${best_occ})  (第 ${NEVAL} 个评测点)"
    if [ $bad -ge $STOP_AFTER ]; then
      echo "[$(date +%H:%M:%S)] == 判定收敛（连续 ${STOP_AFTER} 个新评测点里 中位/clean/遮挡 三项全部停滞），停训练 pid=$PID =="
      kill -TERM $PID 2>/dev/null
      for i in $(seq 1 12); do
        sleep 10
        pgrep -f "kb_trai[n].py" >/dev/null || break
        kill -TERM $(pgrep -f "kb_trai[n].py" | head -1) 2>/dev/null
      done
      if pgrep -f "kb_trai[n].py" >/dev/null; then
        echo "[$(date +%H:%M:%S)] !! 训练没停掉，看门狗退出（请人工处理）"
        exit 1
      fi
      echo "[$(date +%H:%M:%S)] 训练已停。"
      if [ "$RUN_TEST_EVAL" = "1" ]; then
        echo "[$(date +%H:%M:%S)] 补跑 验证集+测试集 评测 ..."
        cd $KB
        # ★ 参数是 --sets（旧版叫 --also-val，改过名，别再写错）
        $PY kb_test_eval.py --ckpt "$CKPT" --sets both \
             --out "${CKPT%.pt}_FINAL_EVAL.json" 2>&1 | tail -30
        rc=$?
        if [ $rc -ne 0 ] || [ ! -f "${CKPT%.pt}_FINAL_EVAL.json" ]; then
          echo "[$(date +%H:%M:%S)] !! 评测失败(rc=$rc)，重试一次（去掉 --sets 用默认 test）"
          $PY kb_test_eval.py --ckpt "$CKPT" --sets test \
               --out "${CKPT%.pt}_FINAL_EVAL_testonly.json" 2>&1 | tail -30
        fi
      fi
      echo "[$(date +%H:%M:%S)] 看门狗任务完成，退出"
      exit 0
    fi
  fi
done
