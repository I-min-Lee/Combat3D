#!/bin/bash
# ============================================================================
#  18_eval_abc.sh  —  STAGE 5.2 evaluation — ★ THE PAPER'S MAIN TABLE ★
#
#  This is the authoritative evaluator. Verified 2026-10-05 on the released
#  checkpoints — it reproduces the paper table exactly:
#
#     model             subset            n    MPJPE       PA
#     A_officialGT      9 held-out        9     58.1     26.6
#     B69_ourLabel      9 held-out        9     65.6     38.2
#     B_ourLabel        9 held-out        9     42.4     32.6      <- main result
#     C_selfBuilt_v2    9 held-out        9    116.9     91.6
#
#  Aggregation: per-take median first, then median across takes. Compare with
#  17_eval_heldout.sh, which pools all (take,view,pid) into one median and
#  therefore reports a much larger MPJPE (81.3) for the same checkpoint.
#  Quote THIS one.
#
#  ★ Cross-arm WARNING: kb_train.evaluate() reads validation GT from each arm's
#    OWN npz, so cross-arm *validation* numbers are not comparable. This script
#    fixes the ground-truth source across arms, which is why it is the one to
#    use.
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"

$PY "$B/scripts/tools/ah1_abc.py"
