#!/bin/bash
# ============================================================================
#  17_eval_heldout.sh  —  STAGE 5.1 evaluation — held-out vs OFFICIAL GT
#
#  ★★ THIS IS *NOT* THE SCRIPT THE PAPER'S MAIN TABLE COMES FROM. ★★
#
#     The paper's table — A 58.1 / B-69 65.6 / B 42.4 / C 116.9 — is produced
#     by 18_eval_abc.sh  ->  scripts/tools/ah1_abc.py.
#
#     The two scripts aggregate differently, and you WILL get a different
#     number from the same checkpoint:
#
#        z25_gtEval.py : median over all (take,view,pid) medians  ->  81.3
#        ah1_abc.py    : median over the per-take medians         ->  42.4
#
#     Same weights, same ground truth, same joint set and metric — only the
#     pooling differs. Quote the ah1_abc.py number. Use this script only when
#     you want a per-(take,view) breakdown.
#
#     (Verified 2026-10-05: ah1_abc.py reproduces the paper table exactly —
#      A 58.1/26.6, B-69 65.6/38.2, B 42.4/32.6, C_v2 116.9/91.6.)
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"

CK=${1:-$B/mb/ckpt/Combat3D_FULL.pt}
$PY "$B/scripts/tools/z25_gtEval.py" "$CK"
