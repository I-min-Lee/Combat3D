#!/bin/bash
# ============================================================================
#  21_to_final13.sh  —  STAGE 7.1 visualization / single-view deployment
#  Runs the fine-tuned lifter on ONE view and recovers world placement
#  (root-regression head, or the geometric bone-length/ground fallback).
#    reads  : 2D annots (this view), calib, ckpt, [root head]
#    writes : $B/final13_mono_<take>
#  Underlying script: monocular/lifter/mb_to_final13.py
#  Full explanation: docs/REPRODUCE.md §7
#
#  ★ This is the ONLY place in the released pipeline where the prediction path
#    is monocular. Legal inputs are exactly: this view's 2D + camera
#    intrinsics/extrinsics + model weights. No triangulation.
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"

usage() {
  cat <<EOF
usage: 21_to_final13.sh <take> <tag> <view> [--annots DIR] [opts...]

  take   sequence name, e.g. 016_mma4
  tag    short tag used in the annots path, e.g. 016mma4
  view   the single camera view to use, e.g. 04

common overrides (env):
  CK=<ckpt>            default \$B/mb/ckpt/Combat3D_FULL.pt
  ROOTHEAD=<ckpt>      default \$B/mb/ckpt/rh_h4d_v04_w0_full.pt  ('' to disable)
  ANNOTS=<dir>         default \$B/easymocap_<tag>/${take//./_}/annots
  CALIB=<dir>          default \$B/calib_<tag>

If your 2D comes from raw ViTPose rather than an EasyMocap-style annots tree,
pass --src vitpose and set --fps-src (see the script's --help).
EOF
  exit 1
}

TAKE=${1:-}; TAG=${2:-}; VIEW=${3:-}
[ -z "$TAKE" ] || [ -z "$TAG" ] || [ -z "$VIEW" ] && usage
shift 3; EXTRA=("$@")

CK=${CK:-$B/mb/ckpt/Combat3D_FULL.pt}
ROOTHEAD=${ROOTHEAD-$B/mb/ckpt/rh_h4d_v04_w0_full.pt}
ANNOTS=${ANNOTS:-$B/easymocap_${TAG}/${TAKE//./_}/annots}
CALIB=${CALIB:-$B/calib_${TAG}}

RH_ARGS=()
[ -n "$ROOTHEAD" ] && RH_ARGS=(--root-head "$ROOTHEAD")

$PY "$B/monocular/lifter/mb_to_final13.py" \
    --take "$TAKE" --tag "$TAG" --view "$VIEW" --src annots \
    --annots "$ANNOTS" --calib "$CALIB" --ckpt "$CK" \
    "${RH_ARGS[@]}" \
    --out "$B/final13_mono_$TAKE" --stride 8 \
    "${EXTRA[@]}"

# Notes
#  * MotionBERT lives in a 25 fps domain (the original data was subsampled by
#    stride 8), so feed 2D at stride 8 and expect a 25 fps output.
#  * The root head ckpt carries feature-normalisation mu/fstd inside it; if you
#    copy the head without them it is useless.
#  * --despike / --sg / --pix-sg are zero-phase smoothing (despike, then savgol).
#    DO NOT substitute a median filter: on continuous trajectories it creates
#    plateau-then-jump artefacts that read as stuttering.
#  * The kendo-domain variant of this script takes extra --src vitpose /
#    --fps-src args; see docs/SECOND_DOMAIN.md.
