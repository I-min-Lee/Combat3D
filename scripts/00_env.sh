#!/bin/bash
# ============================================================================
#  00_env.sh  —  environment check + checkpoint placement
#  Verifies torch/cuDNN/mmcv versions and that weights are in place. Run this FIRST.
#    reads  : weights/ckpt/
#    writes : nothing (prints a report)
#  Underlying script: see the $PY line below.
#  Full explanation: docs/REPRODUCE.md
# ============================================================================
set -euo pipefail
B="${COMBAT3D_ROOT:-/workshop/Lym/combat3d}"
PY="$B/envs/miniconda3/envs/pose312/bin/python"
V="${V:-01,03,04,07,09,14}"

echo "=== Combat3D environment check ==="
echo "COMBAT3D_ROOT = $B"
$PY - <<'PY'
import sys
print("python      ", sys.version.split()[0])
try:
    import torch
    print("torch       ", torch.__version__, " cuda:", torch.cuda.is_available())
    import torch.version
    print("torch cuda  ", torch.version.cuda)
except Exception as e:
    print("torch       !! ", e)
try:
    import torch.backends.cudnn as c
    print("cudnn       ", c.version())
    if not str(c.version()).startswith(("9.1", "9.0", "8.")):
        print("           !! cuDNN does not match a cu121 build -> every conv layer will fail.")
        print("           !! fix: pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.1.0.70")
except Exception as e:
    print("cudnn       !! ", e)
for m in ("numpy", "cv2", "scipy", "matplotlib"):
    try:
        mod = __import__(m)
        print(f"{m:<12}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"{m:<12}!! ", e)
n = int(__import__("numpy").__version__.split(".")[0])
if n >= 2:
    try:
        import cv2 as _c
        _c.np.zeros((2, 2))
        print("cv2/numpy2  ok")
    except Exception as e:
        print("cv2/numpy2  !! ", e, " -> need opencv-python-headless>=5.0")
PY

echo
echo "=== checkpoints ==="
for f in Combat3D_FULL.pt Combat3D_GTlabel2.pt Combat3D_OurLabel69.pt \
         Combat3D_SelfLabel_v2.pt rh_h4d_v04_w0_full.pt; do
  for d in "$B/mb/ckpt" "$B/weights/ckpt" "$(dirname "$0")/../weights/ckpt"; do
    [ -f "$d/$f" ] && { echo "  ok    $f  ($d)"; break; }
  done
done
echo
echo "If any checkpoint is missing, download Combat3D-weights-20261005.zip from"
echo "the GitHub Releases page and see weights/README.md."
