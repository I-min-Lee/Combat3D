# `pose312` 环境配方（222 服务器，2026-09-29 实测跑通）

> **为什么不能直接照搬 westc 的环境**：222 驱动 **535.104.05 = CUDA 12.2**，
> 而 westc 的 torch 是 **2.8.0+cu128**（需驱动 ≥570）。用户红线：不擅自升级容器/CUDA。
> 因此按 **cu121** 重建；同时 **mmcv 必须用官方预编译 wheel**（见下，这是最容易卡住的一步）。

## 一、最终版本清单（与 westc 逐项对齐，仅 torch/CUDA 不同）

| 包 | 版本 | 来源 |
|---|---|---|
| python | 3.12.14 | conda（tuna 频道，`--override-channels` 绕开 Anaconda ToS） |
| **torch / torchvision** | **2.4.1+cu121 / 0.19.1** | PyPI(tuna)。cu121 是驱动 535 **原生支持**的版本 |
| **mmcv** | **2.2.0（含 `_ext`，官方预编译）** | `download.openmmlab.com/mmcv/dist/cu121/torch2.4/`（**可达**） |
| mmdet | 3.3.0（`MMDET_WITH_OPS=0` 纯 python） | PyPI(tuna)，`--no-deps --no-build-isolation` |
| mmpretrain | 1.2.0 | PyPI(tuna)，`--no-deps` |
| mmpose | 1.3.2 | PyPI(tuna)，`--no-deps` |
| mmengine | 0.10.7 | PyPI(tuna) |
| numpy | 2.5.3 | 随 torch |
| opencv | **opencv-python-headless 5.0.0** | ⚠️ 4.8.1.78 是 numpy1 编译的，numpy2 下**加载不了** |
| xtcocotools | 1.14.3（**源码编译**） | 无 cp312 wheel → 用 **zig 当编译器**（见下） |
| setuptools | **<81**（80.10.2） | ≥81 移除了 `pkg_resources`，mmengine 需要 |
| ziglang | 最新 | 仅用作 C 编译器，**不碰系统盘、不改容器** |
| ultralytics | 8.4.113 | 检测层备用（Harmony4D 主结果用真值框，不依赖它） |

## 二、踩过的 8 个坑（按顺序）

1. **`pip` 都不存在**：容器 `/usr/bin/python3` 无 pip、无 ensurepip → 用 miniconda
2. **conda 默认频道卡 ToS** → `--override-channels -c tuna/main -c tuna/r`
3. **新版 Miniconda 带 Python 3.14**，太新（无 cu121 wheel）→ 必须建 py3.12 子环境
4. **`chumpy` / `xtcocotools` 无 cp312 wheel**，且**容器里没有任何编译器**（gcc/cc/make/ld 全无）
   → `pip install ziglang` 拿到 `zig cc`，`CC="<zig> cc" pip install xtcocotools --no-build-isolation`（43 秒搞定）
5. **OpenCV 4.8.1.78 在 numpy 2 下报 `_ARRAY_API not found`** → 改用 headless 5.0
6. **`mmpose` 硬依赖 `mmdet`**（`rtmo_head.py` 无条件 `from mmdet.utils import ConfigType, reduce_mean`），
   而 mmdet 3.3.0 断言 `mmcv<2.2.0` → **按 westc 的改法**把 `mmcv_maximum_version` 放宽为 `'9.9.0'`
7. **`mmpose` 还需 `mmcv.ops.MultiScaleDeformableAttention`**（EDPose/Deformable-DETR 头）→ **mmcv-lite 不够**，
   必须编译版 mmcv；搬 westc 的 `_ext.so` **ABI 不匹配**（`c10::Error` 符号签名在 torch2.8/2.5 间变了）
   → 正解是 **torch 降到 2.4.1 + 官方 cu121/torch2.4 预编译 wheel**
8. **`init_model` 找不到 `mmpretrain.VisionTransformer`** → 脚本里**预导入 `import mmpretrain.models`**
   （注册 scope），并装 `modelindex`

## 三、可复现的安装命令

```bash
B=/workshop/Lym/combat3d ; PY=$B/envs/miniconda3/envs/pose312/bin/python
T=https://pypi.tuna.tsinghua.edu.cn/simple
OL=https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
export TMPDIR=$B/tmp PIP_CACHE_DIR=$B/.pipcache     # ★ 全部落数据盘，系统盘只有 78G

# 1) torch cu121
$PY -m pip install "torch==2.4.1" "torchvision==0.19.1" -i $T
# 2) 官方预编译 mmcv（含 _ext）
$PY -m pip install addict yapf -i $T
$PY -m pip install "mmcv==2.2.0" --no-deps --no-index -f $OL
# 3) mmpose 栈（--no-deps 绕开 chumpy/xtcocotools 的依赖解析）
$PY -m pip install "mmengine==0.10.7" json-tricks munkres matplotlib pillow scipy \
     opencv-python-headless timm einops pyyaml -i $T
$PY -m pip install "mmpose==1.3.2" --no-deps -i $T
$PY -m pip install "mmpretrain==1.2.0" --no-deps -i $T
$PY -m pip install modelindex rich pycocotools shapely terminaltables pyparsing prettytable -i $T
# 4) mmdet（纯 python，不编算子）+ 版本上限放宽（与 westc 同改法）
MMDET_WITH_OPS=0 $PY -m pip install "mmdet==3.3.0" --no-deps --no-build-isolation -i $T
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '9.9.0'/" \
    $B/envs/miniconda3/envs/pose312/lib/python3.12/site-packages/mmdet/__init__.py
# 5) xtcocotools（zig 编译）
$PY -m pip install ziglang -i $T
$PY -m pip install -C pip Cython -i $T; $PY -m pip install Cython -i $T
ZIG=$($PY -c "import ziglang,os;print(os.path.join(os.path.dirname(ziglang.__file__),'zig'))")
CC="$ZIG cc" $PY -m pip install xtcocotools --no-build-isolation --no-cache-dir -i $T
# 6) setuptools 回退（pkg_resources）
$PY -m pip install "setuptools<81" -i $T
```

## 四、运行 ViTPose 的最低要求（★ 与 westc 脚本的差异只有 2 行）

```python
import numpy as np, torch
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})   # westc 脚本已有
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct]) # westc 脚本已有
import mmpretrain.models                    # ★ 新增：注册 mmpretrain.* scope
from mmpose.apis import inference_topdown, init_model
CFG = '<env>/lib/python3.12/site-packages/mmpose/.mim/configs/body_2d_keypoint/' \
      'topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
CKPT = '/workshop/Lym/combat3d/port/autodl-tmp/vitpose_ft.pth'
```

## 五、验收结果（2026-09-29）

| 检验 | 结果 |
|---|---|
| `import mmcv.ops` / `mmcv._ext` | ✅ |
| `from mmpose.apis import inference_topdown, init_model` | ✅ |
| `init_model` 加载 `vitpose_ft.pth` | ✅ 1.4 s，`TopdownPoseEstimator / VisionTransformer / HeatmapHead` |
| 真帧推理（Harmony4D cam09 f400，真值框） | conf 中位 **0.80 / 0.88**；与数据集 2D 最近邻 **10.8 / 9.5 px** |
| 真帧推理（cam04 f001） | conf 中位 0.32 / 0.51 ⚠️ 待查（疑似起始/贴身帧） |
| 吞吐 | **14 ms/实例**（单实例，含 CPU 预处理） |
| CUDA | `cuda_avail=True`，fp16 4096³ 实测通过（卡5） |

## 六、目录约定

```
/workshop/Lym/combat3d/
  envs/miniconda3/envs/pose312/   ← 主环境（跑 stage1-5 + eval）
  port/autodl-tmp/vitpose_ft.pth  ← 从 westc 拉的权重（与源逐字节一致 360,102,585 B）
  port/emcore/                    ← EasyMocap core（read_camera + fit，stage5/6 用）
  data/harmony4d/{zips,raw}/      ← 数据集
  logs/{ops,setup_env,...}.log
  status.sh                       ← 用户侧实时看板
```
