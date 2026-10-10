# Harmony4D 适配层（`adapters/harmony4d/`）

> **只做"数据集 → 本管线最小输入契约"的转换，管线本体一行不改。**
> 2026-09-29 建立；标定链路已实测验收（GT3D 反投影中位 **9.2 px**，门限 20 px）。

## 一、四个脚本与顺序

| 顺序 | 脚本 | 输入 → 输出 |
|---|---|---|
| 1 | `h4d_calib.py` | `colmap/workplace/{cameras.txt,images.txt,scale.npy}` → `intri.yml` / `extri.yml`（EasyMocap 风格）+ `scale_metric.npy` + `h4d_meta.json` |
| 2 | `h4d_frames.py` | `exo/cam{NN}/images/%05d.jpg` → `{frames}/{match}/{seg}/{view}/%06d.png`（**软链**，不占盘） |
| 3 | `h4d_boxes.py` | `processed_data/bbox/cam{NN}/%05d.npy` → LabelMe 检测 JSON（**带真值身份**） |
| 4 | `h4d_gt.py` | `processed_data/{poses3d,poses2d}` → `gt3d_metric/` + `gt3d_colmap/` + `ref2d/` |

```bash
SEQ=/workshop/Lym/combat3d/data/harmony4d/raw/15_mma4/016_mma4
OUT=/workshop/Lym/combat3d
PY=$OUT/envs/miniconda3/envs/pose312/bin/python
V=01,03,04,07,09,14                      # 实测推荐 6 视角（最大方位间隙 64.8°）

$PY adapters/harmony4d/h4d_calib.py  --seq-root $SEQ --out $OUT/calib_h4d_016mma4 --views $V
$PY adapters/harmony4d/h4d_frames.py --seq-root $SEQ --frames-root $OUT/frames --match 15 --seg 4 --views $V
$PY adapters/harmony4d/h4d_boxes.py  --seq-root $SEQ --out $OUT/det_h4d_016mma4 --views $V --tag 016_mma4
$PY adapters/harmony4d/h4d_gt.py     --seq-root $SEQ --out $OUT/gt_h4d_016mma4 --views $V
```

## 二、实测结论（决定适配层为什么这么写）

| 事实 | 值 | 影响 |
|---|---|---|
| 相机模型 | **23/23 全 `OPENCV_FISHEYE`**，4 径向参数 | ★ 管线 `undistort()` 用 OpenCV 标准模型 — **不是"不够准"，是模型错了**（4 个鱼眼系数会被当成 `[k1,k2,p1,p2]`）→ 必须加鱼眼分支 |
| COLMAP id 映射 | 1,2=aria01/02（移动）；**3..22 = exo/cam01..cam20**；23=`mobile/`（手持） | 只保留 3..22；`id = exo 序号 + 2` |
| 外参粒度 | 每台固定机位仅 ~5 个 SfM 关键帧 | **四元数符号对齐后平均** → 单组固定外参（实测 t_std ≤ 0.0044 m、R_std ≤ 0.11°） |
| `scale.npy` | **4×4 相似变换**，s=1.375482 | 米制化：`X_metric = S @ X_colmap` |
| 米制系 | 人物 z 展布 1.80 m、xy 展布 4.6×4.2 m | ✅ 重力对齐、z 向上 |
| 机位几何 | 高度 0.38–0.82 m、距离 4.0–6.0 m、仰角 2.5–6.3° | ⚠️ **低机位环绕** → 头/脚遮挡严重，脸点（眼/耳）预计是弱项 |
| bbox | `dict{subject: [x1,y1,x2,y2]}`（**按受试者**） | ★ **身份真值白给** → `group_id` 写 subject，`IDFLIP` 全 0 即真值身份 |
| `poses2d` | `(45,2)`，**45 个 SMPL 关节投影、无 conf**，是数据集自己 fit 的产物 | ⚠️ **与真值同源，不能当独立 2D**；只用我们自己的 ViTPose，它只作参考 |
| 帧 | 741 帧 × 20 机位，**jpg 已抽好** | 省掉整段视频解码 |
| 真值 | `poses3d` = COCO17 + conf，**米制**；`smpl/` 全参数 | MPJPE 用前者；后者可算 **PVE** 对标 HMR 系 |

## 三、★ 坐标系的处理（关键设计）

- **`extri.yml` 直接写 COLMAP 世界系**，不做米制化 → 与 `read_camera` 的 `P = K@[R|T]`、
  以及实测验收链路**完全一致**，管线内部零改动。
- 因此管线输出的 3D 在 **COLMAP 系**；官方 MPJPE 在**米制系**。
  `h4d_gt.py` 两套都给（`gt3d_colmap/` 与 `gt3d_metric/`），评测时二选一，无隐式近似。
- 换算：`X_metric = scale_metric.npy @ X_colmap`（齐次）。

## 四、下游接线（尚未做，见操作记录 §七）

1. **`conv_rtd_v3` 的 IDFLIP 全 0**：用真值身份 → 只量三角化/节点层（官方 top-down 协议允许）
2. **鱼眼分支**：`tri_temporal_ab.py` **副本**里给 `undistort()` 加判断，走 `cv2.fisheye.undistortPoints`
3. **`final17_from_dp.py`**：13 节点索引表加 4 个脸点键 → 真 COCO17
   （body25→COCO17 只需在现有 `B25_TO_COCO17` 上补 `{15:2, 16:1, 17:4, 18:3}`）
4. **`eval/mpjpe.py`**：MPJPE(去根/含根) / PA-MPJPE / P95 / 骨长误差 / 3DPCK，按接触强度分层

## 五、红线

- 只动本目录 + **副本**（`conv_rtd_v5.py` / `tri_h4d_*.py`），**定稿脚本一行不改**
- 身份层（per-视角分段常数 Viterbi + 双通道证据）与三角化层
  （per-视角加权 `conf × border_weight × W_BOXTOUCH`、逐关节子集枚举、λ 二阶差分防跳变）**保持原样**
