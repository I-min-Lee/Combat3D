#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p2_gt.py — Panoptic `hdPose3d_stage1_coco19` → 我们评测用的 COCO17 GT

输入：`body3DScene_%08d.json`，`bodies=[{id, joints19:[x,y,z,conf]×19}]`，**cm 世界系**。
输出：`{out}/gt3d_colmap/%06d.json` = `{"aria01": [[x,y,z,conf]×17], "aria02": ...}`（COCO17 顺序）

★COCO19 布局（由工具箱 `body_edges` 的官方骨架连边 + 逐关节平均高度实测共同确定）：
    0=neck(毂)  1=nose  2=hip中心  3/4/5=左肩/左肘/左腕  9/10/11=右肩/右肘/右腕
    6/7/8=左髋/膝/踝  12/13/14=右髋/膝/踝  15..18=眼/耳
  → COCO19 索引 16..18 与 19 号关节不参与 COCO17。

★身份：Panoptic 每帧 `bodies[].id` 实测稳定为 {0,1}，且**所有双人帧都恰好是 (0,1)**
  → 直接约定 **id 0 → aria01、id 1 → aria02**。这是数据集自带的真值身份
  （自建管线没有这个锚点 —— 这正是「真值框版」白送的那部分）。

★左右：3/9 到底是左肩还是右肩，靠几何无法确定（互换只是整体镜像，自洽性一样）。
  这里按社区通用的 cod19→coco17 约定写死，并用 `--swap-lr` 留一个开关，
  最终由"真值框版的 MPJPE"来判（左右写反 → 四肢误差巨大）。实测见记录。

用法：
  python p2_gt.py --pose-dir <.../hdPose3d_stage1_coco19> --out <gt 目录> \
      --start 4256 --n 600 [--swap-lr] [--allow-missing]
"""
import os, json, glob, argparse
import numpy as np

# COCO17 索引 -> COCO19 索引
M_LR = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]      # 左肩=3
M_SWAP = [1, 17, 15, 18, 16, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 8]    # 左肩=9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--swap-lr', action='store_true')
    ap.add_argument('--allow-missing', action='store_true',
                    help='缺帧/缺人时也写空 json（默认严格：缺人就跳过该帧）')
    a = ap.parse_args()
    m = M_SWAP if a.swap_lr else M_LR

    od = os.path.join(a.out, 'gt3d_colmap')
    os.makedirs(od, exist_ok=True)
    n_w = n_two = 0
    for fr in range(a.start, a.start + a.n):
        f = os.path.join(a.pose_dir, 'body3DScene_%08d.json' % fr)
        rec = {}
        if os.path.exists(f):
            d = json.load(open(f))
            for b in d.get('bodies', []):
                if len(b.get('joints19', [])) < 76:
                    continue
                bid = int(b['id'])
                if bid not in (0, 1):
                    continue
                J = np.array(b['joints19'], float).reshape(-1, 4)
                rec['aria%02d' % (bid + 1)] = J[m, :].tolist()
        if len(rec) < 2 and not a.allow_missing:
            continue
        with open(os.path.join(od, '%06d.json' % fr), 'w') as fo:
            json.dump(rec, fo)
        n_w += 1
        n_two += (len(rec) == 2)
    print('[p2_gt] 写出 %d 帧（其中两人齐全 %d 帧）-> %s' % (n_w, n_two, od))
    print('       左右映射: %s' % ('swap-lr' if a.swap_lr else 'default'))


if __name__ == '__main__':
    main()
