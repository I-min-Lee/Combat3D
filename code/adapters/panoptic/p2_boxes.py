#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  ★★★  SCENE-SPECIFIC LAYER — DO NOT REUSE AS-IS  ★★★
#
#  这是 Combat3D-Label 五层里【两个场景相关层】之一（检测框层 / 身份层）。
#  我们的测量证明：这两层换场景必须【换算法】，不是调参数 ——
#
#        changing scene  =>  changing algorithm
#
#  实测：Harmony4D 上我们用数据集自带的框与身份；而在这套代码的第二个域
#  （我们自己采集的盔甲对抗棍棒格斗）上，同一层必须从零重写
#  （RT-DETR + 位置先验；以及基于颜色证据的身份定标）。
#  两份实现都在本仓库里 —— 见 docs/SECOND_DOMAIN.md 的逐层对照表。
#
#  通用的是【三角化层及以下】，它们原样复用。层契约见 code/adapters/README.md。
#
#  ★ 连数据格式都是场景相关的：这个文件读的框/身份格式来自特定数据集，
#    换场景时要连 I/O 一起改，不只是改阈值。
#
#  Reusable? This layer: NO — rewrite it. Triangulation & below: YES.
# =============================================================================
"""p2_boxes.py — Panoptic「真值框版」检测树 + `det_vs_gt.py` 用的框 npy

Panoptic 不直接给 2D 人体框（它给 3D 骨架），所以这里**把 3D 真值投影进各视角**
取紧致外接框，再各边放 5% 余量，得到"人体框"。产物两份：

1. `{out}/{view}/{tag}_%06d.json`  —— 管线用的 LabelMe（group_id：aria01→0, aria02→1），
   格式与 `h4d_boxes.py` 逐字段一致，下游 `vp_h4d.py` / `assemble_h4d.py` 零改动。
2. `{bbox-out}/cam{view}/%05d.npy` —— `det_vs_gt.py` 读的真值框
   （dict{subject:[x1,y1,x2,y2]}，**未外扩**）。

★与 Harmony4D 的差别（如实记录）：H4D 的框是数据集自带的；这里的框是**由 3D 真值投影得到**，
  因此与 3D 真值天然一致，"真值框版"的数字会偏乐观。它只用来做**方法链路的正确性基线**，
  不能当成独立框质量证据。

用法：
  python p2_boxes.py --pose-dir <.../hdPose3d_stage1_coco19> --calib-dir <calib> \
      --views 00_01,... --start 4256 --n 600 --out <det 目录> --bbox-out <processed_data 根> \
      --tag ian2 [--swap-lr] [--margin 0.05]
"""
import os, json, argparse, collections
import numpy as np
import cv2

M_LR = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
M_SWAP = [1, 17, 15, 18, 16, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 8]
SUBJ_TO_GID = {'aria01': 0, 'aria02': 1}


def read_cams(calibdir, views):
    import sys
    sys.path.insert(0, os.environ.get('EMCORE', '/workshop/Lym/combat3d/port/emcore'))
    from easymocap.mytools.camera_utils import read_camera
    C = read_camera(os.path.join(calibdir, 'intri.yml'), os.path.join(calibdir, 'extri.yml'))
    out = {}
    for v in views:
        c = C[v]
        out[v] = dict(K=np.array(c['K'], float).reshape(3, 3),
                      D=np.asarray(c['dist'], float).reshape(-1)[:5],
                      R=np.array(c['R'], float).reshape(3, 3),
                      T=np.array(c['T'], float).reshape(3),
                      W=int(c.get('W', 1920)), H=int(c.get('H', 1080)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose-dir', required=True)
    ap.add_argument('--calib-dir', required=True)
    ap.add_argument('--views', required=True)
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bbox-out', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--margin', type=float, default=0.05)
    ap.add_argument('--swap-lr', action='store_true')
    a = ap.parse_args()
    views = [v.strip() for v in a.views.split(',')]
    m = M_SWAP if a.swap_lr else M_LR
    cams = read_cams(a.calib_dir, views)
    for v in views:
        os.makedirs(os.path.join(a.out, v), exist_ok=True)
        os.makedirs(os.path.join(a.bbox_out, 'cam%s' % v), exist_ok=True)

    n_w = 0
    hist = collections.Counter()
    for fr in range(a.start, a.start + a.n):
        f = os.path.join(a.pose_dir, 'body3DScene_%08d.json' % fr)
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        bodies = {}
        for b in d.get('bodies', []):
            if len(b.get('joints19', [])) < 76:
                continue
            bid = int(b['id'])
            if bid in (0, 1):
                bodies['aria%02d' % (bid + 1)] = np.array(b['joints19'], float).reshape(-1, 4)
        if len(bodies) < 2:
            continue
        boxes = {}
        for v in views:
            c = cams[v]
            shapes = []
            for subj, J in sorted(bodies.items(), key=lambda kv: SUBJ_TO_GID[kv[0]]):
                P = J[m, :3]
                if ((c['R'] @ P.T).T + c['T'])[:, 2].min() <= 0:
                    continue
                uv, _ = cv2.projectPoints(np.ascontiguousarray(P).reshape(-1, 1, 3),
                                          cv2.Rodrigues(np.ascontiguousarray(c['R']))[0].reshape(3, 1),
                                          c['T'].reshape(3, 1),
                                          np.ascontiguousarray(c['K'], float),
                                          np.ascontiguousarray(c['D'], float))
                uv = uv.reshape(-1, 2)
                x1, y1 = uv[:, 0].min(), uv[:, 1].min()
                x2, y2 = uv[:, 0].max(), uv[:, 1].max()
                w, h = x2 - x1, y2 - y1
                x1 -= w * a.margin; x2 += w * a.margin
                y1 -= h * a.margin; y2 += h * a.margin
                x1 = float(np.clip(x1, 0, c['W'] - 1)); x2 = float(np.clip(x2, 1, c['W']))
                y1 = float(np.clip(y1, 0, c['H'] - 1)); y2 = float(np.clip(y2, 1, c['H']))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                shapes.append({"label": "person", "points": [[x1, y1], [x2, y2]],
                               "group_id": SUBJ_TO_GID[subj], "description": "subject:%s" % subj,
                               "shape_type": "rectangle", "flags": {}})
                boxes[subj] = np.array([x1, y1, x2, y2], float)
            shapes.sort(key=lambda s: s['group_id'])
            with open(os.path.join(a.out, v, '%s_%06d.json' % (a.tag, fr)), 'w') as fo:
                json.dump(dict(version='5.3.1', flags={}, shapes=shapes,
                               imagePath='%s_%06d.jpg' % (a.tag, fr), imageData=None,
                               imageHeight=c['H'], imageWidth=c['W']), fo)
            np.save(os.path.join(a.bbox_out, 'cam%s' % v, '%05d.npy' % fr), boxes)
            hist[len(shapes)] += 1
        n_w += 1
    print('[p2_boxes] 写出 %d 帧 × %d 视角 | 每(帧,视角)框数分布=%s'
          % (n_w, len(views), sorted(hist.items())))
    print('  LabelMe -> %s   GT npy -> %s/cam<view>/%%05d.npy' % (a.out, a.bbox_out))


if __name__ == '__main__':
    main()
