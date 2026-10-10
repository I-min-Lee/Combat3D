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
"""h4d_boxes.py — Harmony4D 真值框 → 管线 LabelMe 检测 JSON（★带真值身份）

Harmony4D 的框是**按受试者**给的（`processed_data/bbox/cam{NN}/{frame:05d}.npy`
= dict {'aria01': [x1,y1,x2,y2], 'aria02': [...]}，像素坐标，3840×2160）。

→ 本适配层因此**白拿身份真值**：把 subject 直接写进 group_id：
     aria01 → group_id 0
     aria02 → group_id 1
   于是 `conv_rtd_v3.py:138` 的 `gid ^ IDFLIP` 只要 IDFLIP 全 0，
   personID 就是**真值身份** —— 这是身份层的消融基线（"身份白给"，
   只量三角化/节点层），也是官方 top-down 协议允许的用法。

下游契约（vp_rtd_batch.py:66-77 / conv_rtd_v3.py:126-141）：
    {out}/{view}/<任意>_%06d.json   末段必须是帧号（rsplit('_',1)[-1]）
    shapes[].label == 'person'，group_id ∈ {0,1}
    points == [[x1,y1],[x2,y2]]（像素，非归一化）

用法：
  python h4d_boxes.py --seq-root <...>/016_mma4 --out <det 目录> \
                      --views 01,03,04,07,09,14 [--start 1 --end 741] [--tag 016_mma4]
输出：{out}/{view}/{tag}_%06d.json
"""
import os, json, argparse
import numpy as np

SUBJ_TO_GID = {'aria01': 0, 'aria02': 1}     # ★ 身份真值映射（固定）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--tag', default=None, help='文件名前缀，默认取 sequence 目录名')
    ap.add_argument('--W', type=int, default=3840)
    ap.add_argument('--H', type=int, default=2160)
    a = ap.parse_args()

    tag = a.tag or os.path.basename(os.path.normpath(a.seq_root))
    views = [v.zfill(2) for v in a.views.split(',')]
    print('=== 框导出报告 (tag=%s) ===' % tag)
    allsubj = set()
    for v in views:
        src = os.path.join(a.seq_root, 'processed_data', 'bbox', 'cam%s' % v)
        dst = os.path.join(a.out, v)
        os.makedirs(dst, exist_ok=True)
        files = sorted(f for f in os.listdir(src) if f.endswith('.npy')) if os.path.isdir(src) else []
        if not files:
            print('  view %s : ❌ 无框 (%s)' % (v, src)); continue
        n, nsubj_hist, skipped = 0, {}, 0
        for fn in files:
            fi = int(fn.split('.')[0])
            if not (a.start <= fi <= a.end):
                continue
            d = np.load(os.path.join(src, fn), allow_pickle=True)
            d = d.item() if getattr(d, 'dtype', None) == object else d
            shapes = []
            for subj, box in d.items():
                if subj not in SUBJ_TO_GID:
                    skipped += 1; continue
                x1, y1, x2, y2 = [float(t) for t in np.asarray(box, float).reshape(-1)[:4]]
                shapes.append(dict(label='person',
                                   points=[[x1, y1], [x2, y2]],
                                   group_id=SUBJ_TO_GID[subj],
                                   description='subject:%s' % subj,
                                   shape_type='rectangle', flags={}))
                allsubj.add(subj)
            nsubj_hist[len(shapes)] = nsubj_hist.get(len(shapes), 0) + 1
            with open(os.path.join(dst, '%s_%06d.json' % (tag, fi)), 'w') as f:
                json.dump(dict(version='5.3.1', flags={}, shapes=shapes,
                               imagePath='%06d.jpg' % fi, imageData=None,
                               imageHeight=a.H, imageWidth=a.W), f)
            n += 1
        print('  view %s : %d 帧  每帧人数分布=%s%s'
              % (v, n, sorted(nsubj_hist.items()), ('  未识别 subject=%d' % skipped) if skipped else ''))
    print('[OK] 受试者=%s  身份映射=%s' % (sorted(allsubj), SUBJ_TO_GID))
    print('     下游用法：IDFLIP_JSON 指向全 0 的翻转表 → conv 后 personID 即真值身份')


if __name__ == '__main__':
    main()
