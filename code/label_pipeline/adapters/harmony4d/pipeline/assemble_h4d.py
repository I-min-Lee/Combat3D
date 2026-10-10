#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  ✅  UNIVERSAL LAYER — REUSE AS-IS ACROSS SCENES
#
#  这是 Combat3D-Label 五层里【通用】的那几层之一（2D 提取 / 三角化 / SMPL 拟合 /
#  单目 lifter）。两个域上的实测：换场景时这几层【一行未改】。
#
#  与它对照的是【场景相关】的检测框层与身份层 —— 那两个必须换算法，
#  文件头带 "SCENE-SPECIFIC LAYER" 横幅。层契约见 code/adapters/README.md。
# =============================================================================
"""assemble_h4d.py — Harmony4D 的 annots 装配（**身份白给**模式）

为什么不用 `conv_rtd_v3.py`：
  ViTPose 是**按每个受试者的真值框**推理的（h4d_boxes.py 把 subject 写进 group_id），
  所以 `vp_h4d.py` 输出的 annot 里 `personID` **就是真值身份**。
  而 `conv_rtd_v3.py` 的 pass1 会用"躯干中心落在框内 + conf 降序"**重新指派**——
  在贴身缠斗（两框大面积重叠）时必然互换，而它消解重叠唯一的手段是颜色挑，
  Harmony4D 上无颜色先验（实测 view 04 的 pid0 只有 3/12 帧指派正确）。
  → 本脚本只做**格式转换 + 占位**，不做任何身份推断。

  原 `conv_rtd_v3.py` 仍然保留：它属于**另一组实验**
  （自检测框 + 自己的身份层，对 GT subject 算 identity switch rate）。

输出契约与 conv_rtd_v3 完全一致（下游 tri/评测不用改）：
  {OUT}/annots/{view}/%06d.json = {"height":H,"width":W,
      "annots":[{"personID":0,"keypoints":[[x,y,conf]x17]},
                {"personID":1,"keypoints":[[x,y,conf]x17]}]}
  缺观测的 pid 写全 0 占位（三角化会按 min_conf 丢弃该 (帧,视角)）。

用法：
  python assemble_h4d.py --raw <vp 输出> --out <annots 目录> --views 01,03,04,07,09,14 \
                         [--start 1 --end 741] [--H 2160 --W 3840]
"""
import os, json, glob, argparse
import numpy as np

PIDS = (0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='vp_h4d.py 的输出目录（含 {view}/%06d.json）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--H', type=int, default=2160)
    ap.add_argument('--W', type=int, default=3840)
    a = ap.parse_args()
    views = [v.zfill(2) for v in a.views.split(',')]

    for v in views:
        os.makedirs(os.path.join(a.out, 'annots', v), exist_ok=True)
    n_written = n_ph = 0
    per_view = {}
    for v in views:
        files = sorted(glob.glob(os.path.join(a.raw, v, '*.json')))
        nv = 0
        for fp in files:
            fr = int(os.path.basename(fp).split('.')[0])
            if not (a.start <= fr <= a.end):
                continue
            d = json.load(open(fp))
            got = {}
            for an in d.get('annots', []):
                pid = int(an['personID'])
                kp = np.asarray(an['keypoints'], float)
                if kp.shape != (17, 3):
                    continue
                got[pid] = kp
            annots = []
            for pid in PIDS:                      # ★ 直接按 personID 落位，不推断
                if pid in got:
                    annots.append({'personID': pid, 'keypoints': np.round(got[pid], 4).tolist()})
                else:
                    annots.append({'personID': pid,
                                   'keypoints': [[0.0, 0.0, 0.0] for _ in range(17)]})
                    n_ph += 1
            json.dump({'height': a.H, 'width': a.W, 'annots': annots},
                      open(os.path.join(a.out, 'annots', v, '%06d.json' % fr), 'w'))
            nv += 1; n_written += 1
        per_view[v] = nv
        print('  view %s : %d 帧' % (v, nv))
    print('[OK] 写出 %d 个 annots 文件（占位 %d 处）-> %s/annots'
          % (n_written, n_ph, a.out))
    print('     ★ 身份 = 真值（personID 直通），未做任何身份推断')


if __name__ == '__main__':
    main()
