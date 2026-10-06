#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文用【场景复杂度对照表】的量化列。
输出：运动速度 / 出框率 / 两人髋间距 / 帧数，对比 Harmony4D vs Panoptic。"""
import os, sys, json, glob
import numpy as np
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_train as kt
C17B25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
MSCALE = json.load(open(f'{B}/h4d_metric_scale.json'))


def stats(datadir, name, is_p2=False):
    seqs = kt.load_all(datadir)
    takes = sorted(set(s['take'] for s in seqs))
    spd, oof, ipd, nfr = [], [], [], 0
    for t in takes:
        sub = [s for s in seqs if s['take'] == t]
        bypid = {}
        for s in sub:
            n = len(s['k2d'])
            nfr += n
            k2 = s['k2d']
            W, H = (1920, 1080) if is_p2 else (3840, 2160)
            vis = k2[:, :, 2] > 0.05
            outside = ((k2[:, :, 0] < 0) | (k2[:, :, 0] >= W) |
                       (k2[:, :, 1] < 0) | (k2[:, :, 1] >= H))
            oof.append(float((outside & ~vis).sum()) / max(vis.size, 1))
            c = s['k3d'][:, :, :3].astype(float)
            ms = MSCALE.get(t, 1.0)
            if is_p2:
                ms = 0.01
            c = (c - c[:, 0:1, :]) * ms
            v = s['valid']
            if v.ndim == 2:
                d = np.linalg.norm(np.diff(c, axis=0), axis=2)
                msk = v[1:] & v[:-1]
                if msk.sum() > 50:
                    spd.append(float(np.median(d[msk])) * 1000)
            bypid.setdefault(s['pid'], []).append(c[:, 0, :])
        # 两人髋间距（用根位置）
        if len(bypid) >= 2 and 0 in bypid and 1 in bypid:
            a, b = bypid[0][0], bypid[1][0]
            n = min(len(a), len(b))
            ipd.append(float(np.median(np.linalg.norm(a[:n] - b[:n], axis=1))) * 1000)
    print('  %-12s take %2d  帧 %6d | 帧间位移 %6.1f mm/帧 | 出框率 %5.2f%% | 两人髋间距 %6.0f mm'
          % (name, len(takes), nfr,
             np.median(spd) if spd else float('nan'),
             100 * np.median(oof) if oof else float('nan'),
             np.median(ipd) if ipd else float('nan')))


print('=== 场景复杂度对照（论文表用）===')
stats(f'{B}/mb/data_h4d_offtri', 'Harmony4D')
stats(f'{B}/mb/data_p2', 'Panoptic', is_p2=True)
