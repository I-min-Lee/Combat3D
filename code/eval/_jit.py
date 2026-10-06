#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_jit.py <base> [<base2> ...]  逐帧位移统计（帧间抖动）—— 每个 base 下找 pid{0,1}/keypoints3d
输出：中位/p99/max 帧间位移(m)，以及骨长 CV（13 或 25 节点自适应）。"""
import os, sys, json, glob
import numpy as np

IDX13 = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
# 13 节点局部索引: 0鼻 1Rsh 2Rel 3Rwr 4Lsh 5Lel 6Lwr 7Rhip 8Rknee 9Rank 10Lhip 11Lknee 12Lank
BONES = [(0,1),(0,4),(1,2),(4,5),(2,3),(5,6),(1,7),(4,10),(7,8),(10,11),(8,9),(11,12)]

for base in sys.argv[1:]:
    print('=== %s' % base)
    for pid in (0, 1):
        d = '%s/pid%d/keypoints3d' % (base, pid)
        fs = sorted(glob.glob(d + '/*.json'))
        if not fs:
            print('  pid%d 无数据' % pid); continue
        X = []
        for f in fs:
            j = json.load(open(f))
            a = j['annots'][0] if isinstance(j, dict) and 'annots' in j else j[0]
            X.append(np.array(a['keypoints3d'], float))
        X = np.stack(X)
        P = X[:, :, :3]
        if P.shape[1] == 25:
            P = P[:, IDX13]
        dv = np.linalg.norm(np.diff(P, axis=0), axis=-1)          # (N,13)
        mx = dv.max(axis=1)
        bl = np.array([np.linalg.norm(P[:, a] - P[:, b], axis=-1) for a, b in BONES])
        cv = (bl.std(axis=1) / bl.mean(axis=1))
        print('  pid%d N=%d | 帧间位移 中位%.4f p99=%.4f max=%.4f m | 骨长CV 中位%.4f max%.4f'
              % (pid, len(P), np.median(mx), np.percentile(mx, 99), mx.max(),
                 np.median(cv), cv.max()))
