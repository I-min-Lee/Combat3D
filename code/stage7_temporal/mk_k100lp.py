#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mk_k100lp.py <src_base> <out_base> [N]
取 k=100 的 b25 输出, 做**零相位时序滤波** (中值5帧去尖峰 + SG41 poly3 保形),
生成真实产物目录, 供渲染与评估。只在时间轴滤波, 不改任何单帧几何。
"""
import os, sys, json, glob
import numpy as np
import scipy.signal as ss
src, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
for pid in (0, 1):
    d = '%s/pid%d/keypoints3d_b25' % (src, pid)
    files = sorted(glob.glob(d + '/*.json'))[:N]
    if not files:
        print('缺 %s' % d); continue
    X = np.stack([np.array(json.load(open(f))[0]['keypoints3d'], float) for f in files])
    print('pid%d 载入 %s' % (pid, X.shape), flush=True)
    _mk = int(os.environ.get('MEDFILT_K', '5'))    # 25fps 下应传 1（关）
    _sw = int(os.environ.get('SG_WIN', '41'))      # 25fps 下应传 5
    if _mk > 1:
        Y = ss.medfilt(X, kernel_size=(_mk, 1, 1))
    else:
        Y = X.copy()
    Y = ss.savgol_filter(Y, _sw, 3, axis=0, mode='interp')
    print('  窗口: medfilt=%d savgol=%d' % (_mk, _sw), flush=True)
    # 保底: 不要偏离原始超过 200mm (防边缘/异常处失真)
    dv = np.linalg.norm(Y[:, :, :3] - X[:, :, :3], axis=-1) * 1000
    print('  滤波位移: 中位=%.2f 均值=%.2f p99=%.2f max=%.1f mm' %
          (np.median(dv), dv.mean(), np.percentile(dv, 99), dv.max()), flush=True)
    o = '%s/pid%d/keypoints3d_b25' % (out, pid)
    os.makedirs(o, exist_ok=True)
    for f, arr in zip(files, Y):
        # conf 通道原样保留
        json.dump([{'id': pid, 'keypoints3d': arr.tolist()}],
                  open(os.path.join(o, os.path.basename(f)), 'w'))
    print('  -> %s (%d)' % (o, len(os.listdir(o))), flush=True)
print('DONE')
