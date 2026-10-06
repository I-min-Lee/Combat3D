#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★修正官方 GT 的对齐方向（上一版搞反了）：
   要的是 T 使 T(official) ≈ em_off（em_off 在标定世界系）。
   umeyama(a=em_off, b=official) 给的是 official ≈ s R em_off + t，
   所以逆变换 = R^T (official - t)/s。
"""
import os, sys, json, glob, shutil
import numpy as np
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_common as K, kb_train as kt
C17B25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def umeyama(a, b):
    mu_a, mu_b = a.mean(0), b.mean(0)
    A, Bm = a - mu_a, b - mu_b
    U, D, Vt = np.linalg.svd(Bm.T @ A / len(A))
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / max((A ** 2).sum(1).mean(), 1e-9))
    t = mu_b - s * (R @ mu_a)
    return s, R, t


def load_gt(take, pid, src='gt'):
    d = f'{B}/{src}_{take}/gt3d_colmap'
    fs = sorted(glob.glob(f'{d}/*.json'))
    key = 'aria01' if pid == 0 else 'aria02'
    out = []
    for f in fs:
        v = json.load(open(f)).get(key)
        if v is None:
            break
        out.append(np.array(v, float)[:, :3])
    return np.stack(out) if out else None


fixed = 0
for gt in sorted(glob.glob(f'{B}/gt_*')):
    take = os.path.basename(gt)[3:]
    if take.endswith('_aligned'):
        continue
    de = f'{B}/em_off_{take}/lam1.0'
    if not os.path.isdir(os.path.join(gt, 'gt3d_colmap')) or not os.path.isdir(de):
        continue
    # 用【原始官方 GT】和我们的 em_off 解变换
    G = load_gt(take, 0, src='gt')
    fs = sorted(glob.glob(f'{de}/pid0/keypoints3d/*.json'))
    if G is None or len(fs) < 100:
        continue
    n = min(len(G), len(fs))
    E = np.stack([np.array(json.load(open(fs[i]))[0]['keypoints3d'], float)[:, :3] for i in range(n)])
    E17 = E[:, C17B25, :]
    a = E17.reshape(-1, 3); b = G[:n].reshape(-1, 3)
    s, R, t = umeyama(a, b)                      # official ≈ s R a + t
    # ★逆变换：把 official 映射回 em_off 的（=标定）世界系
    outd = f'{B}/gt_{take}_aligned/gt3d_colmap'
    os.makedirs(outd, exist_ok=True)
    for f in sorted(glob.glob(f'{gt}/gt3d_colmap/*.json')):
        d0 = json.load(open(f)); rec = {}
        for k, arr in d0.items():
            A = np.array(arr, float)
            A[:, :3] = (R.T @ (A[:, :3] - t).T).T / s
            rec[k] = A.tolist()
        json.dump(rec, open(f'{outd}/{os.path.basename(f)}', 'w'))
    if os.path.exists(f'{gt}/scale_metric.npy'):
        shutil.copy2(f'{gt}/scale_metric.npy', f'{B}/gt_{take}_aligned/scale_metric.npy')
    fixed += 1
print('重写对齐 %d 个场次' % fixed)
