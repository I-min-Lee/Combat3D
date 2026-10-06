#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★建 C 臂 npz：完全自建的 em_self2_<take> -> mb/data_h4d_selfnpz
k2d/valid 取自自建 2D（asm_self2_<take>），k3d 取自自建三角化（em_self2_<take>）。
布局：em_self2 是 body25，按 C17B25 取 17 关节（与 offtri 同口径）。
"""
import os, sys, json, glob
import numpy as np
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_common as K

C17B25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
V = ['01', '03', '04', '07', '09', '14']
DST = f'{B}/mb/data_h4d_selfnpz'
os.makedirs(DST, exist_ok=True)

# 自建 2D 的 JSON 结构：asm_self2_<take>/annots/<view>/%06d.json（LabelMe 风格，与 asm_off 同）
def load_2d(take, view, pid):
    d = f'{B}/asm_self2_{take}/annots/{view}'
    fs = sorted(glob.glob(f'{d}/*.json'))
    if not fs:
        return None
    out = []
    for f in fs:
        rec = json.load(open(f))
        cand = [a for a in rec['annots'] if int(a['personID']) == pid]
        if not cand:
            out.append(np.zeros((17, 3), np.float32)); continue
        k = np.array(cand[0]['keypoints'], float).reshape(17, 3)
        out.append(k.astype(np.float32))
    return np.stack(out)


def load_3d(take, pid):
    d = f'{B}/em_self2_{take}/lam1.0/pid{pid}/keypoints3d'
    fs = sorted(glob.glob(f'{d}/*.json'))
    if not fs:
        return None
    out = []
    for f in fs:
        rec = json.load(open(f))
        k = np.array(rec[0]['keypoints3d'], float)
        out.append(k)
    return np.stack(out)


n_ok = 0
for take in sorted(set(d.split('asm_self2_')[-1] for d in glob.glob(f'{B}/asm_self2_*'))):
    for view in V:
        for pid in (0, 1):
            k2d = load_2d(take, view, pid)
            k3d25 = load_3d(take, pid)
            if k2d is None or k3d25 is None:
                continue
            n = min(len(k2d), len(k3d25))
            k2d, k3d25 = k2d[:n], k3d25[:n]
            k3d = np.zeros((n, 17, 4), np.float32)
            k3d[:, :, :3] = k3d25[:, C17B25, :3]
            k3d[:, :, 3] = 1.0
            valid = (k2d[:, :, 2] > 0.05)
            meta = dict(group='h4d_%s' % take, take=take, view=view, pid=pid,
                        fps=20.0, n=n, n_ok=int(valid.sum()),
                        src2d='self2', src3d='em_self2_tri', frame0=1)
            np.savez_compressed(f'{DST}/{take}_v{view}_p{pid}.npz', k2d=k2d, k3d=k3d,
                                valid=valid, meta=json.dumps(meta))
            n_ok += 1
print('写出 %d 个 npz -> %s' % (n_ok, DST))
print('覆盖 take 数:', len(set(os.path.basename(f)[:-4].rsplit('_', 2)[0]
                                for f in glob.glob(f'{DST}/*.npz'))))
