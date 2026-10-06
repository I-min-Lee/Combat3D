#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★受控 A/B 数据：把 offtri npz 的 k3d 换成【对齐后的官方 GT】，其余（k2d/valid/meta）完全不动。
这样两条训练臂只差"3D 标签来源"，是最干净的对照。
官方 GT 是 COCO17；我们 npz 是 H36M17 -> 只回填 13 个可映射关节，其余 valid=0。
"""
import os, sys, json, glob
import numpy as np
B = '/workshop/Lym/combat3d'
C17B25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]   # COCO17 -> body25
# H36M17 -> COCO17（由 IDX17_FOR_13 + F13_TO_COCO 复合）
IDX17_FOR_13 = [10, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6]
F13_TO_COCO = {0: 0, 4: 5, 5: 7, 6: 9, 1: 6, 2: 8, 3: 10, 10: 11, 11: 13, 12: 15, 7: 12, 8: 14, 9: 16}
H17_TO_C17 = {int(IDX17_FOR_13[s]): int(F13_TO_COCO[s]) for s in F13_TO_COCO}
C17_OF = sorted(H17_TO_C17.values())
H17_OF = [k for k, v in sorted(H17_TO_C17.items(), key=lambda kv: kv[1])]

SRC = f'{B}/mb/data_h4d_offtri'
DST = f'{B}/mb/data_h4d_gtalign'
os.makedirs(DST, exist_ok=True)

cache = {}
def gt_take(take):
    if take in cache:
        return cache[take]
    d = f'{B}/gt_{take}_aligned/gt3d_colmap'
    if not os.path.isdir(d):
        cache[take] = None; return None
    fs = sorted(glob.glob(f'{d}/*.json'))
    out = {}
    for i, f in enumerate(fs):
        d0 = json.load(open(f))
        out[i + 1] = {k: np.array(v, float) for k, v in d0.items()}
    cache[take] = out
    return out

n_ok = n_skip = 0
for p in sorted(glob.glob(f'{SRC}/*.npz')):
    base = os.path.basename(p)[:-4]
    take, v, pid = base.rsplit('_', 2)
    v = v[1:]; pid = int(pid[1:])
    G = gt_take(take)
    if G is None:
        n_skip += 1; continue
    z = np.load(p, allow_pickle=True)
    k2d, valid = z['k2d'], np.array(z['valid'], bool).copy()
    n = len(k2d)
    # ★未映射到官方 COCO17 的 4 个关节必须置 valid=False，否则模型被要求预测 (0,0,0)
    unmapped = [j for j in range(17) if (j not in H17_OF) and j != 0]
    if valid.ndim == 2:
        valid[:, unmapped] = False
    else:
        valid = valid[:, None] & np.array([j not in unmapped for j in range(17)])[None, :]
    k3d = np.zeros((n, 17, 4), np.float32)
    key = 'aria01' if pid == 0 else 'aria02'
    got = 0
    for t in range(n):
        rec = G.get(t + 1)
        if rec is None or key not in rec:
            continue
        arr = rec[key]                       # (17,3) COCO17，已在对齐世界系
        k3d[t, H17_OF, :3] = arr[C17_OF, :3]
        k3d[t, H17_OF, 3] = 1.0
        # ★★ H36M17 的 joint0 = 骨盆，官方 COCO17 里没有 -> 用双髋中点(COCO 11/12)。
        #    gt_camera_mm() 用 joint0 做根置零，不补这个整条管道就废了。
        k3d[t, 0, :3] = arr[[11, 12], :3].mean(0)
        k3d[t, 0, 3] = 1.0
        got += 1
    if got < 10:
        n_skip += 1; continue
    meta = json.loads(str(z['meta']))
    meta['src3d'] = 'official_gt3d_aligned'
    meta['n_gt'] = got
    np.savez_compressed(f'{DST}/{base}.npz', k2d=k2d, k3d=k3d,
                        valid=valid, meta=json.dumps(meta))
    n_ok += 1
print('写出 %d 个 npz（跳过 %d 个无对齐官方GT的 take）-> %s' % (n_ok, n_skip, DST))
print('  对应 take 数:', len(set(os.path.basename(f)[:-4].rsplit('_', 2)[0]
                                  for f in glob.glob(f'{DST}/*.npz'))))
