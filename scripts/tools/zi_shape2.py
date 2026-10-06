#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zi_shape2.py —— zi_shape.py 的可换数据/权重版本（不改原脚本）。

用法: ZI_DATA=.../mb/data_h4d_offtri ZI_CK=.../mb/ckpt/kb_h4d_offtri.pt zi_shape2.py
"""
import sys, os
import numpy as np, torch
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_common as K, kb_train as kt

DATA = os.environ.get('ZI_DATA', f'{B}/mb/data_h4d')
CLIP, VAL = 121, 'train01_hugging'
CKS = [('零样本 kendo', f'{B}/mb/ckpt/kb_full_v3.pt')]
if os.environ.get('ZI_CK'):
    CKS.append((os.environ.get('ZI_TAG', '本次'), os.environ['ZI_CK']))


def umeyama_r(A, Bm):
    H = A.T @ Bm; U, D, Vt = np.linalg.svd(H)
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ S @ U.T
    s = float(np.trace(np.diag(D) @ S) / max((A ** 2).sum(), 1e-9))
    return s, R


def ang(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


seqs = kt.load_all(DATA)
tr = [s for s in seqs if s['take'] != VAL]
va = [s for s in seqs if s['take'] == VAL]
print(f'DATA={DATA}')
print(f'训练 {len(tr)} 序列 / 验证 {len(va)} 序列')

for tag, ck in CKS:
    if not os.path.exists(ck):
        print(f'!! 缺权重 {ck}'); continue
    m = kt.build_official(); m = kt.load_ckpt_weights(m, ck, 'full')
    m.to(kt.DEV).eval()
    kt.CAL.clear()
    # ★ 与训练同口径：h4d 按 (take,view) 分组，验证场次也要有自己的键
    _cs = tr + va if str(tr[0].get('group', '')).startswith('h4d_') else tr
    kt.calibrate(m, _cs, CLIP)
    rangs = [ang(R) for _, R in kt.CAL.values()]
    r = kt.evaluate(m, va, clip_len=CLIP, max_clips=40, log=lambda *a: None)
    rtr = kt.evaluate(m, tr[:40], clip_len=CLIP, max_clips=40, log=lambda *a: None)

    pa = []
    with torch.no_grad():
        for seq in va:
            x = kt.norm2d(seq['k2d']); G = kt.gt_camera_mm(seq, kt.get_cal(seq))
            n = len(x)
            if n < CLIP:
                continue
            for t in np.linspace(0, n - CLIP, min(8, n - CLIP + 1)).astype(int):
                nat = m(torch.from_numpy(x[t:t + CLIP][None]).to(kt.DEV)).cpu().numpy()[0]
                for f in range(0, CLIP, 20):
                    ok = seq['valid'][t + f] & (seq['k2d'][t + f, :, 2] > 0.05)
                    if ok.sum() < 8:
                        continue
                    a = nat[f][ok]; g = G[t + f][ok]
                    ac = a - a.mean(0); gc = g - g.mean(0)
                    s, R = umeyama_r(ac, gc)
                    pa.append(np.linalg.norm(ac @ R.T * s - gc, axis=1))

    print(f'\n===== {tag}  ({os.path.basename(ck)}) =====')
    print(f'  ① PA 形状残差(逐帧最优对齐) 中位 = {np.median(np.concatenate(pa)):7.1f} mm')
    print(f'  ② 验证集端到端 MPJPE       中位 = {r["med_mm"]:7.1f} mm')
    print(f'  ② 训练集端到端 MPJPE       中位 = {rtr["med_mm"]:7.1f} mm   ← 欠拟合判据')
    print(f'  ③ 标定 R 旋转角 中位 = {np.median(rangs):5.1f}°   s = '
          f'{np.median([s for s, _ in kt.CAL.values()]):.1f} mm/unit')
print('SHAPE_DONE')
