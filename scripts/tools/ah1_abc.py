#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★★ ABC 三臂同口径评测：都在【同一批 held-out test take】上、都【对官方 GT】评测。

A = Combat3D_GTlabel.pt   （训练用官方 GT 标签）
B = Combat3D_FULL.pt      （训练用我们自建标签：官方2D + 我们三角化）
C = Combat3D_SelfLabel.pt （训练用完全自建标签：我们的框/身份/2D/三角化）

评测口径与 z25_gtEval.py 完全一致（13 个共同关节、H36M17→COCO17、相似变换对齐）。
★ 同时输出「全部 15 条」与「剔除 6 条泄漏 sword3 后的 9 条」。
"""
import os, sys, json, glob
import numpy as np, torch
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_common as K, kb_train as kt

CLIP = 121
LEAK = {'001_sword3', '002_sword3', '003_sword3', '004_sword3', '005_sword3', '006_sword3'}
IDX17_FOR_13 = [10, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6]
F13_TO_COCO = {0: 0, 4: 5, 5: 7, 6: 9, 1: 6, 2: 8, 3: 10,
               10: 11, 11: 13, 12: 15, 7: 12, 8: 14, 9: 16}
H17_TO_C17 = {int(IDX17_FOR_13[s]): int(F13_TO_COCO[s]) for s in F13_TO_COCO}
C17 = sorted(H17_TO_C17.values())
H17 = [k for k, v in sorted(H17_TO_C17.items(), key=lambda kv: kv[1])]
PELVIS_C17 = [11, 12]
MSCALE = json.load(open(f'{B}/h4d_metric_scale.json'))

MODELS = [('A_officialGT', f'{B}/mb/ckpt/Combat3D_GTlabel2.pt'),
          ('B69_ourLabel', f'{B}/mb/ckpt/Combat3D_OurLabel69.pt'),
          ('B_ourLabel',   f'{B}/mb/ckpt/Combat3D_FULL.pt'),
          ('C_selfBuilt_v1', f'{B}/mb/ckpt/Combat3D_SelfLabel.pt'),
          ('C_selfBuilt_v2', f'{B}/mb/ckpt/Combat3D_SelfLabel_v2.pt')]


def gt_official_mm(take, view, pid, n):
    d = f'{B}/gt_{take}/gt3d_colmap'
    key = 'aria01' if pid == 0 else 'aria02'
    fs = sorted(glob.glob(f'{d}/*.json'))[:n]
    X = []
    for f in fs:
        v = json.load(open(f)).get(key)
        if v is None:
            return None
        X.append(np.array(v, float)[:, :3])
    if len(X) < 10:
        return None
    X = np.stack(X)
    cal = kt.get_cal(dict(group='h4d_%s' % take))
    R, T = cal.R[view], cal.T[view]
    Xc = (R @ X.reshape(-1, 3).T).T.reshape(X.shape) + T
    Xc = Xc * 1000.0 * MSCALE.get(take, 1.0)
    Xc -= Xc[:, PELVIS_C17, :].mean(1, keepdims=True)
    return Xc


def kabsch(a, g):
    H = a.T @ g
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    return float(np.trace(np.diag(D) @ S) / max((a ** 2).sum(), 1e-9)), Vt.T @ S @ U.T


def predict(mb, s):
    x = kt.norm2d(s['k2d'])
    n = len(x)
    out = np.zeros((n, 17, 3)); cnt = np.zeros(n)
    with torch.no_grad():
        for st in range(0, max(1, n - CLIP + 1), CLIP // 2):
            seg = x[st:st + CLIP]
            if len(seg) < CLIP:
                seg = np.concatenate([seg, np.repeat(seg[-1:], CLIP - len(seg), 0)], 0)
            o = mb(torch.from_numpy(seg[None]).to(kt.DEV)).cpu().numpy()[0]
            m_ = min(CLIP, n - st); out[st:st + m_] += o[:m_]; cnt[st:st + m_] += 1
    return out / np.maximum(cnt, 1)[:, None, None]


seqs = kt.load_all(f'{B}/mb/data_h4d_test15')
# 先备好官方 GT
GT = {}
for s in seqs:
    if len(s['k2d']) < CLIP:
        continue
    G = gt_official_mm(s['take'], s['view'], s['pid'], len(s['k2d']))
    if G is not None and len(G) >= CLIP:
        GT[(s['take'], s['view'], s['pid'])] = G
print('有效序列-视角: %d' % len(GT))

print('\n%-14s %-8s %6s %8s %8s' % ('模型', '子集', 'n', 'MPJPE', 'PA'))
for name, ck in MODELS:
    if not os.path.exists(ck):
        print('  %-14s 权重缺失 %s' % (name, os.path.basename(ck))); continue
    mb = kt.build_official(); mb = kt.load_ckpt_weights(mb, ck, 'full'); mb.to(kt.DEV).eval()
    rows = []
    for t in sorted(set(s['take'] for s in seqs)):
        sub = [s for s in seqs if s['take'] == t and (t, s['view'], s['pid']) in GT]
        if not sub:
            continue
        A_, B_ = [], []
        for s in sub:
            P = predict(mb, s)[:, H17, :]
            G = GT[(t, s['view'], s['pid'])][:, C17, :]
            n = min(len(P), len(G))
            P = P[:n] - P[:n].mean(1, keepdims=True)
            G = G[:n] - G[:n].mean(1, keepdims=True)
            A_.append(P.reshape(-1, 3)); B_.append(G.reshape(-1, 3))
        A2 = np.concatenate(A_); B2 = np.concatenate(B_)
        sv, Rv = kabsch(A2 - A2.mean(0), B2 - B2.mean(0))
        # 只有一个 pid 时也能评
        errs, pas = [], []
        for s in sub:
            P = predict(mb, s)[:, H17, :]
            G = GT[(t, s['view'], s['pid'])][:, C17, :]
            n = min(len(P), len(G))
            P = P[:n] - P[:n].mean(1, keepdims=True)
            G = G[:n] - G[:n].mean(1, keepdims=True)
            Pp = (P.reshape(-1, 3) @ Rv.T * sv).reshape(P.shape)
            errs.append(np.linalg.norm(Pp - G, axis=2).ravel())
            for f in range(0, n, 5):
                a = P[f] - P[f].mean(0); g = G[f] - G[f].mean(0)
                sc, R = kabsch(a, g)
                pas.append(np.linalg.norm(a @ R.T * sc - g, axis=1))
        rows.append((t, float(np.median(np.concatenate(errs))),
                     float(np.median(np.concatenate(pas)))))
    for lab, rr in (('全部', rows), ('9条held-out', [r for r in rows if r[0] not in LEAK])):
        if rr:
            print('%-14s %-8s %6d %8.1f %8.1f' %
                  (name, lab, len(rr), np.median([r[1] for r in rr]), np.median([r[2] for r in rr])))
    del mb
    torch.cuda.empty_cache()
