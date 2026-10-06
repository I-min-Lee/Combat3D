#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★★ 主结果：对【官方 Harmony4D GT3D】评测 held-out test（不再用自己的标签自证）

关键映射：官方 gt3d 是 **COCO17** 序；我们 npz 的 17 是 **H36M17** 序。
    H36M17 -> final13 -> COCO17（由 IDX17_FOR_13 + F13_TO_COCO 复合得到）
共同可比的 13 个关节（COCO17 的眼/耳我们没有）。
两个世界系都在 COLMAP 单位 -> Xc = R@Xw + T，再 *1000*ms 得 mm。
"""
import os, sys, json, glob
import numpy as np, torch
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_common as K, kb_train as kt

CLIP = 121
CK = sys.argv[1] if len(sys.argv) > 1 else f'{B}/mb/ckpt/Combat3D_FULL.pt'
NAME = os.path.basename(CK)
LEAK = {'001_sword3', '002_sword3', '003_sword3', '004_sword3', '005_sword3', '006_sword3'}

# H36M17 -> 13 槽（mb_to_final13）-> COCO17（s221_f13to25）
IDX17_FOR_13 = [10, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6]
F13_TO_COCO = {0: 0, 4: 5, 5: 7, 6: 9, 1: 6, 2: 8, 3: 10,
               10: 11, 11: 13, 12: 15, 7: 12, 8: 14, 9: 16}
H17_TO_C17 = {int(IDX17_FOR_13[s]): int(F13_TO_COCO[s]) for s in F13_TO_COCO}
C17 = sorted(H17_TO_C17.values())          # 共同可比的 COCO17 关节
H17 = [k for k, v in sorted(H17_TO_C17.items(), key=lambda kv: kv[1])]
print('[map] 共同关节数 %d  COCO17 %s' % (len(C17), C17))
PELVIS_C17 = [11, 12]                       # 用双髋中点当根（COCO17 没有骨盆）

mb = kt.build_official(); mb = kt.load_ckpt_weights(mb, CK, 'full'); mb.to(kt.DEV).eval()
MSCALE = json.load(open(f'{B}/h4d_metric_scale.json'))


def gt_official_mm(take, view, pid, n):
    """官方 gt3d -> 相机系 mm（COCO17 序），逐帧根(双髋中点)置零"""
    d = f'{B}/gt_{take}/gt3d_colmap'
    key = 'aria01' if pid == 0 else 'aria02'
    fs = sorted(glob.glob(f'{d}/*.json'))[:n]
    X = []
    for f in fs:
        v = json.load(open(f)).get(key)
        if v is None:
            return None
        X.append(np.array(v, float)[:, :3])
    X = np.stack(X)                                   # (n,17,3) COLMAP 单位
    cal = kt.get_cal(dict(group='h4d_%s' % take))
    R, T = cal.R[view], cal.T[view]
    Xc = (R @ X.reshape(-1, 3).T).T.reshape(X.shape) + T
    Xc = Xc * 1000.0 * MSCALE.get(take, 1.0)
    Xc -= Xc[:, PELVIS_C17, :].mean(1, keepdims=True)  # ★根=双髋中点
    return Xc


def kabsch(a, g):
    H = a.T @ g
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    return float(np.trace(np.diag(D) @ S) / max((a ** 2).sum(), 1e-9)), Vt.T @ S @ U.T


seqs = kt.load_all(f'{B}/mb/data_h4d_test15')
rows = []
for t in sorted(set(s['take'] for s in seqs)):
    sub = [s for s in seqs if s['take'] == t]
    ch = {}
    for s in sub:
        n = len(s['k2d'])
        if n < CLIP:
            continue
        G = gt_official_mm(t, s['view'], s['pid'], n)
        if G is None or len(G) < CLIP:
            continue
        ch.setdefault((t, s['view']), []).append((s, G))
    for k, items in ch.items():
        A_, B_ = [], []
        for s, G in items:
            x = kt.norm2d(s['k2d'])
            out = np.zeros((len(x), 17, 3))
            with torch.no_grad():
                for st in range(0, max(1, len(x) - CLIP + 1), CLIP // 2):
                    seg = x[st:st + CLIP]
                    if len(seg) < CLIP:
                        seg = np.concatenate([seg, np.repeat(seg[-1:], CLIP - len(seg), 0)], 0)
                    o = mb(torch.from_numpy(seg[None]).to(kt.DEV)).cpu().numpy()[0]
                    m_ = min(CLIP, len(x) - st); out[st:st + m_] += o[:m_]
            P = out[:, H17, :]                       # -> COCO17 序（仅共同关节）
            P = P - P.mean(1, keepdims=True)
            Gc = G[:, C17, :] - G[:, C17, :].mean(1, keepdims=True)
            n = min(len(P), len(Gc))
            A_.append(P[:n].reshape(-1, 3)); B_.append(Gc[:n].reshape(-1, 3))
        A2 = np.concatenate(A_); B2 = np.concatenate(B_)
        sv, Rv = kabsch(A2 - A2.mean(0), B2 - B2.mean(0))
        errs, pas = [], []
        for s, G in items:
            x = kt.norm2d(s['k2d'])
            out = np.zeros((len(x), 17, 3))
            with torch.no_grad():
                for st in range(0, max(1, len(x) - CLIP + 1), CLIP // 2):
                    seg = x[st:st + CLIP]
                    if len(seg) < CLIP:
                        seg = np.concatenate([seg, np.repeat(seg[-1:], CLIP - len(seg), 0)], 0)
                    o = mb(torch.from_numpy(seg[None]).to(kt.DEV)).cpu().numpy()[0]
                    m_ = min(CLIP, len(x) - st); out[st:st + m_] += o[:m_]
            P = out[:, H17, :]
            P = ((P - P.mean(1, keepdims=True)) @ Rv.T) * sv
            Gc = G[:, C17, :] - G[:, C17, :].mean(1, keepdims=True)
            n = min(len(P), len(Gc))
            errs.append(np.linalg.norm(P[:n] - Gc[:n], axis=2).ravel())
            for f in range(0, n, 5):
                a = P[f] - P[f].mean(0); g = Gc[f] - Gc[f].mean(0)
                sc, R = kabsch(a, g)
                pas.append(np.linalg.norm(a @ R.T * sc - g, axis=1))
        rows.append(dict(take=t, mpjpe=float(np.median(np.concatenate(errs))),
                         pa=float(np.median(np.concatenate(pas)))))
    print('  ', t, flush=True)

print('\n=== %s  vs  【官方 Harmony4D GT3D】（held-out test）===' % NAME)
for lab, rr in (('全部 15 条', rows), ('9 条真 held-out', [r for r in rows if r['take'] not in LEAK])):
    if rr:
        print('  %-16s n=%2d  MPJPE 中位 %6.1f mm   PA %6.1f mm' %
              (lab, len(rr), np.median([r['mpjpe'] for r in rr]), np.median([r['pa'] for r in rr])))
