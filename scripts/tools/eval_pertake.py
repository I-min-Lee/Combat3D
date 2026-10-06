#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_pertake.py —— 逐 take 评测（test 集用）

对每个 take 报三个口径：
  ① PA 形状残差   —— 逐帧最优 (旋转+尺度+平移) 对齐后的误差 = 形状下界
  ② 对齐后端到端  —— 用【该 take 自己】的 clips 拟合 (s,R) 再评（部署时的合法做法：
                     在目标场次上先标一次）
  ③ 标定出的 R    —— 越大说明模型输出系离相机系越远

用法: eval_pertake.py <ckpt.pt> <data_dir> [--n 12]
"""
import sys, os, argparse
import numpy as np, torch
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_common as K, kb_train as kt

ap = argparse.ArgumentParser()
ap.add_argument('ckpt')
ap.add_argument('data')
ap.add_argument('--clip', type=int, default=121)
ap.add_argument('--n', type=int, default=12)
a = ap.parse_args()
CLIP = a.clip

seqs = kt.load_all(a.data)
takes = sorted(set(s['take'] for s in seqs))
m = kt.build_official(); m = kt.load_ckpt_weights(m, a.ckpt, 'full'); m.to(kt.DEV).eval()
print('ckpt = %s' % os.path.basename(a.ckpt))
print('%-16s %6s %9s %9s %8s %8s' % ('take', 'frames', 'PA形状', '端到端', 'R(deg)', 's'))


def ang(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


rows = []
for t in takes:
    sub = [s for s in seqs if s['take'] == t]
    nfr = sum(len(s['k2d']) for s in sub) // max(1, len(sub))
    if nfr < CLIP:
        continue
    # ② 用该 take 自己的 clips 标定
    kt.CAL.clear()
    try:
        kt.calibrate(m, sub, CLIP)
    except Exception as e:
        print('%-16s %6d  标定失败 %s' % (t, nfr, str(e)[:40])); continue
    r = kt.evaluate(m, sub, clip_len=CLIP, max_clips=a.n, log=lambda *x: None)
    Rs = [ang(v[1]) for v in kt.CAL.values()]
    ss = [v[0] for v in kt.CAL.values()]
    # ① PA 形状
    pa = []
    with torch.no_grad():
        for s in sub:
            x = kt.norm2d(s['k2d']); G = kt.gt_camera_mm(s, kt.get_cal(s))
            n = len(x)
            if n < CLIP:
                continue
            for st in np.linspace(0, n - CLIP, min(6, n - CLIP + 1)).astype(int):
                nat = m(torch.from_numpy(x[st:st + CLIP][None]).to(kt.DEV)).cpu().numpy()[0]
                for f in range(0, CLIP, 20):
                    ok = s['valid'][st + f] & (s['k2d'][st + f, :, 2] > 0.05)
                    if ok.sum() < 8:
                        continue
                    A = nat[f][ok]; Gg = G[st + f][ok]
                    Ac = A - A.mean(0); Gc = Gg - Gg.mean(0)
                    # Umeyama A->B: H = A^T B; SVD(H)=U D Vt; R = Vt^T S U^T;
                    # s = trace(D S) / ||A||_F^2   ← ★ 必须除以【平方和】不是均方
                    H = Ac.T @ Gc
                    U, D, Vt = np.linalg.svd(H)
                    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
                    R = Vt.T @ S @ U.T
                    sc = float(np.trace(np.diag(D) @ S) / max((Ac ** 2).sum(), 1e-9))
                    pa.append(np.linalg.norm(Ac @ R.T * sc - Gc, axis=1))
    pa_med = np.median(np.concatenate(pa)) if pa else float('nan')
    rows.append((t, nfr, pa_med, r['med_mm'], float(np.median(Rs)), float(np.median(ss))))
    print('%-16s %6d %9.1f %9.1f %8.1f %8.1f'
          % (t, nfr, pa_med, r['med_mm'], np.median(Rs), np.median(ss)), flush=True)

if rows:
    arr = np.array([[x[2], x[3]] for x in rows])
    print('-' * 62)
    print('%-16s %6s %9.1f %9.1f' % ('中位(%d条)' % len(rows),
                                     '', np.median(arr[:, 0]), np.median(arr[:, 1])))
