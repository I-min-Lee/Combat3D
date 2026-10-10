#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  ✅  UNIVERSAL LAYER — REUSE AS-IS ACROSS SCENES
#
#  这是 Combat3D-Label 五层里【通用】的那几层之一（2D 提取 / 三角化 / SMPL 拟合 /
#  单目 lifter）。两个域上的实测：换场景时这几层【一行未改】。
#
#  与它对照的是【场景相关】的检测框层与身份层 —— 那两个必须换算法，
#  文件头带 "SCENE-SPECIFIC LAYER" 横幅。层契约见 code/adapters/README.md。
# =============================================================================
"""roothead_h4d_train.py —— Harmony4D 版根回归头（时序窗 / 逐帧两种）

与 kendo 版 `roothead_take_train.py` 的差异（3 处）：
  1. 数据 = `mb/data_h4d_offtri`（干净标签），取 view ∈ {01,03,04,07,09,14}
  2. 标定按 `h4d_<tag>` 惰性加载；图像尺寸 3840×2160
  3. ★ k3d 先乘该场次的米制尺度（`h4d_metric_scale.json`）—— 否则 λ 真值在
     不同场次间差最多 2.73 倍（COLMAP 单目 SfM 的尺度是任意的），头根本学不动。
     同 [[h4d-world-frame-scale-npy-bug]] 里 `gt_camera_mm` 的修法。

take 级划分（避免时序窗跨训练/验证泄漏）。

用法:
  RH_DATA=.../mb/data_h4d_offtri python roothead_h4d_train.py \
      --view 01 --win 8 --out .../mb/ckpt/rh_h4d_v01_w8.pt
"""
import os, sys, json, argparse
import numpy as np
import torch

B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_common as K
import kb_train as kt
import roothead as RH
import roothead2 as RH2

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = os.environ.get('RH_DATA', f'{B}/mb/data_h4d_offtri')
SCALE = json.load(open(f'{B}/h4d_metric_scale.json'))
H4D_WH = (3840, 2160)


def build(view, win):
    """返回 dict: take -> (FX, LY, L0)；窗口只在同一 take 内开。"""
    seqs = kt.load_all(DATA, views=[view])
    cals = {}
    rec = {}
    for s in seqs:
        g = s['group']
        if not g.startswith('h4d_'):
            continue
        if g not in cals:
            cals[g] = K.Calib(K.CALIB[g])
        cal = cals[g]
        W_, H_ = H4D_WH
        R_, T_, Kk = cal.R[view], cal.T[view], cal.K[view]
        ms = SCALE.get(s['take'], 1.0)                 # ★ 米制尺度
        F = RH.person_feats(s['k2d'], R_, T_, Kk, W_, H_)
        # ★ 米制化只在【相机系深度】上乘尺度：缩 3D 而不缩外参 T 会把变换算错
        # ★2026-10-04 修 bug：k3d 不能先乘 ms —— R_/T_ 仍是 COLMAP 单位，几何不自洽。
        #   正确写法（见 docs/PITFALLS.md）：true_lambda(k2d, k3d, R, T) * ms。
        #   实测 train01_hugging view04：旧写法 λ=3.20/3.42；正确写法 1.99/2.07；真值 2.16 ✓
        lam = RH.true_lambda(s['k2d'], s['k3d'].astype(np.float64), R_, T_, Kk) * ms
        m = np.isfinite(lam) & (lam > 0.3) & (lam < 60.0)
        if m.sum() < max(3, 2 * win + 1):
            continue
        FW = RH2.windowize(F, win)
        rec.setdefault(s['take'], []).append((FW[m], np.log(lam[m]), F[m, 6]))
    return rec


def cat(chunks):
    if not chunks:
        return None
    return (np.concatenate([c[0] for c in chunks]),
            np.concatenate([c[1] for c in chunks]),
            np.concatenate([c[2] for c in chunks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default='01')
    ap.add_argument('--win', type=int, default=0)
    ap.add_argument('--holdout', default='train01_hugging')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--out', default=f'{B}/mb/ckpt/rh_h4d.pt')
    a = ap.parse_args()

    rec = build(a.view, a.win)
    if not rec:
        print('!! 没有数据（检查 RH_DATA / --view）'); return
    va_t = {a.holdout}
    tr = cat([c for t, v in rec.items() if t not in va_t for c in v])
    va = cat([c for t, v in rec.items() if t in va_t for c in v])
    assert tr and va, '划分后为空：holdout=%s，takes=%s' % (a.holdout, sorted(rec))
    print(f'[data] take {len(rec)}  train {len(rec)-len(va_t)}  val {len(va_t)}')
    print(f'[data] 训练帧 {len(tr[0])}  验证帧 {len(va[0])}  输入维 {tr[0].shape[1]}', flush=True)

    mu, std_ = tr[0].mean(0), tr[0].std(0) + 1e-6
    t_ = lambda x, d: torch.tensor(x, dtype=torch.float32, device=d)
    Xtr, Ytr, L0tr = t_((tr[0] - mu) / std_, DEV), t_(tr[1], DEV), t_(tr[2], DEV)
    Xva, Yva, L0va = t_((va[0] - mu) / std_, DEV), t_(va[1], DEV), t_(va[2], DEV)

    net = RH2.Net(tr[0].shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    with torch.no_grad():
        base = (torch.exp(L0va) - torch.exp(Yva)).abs()
    print(f'[baseline] 几何先验 λ₀ 在 val: 中位 {base.median().item()*1000:.0f} mm', flush=True)

    best = 1e9
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        tot, nb = 0.0, 0
        for b in range(0, len(Xtr), a.batch):
            i = perm[b:b + a.batch]
            loss = (net(Xtr[i], L0tr[i]) - Ytr[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sch.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            net.eval()
            with torch.no_grad():
                e = (torch.exp(net(Xva, L0va)) - torch.exp(Yva)).abs()
                med = e.median().item()
            flag = ''
            if med < best:
                best = med
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                torch.save(dict(sd=net.state_dict(), mu=mu, fstd=std_, view=a.view,
                                feat_dim=RH.FEAT_DIM, win=a.win, split='take',
                                val_takes=sorted(va_t)), a.out)
                flag = ' ★存'
            print(f'  ep{ep+1:3d}  trainL1={tot/max(nb,1):.4f}  '
                  f'VAL(take级) λ 中位 {med*1000:6.0f} mm{flag}', flush=True)
    print(f'完成 -> {a.out}  (take级验证最优 {best*1000:.0f} mm)', flush=True)


if __name__ == '__main__':
    main()
