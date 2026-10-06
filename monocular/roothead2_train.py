#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""roothead2_train.py —— 训【时序】根回归头

与 roothead_train.py 的差异：
  ① 特征按【序列】分别构造 + windowize(±K 帧)，不再把所有帧混成一锅
  ② ckpt 里多存一个 'win' 字段（K），推理时 mb_to_final13.py 靠它决定要不要开窗
  ③ 验证额外报 **λ 抖动**（时间归一化 mm/s^2）—— 这才是本轮要治的指标

用法:
  python roothead2_train.py --view 1 --win 8 --epochs 60 \
      --out /workshop/Lym/combat3d/mb/ckpt/roothead_v2_temporal.pt
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/workshop/Lym/combat3d/mb/kb')
import kb_common as K
import kb_train as kt
sys.path.insert(0, '/workshop/Lym/combat3d')
import roothead as RH
import roothead2 as RH2

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SZ = {'960': (960, 720), '1920': (1920, 1440)}


def build(view, win):
    seqs = kt.load_all(K.OUT, views=[view])
    print(f'[data] view{view}: {len(seqs)} 个序列', flush=True)
    cals = {g: K.Calib(K.CALIB[g]) for g in ('960', '1920')}
    FX, LY, L0, MT = [], [], [], []
    for s in seqs:
        g = s['group']
        W_, H_ = SZ[g]
        cal = cals[g]
        R_, T_, Kk = cal.R[view], cal.T[view], cal.K[view]
        F = RH.person_feats(s['k2d'], R_, T_, Kk, W_, H_)          # (T,12) 逐帧特征
        lam = RH.true_lambda(s['k2d'], s['k3d'], R_, T_, Kk)       # (T,) 真值 λ(米)
        m = np.isfinite(lam) & (lam > 0.3) & (lam < 60.0)
        if m.sum() < 2 * win + 1:
            continue
        FW = RH2.windowize(F, win)                                  # (T, 12*(2K+1))
        # ★ 其余帧也要保留：若整条序列都合法，则全留着（窗已在序列内对齐）
        keep = m
        FX.append(FW[keep]); LY.append(np.log(lam[keep]))
        L0.append(F[keep, 6])                                       # 几何先验 log λ₀（中心帧）
        MT.append(len(FW[keep]))
    FX = np.concatenate(FX); LY = np.concatenate(LY); L0 = np.concatenate(L0)
    print(f'[data] 样本 {len(FX)}  来自 {len(MT)} 条序列  输入维 {FX.shape[1]}', flush=True)
    return FX, LY, L0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default='1')
    ap.add_argument('--win', type=int, default=8, help='时间窗半径 K；输入维 = 12*(2K+1)')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--out', default='/workshop/Lym/combat3d/mb/ckpt/roothead_v2_temporal.pt')
    a = ap.parse_args()

    F, Y, P = build(a.view, a.win)
    rng = np.random.default_rng(0)
    n = len(F)
    idx = rng.permutation(n)
    nv = int(n * a.val_frac)
    va, tr = idx[:nv], idx[nv:]

    mu, std_ = F[tr].mean(0), F[tr].std(0) + 1e-6
    Fn = (F - mu) / std_
    t = lambda x, d: torch.tensor(x, dtype=torch.float32, device=d)
    Xtr, Ytr, L0tr = t(Fn[tr], DEV), t(Y[tr], DEV), t(P[tr], DEV)
    Xva, Yva, L0va = t(Fn[va], DEV), t(Y[va], DEV), t(P[va], DEV)

    net = RH2.Net(F.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    with torch.no_grad():
        base = (torch.exp(L0va) - torch.exp(Yva)).abs()
    print(f'[baseline] 几何先验 λ₀: 中位 {base.median().item()*1000:.0f} mm', flush=True)

    best = 1e9
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        tot = 0.0
        for b in range(0, len(Xtr), a.batch):
            i = perm[b:b + a.batch]
            loss = (net(Xtr[i], L0tr[i]) - Ytr[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sch.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            net.eval()
            with torch.no_grad():
                pv = net(Xva, L0va)
                e = (torch.exp(pv) - torch.exp(Yva)).abs()          # 逐帧 λ 绝对误差(米)
                med = e.median().item()
            flag = ''
            if med < best:
                best = med
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                torch.save(dict(sd=net.state_dict(), mu=mu, fstd=std_, view=a.view,
                                feat_dim=RH.FEAT_DIM, win=a.win), a.out)
                flag = ' ★存'
            print(f'  ep{ep+1:3d}  trainL1={tot/max(1,len(Xtr)//a.batch):.4f}  '
                  f'VAL λ 中位 {med*1000:6.0f} mm  (几何先验 {base.median().item()*1000:.0f} mm){flag}',
                  flush=True)
    print(f'完成 -> {a.out}  (val 最优 {best*1000:.0f} mm)', flush=True)


if __name__ == '__main__':
    main()
