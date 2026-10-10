#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""roothead_train.py —— 训练【第1视角专用】的单目根回归头

输入特征（全部做分辨率归一化，960/1920 两组可混训）：
    骨盆像素(u/W, v/H) · log(人体span/W) · 2D均值conf · 可见关节占比
    · 最低可见关节 v/H · **几何先验 log(λ₀)** · 相机高 · 相机俯角
    · 归一化焦距 fx/W · 主点(cx/W, cy/H)
输出：log(λ) —— 骨盆沿相机射线的深度
监督：三角化的世界根投到该射线上（★ 5视角只当标注员，**不是输入**）

用法:
  python roothead_train.py --view 1 --epochs 60 --out /workshop/Lym/combat3d/mb/ckpt/roothead_v1.pt
"""
import os, sys, json, glob, argparse, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/workshop/Lym/combat3d/mb/kb')
import kb_common as K
import kb_train as kt
sys.path.insert(0, '/workshop/Lym/combat3d')
import roothead as RH

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SZ = {'960': (960, 720), '1920': (1920, 1440)}


def build(view):
    seqs = kt.load_all(K.OUT, views=[view])
    print(f'[data] view{view}: {len(seqs)} 个序列', flush=True)
    cals = {g: K.Calib(K.CALIB[g]) for g in ('960', '1920')}
    F, Y, meta, prior = [], [], [], []
    for s in seqs:
        g = s['group']
        W, H = SZ[g]
        cal = cals[g]
        Rw2c, Tv, Kk = cal.R[view], cal.T[view], cal.K[view]
        X = RH.person_feats(s['k2d'], Rw2c, Tv, Kk, W, H)
        lam = RH.true_lambda(s['k2d'], s['k3d'], Rw2c, Tv, Kk)
        m = np.isfinite(lam) & (lam > 0.3) & (lam < 60.0)
        if m.sum() == 0:
            continue
        F.append(X[m]); Y.append(np.log(lam[m]))
        prior.append(X[m, 6])                       # 几何先验(log λ₀)作基线对比
        meta.append((s['take'], s['pid']))
    F = np.concatenate(F); Y = np.concatenate(Y); P = np.concatenate(prior)
    print(f'[data] 样本 {len(F)}  来自 {len(meta)} 条序列', flush=True)
    return F, Y, P, meta


class Net(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.LayerNorm(h),
                               nn.Linear(h, h), nn.ReLU(), nn.LayerNorm(h),
                               nn.Linear(h, h // 2), nn.ReLU(),
                               nn.Linear(h // 2, 1))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)   # 零初始化 -> 起点=几何先验

    def forward(self, x, lam0_log):
        return lam0_log + self.f(x).squeeze(-1)      # ★ 残差：学的是对几何先验的修正


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default='1')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--out', default='/workshop/Lym/combat3d/mb/ckpt/roothead_v1.pt')
    a = ap.parse_args()

    F, Y, P, meta = build(a.view)
    rng = np.random.default_rng(0)
    n = len(F)
    idx = rng.permutation(n)
    nv = int(n * a.val_frac)
    va, tr = idx[:nv], idx[nv:]

    mu, std_ = F[tr].mean(0), F[tr].std(0) + 1e-6
    Fn = (F - mu) / std_

    Xtr = torch.tensor(Fn[tr], dtype=torch.float32, device=DEV)
    Ytr = torch.tensor(Y[tr], dtype=torch.float32, device=DEV)
    L0tr = torch.tensor(P[tr], dtype=torch.float32, device=DEV)
    Xva = torch.tensor(Fn[va], dtype=torch.float32, device=DEV)
    Yva = torch.tensor(Y[va], dtype=torch.float32, device=DEV)
    L0va = torch.tensor(P[va], dtype=torch.float32, device=DEV)

    net = Net(F.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    # 基线：几何先验本身
    with torch.no_grad():
        base = (torch.exp(L0va) - torch.exp(Yva)).abs()
    print(f'[baseline] 几何先验 λ₀: 中位 {base.median().item()*1000:.0f} mm  '
          f'均值 {base.mean().item()*1000:.0f} mm', flush=True)

    best = 1e9
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        tot = 0.0
        for b in range(0, len(Xtr), a.batch):
            i = perm[b:b + a.batch]
            pred = net(Xtr[i], L0tr[i])
            loss = (pred - Ytr[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sch.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            net.eval()
            with torch.no_grad():
                pv = net(Xva, L0va)
                e = (torch.exp(pv) - torch.exp(Yva)).abs()
                med, mean = e.median().item(), e.mean().item()
                rel = (e / torch.exp(Yva)).median().item()
            flag = ''
            if med < best:
                best = med
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                torch.save(dict(sd=net.state_dict(), mu=mu, fstd=std_, view=a.view,
                                feat_dim=F.shape[1]), a.out)
                flag = ' ★存'
            print(f'  ep{ep+1:3d}  trainL1={tot/max(1,len(Xtr)//a.batch):.4f}  '
                  f'VAL 中位 {med*1000:6.0f} mm  均值 {mean*1000:6.0f} mm  '
                  f'相对 {rel*100:5.1f}%  (几何先验 {base.median().item()*1000:.0f} mm){flag}', flush=True)
    print(f'完成 -> {a.out}  (val 最优中位 {best*1000:.0f} mm，几何先验基线 {base.median().item()*1000:.0f} mm)', flush=True)


if __name__ == '__main__':
    main()
