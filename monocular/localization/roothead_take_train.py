#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""roothead_take_train.py —— 根回归头（支持【take 级划分】+【时间窗】）

为什么另写一个：roothead_train.py 是**逐帧随机 85/15** 划分，训练/验证帧来自同一批 take。
对逐帧 MLP 危害有限，但时序头(±K 帧窗口)会让相邻帧在训练/验证间共享窗口 ->
验证数字虚高。要做"单目泛化"的诚实评估，必须**按 take 划分**。

--split-json 指向 mb/data/_split.json（57 train / 8 val / 8 test，按分辨率分层）。
--win 0 = 逐帧头（复刻 v1 结构）；>0 = 时序头。

用法:
  python roothead_take_train.py --view 1 --win 0 --split-json /workshop/Lym/combat3d/mb/data/_split.json \
      --out /workshop/Lym/combat3d/mb/ckpt/rh_take_w0.pt
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, '/workshop/Lym/combat3d/mb/kb')
import kb_common as K
import kb_train as kt
sys.path.insert(0, '/workshop/Lym/combat3d')
import roothead as RH
import roothead2 as RH2

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SZ = {'960': (960, 720), '1920': (1920, 1440)}


def build(view, win):
    """返回 dict: take -> (FX, LY, L0)，窗口只在**同一 take 内**开。"""
    seqs = kt.load_all(K.OUT, views=[view])
    cals = {g: K.Calib(K.CALIB[g]) for g in ('960', '1920')}
    rec = {}
    for s in seqs:
        g = s['group']
        W_, H_ = SZ[g]
        cal = cals[g]
        R_, T_, Kk = cal.R[view], cal.T[view], cal.K[view]
        F = RH.person_feats(s['k2d'], R_, T_, Kk, W_, H_)
        lam = RH.true_lambda(s['k2d'], s['k3d'], R_, T_, Kk)
        m = np.isfinite(lam) & (lam > 0.3) & (lam < 60.0)
        if m.sum() < max(3, 2 * win + 1):
            continue
        FW = RH2.windowize(F, win)
        rec.setdefault(s['take'], []).append(
            (FW[m], np.log(lam[m]), F[m, 6]))
    return rec


def cat(chunks):
    if not chunks:
        return None
    F = np.concatenate([c[0] for c in chunks])
    Y = np.concatenate([c[1] for c in chunks])
    L = np.concatenate([c[2] for c in chunks])
    return F, Y, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default='1')
    ap.add_argument('--win', type=int, default=0)
    ap.add_argument('--split-json', default='/workshop/Lym/combat3d/mb/data/_split.json')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--out', default='/workshop/Lym/combat3d/mb/ckpt/rh_take_w0.pt')
    a = ap.parse_args()

    rec = build(a.view, a.win)
    sp = json.load(open(a.split_json))
    tr_t, va_t = set(sp['train']), set(sp['val'])
    print(f'[data] take 总数 {len(rec)}  train {len(tr_t)}  val {len(va_t)}', flush=True)

    tr = cat([c for t, v in rec.items() if t in tr_t for c in v])
    va = cat([c for t, v in rec.items() if t in va_t for c in v])
    assert tr and va, '划分后为空，检查 take 命名'
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
        tot = 0.0
        for b in range(0, len(Xtr), a.batch):
            i = perm[b:b + a.batch]
            loss = (net(Xtr[i], L0tr[i]) - Ytr[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
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
                                feat_dim=RH.FEAT_DIM, win=a.win,
                                split='take', val_takes=sorted(va_t)), a.out)
                flag = ' ★存'
            print(f'  ep{ep+1:3d}  trainL1={tot/max(1,len(Xtr)//a.batch):.4f}  '
                  f'VAL(take级) λ 中位 {med*1000:6.0f} mm{flag}', flush=True)
    print(f'完成 -> {a.out}  (take级验证最优 {best*1000:.0f} mm)', flush=True)


if __name__ == '__main__':
    main()
