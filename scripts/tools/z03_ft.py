#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★ baseline 同协议微调器：任意架构，**同一份 H4D 训练集 + 同一套 (take,view) 对齐评测**。

用法: z03_ft.py --arch vp3d|mixste --gpu N --epochs E --out <ckpt>
数据 = mb/data_h4d_offtri（1800 npz / 135 take），val = train01_hugging
训练目标 = 相机系、根相对、米制的 3D（与我们的 gt_native 同口径，只是不做 R 强制）
"""
import os, sys, json, glob, time, argparse
import numpy as np, torch
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
sys.path.insert(0, f'{B}/baselines/videopose3d')
import kb_common as K, kb_train as kt

ap = argparse.ArgumentParser()
ap.add_argument('--arch', required=True, choices=['vp3d', 'mixste'])
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--clip', type=int, default=243)
ap.add_argument('--stride', type=int, default=81)
ap.add_argument('--cpe', type=int, default=600, help='每 epoch clip 数')
ap.add_argument('--out', required=True)
a = ap.parse_args()
DEV = 'cuda'
W_IMG, H_IMG = 3840, 2160

# ---------------- 数据 ----------------
tr = kt.load_all(f'{B}/mb/data_h4d_offtri')
va = [s for s in tr if s['take'] == 'train01_hugging']
tr = [s for s in tr if s['take'] != 'train01_hugging']
print('[data] train seq %d  val seq %d' % (len(tr), len(va)), flush=True)


def norm_screen(X, w, h):
    return X / w * 2 - np.array([1, h / w])


def prep(seq):
    """-> 2D 归一化输入 (T,17,2), 3D 目标 (T,17,3) 相机系米、根相对"""
    G = kt.gt_camera_mm(seq, kt.get_cal(seq)) / 1000.0      # mm -> m
    G = G - G[:, 0:1, :]
    xy = seq['k2d'][:, :, :2].astype(np.float64).copy()
    v = seq['k2d'][:, :, 2] > 0.05
    for t in range(len(xy)):
        if v[t].sum() >= 4 and (~v[t]).any():
            xy[t][~v[t]] = xy[t][v[t]].mean(0)
    return norm_screen(xy, W_IMG, H_IMG).astype(np.float32), G.astype(np.float32), (seq['valid'] & v)


CLIP, STR = a.clip, a.stride
clips = []
for s in tr:
    n = len(s['k2d'])
    if n < CLIP:
        continue
    for st in range(0, n - CLIP + 1, STR):
        if s['valid'][st:st + CLIP].mean() > 0.5:
            clips.append((s, st))
print('[data] clips %d' % len(clips), flush=True)
rng = np.random.RandomState(0)

# ---------------- 模型 ----------------
if a.arch == 'vp3d':
    from common.model import TemporalModel
    import torch as _t
    net = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=True,
                        dropout=0.25, channels=1024).to(DEV)
    RF = net.receptive_field()
    # ★ 从【公开预训练权重】热启动（而非从零训）——这才是公平的"同协议微调"
    init = os.environ.get('FT_INIT')
    if init and os.path.exists(init):
        sd = _t.load(init, map_location='cpu', weights_only=False)
        sd = sd.get('model_pos', sd.get('model', sd)) if isinstance(sd, dict) else sd
        miss = net.load_state_dict(sd, strict=False)
        print('[init] 从公开权重热启动 %s  missing=%d unexpected=%d'
              % (os.path.basename(init), len(miss.missing_keys), len(miss.unexpected_keys)),
              flush=True)
elif a.arch == 'mixste':
    sys.path.insert(0, f'{B}/baselines/mixste')
    from common.model_cross import MixSTE2
    net = MixSTE2(243, 17, 2, 512, 8, 8).to(DEV)
    RF = 243
print('[model] %s  params %.1fM  RF=%d' % (a.arch, sum(p.numel() for p in net.parameters()) / 1e6, RF), flush=True)

opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
t_ = lambda x: torch.tensor(x, dtype=torch.float32, device=DEV)


def evaluate(seqs):
    net.eval()
    errs = []
    with torch.no_grad():
        for s in seqs:
            n = len(s['k2d'])
            if n < RF:
                continue
            X, G, ok = prep(s)
            out = np.zeros((n, 17, 3), np.float32); cnt = np.zeros(n)
            for st in range(0, max(1, n - RF + 1), RF // 2):
                seg = X[st:st + RF]
                if len(seg) < RF:
                    seg = np.concatenate([seg, np.repeat(seg[-1:], RF - len(seg), 0)], 0)
                o = net(t_1(seg[None]))[0].cpu().numpy() * 1000.0     # -> mm
                m_ = min(RF, n - st); out[st:st + m_] += o[:m_]; cnt[st:st + m_] += 1
            P = (out / np.maximum(cnt, 1)[:, None, None]) - 0
            Gmm = G * 1000.0
            e = np.linalg.norm(P - Gmm, axis=2)[ok]
            if ok.sum():
                errs.append(np.median(e))
    net.train()
    return float(np.median(errs)) if errs else float('nan')


t_1 = lambda x: torch.tensor(x, dtype=torch.float32, device=DEV)
best = 1e9
# ★ 预计算每个序列的 (X, G, ok)，避免每个 epoch 重算 gt_camera_mm
_pc = {}


def get_prep(s):
    k = (s['take'], s['view'], s['pid'])
    if k not in _pc:
        _pc[k] = prep(s)
    return _pc[k]


for ep in range(a.epochs):
    net.train()
    idx = rng.permutation(len(clips))[:a.cpe]
    tot, nb = 0.0, 0
    for b0 in range(0, len(idx), a.batch):
        bidx = idx[b0:b0 + a.batch]
        XB, GB = [], []
        for i in bidx:
            s, st = clips[i]
            X, G, _ = get_prep(s)
            XB.append(X[st:st + CLIP]); GB.append(G[st:st + CLIP])
        xb = t_1(np.stack(XB)); gb = t_1(np.stack(GB))
        pr = net(xb)
        loss = (pr - gb).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    sch.step()
    if (ep + 1) % 5 == 0 or ep == 0:
        v = evaluate(va)
        flag = ''
        if v < best:
            best = v
            torch.save(dict(sd=net.state_dict(), arch=a.arch, rf=RF), a.out)
            flag = ' ★存'
        print('  [%s ep %d/%d] loss=%.4f  val_med=%.1fmm%s' %
              (a.arch, ep + 1, a.epochs, tot / max(nb, 1), v, flag), flush=True)
print('[done] %s best %.1fmm -> %s' % (a.arch, best, a.out), flush=True)
