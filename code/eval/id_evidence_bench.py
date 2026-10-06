#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""id_evidence_bench.py — B 线：身份证据源横评

  B1 颜色判据（red_black_cls.pt，需着装约定）在 Harmony4D 上的失效形态
  B2 体型/骨长签名（数据无关）的可分离性 —— 判别比 = 两人间距离 / 人内时序变异

用法：
  python id_evidence_bench.py --seq-root <.../016_mma4> --views 01,03,04,07,09,14
"""
import os, sys, json, glob, argparse
import numpy as np
import cv2

B = '/workshop/Lym/combat3d'
SUBJ = ('aria01', 'aria02')


def sec_b1(a, V):
    """颜色判据：对真值框裁图跑 red_black_cls，看置信度分布 / 弃权率 / 强取 argmax 的倾向"""
    from ultralytics import YOLO
    m = YOLO(B + '/port/weights/red_black_cls.pt')
    names = list(m.names.values())
    CLS_CONF = 0.80
    stat = {s: {'n': 0, 'mx': [], 'red': 0, 'black': 0, 'abst': 0} for s in SUBJ}
    for v in V:
        for fr in range(a.start, a.end):
            gtf = os.path.join(a.seq_root, 'processed_data/bbox/cam%s/%05d.npy' % (v, fr))
            fp = os.path.join(B, 'frames/15/4', v, '%06d.png' % fr)
            if not (os.path.exists(gtf) and os.path.exists(fp)):
                continue
            gt = np.load(gtf, allow_pickle=True)
            gt = gt.item() if getattr(gt, 'dtype', None) == object else gt
            img = cv2.imread(fp)
            if img is None:
                continue
            H, W = img.shape[:2]
            for s in SUBJ:
                if s not in gt:
                    continue
                x1, y1, x2, y2 = [int(t) for t in np.asarray(gt[s], float).reshape(-1)[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 < 20 or y2 - y1 < 40:
                    continue
                r = m.predict(img[y1:y2, x1:x2], imgsz=224, verbose=False)[0]
                p = r.probs.data.cpu().numpy()
                d = {names[i]: float(p[i]) for i in range(len(names))}
                mx = max(d.get('red', 0.0), d.get('black', 0.0))
                st = stat[s]
                st['n'] += 1
                st['mx'].append(mx)
                if mx >= CLS_CONF:
                    if d.get('red', 0) > d.get('black', 0):
                        st['red'] += 1
                    else:
                        st['black'] += 1
                else:
                    st['abst'] += 1
    print('=== B1 颜色判据（red_black_cls.pt, CLS_CONF=0.80）在 Harmony4D 上 ===')
    print('%-10s | %6s | %-24s | %-30s | %s' % ('真值受试者', '帧数',
          'max(red,black) 中位/p90', '有效证据(>=0.80) / 弃权', '强取 argmax 的倾向'))
    for s in SUBJ:
        st = stat[s]
        if not st['n']:
            continue
        mx = np.array(st['mx'])
        eff = st['n'] - st['abst']
        print('%-10s | %6d | %10.3f / %10.3f | %5d (%5.1f%%) / %5.1f%%     | red %d / black %d'
              % (s, st['n'], np.median(mx), np.percentile(mx, 90), eff,
                 100.0 * eff / st['n'], 100.0 * st['abst'] / st['n'], st['red'], st['black']))
    print('  判读：弃权率接近 100%，或两受试者倾向无差别 ⇒ 该判据在此无信号（依赖着装约定）')


def sec_b2(a):
    """体型/骨长签名：从 GT SMPL 生成 b25 关节，算归一化骨长向量，比较两人间 vs 人内时序变异"""
    sys.path.insert(0, B + '/port/EasyMocap-master')
    import torch
    from easymocap.bodymodel.smpl import SMPLModel
    m = SMPLModel(model_path=B + '/port/smpl/smpl.pkl',
                  regressor_path=B + '/port/smpl/J_regressor_body25.npy',
                  device='cuda', use_pose_blending=True, use_joints=True, NUM_SHAPES=10)
    BONES = [(5, 6), (5, 7), (6, 8), (2, 3), (3, 4), (12, 13), (13, 14),
             (9, 10), (10, 11), (1, 8), (0, 1), (5, 12), (6, 9), (2, 5), (1, 2)]
    sig = {s: [] for s in SUBJ}
    for fr in range(a.start, a.end, 5):
        f = os.path.join(a.seq_root, 'processed_data/smpl/%05d.npy' % fr)
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True).item()
        for s in SUBJ:
            g = d.get(s)
            if g is None:
                continue
            p = dict(shapes=torch.tensor(np.asarray(g['betas'], np.float32)[None]).cuda(),
                     poses=torch.tensor(np.asarray(g['body_pose'], np.float32)[None]).cuda(),
                     Rh=torch.tensor(np.asarray(g['global_orient'], np.float32)[None]).cuda(),
                     Th=torch.tensor(np.asarray(g['transl'], np.float32)[None]).cuda())
            with torch.no_grad():
                kp = m(return_verts=False, **p)[0].cpu().numpy()
            bl = np.array([np.linalg.norm(kp[i, :3] - kp[j, :3]) for i, j in BONES])
            sig[s].append(bl / max(bl.sum(), 1e-9))     # 归一化 ⇒ 与身高/尺度无关
    A = np.stack(sig[SUBJ[0]])
    Bb = np.stack(sig[SUBJ[1]])
    print('')
    print('=== B2 体型/骨长签名（GT SMPL 生成的 b25 骨长，归一化）===')
    print('  采样帧数: %s=%d, %s=%d' % (SUBJ[0], len(A), SUBJ[1], len(Bb)))
    mA, mB = A.mean(0), Bb.mean(0)
    between = float(np.linalg.norm(mA - mB))
    within = float(0.5 * (np.linalg.norm(A - mA, axis=1).mean() + np.linalg.norm(Bb - mB, axis=1).mean()))
    ratio = between / max(within, 1e-9)
    print('  两人签名距离(间) = %.5f ; 人内时序变异(均) = %.5f' % (between, within))
    print('  ★ 判别比 = 间/内 = %.2f  %s' % (ratio, '(>3 表示可用)' if ratio > 3 else '(偏弱，需与其它证据融合)'))
    acc = 0
    for i in range(len(A)):
        d0 = np.linalg.norm(A - A[i], axis=1).min() if len(A) > 1 else 0
        d1 = np.linalg.norm(A[i] - Bb, axis=1).min()
        acc += int(d0 < d1)
    for i in range(len(Bb)):
        d0 = np.linalg.norm(Bb[i] - A, axis=1).min()
        d1 = np.linalg.norm(Bb - Bb[i], axis=1).min() if len(Bb) > 1 else 0
        acc += int(d1 < d0)
    print('  最近邻分类正确率 = %.1f%%（两类各 %d 帧）' % (100.0 * acc / (len(A) + len(Bb)), len(A)))
    diff = np.abs(mA - mB)
    print('  逐骨差异(归一化) 前6条 = %s ; 中位 %.4f vs 人内 %.4f' %
          (np.round(diff[:6], 4), np.median(diff), within))


def sec_b3(a, V):
    """通用外观嵌入（CLIP 视觉塔，timm 走 hf-mirror）：与 B2 同口径，另加**跨视角迁移**"""
    import torch
    import timm
    dev = 'cuda'
    m = timm.create_model('vit_base_patch32_clip_224.openai', pretrained=True, num_classes=0).eval().to(dev)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    def embed(crops):
        if not crops:
            return None
        x = np.stack(crops).astype(np.float32) / 255.0
        t = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
        t = (t - mean) / std
        t = torch.nn.functional.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            f = m(t)
        f = torch.nn.functional.normalize(f, dim=1)
        return f.cpu().numpy()

    E = {v: {s: [] for s in SUBJ} for v in V}
    for v in V:
        for fr in range(a.start, a.end, 5):
            gtf = os.path.join(a.seq_root, 'processed_data/bbox/cam%s/%05d.npy' % (v, fr))
            fp = os.path.join(B, 'frames/15/4', v, '%06d.png' % fr)
            if not (os.path.exists(gtf) and os.path.exists(fp)):
                continue
            gt = np.load(gtf, allow_pickle=True)
            gt = gt.item() if getattr(gt, 'dtype', None) == object else gt
            img = cv2.imread(fp)
            if img is None:
                continue
            H, W = img.shape[:2]
            for s in SUBJ:
                if s not in gt:
                    continue
                x1, y1, x2, y2 = [float(t) for t in np.asarray(gt[s], float).reshape(-1)[:4]]
                dx, dy = (x2 - x1) * 0.15, (y2 - y1) * 0.15        # 与你 conv 的裁图 pad 一致
                x1i, y1i = int(max(0, x1 - dx)), int(max(0, y1 - dy))
                x2i, y2i = int(min(W, x2 + dx)), int(min(H, y2 + dy))
                if x2i - x1i < 24 or y2i - y1i < 48:
                    continue
                crop = img[y1i:y2i, x1i:x2i]
                # ★ CLIP 标准预处理：短边缩到 224 再中心裁 224×224。
                #   直接 resize 成 224×224 会把细长人框压成 1:1，毁掉外观信息（踩过）
                hh, ww = crop.shape[:2]
                sc = 224.0 / max(min(hh, ww), 1)
                nh, nw = max(224, int(round(hh * sc))), max(224, int(round(ww * sc)))
                r2 = cv2.resize(crop, (nw, nh))
                y0, x0 = (nh - 224) // 2, (nw - 224) // 2
                E[v][s].append(r2[y0:y0 + 224, x0:x0 + 224])
        for s in SUBJ:
            E[v][s] = embed(E[v][s])
    print('')
    print('=== B3 通用外观嵌入（CLIP ViT-B/32 视觉塔，timm+hf-mirror）===')
    print('  采样：%d 视角 × 每 %d 帧；各视角样本数 %s'
          % (len(V), 5, {v: len(E[v][SUBJ[0]]) for v in V}))
    # 类内 / 类间（全视角合并）
    A = np.concatenate([E[v][SUBJ[0]] for v in V])
    Bb = np.concatenate([E[v][SUBJ[1]] for v in V])
    mA, mB = A.mean(0), Bb.mean(0)
    between = float(np.linalg.norm(mA - mB))
    within = float(0.5 * (np.linalg.norm(A - mA, axis=1).mean() + np.linalg.norm(Bb - mB, axis=1).mean()))
    print('  类间距离 = %.4f ; 类内单样本变异 = %.4f ; ★判别比 = %.2f  %s'
          % (between, within, between / max(within, 1e-9),
             '(>3 可用)' if between / max(within, 1e-9) > 3 else '(偏弱)'))
    # 逐帧最近邻（模板=各自类均值，模拟实际使用）
    acc = 0; n = 0
    for X, y in ((A, 0), (Bb, 1)):
        d0 = np.linalg.norm(X - mA, axis=1); d1 = np.linalg.norm(X - mB, axis=1)
        acc += int(((d0 < d1) == (y == 0)).sum()); n += len(X)
    print('  同视角逐帧判定（模板=全视角类均值）正确率 = %.1f%% (%d/%d)'
          % (100.0 * acc / n, acc, n))
    # ★ 跨视角迁移：用视角 v0 的类均值当模板，去判其他视角的帧
    v0 = V[0]
    tA, tB = E[v0][SUBJ[0]].mean(0), E[v0][SUBJ[1]].mean(0)
    print('  ★ 跨视角迁移（模板取自 view %s，测试其他视角）:' % v0)
    tot = ok = 0
    for v in V[1:]:
        a2, b2 = E[v][SUBJ[0]], E[v][SUBJ[1]]
        d0 = np.linalg.norm(a2 - tA, axis=1); d1 = np.linalg.norm(a2 - tB, axis=1)
        right = int((d0 < d1).sum())
        d0b = np.linalg.norm(b2 - tA, axis=1); d1b = np.linalg.norm(b2 - tB, axis=1)
        right += int((d1b < d0b).sum())
        print('     view %s : %.1f%% (%d/%d)' % (v, 100.0 * right / (len(a2) + len(b2)),
              right, len(a2) + len(b2)))
        ok += right; tot += len(a2) + len(b2)
    print('     ⇒ 跨视角平均 %.1f%% (%d/%d)' % (100.0 * ok / max(tot, 1), ok, tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=61)
    ap.add_argument('--only', default='all', choices=['all', 'b1', 'b2', 'b3'])
    a = ap.parse_args()
    V = [v.zfill(2) for v in a.views.split(',')]
    if a.only in ('all', 'b1'):
        sec_b1(a, V)
    if a.only in ('all', 'b2'):
        sec_b2(a)
    if a.only in ('all', 'b3'):
        sec_b3(a, V)


if __name__ == '__main__':
    main()
