#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""close_contact_audit.py — **贴身帧审计**：框合并到底吃掉多少选人正确率

问三个问题（全部用 Harmony4D 真值回答）：
  1. 两受试者在真值上就贴在一起（GT-GT IoU ≥ 0.3）的帧占多少？  → 贴身有多常见
  2. 在这些**贴身帧**上，候选池里**能不能同时找到覆盖两人的独立框**？
     （每个受试者各有一个 IoU≥0.5 的候选，且这两个候选彼此 IoU<0.5）
  3. 在**非贴身帧**上，同样的覆盖率是多少？

结论用途：把"选人正确率"拆成
    上界 = P(非贴身)·覆盖(非贴身) + P(贴身)·覆盖(贴身)
  覆盖(贴身) 低 = **检测器在贴身帧上给不出两个人的独立框** → 选人规则再强也白搭
  覆盖(非贴身) 低 = 选人规则的问题
"""
import os, argparse, importlib.util
import numpy as np, cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--contact-iou', type=float, default=0.3)
    a = ap.parse_args()
    V = [v.zfill(2) for v in a.views.split(',')]
    sp = importlib.util.spec_from_file_location('rtd', B + '/code/stage1_detect/rtdetr_pipeline.py')
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    from ultralytics import RTDETR
    model = RTDETR(B + '/port/weights/rtdetr-l.pt')
    SUBJ = ('aria01', 'aria02')

    stat = {}
    for v in V:
        n = n_contact = 0
        cov_c = cov_n = 0          # 贴身帧/非贴身帧里"两人各自都有独立候选"的帧数
        for fr in range(a.start, a.end + 1):
            gtf = os.path.join(a.seq_root, 'processed_data/bbox/cam%s/%05d.npy' % (v, fr))
            if not os.path.exists(gtf):
                continue
            gt = np.load(gtf, allow_pickle=True)
            gt = gt.item() if getattr(gt, 'dtype', None) == object else gt
            gt = [np.asarray(gt[s], float).reshape(-1)[:4] for s in SUBJ if s in gt]
            if len(gt) < 2:
                continue
            img = cv2.imread(os.path.join(B, 'frames/15/4', v, '%06d.png' % fr))
            if img is None:
                continue
            r = model.predict(img, conf=a.conf, imgsz=a.imgsz, classes=[0], device=0, verbose=False)[0]
            cand = [] if r.boxes is None else [b for b in r.boxes.xyxy.cpu().numpy()]
            contact = iou(gt[0], gt[1]) >= a.contact_iou
            n += 1
            if contact:
                n_contact += 1
            ok = True
            for g in gt:                                  # 每个受试者都要有独立候选
                hit = [c for c in cand if iou(c, g) >= 0.5]
                if not hit:
                    ok = False; break
            if ok:
                # 且这两个候选彼此要能分开（不是同一个框同时覆盖两人）
                h0 = [c for c in cand if iou(c, gt[0]) >= 0.5]
                h1 = [c for c in cand if iou(c, gt[1]) >= 0.5]
                sep = any(iou(x, y) < 0.5 for x in h0 for y in h1)
                ok = sep
            if contact:
                cov_c += int(ok)
            else:
                cov_n += int(ok)
        stat[v] = (n, n_contact, cov_c, cov_n)
        print('  view %s 采完' % v, flush=True)

    print('\n=== 贴身帧审计（imgsz=%d conf=%.2f，帧 %d..%d）===' % (a.imgsz, a.conf, a.start, a.end))
    print('%-6s | %6s | %-22s | %-30s | %s' %
          ('view', '帧数', '贴身帧(GT-GT IoU>=%.1f)' % a.contact_iou, '贴身帧里两人都有独立候选', '非贴身帧里两人都有独立候选'))
    print('-' * 112)
    T = [0, 0, 0, 0]
    for v in V:
        n, nc, cc, cn = stat[v]
        T[0] += n; T[1] += nc; T[2] += cc; T[3] += cn
        print('%-6s | %6d | %5d (%5.1f%%)          | %5d / %-5d = %5.1f%%              | %5d / %-5d = %5.1f%%'
              % (v, n, nc, 100.0 * nc / max(n, 1), cc, nc, 100.0 * cc / max(nc, 1),
                 cn, n - nc, 100.0 * cn / max(n - nc, 1)))
    n, nc, cc, cn = T
    print('-' * 112)
    print('整体  | %6d | %5d (%5.1f%%)          | %5d / %-5d = %5.1f%%              | %5d / %-5d = %5.1f%%'
          % (n, nc, 100.0 * nc / max(n, 1), cc, nc, 100.0 * cc / max(nc, 1),
             cn, n - nc, 100.0 * cn / max(n - nc, 1)))
    print('\n可达到的选人上界 = P(非贴身)*覆盖(非贴身) + P(贴身)*覆盖(贴身) = %.1f%%*%.1f%% + %.1f%%*%.1f%% = %.1f%%'
          % (100.0 * (n - nc) / n, 100.0 * cn / max(n - nc, 1),
             100.0 * nc / n, 100.0 * cc / max(nc, 1),
             100.0 * ((n - nc) / n * cn / max(n - nc, 1) + nc / n * cc / max(nc, 1))))


if __name__ == '__main__':
    main()
