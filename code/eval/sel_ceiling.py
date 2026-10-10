#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sel_ceiling.py — 选人规则的**天花板分析**（回答"能改到多准"）

要分清两件事，它们的上限完全不同：
  (1) **候选池召回上限**：真受试者的框在候选池里出现过吗？
      → 若真值框在池里找不到 IoU>=0.5 的候选，那任何选人规则都救不了（检测召回问题）
  (2) **选人规则上限**：候选池里**有**真受试者时，规则能不能挑对？
      → 上限 = (1) 的召回率

同时扫两个可调旋钮，看各自能把上限推到哪：
  · conf 阈值（0.05 ~ 0.50）：低 conf 召回高但假检多；高 conf 假检少但可能漏真人
  · 取前 K 大（K=2/4/6/8/12/全部）：K=2 是现状；K 大了召回不再涨说明瓶颈在检测本身

输出：每个 (conf, K) 组合下的
  · **真值覆盖数**（2 人里覆盖到几个）→ 选人正确率的绝对上限
  · 候选池大小中位数（规则要处理的信噪比）
用法：
  python sel_ceiling.py --seq-root <.../016_mma4> --views 01,03,04,07,09,14 --start 1 --end 60
"""
import os, argparse, importlib.util
import numpy as np
import cv2
from collections import defaultdict

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
SUBJ = ('aria01', 'aria02')


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
    ap.add_argument('--confs', default='0.05,0.15,0.25,0.35,0.50')
    ap.add_argument('--ks', default='2,4,6,8,12,999')
    a = ap.parse_args()
    V = [v.zfill(2) for v in a.views.split(',')]
    confs = [float(x) for x in a.confs.split(',')]
    ks = [int(x) for x in a.ks.split(',')]

    sp = importlib.util.spec_from_file_location('rtd', B + '/code/label_pipeline/stages/1_detect/rtdetr_pipeline.py')
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    from ultralytics import RTDETR
    model = RTDETR(B + '/port/weights/rtdetr-l.pt')

    # 一次性统计：每 (view, frame) 存下全部候选（conf>=0.05）的框与 conf
    data = defaultdict(list)          # (v,fr) -> [(box, conf)]
    gts = {}
    for v in V:
        for fr in range(a.start, a.end + 1):
            img = cv2.imread(os.path.join(B, 'frames/15/4', v, '%06d.png' % fr))
            if img is None:
                continue
            gtf = os.path.join(a.seq_root, 'processed_data/bbox/cam%s/%05d.npy' % (v, fr))
            if not os.path.exists(gtf):
                continue
            gt = np.load(gtf, allow_pickle=True)
            gt = gt.item() if getattr(gt, 'dtype', None) == object else gt
            gts[(v, fr)] = {s: np.asarray(x, float).reshape(-1)[:4] for s, x in gt.items()}
            r = model.predict(img, conf=0.05, imgsz=a.imgsz, classes=[0], device=0, verbose=False)[0]
            if r.boxes is None:
                data[(v, fr)] = []
                continue
            bs = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            data[(v, fr)] = [(b, float(c)) for b, c in zip(bs, cf)]
        print('  view %s 采集完成' % v, flush=True)

    print('\n=== 天花板分析：imgsz=%d，帧 %d..%d，%d 视角 ===' % (a.imgsz, a.start, a.end, len(V)))
    print('%-8s %-8s | %-28s | %-22s | %s' %
          ('conf', '取前K', '真值覆盖 2/2, 1/2, 0/2', '覆盖2/2占比(=规则上限)', '候选池中位/最大'))
    print('-' * 108)
    for cf_th in confs:
        for K in ks:
            n2 = n1 = n0 = 0
            pool = []
            for key, boxes in data.items():
                gt = gts.get(key)
                if not gt:
                    continue
                b2 = [b for b, c in boxes if c >= cf_th]
                b2.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                b2 = b2[:K] if K < 999 else b2
                pool.append(len(b2))
                hit = 0
                for s in SUBJ:
                    if s not in gt:
                        continue
                    if any(iou(b, gt[s]) >= 0.5 for b in b2):
                        hit += 1
                if hit >= 2: n2 += 1
                elif hit == 1: n1 += 1
                else: n0 += 1
            tot = n2 + n1 + n0
            print('%-8.2f %-8s | %6d / %6d / %6d       | %6.1f%%                  | %.0f / %d'
                  % (cf_th, ('全部' if K >= 999 else str(K)), n2, n1, n0,
                     100.0 * n2 / max(tot, 1), np.median(pool) if pool else 0, max(pool) if pool else 0))


if __name__ == '__main__':
    main()
