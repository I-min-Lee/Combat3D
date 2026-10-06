#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""track_gt_audit.py — **轨迹级真值判别力分析**（用真值标每条轨迹，找正确的排序函数）

停止"猜打分函数"，改为量化回答两个问题：
  (1) 正确的两条轨迹（受试者）**在候选轨迹里吗**？→ 决定选人的**可达上界**
  (2) 它们在的话，**按什么量能被排到最前**？→ 决定排序函数该用什么

对每个 view：
  · 跑 v2 的检测/跟踪/合并管线（复用 detect_h4d_v2 的代码，不重写）
  · 每条轨迹算特征：cover / motion / area_med / prox(与别的轨迹的最小中位距离) …
  · 用真值给每条轨迹打标：**该轨迹的框与某受试者按中心口径命中的帧占比** → 命中率高即"是受试者"
  · 报告：受试者轨迹的排名（按面积 / 按与另一条受试者轨迹的 prox / 按 motion）

用法：
  python track_gt_audit.py --seq-root <.../016_mma4> --views 01,... --start 1 --end 60
"""
import os, sys, json, argparse, importlib.util
import numpy as np, cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
SUBJ = ('aria01', 'aria02')


def center_hit(db, gb):
    """中心口径：我们框中心落在 GT 框内 或 中心距 < 0.5×GT对角线"""
    cx, cy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
    gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
    if gb[0] <= cx <= gb[2] and gb[1] <= cy <= gb[3]:
        return True
    return np.linalg.norm([cx - gcx, cy - gcy]) / max(np.hypot(gb[2] - gb[0], gb[3] - gb[1]), 1e-9) < 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--nmax', type=int, default=12)
    ap.add_argument('--border-margin', type=float, default=0.10)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(B, 'code', 'adapters', 'harmony4d', 'pipeline'))
    import detect_h4d_v2 as V2
    from ultralytics import RTDETR
    rtd = V2.RTD
    rtd.MODEL_PATH = B + '/port/weights/rtdetr-l.pt'
    rtd.ROI_JSON = B + '/empty_roi.json'
    rtd.IMG_SIZE, rtd.PERSON_CONF, rtd.DEVICE = a.imgsz, a.conf, 0
    model = RTDETR(rtd.MODEL_PATH)

    print('=== 轨迹级真值判别力分析（imgsz=%d conf=%.2f topk=%d margin=%.2f，帧 %d..%d）==='
          % (a.imgsz, a.conf, a.topk, a.border_margin, a.start, a.end))
    for v in [x.zfill(2) for x in a.views.split(',')]:
        frames = list(range(a.start, a.end + 1))
        tr = V2.NSlotTracker(rtd, nmax=a.nmax)
        hist, gtmap = {}, {}
        for fr in frames:
            img = cv2.imread(os.path.join(B, 'frames/15/4', v, '%06d.png' % fr))
            if img is None:
                continue
            gtf = os.path.join(a.seq_root, 'processed_data/bbox/cam%s/%05d.npy' % (v, fr))
            if not os.path.exists(gtf):
                continue
            gt = np.load(gtf, allow_pickle=True)
            gt = gt.item() if getattr(gt, 'dtype', None) == object else gt
            gtmap[fr] = [np.asarray(gt[s], float).reshape(-1)[:4] for s in SUBJ if s in gt]
            cands = V2.candidates_k(rtd, img, model, v, a.topk, a.border_margin)
            for s in tr.step(cands):
                hist.setdefault(s.name, {})[fr] = list(s.box)
        T = len(frames)
        st = V2.track_stats(hist, T)
        hist, st, _ = V2.merge_tracks(hist, st, T)
        if not st:
            print('  view %s: 无轨迹' % v); continue
        # 每条轨迹的"受试者命中率"
        rows = []
        for nm, d in st.items():
            hit = 0; tot = 0
            for k, fr in enumerate(d['frames']):
                if fr not in gtmap:
                    continue
                tot += 1
                if any(center_hit(d['boxes'][k], g) for g in gtmap[fr]):
                    hit += 1
            d['subj_rate'] = hit / max(tot, 1)
            rows.append((nm, d))
        # 正确的两条 = subj_rate 最高的两条（且 >0.5）
        rows.sort(key=lambda z: -z[1]['subj_rate'])
        true_pair = [nm for nm, d in rows if d['subj_rate'] > 0.5][:2]
        # 当前打分（v2 的 pick_pair 逻辑）会选谁
        picked = V2.pick_pair(st, T, mincover=0.5, excl_iou=0.45)
        print('\n  view %s | 轨迹 %d 条 | 真值受试者轨迹 = %s' % (v, len(st), true_pair))
        print('     %-4s %-8s %-7s %-9s %-9s %s' % ('名', 'subj率', 'cover', 'motion', '面积中位', '被pick_pair选中?'))
        for nm, d in rows[:8]:
            print('     %-4s %-8.2f %-7.2f %-9.3f %-9.0f %s'
                  % (nm, d['subj_rate'], d['cover'], d['motion'], d['area_med'],
                     '★' if nm in picked else ''))
        if len(true_pair) == 2:
            ok = set(true_pair) == set(picked)
            print('     → 正确对 %s，我方选 %s  ⇒ %s' % (true_pair, picked, '✓命中' if ok else '✗错'))
        else:
            print('     → ⚠️ 正确的两条轨迹**不在候选里**（受试者轨迹不足 2 条）⇒ 选人不可达')


if __name__ == '__main__':
    main()
