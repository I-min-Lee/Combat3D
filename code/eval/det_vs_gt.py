#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""det_vs_gt.py — 行 B：**用 Harmony4D 的真值框给我们的检测层打分**

Harmony4D 的 `processed_data/bbox/camNN/%05d.npy` = `{subject: [x1,y1,x2,y2]}`，
是**逐帧、逐人、带身份**的真值框 —— 这正是我们自采数据从来没有的尺子。

报什么（按你要论证的东西排序）：
  1. ★**选人正确率**：该帧选出的框命中几个真值受试者 —— 2/2、1/2、0/2 的帧占比
     （你的主张①说"真实场景里更致命的是选人错误"，这里第一次可量化）
  2. **检测 F1@IoU0.5**：一对一无重复匹配下的 precision/recall/F1
  3. **IoU 分布 / 中心误差**（中心误差按 GT 框对角线归一，对框外扩不敏感）
  4. **身份一致率**：匹配上的对里 `group_id=0` 落到哪个 subject —— 报"多数派占比"
     与**翻转次数**（逐帧跳变），即检测层的身份稳定性

注意：我们的框经 `expand_head` 外扩（`BOX_OUT=0.02/TOP_EXTRA=0.05/BOTTOM_EXTRA=0.03`），
会比 GT 稍大 → IoU 略降是预期的；所以同时给"中心误差/框对角线"这个与尺度无关的口径。

用法：
  python det_vs_gt.py --det <det_ours> --seq-root <.../016_mma4> \
      --views 01,03,04,07,09,14 --start 1 --end 741 --tag 016_mma4 [--iou 0.5] [--csv out.csv]
"""
import os, json, glob, argparse
import numpy as np
from scipy.optimize import linear_sum_assignment

SUBJ = ['aria01', 'aria02']


def unexpand(b):
    """反解 expand_head：宽 ×1.04、高 ×1.12（BOX_OUT=0.02, TOP_EXTRA=0.05, BOTTOM_EXTRA=0.03）
    注意：原函数在图像边界有 clamp，贴边时反解不精确（本口径当作近似）"""
    x1, y1, x2, y2 = b
    w = (x2 - x1) / 1.04
    h = (y2 - y1) / 1.12
    return (x1 + w * 0.02, y1 + h * 0.07, x2 - w * 0.02, y2 - h * 0.05)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--tag', default='016_mma4')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--match', default='iou', choices=['iou', 'center'],
                    help='iou=重叠率≥--iou；center=框中心落在GT框内 或 中心距<0.5×GT对角线。'
                         '★下游真正用的口径是"框中心 + 框内crop"，对框大小容错大，'
                         '贴身遮挡把框截断时 IoU 会掉但中心仍准（实测中心误差中位仅 3.6% 框对角线）')
    ap.add_argument('--unexpand', action='store_true', help='先反解框外扩再算 IoU（推荐）')
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    views = [v.zfill(2) for v in a.views.split(',')]

    per_view = {}
    all_iou, all_cerr, sel_hist = [], [], {}
    flip_hist = {}
    for v in views:
        gtdir = os.path.join(a.seq_root, 'processed_data', 'bbox', 'cam%s' % v)
        files = sorted(glob.glob(os.path.join(a.det, v, '*_*.json')))
        tp = fp = fn = 0
        n2 = n1 = n0 = 0                       # 选人正确率：命中 2/1/0 个受试者
        iv, ce, nfr = [], [], 0
        pid_to_subj = {}                       # 统计 gid0 -> which subject
        seq_flip = []                          # 逐帧 gid0 指向哪个 subject（算翻转次数）
        for fp_ in files:
            fr = int(os.path.basename(fp_).rsplit('_', 1)[-1].split('.')[0])
            if not (a.start <= fr <= a.end):
                continue
            gtf = os.path.join(gtdir, '%05d.npy' % fr)
            if not os.path.exists(gtf):
                continue
            gt = np.load(gtf, allow_pickle=True)
            gt = gt.item() if getattr(gt, 'dtype', None) == object else gt
            gt = {s: np.asarray(b, float).reshape(-1)[:4] for s, b in gt.items() if s in SUBJ}
            if not gt:
                continue
            shapes = [s for s in json.load(open(fp_))['shapes'] if s.get('label') == 'person']
            det = [(int(s.get('group_id', -1)),
                    (min(s['points'][0][0], s['points'][1][0]), min(s['points'][0][1], s['points'][1][1]),
                     max(s['points'][0][0], s['points'][1][0]), max(s['points'][0][1], s['points'][1][1])))
                   for s in shapes]
            nfr += 1

            def score(db, gb):
                """iou 口径 或 center 口径（中心落在GT框内 / 中心距<0.5×GT对角线）"""
                if a.match == 'center':
                    cx, cy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
                    gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                    inside = (gb[0] <= cx <= gb[2]) and (gb[1] <= cy <= gb[3])
                    d = np.linalg.norm([cx - gcx, cy - gcy]) / max(np.hypot(gb[2] - gb[0], gb[3] - gb[1]), 1e-9)
                    return 1.0 if (inside or d < 0.5) else 0.0
                return iou(db, gb)

            thr = a.iou if a.match == 'iou' else 0.5
            M = np.zeros((len(det), len(SUBJ)))
            for i, (_, db) in enumerate(det):
                if a.unexpand:
                    db = unexpand(db)
                for j, s in enumerate(SUBJ):
                    if s in gt:
                        M[i, j] = score(db, gt[s])
            hit = 0
            if M.size:
                ri, ci = linear_sum_assignment(-M)
                for i, j in zip(ri, ci):
                    if M[i, j] >= thr:
                        hit += 1
                        iv.append(M[i, j])
                        g = gt[SUBJ[j]]
                        db_ = unexpand(det[i][1]) if a.unexpand else det[i][1]
                        dc = np.array([(db_[0] + db_[2]) / 2 - (g[0] + g[2]) / 2,
                                       (db_[1] + db_[3]) / 2 - (g[1] + g[3]) / 2])
                        ce.append(np.linalg.norm(dc) / max(np.hypot(g[2] - g[0], g[3] - g[1]), 1e-9))
                        if det[i][0] == 0:
                            pid_to_subj[SUBJ[j]] = pid_to_subj.get(SUBJ[j], 0) + 1
                            seq_flip.append(SUBJ[j])   # ★只记 gid0 指向谁；下方按帧取最后一条
            tp += hit
            fp += len(det) - hit
            fn += len(gt) - hit
            if hit >= 2: n2 += 1
            elif hit == 1: n1 += 1
            else: n0 += 1
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        # 翻转次数：gid0 所指 subject 的逐帧跳变次数
        flips = sum(1 for i in range(1, len(seq_flip)) if seq_flip[i] != seq_flip[i - 1])
        per_view[v] = dict(n=nfr, f1=f1, prec=prec, rec=rec, iou=np.median(iv) if iv else np.nan,
                           cerr=np.median(ce) if ce else np.nan, n2=n2, n1=n1, n0=n0,
                           flips=flips, pid0=dict(pid_to_subj))
        all_iou += iv; all_cerr += ce
        sel_hist[v] = (n2, n1, n0)

    print('%-6s | %6s | %6s %6s %6s | %8s | %8s | %-26s | %s' %
          ('view', '帧数', 'P', 'R', 'F1', 'IoU中位', '中心误差', '选人正确率 2/2,1/2,0/2', 'gid0翻转'))
    print('-' * 118)
    for v in views:
        d = per_view.get(v)
        if not d:
            continue
        print('%-6s | %6d | %6.3f %6.3f %6.3f | %8.3f | %8.3f | %4d / %4d / %4d (%5.1f%% 全中) | %d'
              % (v, d['n'], d['prec'], d['rec'], d['f1'], d['iou'], d['cerr'],
                 d['n2'], d['n1'], d['n0'], 100.0 * d['n2'] / max(d['n'], 1), d['flips']))
    if all_iou:
        print('-' * 118)
        print('整体: IoU 中位 %.3f (p10 %.3f) | 中心误差中位 %.4f (按GT对角线归一) | 匹配对数 %d'
              % (np.median(all_iou), np.percentile(all_iou, 10), np.median(all_cerr), len(all_iou)))
        tot2 = sum(d['n2'] for d in per_view.values()); totn = sum(d['n'] for d in per_view.values())
        totf = sum(d['flips'] for d in per_view.values())
        print('      ★选人正确率(全视角): 2/2 = %.1f%%   身份跳变合计 %d 次' % (100.0 * tot2 / max(totn, 1), totf))
    if a.csv:
        with open(a.csv, 'w') as f:
            f.write('view,n,precision,recall,f1,iou_med,center_err_med,n2,n1,n0,gid0_flips\n')
            for v, d in per_view.items():
                f.write('%s,%d,%.4f,%.4f,%.4f,%.4f,%.5f,%d,%d,%d,%d\n'
                        % (v, d['n'], d['prec'], d['rec'], d['f1'], d['iou'], d['cerr'],
                           d['n2'], d['n1'], d['n0'], d['flips']))
        print('CSV -> %s' % a.csv)


if __name__ == '__main__':
    main()
