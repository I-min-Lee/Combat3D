#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""occl_strat.py — **按遮挡强度分层报 MPJPE**（论文"高遮挡"主张的核心证据）

遮挡强度代理：**两个受试者真值框的 IoU**（在某参考视角上）。
  IoU 越大 ⇒ 两人在图像上越重叠 ⇒ 遮挡/接触越严重。
分层：<0.05 / 0.05-0.20 / 0.20-0.35 / ≥0.35（≈实测"贴身帧"阈值 0.3 附近）

同时可对多个 run 目录并列比较（例如"我们 vs 基线"）。

用法：
  python occl_strat.py --runs 自建:<dir> 真值框:<dir> --seq-root <.../016_mma4> \
      --calib <calib> --views 01,03,04,07,09,14 [--ref-view 04]
"""
import os, json, glob, argparse
import numpy as np

SUBJ = ('aria01', 'aria02')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='标签:目录（目录含 pid{p}/keypoints3d）')
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--ref-view', default='04', help='算 GT 框 IoU 的参考视角')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--pids', default='0,1')
    ap.add_argument('--gt-dir', default=None, help='gt3d_colmap 目录（与管线同一世界系）；不给则用 poses3d(米制, 会错配)')
    a = ap.parse_args()
    S = np.load(os.path.join(a.calib, 'scale_metric.npy'))
    sc = float(np.linalg.norm(S[:3, 0]))
    v0 = a.ref_view.zfill(2)
    BINS = [(0.0, 0.05, '分离 <0.05'), (0.05, 0.20, '轻度 0.05-0.20'),
            (0.20, 0.35, '中度 0.20-0.35'), (0.35, 9.9, '贴身 ≥0.35')]

    def iou(x, y):
        ix1, iy1 = max(x[0], y[0]), max(x[1], y[1]); ix2, iy2 = min(x[2], y[2]), min(x[3], y[3])
        inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
        ua = (x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * (y[3] - y[1]) - inter
        return inter / ua if ua > 0 else 0.

    # 每帧的遮挡分（用参考视角的 GT 两人框 IoU）
    occ = {}
    for fr in range(a.start, a.end):
        gtf = os.path.join(a.seq_root, 'processed_data/bbox/cam%s/%05d.npy' % (v0, fr))
        if not os.path.exists(gtf):
            continue
        g = np.load(gtf, allow_pickle=True)
        g = g.item() if getattr(g, 'dtype', None) == object else g
        if not all(s in g for s in SUBJ):
            continue
        occ[fr] = iou(np.asarray(g[SUBJ[0]], float).reshape(-1)[:4],
                      np.asarray(g[SUBJ[1]], float).reshape(-1)[:4])

    print('=== 按遮挡强度分层 MPJPE（遮挡代理 = 参考视角 %s 上两个 GT 框的 IoU）===' % v0)
    for spec in a.runs:
        lab, d = spec.split(':', 1)
        per_bin = {b[2]: [] for b in BINS}
        for pid in [int(x) for x in a.pids.split(',')]:
            sub = SUBJ[pid]
            gtdir = a.gt_dir or os.path.join(a.seq_root, 'processed_data', 'poses3d')
            for f in sorted(glob.glob(os.path.join(d, 'pid%d' % pid, 'keypoints3d', '*.json'))):
                fr = int(os.path.basename(f).split('.')[0])
                if fr not in occ:
                    continue
                if a.gt_dir:
                    # ★ gt3d_colmap：JSON、%06d、已是 COCO17、与管线同一世界系
                    gtf = os.path.join(gtdir, '%06d.json' % fr)
                    if not os.path.exists(gtf):
                        continue
                    gt = json.load(open(gtf)).get(sub)
                    if gt is None:
                        continue
                    G = np.asarray(gt, float)[:, :3]
                else:
                    gtf = os.path.join(gtdir, '%05d.npy' % fr)
                    if not os.path.exists(gtf):
                        continue
                    gt = np.load(gtf, allow_pickle=True).item().get(sub)
                    if gt is None:
                        continue
                    G = np.asarray(gt, float)[:, :3]
                j = json.load(open(f))
                k = np.array(j[0]['keypoints3d'], float)      # (25,4)
                M = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 9: 12, 10: 14, 11: 16,
                     12: 11, 13: 13, 14: 15, 15: 2, 16: 1, 17: 4, 18: 3}
                O = np.zeros((17, 3)); ok = np.zeros(17, bool)
                for b25, c17 in M.items():
                    if np.any(k[b25, :3]):
                        O[c17] = k[b25, :3]; ok[c17] = True
                if ok.sum() < 6:
                    continue
                d_ = np.linalg.norm(O[ok] - G[ok], axis=1) * sc * 1000
                for lo, hi, name in BINS:
                    if lo <= occ[fr] < hi:
                        per_bin[name].append(d_)
                        break
        print('  [%s]' % lab)
        for lo, hi, name in BINS:
            arr = np.concatenate(per_bin[name]) if per_bin[name] else None
            if arr is None:
                print('    %-16s 无样本' % name); continue
            print('    %-16s 关节样本 %7d | 中位 %6.1f mm | P95 %6.1f mm'
                  % (name, len(arr), np.median(arr), np.percentile(arr, 95)))


if __name__ == '__main__':
    main()
