#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gid_timeline.py — 诊断：**每视角 gid0 在时间轴上指向哪个真值 subject**（分段常数曲线）

为什么需要它（2026-09-30）：`det_vs_gt.py` 只报一个"身份跳变次数"总数，看不出**结构**。
134 次跳变可能是"134 个 1 帧毛刺"（无害），也可能是"3 个 200 帧的长段互换"（致命）。
三角化怕的是后者——一个视角错一整段、其余 5 个视角是对的，混出来就是几百 mm。

本脚本把每个视角的 `gid0→subject` 序列做 **RLE 分段**打出来，并同时报：
  ① 身份跳变次数（口径与 `det_vs_gt.py` 的 gid0翻转 一致，便于对数）
  ② **段长分布**（最长段 = 真正该看的量）
  ③ **物理连续性**：同 gid 相邻帧中心位移（px）—— 判别"机制跳变"vs"检测丢失"
  ④ **纯交换签名**：相邻双人帧里 gid0/gid1 同时互换 subject 的帧
     （= 纯编号翻转，与"选人错误"区分开；选人错误只会让一个 gid 换人）

用法：
  python gid_timeline.py --det $B/det_self_05_sword2 --seq-root $S \\
      --views 01,03,04,07,09,14 --start 1 --end 1041 --tag 05_sword2
"""
import os, json, glob, argparse
import numpy as np

SUBJ = ['aria01', 'aria02']


def unexpand(b):
    x1, y1, x2, y2 = b
    w = (x2 - x1) / 1.04
    h = (y2 - y1) / 1.12
    return (x1 + w * 0.02, y1 + h * 0.07, x2 - w * 0.02, y2 - h * 0.05)


def cen(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def diag(b):
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def box_of(s):
    p = s['points']
    return (min(p[0][0], p[1][0]), min(p[0][1], p[1][1]),
            max(p[0][0], p[1][0]), max(p[0][1], p[1][1]))


def which_subject(db, gt):
    """det 框中心落在哪个 GT 框内；都不在（或都在）则取中心距最近的那个（按 GT 对角线归一）"""
    c = cen(db)
    inside = [s for s, gb in gt.items() if gb[0] <= c[0] <= gb[2] and gb[1] <= c[1] <= gb[3]]
    if len(inside) == 1:
        return inside[0]
    return min(gt, key=lambda s: np.linalg.norm(c - cen(gt[s])) / max(diag(gt[s]), 1e-9))


def rle(seq):
    out = []
    for x in seq:
        if out and out[-1][0] == x:
            out[-1][1] += 1
        else:
            out.append([x, 1])
    return [(a, b) for a, b in out]


def load_gt(seq_root, v, fr):
    f = os.path.join(seq_root, 'processed_data', 'bbox', 'cam%s' % v, '%05d.npy' % fr)
    if not os.path.exists(f):
        return None
    gt = np.load(f, allow_pickle=True)
    gt = gt.item() if getattr(gt, 'dtype', None) == object else gt
    gt = {s: np.asarray(b, float).reshape(-1)[:4] for s, b in gt.items() if s in SUBJ}
    return gt or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--tag', default='05_sword2')
    a = ap.parse_args()
    views = [v.zfill(2) for v in a.views.split(',')]

    tot_flip = tot_swap = 0
    for v in views:
        files = sorted(glob.glob(os.path.join(a.det, v, '*_*.json')))
        timeline, frames = [], []          # gid0 的 subject 序列（仅匹配上的帧）
        pair_seq = []                      # (fr, subj(gid0), subj(gid1)) 仅双人帧
        gidc = {0: {}, 1: {}}              # gid -> {fr: 中心}
        nframe = 0
        for fp in files:
            fr = int(os.path.basename(fp).rsplit('_', 1)[-1].split('.')[0])
            if not (a.start <= fr <= a.end):
                continue
            gt = load_gt(a.seq_root, v, fr)
            if gt is None:
                continue
            det = {}
            for s in json.load(open(fp))['shapes']:
                if s.get('label') != 'person':
                    continue
                g = int(s.get('group_id', -1))
                if g in (0, 1):
                    det[g] = unexpand(box_of(s))
            if not det:
                continue
            nframe += 1
            m = {}
            for g, b in det.items():
                gidc[g][fr] = cen(b)
                m[g] = which_subject(b, gt)
            if 0 in m:
                timeline.append(m[0]); frames.append(fr)
            if 0 in m and 1 in m:
                pair_seq.append((fr, m[0], m[1]))

        flips = sum(1 for i in range(1, len(timeline)) if timeline[i] != timeline[i - 1])
        segs = rle(timeline)
        swap_frames = []
        for i in range(1, len(pair_seq)):
            if pair_seq[i][1] == pair_seq[i - 1][2] and pair_seq[i][2] == pair_seq[i - 1][1]:
                swap_frames.append(pair_seq[i][0])
        jumps = {}
        for g in (0, 1):
            ks = sorted(gidc[g])
            jumps[g] = {ks[i]: float(np.linalg.norm(gidc[g][ks[i]] - gidc[g][ks[i - 1]]))
                        for i in range(1, len(ks)) if ks[i] - ks[i - 1] == 1}
        allj = np.array(list(jumps[0].values()) + list(jumps[1].values()) or [0.0])
        # ★三类逐帧位移事件（这是"该不该动 tracker 的匹配门限"的直接依据）：
        #   交换  = 两个 gid 同帧各跳一大步（编号互换）
        #   独跳  = 只有一个 gid 跳一大步（**槽位被非受试者检测夺走** —— 05/06 的主因）
        n_swap_ev = n_solo_ev = 0
        ev_frames = []
        for thr in (40.0, 100.0, 200.0):
            a = sum(1 for f, x in jumps[0].items() if x > thr)
            b = sum(1 for f, x in jumps[1].items() if x > thr)
            both = sum(1 for f in jumps[0] if f in jumps[1]
                       and jumps[0][f] > thr and jumps[1][f] > thr)
            if thr == 100.0:
                n_swap_ev = both
                n_solo_ev = a + b
                ev_frames = sorted([f for f, x in jumps[0].items() if x > thr] +
                                   [f for f, x in jumps[1].items() if x > thr])[:20]
            print('  >%.0fpx: gid0 %d 帧, gid1 %d 帧, 两 gid 同帧 %d' % (thr, a, b, both))

        print('=' * 96)
        print('VIEW %s | 帧数 %d 双人帧 %d | gid0身份跳变 %d | 纯交换帧 %d | RLE段数 %d 最长段 %d'
              % (v, nframe, len(pair_seq), flips, len(swap_frames), len(segs),
                 max(n for _, n in segs) if segs else 0))
        print('  段: %s' % '  '.join('%s:%d' % (s, n) for s, n in segs))
        print('  逐帧中心位移(px): 中位 %.0f P95 %.0f 最大 %.0f'
              % (np.median(allj), np.percentile(allj, 95), allj.max()))
        if ev_frames:
            print('  大跳跃帧号(前20): %s' % ev_frames)
        if swap_frames:
            print('  交换帧号(前40): %s' % swap_frames[:40])
        tot_flip += flips
        tot_swap += len(swap_frames)
    print('=' * 96)
    print('合计: gid0 身份跳变 %d 次 | 纯交换帧 %d 次' % (tot_flip, tot_swap))


if __name__ == '__main__':
    main()
