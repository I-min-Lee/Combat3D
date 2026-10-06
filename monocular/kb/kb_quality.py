#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kb_quality —— 把 _segqa_*.txt 质量报告的 5 秒分箱解析成「逐帧可用掩码」

报告结构（每个场次一份）:
    f21  frames=23153 fps=200  分段数=24  问题箱=0
      骨长CV p0/p1 = 0.0891 / 0.0930   (健康 0.065~0.146)
      跳变阈值(按帧率缩放) = 0.200 m
           0.0-   5.0s  conf=0.828  low=0.004  jumps=0
      ★   35.0-  40.0s  conf=0.515  low=0.084  jumps=0      <- ★ = 报告判定的问题箱

训练用的「好帧」判据（箱级，全部满足才算好）:
    · 不是 ★ 箱
    · jumps == 0            （帧间跳变 -> 该段 3D 序列本身不连续，是硬伤）
    · conf >= --conf-thr    （3D 关节平均置信度）
    · low  <= --low-thr     （低置信关节占比）

输出 _quality.json:
  { "takes": {tag: {"good": [[f0,f1),...] (25fps 帧号), "bad_bins": [[t0,t1),...], "cv": [p0,p1], "kept_ratio": r}},
    "criteria": {...}, "summary": {...} }

用法:
  python kb_quality.py --stats                 # 只看各阈值下丢多少帧，不写文件
  python kb_quality.py --conf-thr 0.6 --low-thr 0.08   # 正式产出
"""
import os, re, sys, json, glob, argparse
import numpy as np

SEGQA = '/workshop/Lym/combat3d'
TARGET_FPS = 25

HEAD = re.compile(r'^(f\d+)\s+frames=(\d+)\s+fps=(\d+)\s+分段数=(\d+)\s+问题箱=(\d+)')
CVL = re.compile(r'骨长CV\s+p0/p1\s*=\s*([\d.]+)\s*/\s*([\d.]+)')
BIN = re.compile(r'^\s*(★)?\s*([\d.]+)-\s*([\d.]+)s\s+conf=([\d.]+)\s+low=([\d.]+)\s+jumps=(\d+)')


def parse(tag):
    p = f'{SEGQA}/_segqa_{tag}.txt'
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8', errors='ignore') as fh:
        lines = fh.readlines()
    head = None
    cv = None
    bins = []
    for ln in lines:
        m = HEAD.match(ln.strip())
        if m:
            head = dict(tag=m.group(1), frames=int(m.group(2)), fps=int(m.group(3)),
                        n_bins=int(m.group(4)), n_prob=int(m.group(5)))
            continue
        m = CVL.search(ln)
        if m:
            cv = [float(m.group(1)), float(m.group(2))]
            continue
        m = BIN.match(ln)
        if m:
            bins.append(dict(star=bool(m.group(1)), t0=float(m.group(2)), t1=float(m.group(3)),
                             conf=float(m.group(4)), low=float(m.group(5)), jumps=int(m.group(6))))
    if head is None:
        return None
    head['cv'] = cv
    head['bins'] = bins
    return head


def good_mask(rec, conf_thr, low_thr, use_star=True):
    """返回 (帧级 bool 掩码 at 25fps, 坏箱列表)"""
    n25 = int(round(rec['frames'] / max(1, rec['fps']) * TARGET_FPS))
    mask = np.zeros(n25, dtype=bool)
    bad = []
    for b in rec['bins']:
        ok = True
        if use_star and b['star']:
            ok = False
        if b['jumps'] > 0:
            ok = False
        if b['conf'] < conf_thr:
            ok = False
        if b['low'] > low_thr:
            ok = False
        f0 = int(round(b['t0'] * TARGET_FPS))
        f1 = min(n25, int(round(b['t1'] * TARGET_FPS)))
        if f1 > f0:
            mask[f0:f1] = ok
        if not ok:
            bad.append([b['t0'], b['t1']])
    return mask, bad


def to_ranges(mask):
    r, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            r.append([s, i]); s = None
    if s is not None:
        r.append([s, len(mask)])
    return r


def scan(conf_thr, low_thr, use_star=True, verbose=False):
    takes = {}
    tot = kept = 0
    for p in sorted(glob.glob(f'{SEGQA}/_segqa_*.txt')):
        tag = os.path.basename(p)[len('_segqa_'):-4]
        if tag == 'summary':
            continue
        rec = parse(tag)
        if rec is None:
            continue
        mask, bad = good_mask(rec, conf_thr, low_thr, use_star)
        rng = to_ranges(mask)
        k = int(mask.sum())
        tot += len(mask); kept += k
        takes[tag] = dict(good=rng, bad_bins=bad, cv=rec['cv'],
                          n25=len(mask), kept=k,
                          kept_ratio=(k / len(mask)) if len(mask) else 0.0)
    return takes, tot, kept


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--conf-thr', type=float, default=0.60)
    ap.add_argument('--low-thr', type=float, default=0.08)
    ap.add_argument('--no-star', action='store_true', help='忽略报告里的 ★ 标记，只用数值判据')
    ap.add_argument('--min-keep', type=float, default=0.10,
                    help='整场保留帧比例低于此值就整场丢弃（避免只剩碎片的场次）')
    ap.add_argument('--out', default='/workshop/Lym/combat3d/mb/data/_quality.json')
    a = ap.parse_args()

    if a.stats:
        print('阈值扫描（保留帧比例，25fps 域）')
        print('%-10s %-10s %-10s %s' % ('conf_thr', 'low_thr', '用★', '保留'))
        for c in (0.0, 0.50, 0.55, 0.60, 0.65, 0.70):
            for l in (1.0, 0.12, 0.08, 0.05, 0.03):
                _, tot, kept = scan(c, l, not a.no_star)
                print('%-10.2f %-10.2f %-10s %5.1f%%  (%d/%d 帧)' % (
                    c, l, 'Y' if not a.no_star else 'N', 100 * kept / max(tot, 1), kept, tot))
        # 只看 ★ 本身
        _, tot, kept = scan(0.0, 1.0, True)
        print('\n只按 ★ + jumps 筛: %.1f%%' % (100 * kept / max(tot, 1)))
        _, tot, kept = scan(0.0, 1.0, False)
        print('只按 jumps 筛:      %.1f%%' % (100 * kept / max(tot, 1)))
        sys.exit(0)

    takes, tot, kept = scan(a.conf_thr, a.low_thr, not a.no_star)
    # 整场丢弃
    dropped = [t for t, d in takes.items() if d['kept_ratio'] < a.min_keep]
    for t in dropped:
        takes[t]['good'] = []
    print(f'场次 {len(takes)}；整场丢弃 {len(dropped)} 个（保留帧<{a.min_keep:.0%}）: {sorted(dropped)[:20]}')
    kept2 = sum(d['kept'] for d in takes.values() if d['kept_ratio'] >= a.min_keep)
    print(f'样本帧筛选: {kept}/{tot} = {100*kept/max(tot,1):.1f}%   （丢弃 {tot-kept} 帧）')
    print('保留率分布: ' + '  '.join('%d%%' % q for q in (0, 5, 25, 50, 75, 100)) +
          '  -> %s' % np.round(np.percentile([d['kept_ratio'] for d in takes.values()], [0, 5, 25, 50, 75, 100]) * 100, 1))
    worst = sorted(takes.items(), key=lambda kv: kv[1]['kept_ratio'])[:15]
    print('保留率最低的场次:')
    for t, d in worst:
        print('   %-6s 保留 %5.1f%%  CV=%.3f/%.3f  n25=%d  bad_bins=%d' % (
            t, 100 * d['kept_ratio'], d['cv'][0] if d['cv'] else -1,
            d['cv'][1] if d['cv'] else -1, d['n25'], len(d['bad_bins'])))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(takes=takes,
                   criteria=dict(conf_thr=a.conf_thr, low_thr=a.low_thr,
                                 use_star=not a.no_star, min_keep=a.min_keep,
                                 target_fps=TARGET_FPS),
                   summary=dict(total_frames_25fps=tot, kept=kept2,
                                kept_ratio=kept2 / max(tot, 1))),
              open(a.out, 'w'), ensure_ascii=False)
    print('->', a.out)
