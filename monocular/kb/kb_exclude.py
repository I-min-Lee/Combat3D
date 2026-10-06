#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kb_exclude —— 从建库审计报告生成排除清单 _exclude.json

判据（与项目一贯口径一致：标烂场 + 跳过 + 保留产物）：
  1. (take,view,pid) 级：3D->2D 重投影中位 > --med-thr (默认 25px) -> 该条不参与训练/评测
  2. 场次级黑名单：已知身份层失效场（两人同色球衣）——用户口径明确要标出来跳过
  3. 2D 命中率 < --hit-thr (默认 0.85) 的 (take,view,pid) 单列"低命中"名单

产出: <data>/_exclude.json = {"pairs": [[take,view,pid],...], "lowhit": [...], "takes": [...]}
kb_train.py 会自动读取它。
"""
import json, sys, argparse, collections
import numpy as np

# 已知身份层失效（记忆: 43.1/43.2 两人同色球衣 -> 颜色指纹无信号）
KNOWN_BAD_TAKES = ["f431", "f432"]

ap = argparse.ArgumentParser()
ap.add_argument('--report', default='/workshop/Lym/combat3d/mb/data/_build_report.json')
ap.add_argument('--out', default='/workshop/Lym/combat3d/mb/data/_exclude.json')
ap.add_argument('--med-thr', type=float, default=25.0)
ap.add_argument('--hit-thr', type=float, default=0.85)
ap.add_argument('--take-thr', type=float, default=20.0, help='场次内中位超过它就整场排除')
a = ap.parse_args()

R = json.load(open(a.report))
ok = [r for r in R if r['status'] == 'ok']

pairs, lowhit = [], []
take_meds = collections.defaultdict(list)
for r in ok:
    i = r['info']
    au = i.get('audit') or {}
    m = au.get('med_px')
    t, v, p = r['take'], r['view'], r['pid']
    if m is not None:
        take_meds[t].append(m)
        if m > a.med_thr:
            pairs.append([t, v, p])
    if i.get('has2d_ratio', 1.0) < a.hit_thr or (au.get('hit') is not None and au['hit'] < a.hit_thr):
        lowhit.append([t, v, p])

bad_takes = set(KNOWN_BAD_TAKES)
for t, ms in take_meds.items():
    if float(np.median(ms)) > a.take_thr:
        bad_takes.add(t)

# 整场排除 -> 展开成 5 视角 x 2 人
for t in sorted(bad_takes):
    for v in ('1', '3', '4', '7', '11'):
        for p in (0, 1):
            if [t, v, p] not in pairs:
                pairs.append([t, v, p])

pairs = sorted({tuple(x) for x in pairs})
lowhit = sorted({tuple(x) for x in lowhit if list(x) not in [list(y) for y in pairs]})

tot = len(ok)
print('总序列 %d；排除 %d (%.1f%%)；低命中单列 %d' % (tot, len(pairs), 100 * len(pairs) / max(tot, 1), len(lowhit)))
print('整场排除的场次:', sorted(bad_takes))
print('低命中 (2D<%.2f):' % a.hit_thr)
for x in lowhit[:30]:
    print('   ', x)

json.dump({'pairs': [list(x) for x in pairs],
           'lowhit': [list(x) for x in lowhit],
           'takes': sorted(bad_takes),
           'criteria': dict(med_thr=a.med_thr, hit_thr=a.hit_thr, take_thr=a.take_thr,
                            known_bad_takes=KNOWN_BAD_TAKES)},
          open(a.out, 'w'), ensure_ascii=False, indent=1)
print('->', a.out)
