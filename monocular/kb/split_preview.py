#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预览 train/val/test 三分（按场次、按分辨率组分层的 stratified）"""
import sys, os, json
sys.path.insert(0, '/workshop/Lym/combat3d/mb/kb')
import kb_common as K

takes = K.load_takes()
tw, qbad = K.take_weights(0.18, 3)
excl = set(qbad)
# 加上 _exclude.json 的场次级排除
p = '/workshop/Lym/combat3d/mb/data/_exclude.json'
if os.path.exists(p):
    excl |= set(json.load(open(p)).get('takes', []))

cand = [t for t in takes if t[0] not in excl]
print("总场次 %d，质量/身份排除 %d，候选 %d" % (len(takes), len(excl), len(cand)))
import collections
g = collections.Counter(t[3] for t in cand)
print("候选按分辨率组:", dict(g), " (960=960x720@200fps, 1920=1920x1440@25fps)")

import random
rnd = random.Random(1234)
groups = {}
for tag, n, fps, grp in takes:
    if tag in excl:
        continue
    groups.setdefault(grp, []).append(tag)

tr, va, te = set(), set(), set()
for grp, lst in sorted(groups.items()):
    lst = sorted(lst); rnd.shuffle(lst)
    nv = max(1, int(round(len(lst) * 0.12)))
    nt = max(1, int(round(len(lst) * 0.12)))
    te  |= set(lst[:nt])
    va  |= set(lst[nt:nt + nv])
    tr  |= set(lst[nt + nv:])
    print("  组 %s: 共 %d -> test %d, val %d, train %d" % (grp, len(lst), nt, nv, len(lst) - nt - nv))

print()
print("train (%d): %s" % (len(tr), sorted(tr)))
print()
print("val   (%d): %s" % (len(va), sorted(va)))
print()
print("test  (%d): %s" % (len(te), sorted(te)))
print()
# 与当前正在跑的 run 的 val 对比
cur_val = {'f01', 'f111', 'f131', 'f132', 'f14', 'f142', 'f32', 'f391'}
print("当前 run 的 val:", sorted(cur_val))
print("新 test 与当前训练集是否有重叠:", "有(说明重启后 test 会泄漏!)" if te & (set(t[0] for t in takes) - excl - cur_val) else "无")
