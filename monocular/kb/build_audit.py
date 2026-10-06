#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 kb_build 报告：分布 + 离群 (take,view,pid)"""
import json, sys, collections
import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else "/workshop/Lym/combat3d/mb/data/_build_report.json"
R = json.load(open(p))
ok = [r for r in R if r["status"] == "ok"]
bad = [r for r in R if r["status"] != "ok"]
print("总 %d  ok %d  bad %d" % (len(R), len(ok), len(bad)))
for r in bad[:20]:
    print("  BAD", r["take"], r["view"], r["pid"], r["status"], str(r["info"])[:120])

meds, hits, b2d, ns = [], [], [], []
rows = []
for r in ok:
    i = r["info"]
    a = i.get("audit") or {}
    m = a.get("med_px")
    if m is None:
        continue
    rows.append((r["take"], r["view"], r["pid"], m, a["n"], a["hit"], i["n"], i["has2d_ratio"]))
    meds.append(m); hits.append(a["hit"]); b2d.append(i["has2d_ratio"]); ns.append(i["n"])

meds = np.array(meds)
print("\n=== 重投影中位分布 (n=%d) ===" % len(meds))
for q in (0, 1, 5, 25, 50, 75, 90, 95, 99, 100):
    print("  p%-3d = %7.2f px" % (q, float(np.percentile(meds, q))))
print("  <10px: %.1f%%   <15px: %.1f%%   >25px: %.1f%%   >50px: %.1f%%" % (
    100 * (meds < 10).mean(), 100 * (meds < 15).mean(),
    100 * (meds > 25).mean(), 100 * (meds > 50).mean()))
print("  2D 命中率 中位 %.3f, <0.9 的比例 %.1f%%" % (np.median(b2d), 100 * (np.array(b2d) < 0.9).mean()))
print("  总帧数(25fps域) = %d" % sum(ns))

print("\n=== 离群: 中位 > 25px 的 (take,view,pid) ===")
w = sorted([r for r in rows if r[3] > 25], key=lambda x: -x[3])
for t, v, pid, m, n, h, N, h2 in w[:60]:
    print("  %-6s v%-3s p%d  med=%7.2f  n=%3d 审计2D命中=%.2f  帧=%d 总2D命中=%.2f" % (t, v, pid, m, n, h, N, h2))
print("  共 %d 个" % len(w))

# 按场次聚合
per = collections.defaultdict(list)
for t, v, pid, m, n, h, N, h2 in rows:
    per[t].append(m)
print("\n=== 最差的 20 个场次（场次内中位的中位）===")
for t, ms in sorted(per.items(), key=lambda kv: -np.median(kv[1]))[:20]:
    print("  %-6s 中位 %7.2f  最差 %7.2f  n=%d" % (t, float(np.median(ms)), float(np.max(ms)), len(ms)))
