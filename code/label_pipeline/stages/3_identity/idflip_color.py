#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""idflip_color.py — 由颜色证据生成「逐视角分段常数」的身份翻转
输入  _idcolor_obs.csv  (idcolor_scan.py 产出: 每(帧,视角)的框内人颜色)
      vitpose/rtdetr json (只读几何, 用于判定"框内是否只有唯一候选 annot")
输出  rtd2/idflip_color.json   frame -> {view: 0/1}  (与 idflip.json 同格式)
      _single_assign.json      frame -> {view: pid}  该视角只有一个人时归谁
      _idflip_color_report.txt 每视角跳变点清单

规则:
  证据帧 = 框内候选唯一(几何不歧义) 且 分类器决定性(max prob>=0.80)
  单人帧 = 两条 annot 都被判成同一种颜色(说明该视角只有这一个人) 或 d01<8px
算法: 每视角独立 2 状态 Viterbi, 发射 PEN / 转移 SWITCH
"""
import json, os, glob, sys
import numpy as np
from collections import defaultdict

B = "/root/autodl-tmp"
CSV = os.environ.get("OBS_IN", B + "/_idcolor_obs.csv")
RAW = os.environ.get("RAW_DIR", B + "/vitpose_rtd5000of/1/1")
DET = os.environ.get("DETECT_DIR", B + "/rtdetr_v2/输出/1/1/json")
OUT_JSON = os.environ.get("FLIP_OUT", B + "/rtd2/idflip_color.json")
OUT_DUP = os.environ.get("DUP_OUT", B + "/_single_assign.json")
OUT_RPT = os.environ.get("RPT_OUT", B + "/_idflip_color_report.txt")
VIEWS = ["1", "3", "4", "7", "11"]
TORSO = [5, 6, 11, 12]
PEN = float(os.environ.get("PEN", "1.0"))
SWITCH = float(os.environ.get("SWITCH", "12.0"))
N = int(os.environ.get("NFR", "14017"))
CLS_CONF = 0.80
DUP_TOL = 8.0        # 中位关节距 < 该值 -> 同一人
SAME_MAX = 50.0      # "两条 annot 同色" 判定为单人时的最大间距
SMALL_PAD = int(os.environ.get("SMALL_PAD", "1"))   # 单人判定向未采样邻帧外推的帧数
USE_BOX = int(os.environ.get("USE_BOX", "1"))       # 是否用框级颜色证据补位
BOX_IN = os.environ.get("BOX_IN", B + "/_idbox_obs*.csv")
SAME_FAR = []        # 同色但间距过大 (不剔除, 仅报告)
ODD = []             # 间距 < DUP_TOL 却判出两种颜色 (矛盾, 仅报告)


def torso(kp):
    p = [kp[i, :2] for i in TORSO if kp[i, 2] > 0.3]
    return np.mean(p, axis=0) if len(p) >= 2 else None


def geom_ok(v, fr, n_ann):
    """每个框内的候选 annot 数 <=1 ? (不歧义)"""
    cand = glob.glob(os.path.join(DET, v, "*_%06d.json" % fr))
    rf = os.path.join(RAW, v, "%06d.json" % fr)
    if not cand or not os.path.exists(rf):
        return False
    boxes = {}
    for s in json.load(open(cand[0])).get("shapes", []):
        if s.get("label") == "person" and s.get("group_id") is not None:
            p = s["points"]
            boxes[int(s["group_id"])] = [p[0][0], p[0][1], p[1][0], p[1][1]]
    d = json.load(open(rf))
    ks = [np.array(a["keypoints"], float) for a in
          (d["annots"] if isinstance(d, dict) else d)]
    ks = [k for k in ks if k.shape == (17, 3)]
    ctr = [torso(k) for k in ks]
    for bx in boxes.values():
        pad = 0.1 * max(bx[2] - bx[0], bx[3] - bx[1])
        n = sum(1 for c in ctr if c is not None and
                bx[0] - pad <= c[0] <= bx[2] + pad and bx[1] - pad <= c[1] <= bx[3] + pad)
        if n > 1:
            return False
    return True


rows = defaultdict(dict)
PATS = [CSV] + glob.glob(os.path.join(os.path.dirname(CSV),
                                      os.path.basename(CSV).replace(".csv", "_*.csv")))
LINES = []
for path in PATS:
    if os.path.exists(path):
        LINES += open(path).read().splitlines()[1:]

for ln in LINES:
    c = ln.split(",")
    if len(c) < 12:
        continue
    fr, v = int(c[0]), c[1]
    d01 = float(c[3]) if c[3] else -1.0
    g0rr = c[6]
    g0bb = c[7]
    g1rr = c[9]
    g1bb = c[10]
    flip = None if c[11] == "" else int(c[11])

    def dec(rr, bb):
        if rr == "" or bb == "":
            return None
        r, b = float(rr), float(bb)
        if max(r, b) < CLS_CONF:
            return None
        return "R" if r > b else "B"

    c0, c1 = dec(g0rr, g0bb), dec(g1rr, g1bb)
    single = False
    owner = None
    if c0 is not None and c1 is not None and c0 == c1 and d01 < SAME_MAX:
        single = True                      # 两条 annot 同色且靠得近 -> 只有这一个人
        owner = 1 if c0 == "R" else 0
    elif 0 <= d01 < DUP_TOL:
        single = True
        cc = c0 or c1
        owner = None if cc is None else (1 if cc == "R" else 0)
    if c0 is not None and c1 is not None and c0 == c1 and d01 >= SAME_MAX:
        SAME_FAR.append((fr, v, round(d01, 1), c0))
    if c0 is not None and c1 is not None and c0 != c1 and 0 <= d01 < DUP_TOL:
        ODD.append((fr, v, round(d01, 1), c0, c1))
    rows[v][fr] = (flip, single, owner)

# ---- 框级颜色证据(可选): frame -> {view: 0/1} ----
BOX = defaultdict(dict)
CAL = [0, 0, 0]      # [两路一致, 两路打架, 仅框级]
if USE_BOX:
    for path in glob.glob(BOX_IN):
        for ln in open(path).read().splitlines()[1:]:
            c = ln.split(",")
            if len(c) < 13:
                continue
            ev = c[12].strip()
            if ev in ("0", "1"):
                BOX[c[1]][int(c[0])] = int(ev)
    print("框级证据载入: %d 个 (帧,视角) 槽, %d 个视角" % (
        sum(len(x) for x in BOX.values()), len(BOX)))

report = []
FLIPMAP, SINGLE = {}, {}
for v in VIEWS:
    obs = rows[v]
    frs = sorted(obs)
    if not frs:
        report.append("v%s: 无数据" % v)
        continue
    # 几何歧义/证据不足 -> 不投票; 框级证据补位(见下)
    good = {}
    for fr in frs:
        f, single, owner = obs[fr]
        a_ev = f if (f is not None and geom_ok(v, fr, 2)) else None
        b_ev = BOX.get(v, {}).get(fr) if USE_BOX else None
        if a_ev is not None and b_ev is not None:
            if a_ev == b_ev:
                good[fr] = a_ev
                CAL[0] += 1
            else:
                good[fr] = None            # 两路证据打架 -> 不投票
                CAL[1] += 1
        elif a_ev is not None:
            good[fr] = a_ev
        else:
            good[fr] = b_ev                # 只用框级(贴靠窗口救的就是这里)
            if b_ev is not None:
                CAL[2] += 1
    T = len(frs)
    dp = np.zeros((T, 2))
    bp = np.zeros((T, 2), int)
    for t in range(1, T):
        f = good[frs[t]]
        em = [0.0, 0.0] if f is None else ([0.0, PEN] if f == 0 else [PEN, 0.0])
        for s in (0, 1):
            cand = [dp[t - 1][0] + (0.0 if s == 0 else SWITCH),
                    dp[t - 1][1] + (0.0 if s == 1 else SWITCH)]
            j = int(np.argmin(cand))
            dp[t][s] = cand[j] + em[s]
            bp[t][s] = j
    st = int(np.argmin(dp[T - 1]))
    path = [st]
    for t in range(T - 1, 0, -1):
        st = bp[t][st]
        path.append(st)
    path = path[::-1]
    for t, fr in enumerate(frs):
        if path[t]:
            FLIPMAP.setdefault(str(fr), {})[v] = 1
        _f, single, owner = obs[fr]
        if single and owner is not None:
            for dfr in range(-SMALL_PAD, SMALL_PAD + 1):
                ff = fr + dfr
                if 0 <= ff < N and ff not in obs:
                    SINGLE.setdefault(str(ff), {})[v] = owner
            SINGLE.setdefault(str(fr), {})[v] = owner
    chg = [frs[t] for t in range(1, T) if path[t] != path[t - 1]]
    nvote = sum(1 for f in frs if good[f] is not None)
    agr = sum(1 for t, f in enumerate(frs)
              if good[f] is not None and good[f] == path[t])
    nsingle = sum(1 for f in frs if obs[f][1])
    report.append(
        "v%-3s 采样=%-5d 有效证据=%-5d(%.0f%%) 状态0=%-6d 状态1=%-6d 一致率=%.1f%% "
        "单人帧=%-5d 跳变点=%d %s" % (
            v, T, nvote, 100.0 * nvote / T,
            sum(1 for p in path if p == 0), sum(1 for p in path if p == 1),
            100.0 * agr / max(nvote, 1), nsingle, len(chg), chg[:16]))

# ★ 逐帧填满: 采样帧之间沿用最近一次采样的 Viterbi 状态(阶梯保持),
#   绝不能让未采样帧默认成 0 —— 那等于又变成逐帧乱翻。
full = {}
for v in VIEWS:
    frs = sorted(rows[v])
    seq = {f: FLIPMAP.get(str(f), {}).get(v, 0) for f in frs}
    cur = seq[frs[0]] if frs else 0
    j = 0
    for fr in range(N):
        while j < len(frs) and frs[j] <= fr:
            cur = seq[frs[j]]
            j += 1
        full.setdefault(str(fr), {})[v] = cur
json.dump(full, open(OUT_JSON, "w"))
json.dump(SINGLE, open(OUT_DUP, "w"))
open(OUT_RPT, "w").write("\n".join(report) + "\n")
print("\n".join(report))
print("\n反转=1 帧数:", {v: sum(1 for fr in range(N) if full[str(fr)][v]) for v in VIEWS})
print("单人帧条目:", sum(len(x) for x in SINGLE.values()))
print("同色但间距>=%.0fpx (未剔除): %d 例 %s" % (SAME_MAX, len(SAME_FAR), SAME_FAR[:10]))
print("间距<%.0fpx 却判两色(矛盾): %d 例 %s" % (DUP_TOL, len(ODD), ODD[:10]))
if USE_BOX:
    tot = CAL[0] + CAL[1]
    print("框级证据: 与 annot 级一致 %d, 打架 %d (一致率 %.2f%%), 仅框级补位 %d 个槽"
          % (CAL[0], CAL[1], 100.0 * CAL[0] / max(tot, 1), CAL[2]))
print("->", OUT_JSON, OUT_DUP, OUT_RPT)
