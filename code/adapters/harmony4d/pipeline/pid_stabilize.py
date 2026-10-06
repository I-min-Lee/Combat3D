#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  ★★★  SCENE-SPECIFIC LAYER — DO NOT REUSE AS-IS  ★★★
#
#  这是 Combat3D-Label 五层里【两个场景相关层】之一（检测框层 / 身份层）。
#  我们的测量证明：这两层换场景必须【换算法】，不是调参数 ——
#
#        changing scene  =>  changing algorithm
#
#  实测：Harmony4D 上我们用数据集自带的框与身份；而在这套代码的第二个域
#  （我们自己采集的盔甲对抗棍棒格斗）上，同一层必须从零重写
#  （RT-DETR + 位置先验；以及基于颜色证据的身份定标）。
#  两份实现都在本仓库里 —— 见 docs/SECOND_DOMAIN.md 的逐层对照表。
#
#  通用的是【三角化层及以下】，它们原样复用。层契约见 code/adapters/README.md。
#
#  ★ 连数据格式都是场景相关的：这个文件读的框/身份格式来自特定数据集，
#    换场景时要连 I/O 一起改，不只是改阈值。
#
#  Reusable? This layer: NO — rewrite it. Triangulation & below: YES.
# =============================================================================
"""pid_stabilize.py — **gid 跨时间连续性修复**（任务1，2026-09-30）

## 病（实测，05_sword2）
`det_self_final.py` 的 ② 用 `TwoSlotTracker` 在"已选的 2 个框"上定 gid。
实测：**1040 帧全部都是双人帧**（tracker 一个人都没丢），框运动也很平滑（逐帧中心位移中位 8px），
但身份**一直在翻**：view07 有 18 次跳变 / 164 个纯交换帧，view09 有 13 次 / 191 个。
→ 所以**不是"槽位清空重建"**（handoff 里的猜测），而是 tracker 的 NO-SWAP 偏好分区
  在两人贴身时**逐帧**把两个候选对到对方槽位上。单看一帧它自洽，跨时间就是身份翻转。
→ 一个视角错一整段、其余 5 个视角是对的，三角化把两人的观测混在一起 → PA-MPJPE 511mm。

## 药
**事后按"物理连续性"重排编号**（零数据集依赖，不碰检测/选人/跟踪的任何行为）：

对每个视角，逐帧维护两个 canonical id（0/1）的**位置+速度**状态，每一帧只做一个决定：
    A: 本轮 raw0→canon0, raw1→canon1
    B: 本轮 raw0→canon1, raw1→canon0
用**上一帧外推**（`cen + vel*dt`）算两种分配的归一化残差和，取小的那个。

★**迟滞**（`--ratio`，默认 0.5）：只有当"换"比"不换"好出 `1/ratio` 倍以上才真的换。
这一点是整个脚本的关键 —— 两人在 2D 上交错而过时两种分配代价几乎相等（ratio≈1）→ 不动；
而 tracker 真翻了时，代价差是"两人的像素间距"级别（ratio≈0.03）→ 果断换。
**没有迟滞就会把正常的 2D 交叉误判成身份交换**，而且一旦误判会自我强化（状态被污染）。因此迟滞不是调优项，是正确性项。

输出只**置换 group_id**，不改任何框的几何。

## 用法
```bash
# 离线（对已有 det 树做事后修复；不需要 GPU，秒级）—— 用于快速迭代与验收
python pid_stabilize.py --det $B/det_self_05_sword2 --out $B/det_self_05_sword2_fix \
    --views 01,03,04,07,09,14 --start 1 --end 1041 --tag 05_sword2
python pid_stabilize.py --det ... --dry-run          # 只看统计不写盘
```
`det_self_final.py` 里以 `stabilize_sel()` 复用同一套逻辑（在 ② 之后、③ 跨视角对齐之前）。
"""
import os, sys, json, glob, argparse
import numpy as np


# ---------- 几何小工具 ----------
def _cen(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _dia(b):
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def _nd(c, p, d_new, d_old):
    """归一化距离：中心距 / 平均框对角线（对"一个人离相机近、框特别大"免疫）"""
    return float(np.linalg.norm(c - p) / max(0.5 * (d_new + d_old), 1e-6))


# ---------- 核心 ----------
def stabilize_view(items, ratio=0.5, vel_a=0.5, vel_frac=0.5, gap_max=30, freeze_max=45):
    """单视角 gid 跨时间重排。

    参数
      items   : [(fr, {raw_gid: box}), ...]，按帧号升序；box = (x1,y1,x2,y2)；允许某些帧只有 1 个框
      ratio   : 迟滞比。换号必须满足 cost_new < ratio * cost_old（默认 0.5 = 要明显更好）
      vel_a   : 速度 EMA 系数
      vel_frac: 速度上限 = vel_frac × 框对角线 / 帧
      gap_max : 帧间隔超过它就当作断点（速度清零）
      freeze_max: ★连续"证据不足"超过这么多帧就重新锚定（防止外推飞掉；默认 45）

    返回 (out_items, stats)
      out_items : [(fr, {0/1: box}), ...]，帧号与内容不变，只有编号被重排
      stats     : dict(toggles=[帧号...], jumps={0:(中位,P95,最大), 1:...}, n=处理帧数)
    """
    state = {0: None, 1: None}        # canonical -> (中心, 速度, 对角线)
    perm = None                       # raw -> canonical（最近一次确定的映射）
    toggles, out = [], []
    last_upd = None                   # ★最后一次"真正更新了运动模型"的帧（不是最后一次处理）
    froze = 0                         # 连续冻结帧数
    n_frozen = 0
    last_cen = {}
    jumps = {0: [], 1: []}

    for fr, got in items:
        raw = sorted(got)
        if not raw:
            continue
        box = {r: np.asarray(got[r], float) for r in raw}
        c = {r: _cen(box[r]) for r in raw}
        d = {r: _dia(box[r]) for r in raw}
        # ★dt 从"上次更新模型的帧"算起，这样冻结期内的外推不会被重置（跨过交错区）
        dt = 1 if last_upd is None else max(1, min(fr - last_upd, gap_max))

        def pred(cc):
            """上一帧的位置按速度外推（dt 已在上面按 gap_max 截断）"""
            s = state[cc]
            return None if s is None else s[0] + s[1] * dt

        # ---- 决定本帧 raw -> canonical ----
        ambiguous = False                  # ★本帧证据不足以支持任何一次改判
        if len(raw) == 1:
            r = raw[0]
            if perm is not None and r in perm:
                assign = {r: perm[r]}
            else:
                cands = [cc for cc in (0, 1) if pred(cc) is not None]
                assign = ({r: min(cands, key=lambda cc: _nd(c[r], pred(cc), d[r], state[cc][2]))}
                          if cands else {r: 0})
        else:
            r0, r1 = raw[0], raw[1]

            def cost(a0):
                a1 = 1 - a0
                t = 0.0
                for r, cc in ((r0, a0), (r1, a1)):
                    p = pred(cc)
                    if p is not None:
                        t += _nd(c[r], p, d[r], state[cc][2])
                return t

            cA, cB = cost(0), cost(1)          # A: r0→0 ; B: r0→1
            cur = perm[r0] if (perm is not None and r0 in perm) else None
            if cur is None:
                a0 = 0 if cA <= cB else 1
            elif cB < ratio * cA:
                a0 = 1
            elif cA < ratio * cB:
                a0 = 0
            else:
                a0 = cur                        # ★迟滞：证据不够强就坚决不动
                ambiguous = True
            assign = {r0: a0, r1: 1 - a0}
            if cur is not None and a0 != cur:
                toggles.append(fr)
            perm = dict(assign)

        # ---- 更新状态 ----
        # ★★ 关键：**证据不足的帧绝不喂给运动模型**。
        #   两人交错时两个框几乎重合 → 判据分不出 → 若照常更新，模型会被"错的那一边"带走，
        #   之后所有帧的预测都跟着错，错误永久锁死（实测 06_sword3 就是这么丢掉 210 帧的）。
        #   冻结后，模型停在交错前的轨迹上并继续外推，交错结束后正确的分配才会胜出。
        if ambiguous and froze < freeze_max:
            froze += 1
            n_frozen += 1
        else:
            for r, cc in assign.items():
                old = state[cc]
                vmax = vel_frac * d[r]
                v = np.zeros(2) if old is None else (c[r] - old[0]) / dt
                v = np.clip(vel_a * v + (1.0 - vel_a) * (np.zeros(2) if old is None else old[1]),
                            -vmax, vmax)
                state[cc] = (c[r], v, d[r])
            for cc in (0, 1):                   # 没被观测到的那个：速度衰减，别外推中心
                if cc not in assign.values() and state[cc] is not None:
                    state[cc] = (state[cc][0], state[cc][1] * 0.9, state[cc][2])
            last_upd = fr
            froze = 0
        # 输出连续性统计**总在更新**（量的是写出去的东西，不是模型）
        for r, cc in assign.items():
            if cc in last_cen:
                jumps[cc].append(float(np.linalg.norm(c[r] - last_cen[cc])))
            last_cen[cc] = c[r]

        out.append((fr, {cc: tuple(box[r]) for r, cc in assign.items()}))

    st = dict(toggles=toggles, n=len(out), jumps={}, n_frozen=n_frozen)
    for cc in (0, 1):
        a = np.array(jumps[cc] or [0.0])
        st['jumps'][cc] = (float(np.median(a)), float(np.percentile(a, 95)), float(a.max()))
    return out, st


def stabilize_sel(sel_v, **kw):
    """对 det_self_final.py 里的 `sel[v] = {fr: {gid: box}}` 原地重排编号。
    返回 stats（写日志用）。"""
    items = sorted(sel_v.items())
    out, st = stabilize_view(items, **kw)
    sel_v.clear()
    sel_v.update({fr: g for fr, g in out})
    return st


# ---------- CLI：对已有 LabelMe 树做事后修复 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--tag', default='05_sword2')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--ratio', type=float, default=0.5)
    ap.add_argument('--vel-a', type=float, default=0.5)
    ap.add_argument('--vel-frac', type=float, default=0.5)
    ap.add_argument('--gap-max', type=int, default=30)
    ap.add_argument('--freeze-max', type=int, default=45,
                    help='★连续"证据不足"超过这么多帧就重新锚定（默认 45）')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    views = [v.zfill(2) for v in a.views.split(',')]

    tot = 0
    for v in views:
        files = sorted(glob.glob(os.path.join(a.det, v, '*_*.json')))
        items = []
        for fp in files:
            fr = int(os.path.basename(fp).rsplit('_', 1)[-1].split('.')[0])
            if not (a.start <= fr <= a.end):
                continue
            shapes = [s for s in json.load(open(fp))['shapes'] if s.get('label') == 'person']
            got = {}
            for s in shapes:
                g = int(s.get('group_id', -1))
                if g in (0, 1):
                    p = s['points']
                    got[g] = (min(p[0][0], p[1][0]), min(p[0][1], p[1][1]),
                              max(p[0][0], p[1][0]), max(p[0][1], p[1][1]))
            if got:
                items.append((fr, got))
        if not items:
            print('  view %s: 无数据' % v, flush=True)
            continue
        out, st = stabilize_view(items, ratio=a.ratio, vel_a=a.vel_a,
                                 vel_frac=a.vel_frac, gap_max=a.gap_max,
                                 freeze_max=a.freeze_max)
        tot += len(st['toggles'])
        j0, j1 = st['jumps'][0], st['jumps'][1]
        print('  view %s: %d 帧 | 换号 %d 次 %s | 冻结 %d 帧 | 逐帧位移中位 gid0=%.0f gid1=%.0f px'
              % (v, st['n'], len(st['toggles']),
                 (st['toggles'][:12] if st['toggles'] else ''), st['n_frozen'],
                 j0[0], j1[0]), flush=True)
        if (j0[2] > 300 or j1[2] > 300):
            print('     ⚠️ 仍有大位移: gid0 最大 %.0f px, gid1 最大 %.0f px' % (j0[2], j1[2]), flush=True)

        if a.dry_run:
            continue
        # 写盘：只改 group_id，几何原样
        newmap = {fr: g for fr, g in out}
        od = os.path.join(a.out, v)
        os.makedirs(od, exist_ok=True)
        for fp in files:
            fr = int(os.path.basename(fp).rsplit('_', 1)[-1].split('.')[0])
            if fr not in newmap:
                continue
            rec = json.load(open(fp))
            # 旧 gid -> 新 gid：按框中心匹配（中心在两步之间不变）
            old = {}
            for s in rec['shapes']:
                if s.get('label') != 'person':
                    continue
                g = int(s.get('group_id', -1))
                if g in (0, 1):
                    p = s['points']
                    old[g] = np.array([(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2])
            newc = {g2: _cen(b) for g2, b in newmap[fr].items()}
            remap = {}
            for g, cc in old.items():
                if not newc:
                    break
                best = min(newc, key=lambda k: np.linalg.norm(newc[k] - cc))
                remap[g] = best
                newc.pop(best)
            for s in rec['shapes']:
                if s.get('label') != 'person':
                    continue
                g = int(s.get('group_id', -1))
                if g in remap:
                    s['group_id'] = remap[g]
                    s['description'] = 'self-fix:%d' % remap[g]
            rec['shapes'].sort(key=lambda s: s.get('group_id', -1))
            json.dump(rec, open(os.path.join(od, os.path.basename(fp)), 'w'),
                      ensure_ascii=False, indent=1)
    print('[%s] 合计换号 %d 次%s'
          % ('dry-run' if a.dry_run else '已写 ' + str(a.out), tot,
             '' if a.dry_run else ''), flush=True)


if __name__ == '__main__':
    main()
