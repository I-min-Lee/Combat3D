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
"""det_ours_select.py v2 — **全自建前端**：不依赖数据集的框与身份

产出与 `h4d_boxes.py` 同契约的 LabelMe 检测树（group_id=0/1），
因此 `vp_h4d → assemble_h4d → tri_h4d → fit_h4d → eval/视频` **整条链一行不改**。

相对 v1（失败版）补的三条约束（v1 把 48 条轨迹并成 1 组，选人 0%）：
  ★A3 **多视角验证分组**：v1 用"两两射线一致"的**连通分量** → 会链式合并。
     改成**增量式**：以面积为种子，逐个尝试把别的轨迹并入，
     并入条件是"**组内已有成员三角化出的 3D 轨迹**反投影到新成员视角后的归一化残差中位 < τ"
     → 等价于要求"与该组**全体**一致"，而不是"与某一个成员一致"。
  ★A3b 组必须覆盖 **≥ min_views 个视角**（真受试者多视角可见；瞬时假检/器材不是）。
  ★B3 **3D 互斥**：第二组与第一组的 3D 轨迹必须保持距离（两人不能占同一 3D 位置）。
  ★D3/D4 **数量硬约束 + 联合选一对**：只输出 2 组，且优先选"面积大 + 两人互相靠近（对打）"的那一对。

用法（容器内）：
  python det_ours_select.py --frames-root $ROOT/frames/15/4 --out $ROOT/det_ours_final \
      --views 01,03,04,07,09,14 --tag 016_mma4 --start 1 --end 61 \
      [--topk 8] [--border-margin 0.10] [--mincover 0.5] [--res-tau 0.30] \
      [--min-views 3] [--pick 2] [--excl-dist 0.5]
"""
import os, sys, json, argparse, importlib.util
import numpy as np
import cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')


def load_v2():
    p = os.path.join(B, 'code', 'adapters', 'harmony4d', 'pipeline', 'detect_h4d_v2.py')
    spec = importlib.util.spec_from_file_location('det2', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--tag', default='016_mma4')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--nmax', type=int, default=12)
    ap.add_argument('--border-margin', type=float, default=0.10)
    ap.add_argument('--mincover', type=float, default=0.15, help='轨迹覆盖率门槛。★不要按帧数缩放：长视频里 tracker 会把受试者分段，0.5×741=371 帧会把它们全排除（踩过）')
    ap.add_argument('--res-tau', type=float, default=0.30, help='★A3 归一化重投影残差阈值')
    ap.add_argument('--min-views', type=int, default=3, help='★A3b 组内最少视角数')
    ap.add_argument('--pick', type=int, default=2, help='★D3 输出几组（2，或有裁判设 3）')
    ap.add_argument('--excl-dist', type=float, default=0.5, help='★B3 组间 3D 最小距离（按人体高度归一）')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--calib', default=B + '/calib_h4d_016mma4')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--W', type=int, default=3840)
    ap.add_argument('--H', type=int, default=2160)
    a = ap.parse_args()
    V = [v.zfill(2) for v in a.views.split(',')]

    det2 = load_v2()
    rtd = det2.RTD
    rtd.MODEL_PATH = os.environ.get('DET_MODEL', B + '/port/weights/rtdetr-l.pt')
    rtd.ROI_JSON = os.environ.get('DET_ROI', B + '/empty_roi.json')
    rtd.IMG_SIZE, rtd.PERSON_CONF, rtd.DEVICE = a.imgsz, a.conf, a.device
    from ultralytics import RTDETR
    model = RTDETR(rtd.MODEL_PATH)

    sys.path.insert(0, os.environ.get('EMCORE', B + '/port/emcore'))
    from easymocap.mytools.camera_utils import read_camera
    C = read_camera(os.path.join(a.calib, 'intri.yml'), os.path.join(a.calib, 'extri.yml'))
    cam = {}
    for v in V:
        c = C[v]
        K = np.asarray(c['K'], float).reshape(3, 3)
        K[2, 2] = 1.0
        cam[v] = dict(K=K, D=np.asarray(c['dist'], float).reshape(-1)[:4].reshape(4, 1),
                      R=np.asarray(c['R'], float).reshape(3, 3),
                      T=np.asarray(c['T'], float).reshape(3),
                      P=K @ np.hstack([np.asarray(c['R'], float).reshape(3, 3),
                                       np.asarray(c['T'], float).reshape(3, 1)]))

    def undist(uv, v):
        o = cv2.fisheye.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2),
                                        cam[v]['K'], cam[v]['D'], None, None, cam[v]['K'])
        return o.reshape(-1, 2)

    def dlt(pts, views):
        if len(pts) < 2:
            return None
        A = []
        for uv, v in zip(pts, views):
            P = cam[v]['P']
            A.append(uv[0] * P[2] - P[0]); A.append(uv[1] * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.stack(A))
        X = Vt[-1]
        return None if abs(X[3]) < 1e-12 else X[:3] / X[3]

    def reproj(X, v):
        p = cam[v]['P'] @ np.append(np.asarray(X, float), 1.0)
        if abs(p[2]) < 1e-9:
            return None
        return np.array([p[0] / p[2], p[1] / p[2]])

    # ---------- ① + ② 检测 + 逐视角跟踪 ----------
    print('=== ① 检测 + ② 逐视角跟踪 ===', flush=True)
    hist = {}
    for v in V:
        tr = det2.NSlotTracker(rtd, nmax=a.nmax)
        fs = sorted(int(f.split('.')[0]) for f in os.listdir(os.path.join(a.frames_root, v))
                    if f.endswith('.png'))
        fs = [f for f in fs if a.start <= f <= a.end]
        for fr in fs:
            img = cv2.imread(os.path.join(a.frames_root, v, '%06d.png' % fr))
            if img is None:
                continue
            for s in tr.step(det2.candidates_k(rtd, img, model, v, a.topk, a.border_margin)):
                hist.setdefault((v, s.name), {})[fr] = list(s.box)
        print('  view %s: %d 帧, 轨迹 %d 条'
              % (v, len(fs), sum(1 for k in hist if k[0] == v)), flush=True)

    allf = sorted(set(f for h in hist.values() for f in h))
    T = len(allf)
    st = {}
    for k, h in hist.items():
        frs = sorted(h)
        cen = np.array([[(h[f][0] + h[f][2]) / 2, (h[f][1] + h[f][3]) / 2] for f in frs])
        st[k] = dict(frames=frs, boxes=[h[f] for f in frs], cen=cen, idx={f: i for i, f in enumerate(frs)},
                     cover=len(frs) / max(T, 1),
                     area=float(np.median([(h[f][2] - h[f][0]) * (h[f][3] - h[f][1]) for f in frs])),
                     cmed=np.median(cen, axis=0))

    # ---------- ★A3 增量式多视角分组（对照"组内 3D"验证，而不是 pairwise 连通）----------
    print('\n=== ★A3 增量式多视角分组（残差τ=%.2f, 最少视角数=%d）===' % (a.res_tau, a.min_views), flush=True)
    def anc(k, f):
        """★锚点 = 头顶 top-center。实测判别力最好（同人 p90 0.299 < 不同人 p10 0.420）；
        框中心勉强可分（0.349 vs 0.357）；脚点重叠（躺地缠斗时脚不是固定接触点）。"""
        b = st[k]['boxes'][st[k]['idx'][f]]
        return np.array([(b[0] + b[2]) / 2, b[1]])

    # ★同视角重复轨迹合并：同一人被 tracker 拆成多条时，只保留面积大的那条。
    #（实测 view03 的 P0/P2/P6 全部命中同一 subject → 不合并则跨视角关联必乱）
    _keep = sorted([k for k in st if st[k]['cover'] >= a.mincover], key=lambda k: -st[k]['area'])
    _drop = set()
    for _v in V:
        _ks = [k for k in _keep if k[0] == _v]
        for _i in range(len(_ks)):
            for _j in range(_i + 1, len(_ks)):
                _k1, _k2 = _ks[_i], _ks[_j]
                _ov = sorted(set(st[_k1]['frames']) & set(st[_k2]['frames']))
                if len(_ov) < 10:
                    continue
                _ious = []
                for _f in _ov:
                    _b1 = st[_k1]['boxes'][st[_k1]['idx'][_f]]
                    _b2 = st[_k2]['boxes'][st[_k2]['idx'][_f]]
                    _ix1, _iy1 = max(_b1[0], _b2[0]), max(_b1[1], _b2[1])
                    _ix2, _iy2 = min(_b1[2], _b2[2]), min(_b1[3], _b2[3])
                    _inter = max(0., _ix2 - _ix1) * max(0., _iy2 - _iy1)
                    _ua = (_b1[2] - _b1[0]) * (_b1[3] - _b1[1]) + \
                          (_b2[2] - _b2[0]) * (_b2[3] - _b2[1]) - _inter
                    _ious.append(_inter / _ua if _ua > 0 else 0.)
                if np.median(_ious) > 0.5:
                    _drop.add(_k1 if st[_k1]['area'] < st[_k2]['area'] else _k2)
    keys = [k for k in _keep if k not in _drop]
    print('  同视角重复轨迹合并：%d -> %d 条（丢弃 %d）' % (len(_keep), len(keys), len(_drop)), flush=True)

    def group_3d(mem):
        """组内成员逐帧三角化 → {frame: X}（用框中心；≥2 视角的帧才有解）"""
        out = {}
        for f in allf:
            us, vs = [], []
            for k in mem:
                if f in st[k]['idx']:
                    us.append(undist(anc(k, f), k[0])[0]); vs.append(k[0])
            if len(us) >= 2:
                X = dlt(us, vs)
                if X is not None:
                    out[f] = X
        return out

    def accept(mem, k, X3):
        """把轨迹 k 并入 mem 的条件：组内 3D 反投影到 k 的视角后，与 k 的框中心一致"""
        err = []
        for f in st[k]['frames']:
            if f not in X3:
                continue
            p = reproj(X3[f], k[0])
            if p is None:
                continue
            cr = anc(k, f)
            h = st[k]['boxes'][st[k]['idx'][f]][3] - st[k]['boxes'][st[k]['idx'][f]][1]
            err.append(np.linalg.norm(p - cr) / max(h, 1e-6))
        need = max(10, min(60, int(0.3 * T)))   # ★绝对上限 60 帧: 长视频不该要求几百帧重叠
        return (len(err) >= need) and (np.median(err) < a.res_tau), (np.median(err) if err else 9e9)

    def resid_against(mem, X3, k):
        """轨迹 k 相对于"某组 3D 轨迹 X3"的归一化残差中位；样本不足返回 None"""
        if not mem or not X3:
            return None
        err = []
        for f in st[k]['frames']:
            if f not in X3:
                continue
            p = reproj(X3[f], k[0])
            if p is None:
                continue
            b = st[k]['boxes'][st[k]['idx'][f]]
            err.append(np.linalg.norm(p - anc(k, f)) / max(b[3] - b[1], 1e-6))
        if len(err) < max(10, min(60, int(0.3 * T))):
            return None
        return float(np.median(err))

    # ---------- ★A3'' 联合两组生长（残差空间 2-means + 每视角每组最多一条）----------
    #  为什么必须"联合"而不是顺序贪心：长视频里 tracker 的"槽位↔人"可能在不同视角互换，
    #  且贴身时两人 3D 重叠 → 顺序生长会让 A 组把 B 的轨迹吸走、B 组被饿死
    #  （实测 741 帧下 B 只关联到 3 个视角、面积 64973 vs 受试者 ~290k → 选组崩）。
    def resid_pair(k1, k2):
        ov = sorted(set(st[k1]['frames']) & set(st[k2]['frames']))
        if len(ov) < max(10, min(60, int(0.3 * T))):
            return None
        er = []
        for f in ov:
            X = dlt([undist(anc(k1, f), k1[0])[0], undist(anc(k2, f), k2[0])[0]],
                    [k1[0], k2[0]])
            if X is None:
                continue
            e = []
            for kk in (k1, k2):
                q = reproj(X, kk[0])
                if q is None:
                    e = None
                    break
                b = st[kk]['boxes'][st[kk]['idx'][f]]
                e.append(np.linalg.norm(q - anc(kk, f)) / max(b[3] - b[1], 1e-6))
            if e:
                er.append(max(e))
        return float(np.median(er)) if len(er) >= 10 else None

    def best_partner(seed, pool, exclude):
        """在 pool\\exclude 里找与 seed 两视角残差最小的伙伴（不同视角）"""
        best = None
        for k in pool:
            if k in exclude or k[0] == seed[0]:
                continue
            r = resid_pair(seed, k)
            if r is not None and r < a.res_tau and (best is None or r < best[0]):
                best = (r, k)
        return best[1] if best else None

    def grow_two(pool):
        """返回 (memA, memB, XA, XB)

        ★种子必须**成对**（单条轨迹三角化不出 3D → 残差全 None → 谁都分不出去，踩过两次）：
          A: seedA=面积最大者 + 与它最兼容的伙伴
          B: 与 A 组不兼容的最大面积轨迹 + 与它最兼容的伙伴
        然后做"每视角每组最多一条"的联合迭代（含改判）。
        """
        if len(pool) < 4:
            return None
        seedA = pool[0]
        pA = best_partner(seedA, pool, set())
        if pA is None:
            return None
        memA = [seedA, pA]
        XA = group_3d(memA)
        seedB = None
        for k in pool:
            if k in memA or k[0] == seedA[0]:
                continue
            rA = resid_against(memA, XA, k)
            if rA is None or rA >= a.res_tau:       # 与 A 不兼容 ⇒ 另一个人
                seedB = k
                break
        if seedB is None:
            return None
        pB = best_partner(seedB, pool, {seedB})
        if pB is None:
            return None
        memB = [seedB, pB]
        if pB in memA and len(memA) > 2:            # 允许从 A 借一条当伙伴（迭代里会正式改判）
            memA.remove(pB)
            XA = group_3d(memA)
        XB = group_3d(memB)
        for _ in range(12):
            slotA, slotB = {}, {}                  # view -> (resid, tracklet)（每视角每组最多一条）
            for k in pool:
                rA = resid_against(memA, XA, k)
                rB = resid_against(memB, XB, k)
                ra = rA if rA is not None else 9e9
                rb = rB if rB is not None else 9e9
                if min(ra, rb) >= a.res_tau:
                    continue
                tgt = slotA if ra < rb else slotB
                cur = tgt.get(k[0])
                if cur is None or min(ra, rb) < cur[0]:
                    tgt[k[0]] = (min(ra, rb), k)
            nA = [v[1] for _, v in sorted(slotA.items())]
            nB = [v[1] for _, v in sorted(slotB.items())]
            if not nA or not nB:
                return None
            same = (set(nA) == set(memA) and set(nB) == set(memB))
            memA, memB = nA, nB
            XA, XB = group_3d(memA), group_3d(memB)
            if same:
                break
        return memA, memB, XA, XB

    groups, used = [], set()
    _rw = grow_two(keys)
    if _rw:
        for _mem, _X3 in ((_rw[0], _rw[2]), (_rw[1], _rw[3])):
            _vs = sorted(set(m[0] for m in _mem))
            print('  [联合] 候选组：视角数=%d 面积=%.0f 成员=%s'
                  % (len(_vs), max(st[k]['area'] for k in _mem),
                     ','.join('%s.%s' % k for k in _mem)), flush=True)
            if len(_vs) >= a.min_views:
                groups.append(dict(mem=_mem, X3=_X3, views=_vs,
                                   area=max(st[k]['area'] for k in _mem)))
    keys = []          # ★让下面的顺序贪心循环空转（旧代码保留以便对照/回退）
    for seed in keys:
        if seed in used:
            continue
        # ★配对启动：单条轨迹无法三角化（v2 死锁的根因），先用 top 锚点找一条两视角残差
        #   最小的伙伴；注意"两视角 DLT"判别力弱 → 门槛用较严的 res_tau
        best2 = None
        for k in keys:
            if k in used or k[0] == seed[0]:
                continue
            ov = sorted(set(st[seed]['frames']) & set(st[k]['frames']))
            if len(ov) < max(10, min(60, int(0.3 * T))):
                continue
            er = []
            for f in ov:
                X = dlt([undist(anc(seed, f), seed[0])[0], undist(anc(k, f), k[0])[0]],
                        [seed[0], k[0]])
                if X is None:
                    continue
                e = []
                for kk in (seed, k):
                    q = reproj(X, kk[0])
                    if q is None:
                        e = None
                        break
                    b = st[kk]['boxes'][st[kk]['idx'][f]]
                    e.append(np.linalg.norm(q - anc(kk, f)) / max(b[3] - b[1], 1e-6))
                if e:
                    er.append(max(e))
            if len(er) >= 10 and np.median(er) < a.res_tau:
                if best2 is None or np.median(er) < best2[0]:
                    best2 = (float(np.median(er)), k)
        if best2 is None:
            continue
        mem = [seed, best2[1]]
        X3 = group_3d(mem)
        improved = True
        while improved:
            improved = False
            for k in keys:
                if k in mem or k in used or k[0] in [m[0] for m in mem]:
                    continue
                ok, md = accept(mem, k, X3)
                if ok:
                    mem.append(k)
                    X3 = group_3d(mem)
                    improved = True
        # ★改判/抢回（治长视频里 A 组误吸收 B 的轨迹）
        #   贴身缠斗时两人的 3D 叠在一起，A 组的 3D 与 B 的框也"一致" → 贪心生长会把 B 的轨迹
        #   吸进 A；而一旦进了 used，B 组就再也长不起来（实测 741 帧下 B 只到 3 个视角、选组崩）。
        #   这里在 B 组生长后，把"A 组里其实更符合 B"的轨迹搬过来。
        if groups:
            A = groups[0]
            for k in list(A['mem']):
                if k[0] in [m[0] for m in mem]:
                    continue
                rB = resid_against(mem, X3, k)
                rA_ = resid_against(A['mem'], A['X3'], k)
                if rB is not None and rB < a.res_tau and (rA_ is None or rB < 0.7 * rA_):
                    A['mem'].remove(k)
                    A['X3'] = group_3d(A['mem'])
                    mem.append(k)
                    X3 = group_3d(mem)
                    if A['mem']:
                        A['views'] = sorted(set(m[0] for m in A['mem']))
                        A['area'] = max(st[m]['area'] for m in A['mem'])
        if len(mem) >= 2:
            vs = set(m[0] for m in mem)
            if len(vs) >= a.min_views:
                groups.append(dict(mem=mem, X3=X3, views=sorted(vs),
                                   area=max(st[k]['area'] for k in mem)))
                used.update(mem)
    for gi, g in enumerate(groups):
        print('  组%d: 视角数=%d 面积=%.0f 成员=%s'
              % (gi, len(g['views']), g['area'], ','.join('%s.%s' % k for k in g['mem'])), flush=True)

    # ---------- ★D3/D4 选 2 组：面积大 + 两人互相靠近 + ★B3 3D 互斥 ----------
    print('\n=== ★D3/D4 选人（面积 × 互相靠近；★B3 3D 互斥 %.2f）===' % a.excl_dist, flush=True)
    def pair_score(g1, g2):
        com = sorted(set(g1['X3']) & set(g2['X3']))
        if len(com) < max(5, 0.3 * T):
            return None
        d = [np.linalg.norm(g1['X3'][f] - g2['X3'][f]) for f in com]
        return float(np.median(d)), len(com)
    ranked = sorted(groups, key=lambda g: -g['area'])
    chosen = []
    if ranked:
        chosen.append(ranked[0])
        best = None
        for g in ranked[1:]:
            ps = pair_score(chosen[0], g)
            if ps is None:
                continue
            dist, ncom = ps
            if dist < a.excl_dist:                       # ★B3 太近 ⇒ 疑同一人/重复组
                continue
            sc = g['area'] * (1.0 / (1.0 + dist))        # 面积大 + 靠近
            if best is None or sc > best[0]:
                best = (sc, g, dist)
        if best is not None:
            chosen.append(best[1])
            print('  pid1 ← 面积=%.0f 与 pid0 中位 3D 距离=%.3f' % (best[1]['area'], best[2]), flush=True)
        else:
            for g in ranked[1:]:
                if g is not chosen[0]:
                    chosen.append(g); print('  pid1 ← 回退取次高面积组', flush=True); break
    chosen = chosen[:a.pick]
    print('  最终输出 %d 组' % len(chosen), flush=True)

    # ---------- ⑤ 写 LabelMe（pid 恒定）----------
    print('\n=== ⑤ 写 LabelMe（group_id 全片恒定）===', flush=True)
    for v in V:
        od = os.path.join(a.out, v); os.makedirs(od, exist_ok=True)
        fs = sorted(int(f.split('.')[0]) for f in os.listdir(os.path.join(a.frames_root, v))
                    if f.endswith('.png'))
        fs = [f for f in fs if a.start <= f <= a.end]
        n = 0
        for fr in fs:
            shapes = []
            for gi, g in enumerate(chosen):
                for k in g['mem']:
                    if k[0] != v:
                        continue
                    b = hist[k].get(fr)
                    if b is None:
                        break
                    bb = rtd.expand_head(b, a.W, a.H)
                    shapes.append({"label": "person",
                                   "points": [[float(bb[0]), float(bb[1])], [float(bb[2]), float(bb[3])]],
                                   "group_id": gi, "description": "ours:%s" % k[1],
                                   "shape_type": "rectangle", "flags": {}})
                    break
            json.dump(rtd.build_labelme('%s_%06d.jpg' % (a.tag, fr), a.H, a.W, shapes),
                      open(os.path.join(od, '%s_%06d.json' % (a.tag, fr)), 'w'),
                      ensure_ascii=False, indent=1)
            n += 1
        print('  view %s: %d 帧' % (v, n), flush=True)
    print('[OK] -> %s（★不含任何数据集框/身份信息）' % a.out, flush=True)


if __name__ == '__main__':
    main()
