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
"""det_ours_v4.py — 全自建前端 v4：**逐帧 3D 引导的两人框分配（E/M 迭代）**

为什么换思路（v3 在 741 帧上崩的原因）：
   v3 在**轨迹级**做跨视角分组 → 一旦 tracker 把一个人拆成多段、或两人贴身时槽位互换，
   分组就会串（实测两组各只覆盖 3–4 视角、MPJPE 1.1m）。轨迹级判据对这三类误差没有回旋余地。

v4 的骨架：
 ① 检测 + 逐视角跟踪（复用 detect_h4d_v2，NO-SWAP + EMA + coast）
 ② ★**轨道段拼接**：同视角内、时间不重叠（或微重叠）、运动连续且尺寸相近的段合并
      → 治"一个人被 tracker 拆成多段"
 ③ 初始化：轨迹级联合两组（面积最大的两条互不兼容轨迹作种子，成对生长）→ 得到**部分帧**的 XA/XB
 ④ ★**逐帧 3D 引导分配（迭代 4 轮）**：
      E：每帧每视角，把该视角的活跃候选框按"反投影到 XA/XB 的距离/框高"分配给 A/B（带 τ 拒绝）
      M：用分配结果**逐帧重新三角化** XA/XB
    → 治"槽位互换"（每帧独立决策，不依赖轨迹编号）、治"贴身重叠"（靠 3D 距离仲裁）、
       治"分段"（拼接后仍缺的帧由 M 步从其他视角补回来）
 ⑤ 写 LabelMe（group_id 恒定为 A=0 / B=1）

用法（容器内）：
  python det_ours_v4.py --frames-root $ROOT/frames/15/4 --out $ROOT/det_v4 \
      --views 01,03,04,07,09,14 --tag 016_mma4 --start 1 --end 742 \
      [--topk 8] [--border-margin 0.10] [--res-tau 0.35] [--stitch-gap 60] [--rounds 4]
"""
import os, sys, json, argparse, importlib.util
import numpy as np
import cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')


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
    ap.add_argument('--res-tau', type=float, default=0.35)
    ap.add_argument('--stitch-gap', type=int, default=60, help='★轨道段拼接允许的最大时间间隔(帧)')
    ap.add_argument('--rounds', type=int, default=4, help='★E/M 迭代轮数')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--calib', default=B + '/calib_h4d_016mma4')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--W', type=int, default=3840)
    ap.add_argument('--H', type=int, default=2160)
    a = ap.parse_args()
    V = [v.zfill(2) for v in a.views.split(',')]

    # ---- 复用 detect_h4d_v2 的检测/跟踪 ----
    p = os.path.join(B, 'code', 'adapters', 'harmony4d', 'pipeline', 'detect_h4d_v2.py')
    spec = importlib.util.spec_from_file_location('det2', p)
    det2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(det2)
    rtd = det2.RTD
    rtd.MODEL_PATH = B + '/port/weights/rtdetr-l.pt'; rtd.ROI_JSON = B + '/empty_roi.json'
    rtd.IMG_SIZE, rtd.PERSON_CONF, rtd.DEVICE = a.imgsz, a.conf, a.device
    from ultralytics import RTDETR
    model = RTDETR(rtd.MODEL_PATH)

    sys.path.insert(0, B + '/port/emcore')
    from easymocap.mytools.camera_utils import read_camera
    C = read_camera(os.path.join(a.calib, 'intri.yml'), os.path.join(a.calib, 'extri.yml'))
    cam = {}
    for v in V:
        c = C[v]
        K = np.asarray(c['K'], float).reshape(3, 3); K[2, 2] = 1.0
        cam[v] = dict(K=K, D=np.asarray(c['dist'], float).reshape(-1)[:4].reshape(4, 1),
                      P=K @ np.hstack([np.asarray(c['R'], float).reshape(3, 3),
                                       np.asarray(c['T'], float).reshape(3, 1)]))

    def und(uv, v):
        return cv2.fisheye.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2),
                                           cam[v]['K'], cam[v]['D'], None, None,
                                           cam[v]['K']).reshape(-1, 2)

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
        q = cam[v]['P'] @ np.append(np.asarray(X, float), 1.0)
        return None if abs(q[2]) < 1e-9 else np.array([q[0] / q[2], q[1] / q[2]])

    def anc(box):
        """锚点：头顶 top-center（实测判别力最好）"""
        return np.array([(box[0] + box[2]) / 2, box[1]])

    def hof(box):
        return max(box[3] - box[1], 1e-6)

    # ---------- ①② 检测 + 跟踪 + 轨道段拼接 ----------
    print('=== ①② 检测 + 跟踪 + 轨道段拼接 ===', flush=True)
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
        print('  view %s: %d 帧, 轨迹 %d 条' % (v, len(fs), sum(1 for k in hist if k[0] == v)), flush=True)

    # 拼接：同视角内，时间不重叠(或微重叠)、运动连续、尺寸相近 → 合并
    n_before = len(hist)
    groups_by_view = {v: [k for k in hist if k[0] == v] for v in V}
    alive = {k: True for k in hist}
    for v in V:
        ks = groups_by_view[v]
        changed = True
        while changed:
            changed = False
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    k1, k2 = ks[i], ks[j]
                    if not (alive[k1] and alive[k2]):
                        continue
                    h1, h2 = hist[k1], hist[k2]
                    f1e, f2s = max(h1), min(h2)
                    if f1e < f2s:
                        pass
                    elif max(h2) < min(h1):
                        k1, k2, f1e, f2s = k2, k1, max(h2), min(h1)
                    else:
                        continue                     # 时间大量重叠 → 不合并（可能是两个人）
                    if 0 < (f2s - f1e) > a.stitch_gap:
                        continue
                    # 运动连续：k1 末尾位移外推到 f2s 处与 k2 首帧中心的距离（按框高归一）
                    fr_a, fr_b = sorted(h1)[-1], sorted(h2)[0]
                    fr_p = sorted(h1)[max(0, len(h1) - 3)]
                    c_a, c_p = anc(h1[fr_a]), anc(h1[fr_p])
                    dur = max(fr_a - fr_p, 1)
                    pred = c_a + (c_a - c_p) / dur * (fr_b - fr_a)
                    gapn = np.linalg.norm(pred - anc(h2[fr_b])) / hof(h1[fr_a])
                    ar1 = np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in h1.values()])
                    ar2 = np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in h2.values()])
                    ratio = max(ar1, ar2) / max(min(ar1, ar2), 1e-6)
                    if gapn < 1.0 and ratio < 2.5:
                        hist[k1].update(hist[k2])
                        alive[k2] = False
                        changed = True
    hist = {k: h for k, h in hist.items() if alive[k]}
    print('  拼接：%d -> %d 条轨迹' % (n_before, len(hist)), flush=True)

    # 轨迹统计
    allf = sorted(set(f for h in hist.values() for f in h))
    T = len(allf)
    st = {}
    for k, h in hist.items():
        frs = sorted(h)
        st[k] = dict(frames=frs, boxes=[h[f] for f in frs],
                     idx={f: i for i, f in enumerate(frs)},
                     area=float(np.median([(h[f][2] - h[f][0]) * (h[f][3] - h[f][1]) for f in frs])),
                     cover=len(frs) / max(T, 1))
    keys = sorted([k for k in st if st[k]['cover'] >= 0.05], key=lambda k: -st[k]['area'])
    print('  可用轨迹 %d 条（cover>=0.05）' % len(keys), flush=True)

    # ---------- ③ 初始化：轨迹级联合两组（成对种子）----------
    def g3d(mem):
        out = {}
        for f in allf:
            us, vs = [], []
            for k in mem:
                if f in st[k]['idx']:
                    us.append(und(anc(st[k]['boxes'][st[k]['idx'][f]]), k[0])[0]); vs.append(k[0])
            if len(us) >= 2:
                X = dlt(us, vs)
                if X is not None:
                    out[f] = X
        return out

    def rpair(k1, k2):
        ov = sorted(set(st[k1]['frames']) & set(st[k2]['frames']))
        if len(ov) < max(10, min(60, int(0.3 * T))):
            return None
        er = []
        for f in ov:
            X = dlt([und(anc(st[k1]['boxes'][st[k1]['idx'][f]]), k1[0])[0],
                     und(anc(st[k2]['boxes'][st[k2]['idx'][f]]), k2[0])[0]], [k1[0], k2[0]])
            if X is None:
                continue
            e = []
            for kk in (k1, k2):
                q = reproj(X, kk[0])
                if q is None:
                    e = None; break
                e.append(np.linalg.norm(q - anc(st[kk]['boxes'][st[kk]['idx'][f]])) /
                         hof(st[kk]['boxes'][st[kk]['idx'][f]]))
            if e:
                er.append(max(e))
        return float(np.median(er)) if len(er) >= 10 else None

    def rgroup(X3, k):
        if not X3:
            return None
        er = []
        for f in st[k]['frames']:
            if f not in X3:
                continue
            q = reproj(X3[f], k[0])
            if q is None:
                continue
            er.append(np.linalg.norm(q - anc(st[k]['boxes'][st[k]['idx'][f]])) /
                      hof(st[k]['boxes'][st[k]['idx'][f]]))
        if len(er) < max(10, min(60, int(0.3 * T))):
            return None
        return float(np.median(er))

    seedA = keys[0]
    pA = None
    for k in keys[1:]:
        if k[0] == seedA[0]:
            continue
        r = rpair(seedA, k)
        if r is not None and r < a.res_tau and (pA is None or r < pA[0]):
            pA = (r, k)
    print('=== ③ 初始化：种子对 ===', flush=True)
    if pA is None:
        print('  ⚠️ 找不到配对伙伴，退出'); return
    memA = [seedA, pA[1]]
    XA = g3d(memA)
    seedB = None
    for k in keys:
        if k in memA or k[0] == seedA[0]:
            continue
        r = rgroup(XA, k)
        if r is None or r >= a.res_tau:
            seedB = k; break
    if seedB is None:
        print('  ⚠️ 找不到第二个种子，退出'); return
    pB = None
    for k in keys:
        if k[0] == seedB[0]:
            continue
        r = rpair(seedB, k)
        if r is not None and r < a.res_tau and (pB is None or r < pB[0]):
            pB = (r, k)
    if pB is None:
        print('  ⚠️ 第二组找不到伙伴，退出'); return
    memB = [seedB, pB[1]]
    XB = g3d(memB)
    print('  A=%s   B=%s' % (','.join('%s.%s' % k for k in memA),
                             ','.join('%s.%s' % k for k in memB)), flush=True)

    # ---------- ④ 逐帧 3D 引导分配（E/M 迭代）----------
    print('=== ④ 逐帧 3D 引导分配（%d 轮）===' % a.rounds, flush=True)
    # 初始分配：来自轨迹级种子组（覆盖部分帧）
    assign = {}                    # (fr, view) -> {0: box, 1: box}
    for f in allf:
        for v in V:
            got = {}
            for gi, mem in enumerate((memA, memB)):
                for k in mem:
                    if k[0] == v and f in st[k]['idx']:
                        got[gi] = st[k]['boxes'][st[k]['idx'][f]]
                        break
            if got:
                assign[(f, v)] = got
    for rd in range(a.rounds):
        # ---- M：逐帧重新三角化 ----
        newXA, newXB = {}, {}
        for f in allf:
            for gi, out in ((0, newXA), (1, newXB)):
                us, vs = [], []
                for v in V:
                    b = assign.get((f, v), {}).get(gi)
                    if b is None:
                        continue
                    us.append(und(anc(b), v)[0]); vs.append(v)
                if len(us) >= 2:
                    X = dlt(us, vs)
                    if X is not None:
                        out[f] = X
        # ---- E：每帧每视角重新分配（含"拒绝"）----
        n_chg = 0
        newa = {}
        for f in allf:
            for v in V:
                # 该视角该帧的候选框 = 所有活跃轨迹的框
                cands = [st[k]['boxes'][st[k]['idx'][f]] for k in st if k[0] == v and f in st[k]['idx']]
                if not cands:
                    continue
                C = np.full((len(cands), 2), 9e9)
                for gi, X3 in ((0, newXA), (1, newXB)):
                    if f not in X3:
                        continue
                    q = reproj(X3[f], v)
                    if q is None:
                        continue
                    for bi, b in enumerate(cands):
                        C[bi, gi] = np.linalg.norm(q - anc(b)) / hof(b)
                best = None
                # 显式枚举两人框（候选 ≤8，规模极小）
                for bi in range(len(cands)):
                    for bj in range(len(cands)):
                        if bi == bj:
                            continue
                        if C[bi, 0] >= a.res_tau or C[bj, 1] >= a.res_tau:
                            continue
                        tot = C[bi, 0] + C[bj, 1]
                        if best is None or tot < best[0]:
                            best = (tot, bi, bj)
                if best is None:
                    continue
                got = {0: cands[best[1]], 1: cands[best[2]]}
                if assign.get((f, v)) != got:
                    n_chg += 1
                newa[(f, v)] = got
        # 未分配的 (帧,视角) 保留上一轮（跨轮 carry）
        for kk, vv in assign.items():
            if kk not in newa:
                newa[kk] = vv
        assign = newa
        cov = {0: sum(1 for (f, v), g in assign.items() if 0 in g),
               1: sum(1 for (f, v), g in assign.items() if 1 in g)}
        print('  轮 %d：变更 %5d 处；A 覆盖 %d 个(帧,视角)，B 覆盖 %d'
              % (rd + 1, n_chg, cov[0], cov[1]), flush=True)
        if n_chg == 0:
            break

    # ---------- ⑤ 写 LabelMe ----------
    print('=== ⑤ 写 LabelMe ===', flush=True)
    for v in V:
        od = os.path.join(a.out, v); os.makedirs(od, exist_ok=True)
        fs = sorted(int(f.split('.')[0]) for f in os.listdir(os.path.join(a.frames_root, v))
                    if f.endswith('.png'))
        fs = [f for f in fs if a.start <= f <= a.end]
        n = 0
        for fr in fs:
            shapes = []
            for gi in (0, 1):
                b = assign.get((fr, v), {}).get(gi)
                if b is None:
                    continue
                bb = rtd.expand_head(b, a.W, a.H)
                shapes.append({"label": "person",
                               "points": [[float(bb[0]), float(bb[1])], [float(bb[2]), float(bb[3])]],
                               "group_id": gi, "description": "v4:%d" % gi,
                               "shape_type": "rectangle", "flags": {}})
            json.dump(rtd.build_labelme('%s_%06d.jpg' % (a.tag, fr), a.H, a.W, shapes),
                      open(os.path.join(od, '%s_%06d.json' % (a.tag, fr)), 'w'),
                      ensure_ascii=False, indent=1)
            n += 1
        print('  view %s: %d 帧' % (v, n), flush=True)
    print('[OK] -> %s（★不含任何数据集框/身份信息）' % a.out, flush=True)


if __name__ == '__main__':
    main()
