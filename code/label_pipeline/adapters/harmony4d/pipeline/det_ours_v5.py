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
"""det_ours_v5.py — 全自建前端 v5：**逐帧 3D 跟踪式分配（前向/后向推进）**

v3（轨迹级分组）与 v4（批式 E/M）在 741 帧上都栽了，病根相同：
  **3D 只来自"本来就已经选对的那些框"** → 一旦某帧选错，误差在批式迭代里全局扩散。
v5 改成**因果推进**：用**上一帧已验证的 3D** 去预测/挑选当前帧的框，每帧独立决策，
误差不跨帧扩散（缺观测时"恒速 coast"，不往回污染）。

流程：
 ① 检测 + 逐视角跟踪（复用 detect_h4d_v2）
 ② 轨道段拼接（同视角、时间不重叠、运动连续、尺寸相近 → 合并）
 ③ 种子对 → 在前若干帧三角化出 XA/XB → 取**最长连续核心段**
 ④ ★前向/后向逐帧推进：预测 X(t) ← X(t-1) → 每视角按反投影距离选框（带 τ 拒绝）
    → ≥2 视角则重新三角化，否则 coast
 ⑤ 写 LabelMe（A=0 / B=1 恒定）

用法同 v4。
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
    ap.add_argument('--stitch-gap', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--calib', default=B + '/calib_h4d_016mma4')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--W', type=int, default=3840)
    ap.add_argument('--H', type=int, default=2160)
    a = ap.parse_args()
    V = [v.zfill(2) for v in a.views.split(',')]

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
        return np.array([(box[0] + box[2]) / 2, box[1]])          # 头顶锚点（实测最优）

    def hof(box):
        return max(box[3] - box[1], 1e-6)

    # ---------- ①② 检测 + 跟踪 + 拼接 ----------
    print('=== ①② 检测 + 跟踪 + 拼接 ===', flush=True)
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
    n_before = len(hist)
    alive = {k: True for k in hist}
    for v in V:
        ks = [k for k in hist if k[0] == v]
        changed = True
        while changed:
            changed = False
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    k1, k2 = ks[i], ks[j]
                    if not (alive[k1] and alive[k2]):
                        continue
                    if max(hist[k2]) < min(hist[k1]):
                        k1, k2 = k2, k1
                    f1e, f2s = max(hist[k1]), min(hist[k2])
                    if f1e >= f2s or (f2s - f1e) > a.stitch_gap:
                        continue
                    fa, fb = sorted(hist[k1])[-1], sorted(hist[k2])[0]
                    fp = sorted(hist[k1])[max(0, len(hist[k1]) - 3)]
                    ca, cp = anc(hist[k1][fa]), anc(hist[k1][fp])
                    pred = ca + (ca - cp) / max(fa - fp, 1) * (fb - fa)
                    gapn = np.linalg.norm(pred - anc(hist[k2][fb])) / hof(hist[k1][fa])
                    a1 = np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in hist[k1].values()])
                    a2 = np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in hist[k2].values()])
                    if gapn < 1.0 and max(a1, a2) / max(min(a1, a2), 1e-6) < 2.5:
                        hist[k1].update(hist[k2]); alive[k2] = False; changed = True
    hist = {k: h for k, h in hist.items() if alive[k]}
    print('  拼接：%d -> %d 条' % (n_before, len(hist)), flush=True)

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

    # ---------- ③ 种子对 + 核心段 ----------
    def tri2(k1, k2):
        """两条轨迹共同帧上的 3D（用它们自己的框）"""
        out = {}
        for f in sorted(set(st[k1]['frames']) & set(st[k2]['frames'])):
            X = dlt([und(anc(st[k1]['boxes'][st[k1]['idx'][f]]), k1[0])[0],
                     und(anc(st[k2]['boxes'][st[k2]['idx'][f]]), k2[0])[0]], [k1[0], k2[0]])
            if X is not None:
                out[f] = X
        return out

    def med2(k1, k2, X3):
        er = []
        for f, X in X3.items():
            e = []
            for kk in (k1, k2):
                q = reproj(X, kk[0])
                if q is None:
                    e = None; break
                e.append(np.linalg.norm(q - anc(st[kk]['boxes'][st[kk]['idx'][f]])) /
                         hof(st[kk]['boxes'][st[kk]['idx'][f]]))
            if e:
                er.append(max(e))
        return float(np.median(er)) if er else None

    print('=== ③ 种子对 + 核心段 ===', flush=True)
    seedA = keys[0]
    pA, bestA = None, None
    for k in keys[1:]:
        if k[0] == seedA[0]:
            continue
        X3 = tri2(seedA, k)
        if len(X3) < max(30, 0.1 * T):
            continue
        m = med2(seedA, k, X3)
        if m is not None and m < a.res_tau and (bestA is None or m < bestA[0]):
            bestA = (m, k)
    if bestA is None:
        print('  ⚠️ 种子 A 找不到伙伴'); return
    pA = bestA[1]
    XA = tri2(seedA, pA)
    print('  A 种子 %s.%s + %s.%s（共同帧 %d，中位残差 %.3f）'
          % (seedA[0], seedA[1], pA[0], pA[1], len(XA), bestA[0]), flush=True)
    seedB, bestB = None, None
    for k in keys:
        if k in (seedA, pA) or k[0] == seedA[0]:
            continue
        X3 = tri2(seedA, k) if k[0] != pA[0] else {}
        # 与 A 组不兼容：用 A 的 3D 反投影到 k 的视角看残差
        er = []
        for f in st[k]['frames']:
            if f not in XA:
                continue
            q = reproj(XA[f], k[0])
            if q is None:
                continue
            er.append(np.linalg.norm(q - anc(st[k]['boxes'][st[k]['idx'][f]])) /
                      hof(st[k]['boxes'][st[k]['idx'][f]]))
        if len(er) < max(30, 0.1 * T):
            continue
        m = float(np.median(er))
        if m >= a.res_tau and (bestB is None or st[k]['area'] > bestB[1]):
            bestB = (m, st[k]['area'], k)
    if bestB is None:
        print('  ⚠️ 找不到第二个人'); return
    seedB = bestB[2]
    pB, bestPB = None, None
    for k in keys:
        if k[0] == seedB[0]:
            continue
        X3 = tri2(seedB, k)
        if len(X3) < max(30, 0.1 * T):
            continue
        m = med2(seedB, k, X3)
        if m is not None and m < a.res_tau and (bestPB is None or m < bestPB[0]):
            bestPB = (m, k)
    if bestPB is None:
        print('  ⚠️ 种子 B 找不到伙伴'); return
    pB = bestPB[1]
    XB = tri2(seedB, pB)
    print('  B 种子 %s.%s + %s.%s（共同帧 %d，中位残差 %.3f）'
          % (seedB[0], seedB[1], pB[0], pB[1], len(XB), bestPB[0]), flush=True)

    both = [f for f in allf if f in XA and f in XB]
    if not both:
        print('  ⚠️ 无共同核心帧'); return
    runs, cur = [], [both[0]]
    for f in both[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            runs.append(cur); cur = [f]
    runs.append(cur)
    core = max(runs, key=len)
    print('  核心连续段 %d..%d（%d 帧）' % (core[0], core[-1], len(core)), flush=True)

    # ---------- ④ 逐帧 3D 跟踪式推进 ----------
    def assign_frame(f, Xa, Xb):
        got = {}
        for v in V:
            cands = [st[k]['boxes'][st[k]['idx'][f]] for k in st if k[0] == v and f in st[k]['idx']]
            if not cands:
                continue
            C = np.full((len(cands), 2), 9e9)
            for gi, Xp in ((0, Xa), (1, Xb)):
                if Xp is None:
                    continue
                q = reproj(Xp, v)
                if q is None:
                    continue
                for bi, b in enumerate(cands):
                    C[bi, gi] = np.linalg.norm(q - anc(b)) / hof(b)
            best = None
            for bi in range(len(cands)):
                for bj in range(len(cands)):
                    if bi == bj:
                        continue
                    if C[bi, 0] >= a.res_tau or C[bj, 1] >= a.res_tau:
                        continue
                    tot = C[bi, 0] + C[bj, 1]
                    if best is None or tot < best[0]:
                        best = (tot, bi, bj)
            if best is not None:
                got[v] = {0: cands[best[1]], 1: cands[best[2]]}
        return got

    def tri_from(got, gi):
        us, vs = [], []
        for v, g in got.items():
            b = g.get(gi)
            if b is None:
                continue
            us.append(und(anc(b), v)[0]); vs.append(v)
        return dlt(us, vs) if len(us) >= 2 else None

    print('=== ④ 逐帧 3D 跟踪式推进 ===', flush=True)
    assign = {}
    for f in core:                                  # 核心段：直接用已解出的 XA/XB
        got = assign_frame(f, XA[f], XB[f])
        if got:
            assign[f] = got

    def sweep(frames, Xa, Xb, tag):
        n = 0
        for f in frames:
            got = assign_frame(f, Xa, Xb)
            if not got:
                continue
            assign[f] = got
            na, nb = tri_from(got, 0), tri_from(got, 1)
            if na is not None:
                Xa = na
            if nb is not None:
                Xb = nb
            n += 1
        print('  %s推进：写入 %d 帧' % (tag, n), flush=True)
        return n
    sweep([f for f in allf if f > core[-1]], XA[core[-1]], XB[core[-1]], '前向')
    sweep([f for f in allf if f < core[0]][::-1], XA[core[0]], XB[core[0]], '后向')

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
                b = assign.get(fr, {}).get(v, {}).get(gi)
                if b is None:
                    continue
                bb = rtd.expand_head(b, a.W, a.H)
                shapes.append({"label": "person",
                               "points": [[float(bb[0]), float(bb[1])], [float(bb[2]), float(bb[3])]],
                               "group_id": gi, "description": "v5:%d" % gi,
                               "shape_type": "rectangle", "flags": {}})
            json.dump(rtd.build_labelme('%s_%06d.jpg' % (a.tag, fr), a.H, a.W, shapes),
                      open(os.path.join(od, '%s_%06d.json' % (a.tag, fr)), 'w'),
                      ensure_ascii=False, indent=1)
            n += 1
        print('  view %s: %d 帧' % (v, n), flush=True)
    print('[OK] -> %s（★不含任何数据集框/身份信息）' % a.out, flush=True)


if __name__ == '__main__':
    main()
