#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""select3d_probe.py — 选人新方案的**第 1 步探针**：跨视角候选三角化 → 3D 点 → 反投影回原图

思路（数据无关）：
  1. 每个视角取**全部**候选框（conf≥0.15，按面积保留前 N）
  2. 框中心当"人体中心"的代理 → 跨视角两两 DLT 三角化 → 残差(按框高归一)小的组合
  3. 组成连通分量（节点 = (视角, 候选)），要求分量覆盖 ≥MIN_V 个视角 → 得到"3D 人候选"
  4. 分量的 3D 点 DLT 求解 → **鱼眼正投影回原图**，画圈 + 标注"覆盖了几个视角"
  5. 同时打印这些 3D 点的世界坐标 → 用来判断"受试者在场地中央、围观者贴边"

⚠️ 认识（重要）：**几何一致性区分不了"受试者 vs 围观者"**——围观者也是真人、也多视角自洽。
   它只能剔除"器材/反光"这类非人误检。区分受试者要靠 **3D 位置(活动区) + 运动量 + 时序覆盖**。
   本探针就是为下一步（在 3D 空间里选人）打地基，并把结果可视化回原图。

用法：
  python select3d_probe.py --view 01 --frames 5,40 --out <dir> --tri-lam 1.0 [--topn 12] [--minv 3]
"""
import os, sys, json, argparse
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
V_ALL = ['01', '03', '04', '07', '09', '14']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default='01')
    ap.add_argument('--frames', default='5,40')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cams', default=B + '/calib_h4d_016mma4')
    ap.add_argument('--views', default=','.join(V_ALL))
    ap.add_argument('--topn', type=int, default=12)
    ap.add_argument('--minv', type=int, default=3)
    ap.add_argument('--res-tau', type=float, default=0.35, help='归一化重投影残差阈值(按框高)')
    ap.add_argument('--scale', type=float, default=0.55)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    V = [v.zfill(2) for v in a.views.split(',')]

    sys.path.insert(0, os.environ.get('EMCORE', B + '/port/emcore'))
    from easymocap.mytools.camera_utils import read_camera
    C = read_camera(os.path.join(a.cams, 'intri.yml'), os.path.join(a.cams, 'extri.yml'))
    cam = {}
    for v in V:
        c = C[v]
        cam[v] = dict(K=np.asarray(c['K'], float).reshape(3, 3),
                      D=np.asarray(c['dist'], float).reshape(-1)[:4].reshape(4, 1),
                      R=np.asarray(c['R'], float).reshape(3, 3),
                      T=np.asarray(c['T'], float).reshape(3),
                      rvec=cv2.Rodrigues(np.ascontiguousarray(np.asarray(c['R'], float).reshape(3, 3)))[0])

    import importlib.util
    from ultralytics import RTDETR
    sp = importlib.util.spec_from_file_location('rtd', B + '/code/label_pipeline/stages/1_detect/rtdetr_pipeline.py')
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    model = RTDETR(B + '/port/weights/rtdetr-l.pt')

    def undist(uv, v):
        o = cv2.fisheye.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2),
                                        cam[v]['K'], cam[v]['D'], None, None, cam[v]['K'])
        return o.reshape(-1, 2)

    def dlt(pts, views):
        A = []
        for uv, v in zip(pts, views):
            P = cam[v]['K'] @ np.hstack([cam[v]['R'], cam[v]['T'].reshape(3, 1)])
            A.append(uv[0] * P[2] - P[0]); A.append(uv[1] * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.stack(A))
        X = Vt[-1]
        return None if abs(X[3]) < 1e-12 else X[:3] / X[3]

    def proj(X, v):
        Xc = cam[v]['R'] @ np.asarray(X, float) + cam[v]['T']
        if Xc[2] <= 0.05:
            return None
        p, _ = cv2.fisheye.projectPoints(Xc.reshape(1, 1, 3), np.zeros(3), np.zeros((3, 1)),
                                         cam[v]['K'], cam[v]['D'])
        return p.reshape(2)

    for fr in [int(x) for x in a.frames.split(',')]:
        # ---- 每视角候选 ----
        cand = {}
        for v in V:
            img = cv2.imread(os.path.join(B, 'frames/15/4', v, '%06d.png' % fr))
            if img is None:
                continue
            r = model.predict(img, conf=0.15, imgsz=640, classes=[0], device=0, verbose=False)[0]
            if r.boxes is None:
                continue
            bs = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            order = np.argsort(-((bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])))
            bs, cf = bs[order][:a.topn], cf[order][:a.topn]
            cand[v] = [(b, c, np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]),
                        float(b[3] - b[1])) for b, c in zip(bs, cf)]
        # ---- 两两三角化 -> 边 ----
        nodes = [(v, i) for v in V for i in range(len(cand.get(v, [])))]
        parent = {n: n for n in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        for vi in range(len(V)):
            for vj in range(vi + 1, len(V)):
                v1, v2 = V[vi], V[vj]
                if v1 not in cand or v2 not in cand:
                    continue
                for i, (b1, c1, u1, h1) in enumerate(cand[v1]):
                    for j, (b2, c2, u2, h2) in enumerate(cand[v2]):
                        X = dlt([undist(u1, v1)[0], undist(u2, v2)[0]], [v1, v2])
                        if X is None:
                            continue
                        e = []
                        for v, u, h in ((v1, u1, h1), (v2, u2, h2)):
                            p = proj(X, v)
                            if p is None:
                                e = None; break
                            e.append(np.linalg.norm(p - u) / max(h, 1e-6))
                        if e and max(e) < a.res_tau:
                            union((v1, i), (v2, j))
        # ---- 连通分量 -> 3D 点 ----
        comp = {}
        for n in nodes:
            comp.setdefault(find(n), []).append(n)
        pts3d = []
        for root, mem in comp.items():
            vs = sorted(set(x[0] for x in mem))
            if len(vs) < a.minv:
                continue
            us, vv = [], []
            for v, i in mem:
                b, c, u, h = cand[v][i]
                us.append(undist(u, v)[0]); vv.append(v)
            X = dlt(us, vv)
            if X is None:
                continue
            pts3d.append(dict(X=X, nviews=len(vs), views=vs))
        # ---- 可视化：目标视角 ----
        v0 = a.view.zfill(2)
        img = cv2.imread(os.path.join(B, 'frames/15/4', v0, '%06d.png' % fr)).copy()
        H, W = img.shape[:2]
        for k, d in enumerate(sorted(pts3d, key=lambda z: -z['nviews'])):
            p = proj(d['X'], v0)
            if p is None:
                continue
            x, y = int(p[0]), int(p[1])
            if not (0 <= x < W and 0 <= y < H):
                continue
            col = (0, 200, 0) if d['nviews'] >= 4 else (0, 165, 255)
            cv2.circle(img, (x, y), 14, col, 4, cv2.LINE_AA)
            cv2.putText(img, '%dv' % d['nviews'], (x + 16, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3, cv2.LINE_AA)
        # 灰框：该视角全部候选
        for b, c, u, h in cand.get(v0, []):
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (150, 150, 150), 2)
        cv2.putText(img, 'view %s f%03d  3D 候选 %d 个（圆圈=跨视角三角化, 标数字=覆盖视角数）'
                    % (v0, fr, len(pts3d)), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        outp = os.path.join(a.out, 'sel3d_%s_%06d.jpg' % (v0, fr))
        cv2.imwrite(outp, cv2.resize(img, None, fx=a.scale, fy=a.scale), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print('  frame %d: 3D 候选 %d 个' % (fr, len(pts3d)))
        for d in sorted(pts3d, key=lambda z: -z['nviews'])[:8]:
            print('     %dv views=%s  X=%s' % (d['nviews'], ','.join(d['views']), np.round(d['X'], 3)))
        print('  -> %s' % outp)


if __name__ == '__main__':
    main()
