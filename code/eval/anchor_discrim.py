#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""anchor_discrim.py — 跨视角关联的**锚点判别力**测量

问题：v1/v2 的"框中心三角化"关联要么链式合并（太松）要么一个都不收（太紧）。
可能的根因是**锚点本身不稳定**（框中心随人转身/移动漂移）。
本脚本用真值衡量三种锚点的判别力：
    A 框中心  B 脚点(bottom-center)  C 头顶(top-center)
对每个视角对、每对轨迹，算"两视角 DLT 归一化重投影残差中位"，
再按"是否同一人（用真值 subject 判）"分两组，报各自的分布与可分性（AUC 近似）。

用法：
  python anchor_discrim.py --frames-root <frames/15/4> --seq-root <.../016_mma4> \
      --views 01,03 --start 1 --end 61
"""
import os, sys, json, argparse, importlib.util
import numpy as np
import cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
SUBJ = ('aria01', 'aria02')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--views', default='01,03,04')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=61)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--border-margin', type=float, default=0.10)
    ap.add_argument('--calib', default=B + '/calib_h4d_016mma4')
    ap.add_argument('--imgsz', type=int, default=640)
    a = ap.parse_args()
    V = [v.zfill(2) for v in a.views.split(',')]

    p = os.path.join(B, 'code', 'adapters', 'harmony4d', 'pipeline', 'detect_h4d_v2.py')
    spec = importlib.util.spec_from_file_location('det2', p)
    det2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(det2)
    rtd = det2.RTD
    rtd.MODEL_PATH = B + '/port/weights/rtdetr-l.pt'; rtd.ROI_JSON = B + '/empty_roi.json'
    rtd.IMG_SIZE, rtd.PERSON_CONF, rtd.DEVICE = a.imgsz, 0.15, 0
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

    def undist(uv, v):
        return cv2.fisheye.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2),
                                           cam[v]['K'], cam[v]['D'], None, None,
                                           cam[v]['K']).reshape(-1, 2)

    def dlt(pts, views):
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

    # 检测+跟踪（每视角）
    hist, gtmap = {}, {}
    for v in V:
        tr = det2.NSlotTracker(rtd, nmax=12)
        fs = sorted(int(f.split('.')[0]) for f in os.listdir(os.path.join(a.frames_root, v))
                    if f.endswith('.png'))
        fs = [f for f in fs if a.start <= f <= a.end]
        for fr in fs:
            img = cv2.imread(os.path.join(a.frames_root, v, '%06d.png' % fr))
            if img is None:
                continue
            gtf = os.path.join(a.seq_root, 'processed_data/bbox/cam%s/%05d.npy' % (v, fr))
            if os.path.exists(gtf):
                g = np.load(gtf, allow_pickle=True)
                g = g.item() if getattr(g, 'dtype', None) == object else g
                gtmap[(v, fr)] = {s: np.asarray(x, float).reshape(-1)[:4] for s, x in g.items()}
            for s in tr.step(det2.candidates_k(rtd, img, model, v, a.topk, a.border_margin)):
                hist.setdefault((v, s.name), {})[fr] = list(s.box)

    def anchor(box, kind):
        x1, y1, x2, y2 = box
        if kind == 'center':
            return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        if kind == 'foot':
            return np.array([(x1 + x2) / 2, y2])
        return np.array([(x1 + x2) / 2, y1])          # top
    # 每条轨迹的主 label（用中心口径与真值比）
    def subj_of(k):
        cnt = {s: 0 for s in SUBJ}
        for fr, b in hist[k].items():
            g = gtmap.get((k[0], fr))
            if not g:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            for s, gb in g.items():
                if (gb[0] <= cx <= gb[2] and gb[1] <= cy <= gb[3]):
                    cnt[s] += 1
        s, c = max(cnt.items(), key=lambda z: z[1])
        return s if c > 0 else None

    lab = {k: subj_of(k) for k in hist}
    print('=== 轨迹真值标签 ===')
    for k in sorted(hist):
        print('  %s.%s  label=%s  帧数=%d' % (k[0], k[1], lab[k], len(hist[k])))

    keys = [k for k in hist if lab[k] is not None and len(hist[k]) > 20]
    print('\n=== 三锚点的跨视角判别力（同人 vs 不同人）===')
    print('%-8s | %-34s | %-34s | %s' % ('锚点', '同人 残差中位 (n)', '不同人 残差中位 (n)', '可分性(同人p90 < 不同人p10 ?)'))
    for kind in ('center', 'foot', 'top'):
        same, diff = [], []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                if k1[0] == k2[0]:
                    continue
                ov = sorted(set(hist[k1]) & set(hist[k2]))
                if len(ov) < 10:
                    continue
                errs = []
                for fr in ov:
                    a1 = anchor(hist[k1][fr], kind); a2 = anchor(hist[k2][fr], kind)
                    X = dlt([undist(a1, k1[0])[0], undist(a2, k2[0])[0]], [k1[0], k2[0]])
                    if X is None:
                        continue
                    e = []
                    for v, a, b in ((k1[0], a1, hist[k1][fr]), (k2[0], a2, hist[k2][fr])):
                        q = reproj(X, v)
                        if q is None:
                            e = None; break
                        e.append(np.linalg.norm(q - a) / max(b[3] - b[1], 1e-6))
                    if e:
                        errs.append(max(e))
                if len(errs) < 5:
                    continue
                m = float(np.median(errs))
                (same if lab[k1] == lab[k2] else diff).append(m)
        if not same or not diff:
            print('%-8s | (样本不足) same=%d diff=%d' % (kind, len(same), len(diff))); continue
        s, d = np.array(same), np.array(diff)
        sep = np.percentile(s, 90) < np.percentile(d, 10)
        print('%-8s | %8.3f / %8.3f / %8.3f (%2d) | %8.3f / %8.3f / %8.3f (%2d) | %s'
              % (kind, np.median(s), np.percentile(s, 90), s.max(), len(s),
                 np.median(d), np.percentile(d, 10), d.min(), len(d),
                 '★可分' if sep else '重叠'))


if __name__ == '__main__':
    main()
