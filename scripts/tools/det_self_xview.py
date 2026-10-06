#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""det_self_xview.py —— 跨视角一致性选人（det_self_final.py 的替代选人规则）

原版 det_self_final.py 的弱点：
    raw[fr] = det2.candidates_k(rtd, img, model, v, 2, margin)   # 每视角【各自独立】取面积前2
某视角只要有一个围观者的面积排进前二，该视角就选错人，且没有任何机制要求
「6 个视角选出来的是同样两个人」。实测把检测分辨率提到 1280 后框更准了，
选人却 3/3 全面变差（93.6→88.7 / 98.7→90.0 / 92.1→84.9）——正是这个原因。

本脚本：
  1) 每视角保留 K 个候选（默认 4）
  2) 逐帧联合搜索：在参考视角挑一对候选 (i0,i1)，对每个其它视角为每个人找
     重投影一致性最好的候选；得分 = 能对上的视角数，同分取误差小者
  3) 输出与 det_self_final 完全相同的 LabelMe 契约（group_id 0/1）

用法:
  python det_self_xview.py --frames-root <frames/M/1> --out <det 目录> --views 01,03,... \
      --tag <tag> --start 1 --end NF --border-margin 0.15 --calib <calib_gt_tag> \
      --K 4 --thr 0.12 --imgsz 640 --device 0
"""
import os, sys, json, argparse, importlib.util, itertools
import numpy as np
import cv2

B = '/workshop/Lym/combat3d'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--tag', default='t')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--border-margin', type=float, default=0.15)
    ap.add_argument('--ref-view', default='04')
    ap.add_argument('--K', type=int, default=4, help='每视角保留的候选数')
    ap.add_argument('--thr', type=float, default=0.12, help='归一化重投影一致性阈值')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--W', type=int, default=3840)
    ap.add_argument('--H', type=int, default=2160)
    a = ap.parse_args()
    V = [v.zfill(2) for v in a.views.split(',')]
    v0 = a.ref_view.zfill(2) if a.ref_view.zfill(2) in V else V[0]

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

    def anc(bx):
        return np.array([(bx[0] + bx[2]) / 2, bx[1]])

    def hof(bx):
        return max(bx[3] - bx[1], 1e-6)

    # ---------- ① 每视角保留 K 个候选 ----------
    print('=== ① 每视角候选池 K=%d（margin=%.2f）===' % (a.K, a.border_margin), flush=True)
    cands = {v: {} for v in V}
    for v in V:
        fs = sorted(int(f.split('.')[0]) for f in os.listdir(os.path.join(a.frames_root, v))
                    if f.endswith('.png'))
        fs = [f for f in fs if a.start <= f <= a.end]
        for fr in fs:
            img = cv2.imread(os.path.join(a.frames_root, v, '%06d.png' % fr))
            if img is None:
                continue
            cands[v][fr] = det2.candidates_k(rtd, img, model, v, a.K, a.border_margin)
        print('  view %s: %d 帧' % (v, len(cands[v])), flush=True)

    # ---------- ② 逐帧跨视角联合选人 ----------
    print('=== ② 逐帧跨视角联合选人（参考视角 %s, thr=%.3f）===' % (v0, a.thr), flush=True)
    frames = sorted(set.intersection(*[set(cands[v]) for v in V]))
    sel = {v: {} for v in V}
    n_single = 0
    for fr in frames:
        cd = {v: cands[v][fr] for v in V}
        if len(cd[v0]) < 1:
            continue
        nref = len(cd[v0])
        best = None
        pairs = list(itertools.permutations(range(nref), 2)) if nref >= 2 \
            else [(i, None) for i in range(nref)]
        for (i0, i1) in pairs:
            pidbox = {0: cd[v0][i0]}
            if i1 is not None:
                pidbox[1] = cd[v0][i1]
            score, tot, ok = 0, 0.0, {v0: {0: i0, 1: i1}}
            for v in V:
                if v == v0:
                    continue
                got = {}
                for pid in pidbox:
                    bref = pidbox[pid]
                    bj, be = None, None
                    for j, b in enumerate(cd[v]):
                        X = dlt([und(anc(bref), v0)[0], und(anc(b), v)[0]], [v0, v])
                        if X is None:
                            continue
                        e = 0.0; good = True
                        for vv, bb in ((v0, bref), (v, b)):
                            q = reproj(X, vv)
                            if q is None:
                                good = False; break
                            e = max(e, float(np.linalg.norm(q - anc(bb)) / hof(bb)))
                        if good and (be is None or e < be):
                            bj, be = j, e
                    if bj is not None:
                        got[pid] = (bj, be)
                ok[v] = got
                if len(got) == len(pidbox):
                    score += 1
                    tot += sum(min(e, 1.0) for _, e in got.values())
            key = (score, -tot)
            if best is None or key > best[0]:
                best = (key, i0, i1, ok)
        if best is None:
            continue
        _, i0, i1, ok = best
        if i1 is None:
            n_single += 1
        for v in V:
            got = {}
            if v == v0:
                got[0] = cd[v][i0]
                if i1 is not None:
                    got[1] = cd[v][i1]
            else:
                for pid, (j, e) in ok.get(v, {}).items():
                    if e <= a.thr:
                        got[pid] = cd[v][j]
            if got:
                sel[v][fr] = got
    print('  逐帧联合选人完成：%d 帧（其中单人选人 %d 帧）' % (len(frames), n_single), flush=True)

    # ---------- ③ 写 LabelMe ----------
    print('=== ③ 写 LabelMe ===', flush=True)
    for v in V:
        od = os.path.join(a.out, v); os.makedirs(od, exist_ok=True)
        n = 0
        for fr, got in sorted(sel[v].items()):
            shapes = []
            for gi, b in sorted(got.items()):
                bb = rtd.expand_head(b, a.W, a.H)
                shapes.append({"label": "person",
                               "points": [[float(bb[0]), float(bb[1])], [float(bb[2]), float(bb[3])]],
                               "group_id": int(gi), "description": "xview:%d" % gi,
                               "shape_type": "rectangle", "flags": {}})
            shapes.sort(key=lambda s: s['group_id'])
            json.dump(rtd.build_labelme('%s_%06d.jpg' % (a.tag, fr), a.H, a.W, shapes),
                      open(os.path.join(od, '%s_%06d.json' % (a.tag, fr)), 'w'),
                      ensure_ascii=False, indent=1)
            n += 1
        print('  view %s: %d 帧' % (v, n), flush=True)
    print('[OK] -> %s（跨视角一致性选人）' % a.out, flush=True)


if __name__ == '__main__':
    main()
