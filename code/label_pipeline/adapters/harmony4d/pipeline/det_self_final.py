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
"""det_self_final.py — 全自建前端（最终版）：**选人 + 单视角 gid 稳定 + 跨视角 pid 对齐**

三步（全部零数据集依赖）：
 ① **选人**：逐帧取面积前 2，候选池先过**位置先验**（框中心不得落在画面外圈 margin）
    —— 实测这一步在全量 741 帧上把选人正确率做到 **95.7%**（margin=0.15 滤掉贴边的摄影师）
 ② **单视角 gid 稳定化**：用你的 `TwoSlotTracker` 跑在"已选的 2 个框"上（只做编号稳定，不选人）
    —— 避免"谁离相机近谁排第一"造成的逐帧编号翻转
 ③ ★**跨视角 pid 对齐**（本版新增，治"不同视角的 pid0 指向不同人"）：
     取一个参考视角 v0，对每个其他视角 v 做**两视角 DLT 残差比较**：
        r(same) = 用 (v0.pid0, v.pid0) 与 (v0.pid1, v.pid1) 三角化的残差
        r(flip) = 用 (v0.pid0, v.pid1) 与 (v0.pid1, v.pid0) 三角化的残差
     若 r(flip) 明显更小 → 该视角要翻转 pid。再以 v0 为基准做一次**传递闭包**保证全局一致。

用法：
  python det_self_final.py --frames-root $ROOT/frames/15/4 --out $ROOT/det_self2 \
      --views 01,03,04,07,09,14 --tag 016_mma4 --start 1 --end 742 \
      [--border-margin 0.15] [--ref-view 04] [--imgsz 640]
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
    ap.add_argument('--border-margin', type=float, default=0.15)
    ap.add_argument('--ref-view', default='04', help='★跨视角对齐的参考视角')
    ap.add_argument('--flip-ratio', type=float, default=0.75, help='翻转判定：r(flip) < ratio*r(same)')
    ap.add_argument('--drop-coast', action='store_true',
                    help='★只写"本帧真的匹配上检测"的框，丢掉 tracker coast（按速度外推）出来的框。'
                         '理由（2026-09-30 实测 06_sword3）：受试者快速移动会超出 '
                         'MATCH_GATE=160px → 槽位丢失 → coast 最多 MAX_MISS=20 帧 → 这 20 帧写出去的是'
                         '**凭速度编的框**（实测落在离两人都 ~150px 的地方），喂给 ViTPose 只会产出'
                         '错的 2D 去污染三角化。丢掉它 = 该视角这一帧不投票，比投假票好。')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--nmax', type=int, default=12)
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

    # ---------- ①② 每视角：选人 + gid 稳定化 ----------
    print('=== ①② 每视角选人 + gid 稳定化（margin=%.2f）===' % a.border_margin, flush=True)
    sel = {v: {} for v in V}          # sel[v][fr] = {0: box, 1: box}
    for v in V:
        fs = sorted(int(f.split('.')[0]) for f in os.listdir(os.path.join(a.frames_root, v))
                    if f.endswith('.png'))
        fs = [f for f in fs if a.start <= f <= a.end]
        raw = {}
        for fr in fs:
            img = cv2.imread(os.path.join(a.frames_root, v, '%06d.png' % fr))
            if img is None:
                continue
            raw[fr] = det2.candidates_k(rtd, img, model, v, 2, a.border_margin)
        tr = rtd.TwoSlotTracker()
        n_coast = 0
        for fr in fs:
            if fr not in raw:
                continue
            got = {}
            for o in tr.step(raw[fr]):
                if a.drop_coast and getattr(o, 'miss', 0) > 0:
                    n_coast += 1
                    continue          # ★本帧没匹配上检测 → 这个框是编的，不写出去
                got[0 if o.name == 'P0' else 1] = list(o.box)
            if got:
                sel[v][fr] = got
        print('  view %s: %d 帧有选中框%s'
              % (v, len(sel[v]),
                 '（丢弃 coast 框 %d 个）' % n_coast if a.drop_coast else ''), flush=True)

    # ---------- ③ 跨视角 pid 对齐 ----------
    print('=== ③ 跨视角 pid 对齐（参考视角 %s）===' % a.ref_view.zfill(2), flush=True)
    v0 = a.ref_view.zfill(2)
    if v0 not in V:
        v0 = V[0]

    def pair_res(vA, vB, swapB=False, ncap=400):
        """用 vA/vB 的 pid 对做两视角 DLT，返回归一化残差中位；swapB=True 时 B 的 pid 互换"""
        er = []
        for fr in sorted(set(sel[vA]) & set(sel[vB])):
            ga, gb = sel[vA][fr], sel[vB][fr]
            for gi in (0, 1):
                ba = ga.get(gi)
                gb_i = (1 - gi) if swapB else gi
                bb = gb.get(gb_i)
                if ba is None or bb is None:
                    continue
                X = dlt([und(anc(ba), vA)[0], und(anc(bb), vB)[0]], [vA, vB])
                if X is None:
                    continue
                e = []
                for vv, b in ((vA, ba), (vB, bb)):
                    q = reproj(X, vv)
                    if q is None:
                        e = None; break
                    e.append(np.linalg.norm(q - anc(b)) / hof(b))
                if e:
                    er.append(max(e))
            if len(er) > ncap:
                break
        return float(np.median(er)) if len(er) >= 30 else None

    flip = {v0: False}
    for v in V:
        if v == v0:
            continue
        r_same = pair_res(v0, v)
        r_flip = pair_res(v0, v, swapB=True)
        if r_same is None or r_flip is None:
            flip[v] = False
            print('  view %s: 样本不足，不翻转' % v, flush=True)
            continue
        # ★翻转后要与基准"绝对一致"：用"与参考的残差"直接比较
        fl = r_flip < a.flip_ratio * r_same
        flip[v] = fl
        print('  view %s: r(same)=%.3f  r(flip)=%.3f  → %s'
              % (v, r_same, r_flip, '翻转 pid' if fl else '保持'), flush=True)
    n_flip = sum(1 for v in flip if flip[v])
    print('  共 %d/%d 个视角需要翻转' % (n_flip, len(V)), flush=True)

    # ---------- ④ 写 LabelMe（按对齐后的 pid）----------
    print('=== ④ 写 LabelMe ===', flush=True)
    for v in V:
        od = os.path.join(a.out, v); os.makedirs(od, exist_ok=True)
        n = 0
        for fr, got in sorted(sel[v].items()):
            shapes = []
            for gi, b in sorted(got.items()):
                g2 = (1 - gi) if flip.get(v, False) else gi
                bb = rtd.expand_head(b, a.W, a.H)
                shapes.append({"label": "person",
                               "points": [[float(bb[0]), float(bb[1])], [float(bb[2]), float(bb[3])]],
                               "group_id": g2, "description": "self:%d" % g2,
                               "shape_type": "rectangle", "flags": {}})
            shapes.sort(key=lambda s: s['group_id'])
            json.dump(rtd.build_labelme('%s_%06d.jpg' % (a.tag, fr), a.H, a.W, shapes),
                      open(os.path.join(od, '%s_%06d.json' % (a.tag, fr)), 'w'),
                      ensure_ascii=False, indent=1)
            n += 1
        print('  view %s: %d 帧' % (v, n), flush=True)
    print('[OK] -> %s（★零数据集框/身份依赖）' % a.out, flush=True)


if __name__ == '__main__':
    main()
