#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""viz_overlay.py — 把每个阶段的结果**投影回原始帧**渲染成图（可复用工具）

用户口径：每个阶段跑完都要回原图看一眼。本脚本一层层叠：
  灰细框  候选检测（全部，未筛选）          ← 检测层输出
  绿框    我们**选中且命中真值**的框        ← 选人层
  品红框  我们**选中但没命中真值**的框      ← 选人层的【抢位】
  黄框    真值受试者里**没人覆盖**的那个    ← 选人层的【漏掉】
  红框    真值框（Harmony4D bbox）
  青骨架  我们自己的 2D（ViTPose）           ← 2D 层
  黄骨架  我们三角化的 3D **用鱼眼正投影回原图**  ← 三角化层（与青骨架的偏差 = 重投影残差）
  红骨架  数据集 3D 真值同样投影回原图        ← 评估层的参照
文字角标：view / frame / 选人是否 2/2 / 该帧重投影中位 / conf

用法：
  python viz_overlay.py --view 01 --frames 1,40 --out <dir> [--no-cand] [--no-tri] ...
"""
import os, sys, json, argparse
import numpy as np
import cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
SEQ = os.environ.get('H4D_SEQ', B + '/data/harmony4d/raw/15_mma4/016_mma4')
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
# ★ body25 -> COCO17（与你 knee_calib.py 的 B25_TO_COCO17 同一张表 + 补 4 个脸点）
#   b25: 0鼻 1颈 2右肩 3右肘 4右腕 5左肩 6左肘 7左腕 8中髋 9右髋 10右膝 11右踝 12左髋 13左膝 14左踝 15右眼 16左眼 17右耳 18左耳
#   coco: 0鼻 1左眼 2右眼 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘 9左腕 10右腕 11左髋 12右髋 13左膝 14右膝 15左踝 16右踝
B25 = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 9: 12, 10: 14, 11: 16,
       12: 11, 13: 13, 14: 15, 15: 2, 16: 1, 17: 4, 18: 3}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.


def skeleton(kp, idx, edges, color, img, thick=3, r=5):
    """kp: (17,2) 像素坐标；无效关节用负值标记 → 整条边不画
    （避免画到"相机后方/图像外"的垃圾点，那会画出横跨全图的假骨架线）"""
    pts = np.asarray(kp, float)
    H, W = img.shape[:2]
    MARGIN = 0.25

    def good(p):
        return (-MARGIN * W < p[0] < W * (1 + MARGIN)) and (-MARGIN * H < p[1] < H * (1 + MARGIN))
    for a, b in edges:
        if a >= len(pts) or b >= len(pts):
            continue
        pa, pb = pts[a], pts[b]
        if not (good(pa) and good(pb)):
            continue
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, thick, cv2.LINE_AA)
    for p in pts:
        if good(p):
            cv2.circle(img, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)


def box(img, b, color, label=None, thick=4, dash=False):
    x1, y1, x2, y2 = [int(v) for v in b]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 12, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', required=True)
    ap.add_argument('--frames', required=True, help='逗号分隔')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cams', default=B + '/calib_h4d_016mma4')
    ap.add_argument('--det', default=B + '/det_ours_i640')
    ap.add_argument('--annots', default=B + '/asm_h4d_full/annots')
    ap.add_argument('--tri', default=B + '/em_h4d_full/lam1.0')
    ap.add_argument('--tag', default='016_mma4')
    ap.add_argument('--scale', type=float, default=0.55, help='输出缩放')
    ap.add_argument('--no-cand', action='store_true')
    ap.add_argument('--no-tri', action='store_true')
    a = ap.parse_args()
    v = a.view.zfill(2)
    os.makedirs(a.out, exist_ok=True)

    # 相机（读我们写的 yml）
    sys.path.insert(0, os.environ.get('EMCORE', B + '/port/emcore'))
    from easymocap.mytools.camera_utils import read_camera
    C = read_camera(os.path.join(a.cams, 'intri.yml'), os.path.join(a.cams, 'extri.yml'))
    cam = C[v]
    K = np.asarray(cam['K'], float).reshape(3, 3)
    D = np.asarray(cam['dist'], float).reshape(-1)[:4].reshape(4, 1)
    R = np.asarray(cam['R'], float).reshape(3, 3)
    T = np.asarray(cam['T'], float).reshape(3)
    rvec = cv2.Rodrigues(np.ascontiguousarray(R))[0]

    # 候选检测（重跑一次，用于画灰框）
    cands = {}
    if not a.no_cand:
        import importlib.util
        from ultralytics import RTDETR
        sp = importlib.util.spec_from_file_location('rtd', B + '/code/stage1_detect/rtdetr_pipeline.py')
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
        model = RTDETR(m.MODEL_PATH if os.path.exists(m.MODEL_PATH)
                       else B + '/port/weights/rtdetr-l.pt')

    def proj3d(X, K, D, rvec, R, T):
        """★ 必须先把世界点变到相机系判 z_cam>0：在相机后方的点若直接喂 projectPoints，
        会得到镜像的垃圾像素坐标 → 画出横跨全图的假骨架（踩过）"""
        X = np.asarray(X, float).reshape(-1, 3)
        Xc = (np.asarray(R, float) @ X.T + np.asarray(T, float).reshape(3, 1)).T
        uv = np.full((len(X), 2), -1.0)
        ok = Xc[:, 2] > 0.05
        if ok.any():
            p, _ = cv2.fisheye.projectPoints(Xc[ok].reshape(-1, 1, 3),
                                             np.zeros(3), np.zeros((3, 1)), K, D)
            uv[ok] = p.reshape(-1, 2)
        return uv

    for fr in [int(x) for x in a.frames.split(',')]:
        fp = os.path.join(B, 'frames/15/4', v, '%06d.png' % fr)
        img0 = cv2.imread(fp)
        if img0 is None:
            print('无帧 %s' % fp); continue
        H, W = img0.shape[:2]
        img = img0.copy()

        # ---- GT 框 + GT 3D 骨架 ----
        gtf = os.path.join(SEQ, 'processed_data/bbox/cam%s/%05d.npy' % (v, fr))
        gtb = {}
        if os.path.exists(gtf):
            gt = np.load(gtf, allow_pickle=True)
            gt = gt.item() if getattr(gt, 'dtype', None) == object else gt
            gtb = {s: np.asarray(x, float).reshape(-1)[:4] for s, x in gt.items()}
        for s, b in gtb.items():
            box(img, b, (0, 0, 255), 'GT ' + s, 5)
        gtf3 = os.path.join(B, 'gt_h4d_016mma4/gt3d_colmap/%06d.json' % fr)
        if os.path.exists(gtf3):
            g3 = json.load(open(gtf3))
            for s, arr in g3.items():
                X = np.asarray(arr, float)[:, :3]
                skeleton(proj3d(X, K, D, rvec, R, T), None, COCO_EDGES, (0, 0, 200), img, 2, 3)

        # ---- 候选（灰） ----
        if not a.no_cand:
            r = model.predict(img0, conf=0.15, imgsz=640, classes=[0], device=0, verbose=False)[0]
            if r.boxes is not None:
                for b in r.boxes.xyxy.cpu().numpy():
                    box(img, b, (150, 150, 150), None, 2)

        # ---- 我们选中的框 ----
        dj = os.path.join(a.det, v, '%s_%06d.json' % (a.tag, fr))
        sel = []
        if os.path.exists(dj):
            for s in json.load(open(dj))['shapes']:
                if s.get('label') != 'person':
                    continue
                p = s['points']
                sel.append((int(s.get('group_id', -1)),
                            (min(p[0][0], p[1][0]), min(p[0][1], p[1][1]),
                             max(p[0][0], p[1][0]), max(p[0][1], p[1][1]))))
        covered = set()
        for g, b in sel:
            best, bs = 0., None
            for s, gb in gtb.items():
                x = iou(b, gb)
                if x > best:
                    best, bs = x, s
            if best >= 0.5:
                covered.add(bs)
                box(img, b, (0, 200, 0), 'PICK gid%d=%s iou%.2f' % (g, bs, best), 5)
            else:
                box(img, b, (255, 0, 255), 'PICK gid%d ✗抢位' % g, 5)
        for s, gb in gtb.items():
            if s not in covered:
                box(img, gb, (0, 255, 255), 'MISSED ' + s, 5)

        # ---- 我们的 2D（青）- 总览画一起便于看整体；两 pid 用不同色深 ----
        asm_f = os.path.join(a.annots, v, '%06d.json' % fr)
        kp2d = {}
        if os.path.exists(asm_f):
            for an in json.load(open(asm_f))['annots']:
                kp2d[int(an['personID'])] = np.asarray(an['keypoints'], float)
                skeleton(kp2d[int(an['personID'])][:, :2], None, COCO_EDGES,
                         (255, 200, 0) if int(an['personID']) == 0 else (200, 255, 0), img, 3, 4)

        # ---- 我们的 3D 反投影（黄）----
        if not a.no_tri:
            for pid in (0, 1):
                tf = os.path.join(a.tri, 'pid%d/keypoints3d/%06d.json' % (pid, fr))
                if not os.path.exists(tf):
                    continue
                k3 = np.array(json.load(open(tf))[0]['keypoints3d'], float)   # (25,4)
                Xc = np.zeros((17, 3)); ok = np.zeros(17, bool)
                for b25, c17 in B25.items():
                    if np.any(k3[b25, :3]):
                        Xc[c17] = k3[b25, :3]; ok[c17] = True
                uv = proj3d(Xc[ok], K, D, rvec, R, T)
                full = np.full((17, 3), -1.0); full[ok, :2] = uv
                skeleton(full, None, COCO_EDGES, (0, 255, 255), img, 4, 5)

        # ---- 角标 ----
        n2 = len(covered)
        txt = 'view %s  f%03d  选人 %d/2' % (v, fr, n2)
        cv2.putText(img, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 200, 0) if n2 == 2 else (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(img, 'gray=cands  green=our pick(ok)  magenta=pick(wrong)  yellowbox=missed  '
                         'cyan/green2=our2D  yellowskel=our3D reproj  red=GT',
                    (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        out = os.path.join(a.out, 'overlay_%s_%06d.jpg' % (v, fr))
        cv2.imwrite(out, cv2.resize(img, None, fx=a.scale, fy=a.scale), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print('  -> %s' % out)


if __name__ == '__main__':
    main()
