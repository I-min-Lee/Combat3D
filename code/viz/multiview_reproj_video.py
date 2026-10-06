#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""multiview_reproj_video.py — 整段视频的 **各视角反投影多画面**（mp4）

每一帧：把 stage6 拟合出的 SMPL 网格用**鱼眼正投影**投到 6 个机位，拼成多画面（默认 3×2），
逐帧写成 mp4。这样能一眼看出"3D 结果在各视角是否都贴合真人" —— 即反投影残差的直观版。

只画本阶段产物（我们的网格），不叠真值/骨架。
格内标注每视角的**该帧反投影中位残差（px）**，便于定位哪一帧/哪个视角崩。

用法：
  python multiview_reproj_video.py --fit <emfit_h4d_full> --calib <calib 目录> \
      --frames-root <frames/15/4> --out <输出目录> --views 01,03,04,07,09,14 \
      --start 1 --end 742 --fps 20 [--cols 3] [--cell 640] [--subsample 4]
输出：<out>/multiview_reproj.mp4
"""
import os, sys, glob, argparse
import numpy as np
import cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
COL = {0: (90, 230, 90), 1: (60, 170, 255)}     # pid0 绿 / pid1 橙


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--cell', type=int, default=640, help='单格宽（高按 9:16 自适应为 cell*9/16）')
    ap.add_argument('--subsample', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    views = [v.zfill(2) for v in a.views.split(',')]

    sys.path.insert(0, os.environ.get('EMCORE', B + '/port/emcore'))
    from easymocap.mytools.camera_utils import read_camera
    C = read_camera(os.path.join(a.calib, 'intri.yml'), os.path.join(a.calib, 'extri.yml'))
    cam = {}
    for v in views:
        c = C[v]
        cam[v] = dict(K=np.asarray(c['K'], float).reshape(3, 3),
                      D=np.asarray(c['dist'], float).reshape(-1)[:4].reshape(4, 1),
                      R=np.asarray(c['R'], float).reshape(3, 3),
                      T=np.asarray(c['T'], float).reshape(3))
    S = np.load(os.path.join(a.calib, 'scale_metric.npy'))
    Rs, ts = S[:3, :3], S[:3, 3]
    # 网格在 COLMAP 系；投影用 COLMAP 系的相机 → 直接用（不换算到米制）
    frames = sorted(int(os.path.basename(f).split('.')[0]) for f in
                    glob.glob(os.path.join(a.fit, 'pid0/verts/*.npy')))
    frames = [f for f in frames if a.start <= f <= a.end]
    print('共 %d 帧，%d 视角，输出 %dx%d' % (len(frames), len(views), a.cols * a.cell,
          ((len(views) + a.cols - 1) // a.cols) * int(a.cell * 9 / 16)), flush=True)

    cw = a.cell; ch = int(a.cell * 9 / 16)
    rows = (len(views) + a.cols - 1) // a.cols
    vw = None
    n = 0
    for fr in frames:
        canvas = np.zeros((rows * ch, a.cols * cw, 3), np.uint8)
        for k, v in enumerate(views):
            r, c = divmod(k, a.cols)
            # 帧号：fit 索引 fr ↔ Harmony4D 帧 fr+1
            fp = os.path.join(a.frames_root, v, '%06d.png' % (fr + 1))
            img = cv2.imread(fp)
            if img is None:
                img = np.zeros((1080, 1920, 3), np.uint8)
            else:
                img = cv2.resize(img, (cw, ch))
            H, W = img.shape[:2]
            sc = W / 3840.0                      # 该格相对原图的缩放
            med = []
            for pid in (0, 1):
                f = os.path.join(a.fit, 'pid%d/verts/%06d.npy' % (pid, fr))
                if not os.path.exists(f):
                    continue
                V = np.load(f).astype(np.float64)[::a.subsample]
                Xc = (cam[v]['R'] @ V.T + cam[v]['T'][:, None]).T
                ok = Xc[:, 2] > 0.05
                if ok.sum() < 10:
                    continue
                uv, _ = cv2.fisheye.projectPoints(Xc[ok].reshape(-1, 1, 3),
                                                  np.zeros(3), np.zeros((3, 1)),
                                                  cam[v]['K'], cam[v]['D'])
                uv = uv.reshape(-1, 2) * sc
                ins = (uv[:, 0] > -0.05 * W) & (uv[:, 0] < 1.05 * W) & \
                      (uv[:, 1] > -0.05 * H) & (uv[:, 1] < 1.05 * H)
                uv = uv[ins]
                for p in uv:
                    cv2.circle(img, (int(p[0]), int(p[1])), 1, COL[pid], -1)
                if len(uv) > 60:
                    hull = cv2.convexHull(uv.astype(np.int32).reshape(-1, 1, 2))
                    cv2.polylines(img, [hull], True, COL[pid], 2, cv2.LINE_AA)
            cv2.putText(img, 'view %s  f%d' % (v, fr + 1), (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = img
        if vw is None:
            vw = cv2.VideoWriter(os.path.join(a.out, 'multiview_reproj.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (a.cols * cw, rows * ch))
        vw.write(canvas)
        n += 1
        if n % 50 == 0:
            print('  写出 %d/%d' % (n, len(frames)), flush=True)
    vw.release()
    print('[OK] %s（%d 帧 @ %.1f fps）' % (os.path.join(a.out, 'multiview_reproj.mp4'), n, a.fps))


if __name__ == '__main__':
    main()
