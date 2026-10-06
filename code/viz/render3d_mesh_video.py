#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""render3d_mesh_video.py — 整段视频的 **SMPL 3D 渲染**（网格点云，mp4）

参照你主线 `rtd2/render3d_5000f13.py` 的画法（matplotlib 3D + 地面参考面 + 两人双色），
但**渲染对象从 13 节点骨架换成 stage6 拟合出的 SMPL 网格顶点**（6890 点/人），因此是真正的
"SMPL 3D 渲染"，且**不需要 pyrender/EGL**（规避你记录里那四个离屏坑）。

坐标：先把网格从 COLMAP 世界系乘 `scale_metric.npy` 变到**米制系**（那里 z 才是竖直向上，实测
人物 z 展布 1.80m = 身高），于是显示映射直接是 (x,y,z)，地面画在 z = 最小脚底附近。

用法：
  python render3d_mesh_video.py --fit <emfit_h4d_full> --calib <calib 目录> \
      --out <输出目录> --start 1 --end 742 --fps 20 [--subsample 6] [--workers 8]
输出：<out>/render3d_mesh.mp4  (+ <out>/frames/ 中间帧)
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

C_P1 = '#00a651'      # pid0 绿（与你管线语义一致）
C_P2 = '#f0a020'      # pid1 橙


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--subsample', type=int, default=6, help='顶点抽稀步长（6890/6≈1150 点/人）')
    ap.add_argument('--elev', type=float, default=22.0)
    ap.add_argument('--azim', type=float, default=-60.0)
    ap.add_argument('--dpi', type=int, default=110)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fr_dir = os.path.join(a.out, 'frames'); os.makedirs(fr_dir, exist_ok=True)

    S = np.load(os.path.join(a.calib, 'scale_metric.npy'))
    Rs, ts = S[:3, :3], S[:3, 3]

    frames = sorted(int(os.path.basename(f).split('.')[0]) for f in
                    glob.glob(os.path.join(a.fit, 'pid0/verts/*.npy')))
    frames = [f for f in frames if a.start <= f <= a.end]
    print('共 %d 帧（%d..%d）' % (len(frames), frames[0], frames[-1]), flush=True)

    # ---- 先扫一遍确定固定视角范围（避免镜头抖动）----
    lo = np.array([1e9] * 3); hi = -lo
    for fr in frames[::max(1, len(frames) // 30)]:
        for pid in (0, 1):
            f = os.path.join(a.fit, 'pid%d/verts/%06d.npy' % (pid, fr))
            if os.path.exists(f):
                V = np.load(f).astype(np.float64) @ Rs.T + ts
                lo = np.minimum(lo, V.min(0)); hi = np.maximum(hi, V.max(0))
    pad = 0.4
    print('范围 x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]' % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]), flush=True)
    zfloor = lo[2]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    written = 0
    for i, fr in enumerate(frames):
        ax.clear()
        # 地面参考面
        gx = np.linspace(lo[0] - pad, hi[0] + pad, 2)
        gy = np.linspace(lo[1] - pad, hi[1] + pad, 2)
        GX, GY = np.meshgrid(gx, gy)
        ax.plot_surface(GX, GY, np.full_like(GX, zfloor), alpha=0.08, color='#888888', linewidth=0)
        for pid, col in ((0, C_P1), (1, C_P2)):
            f = os.path.join(a.fit, 'pid%d/verts/%06d.npy' % (pid, fr))
            if not os.path.exists(f):
                continue
            V = np.load(f).astype(np.float64) @ Rs.T + ts
            V = V[::a.subsample]
            ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=2.0, c=col, alpha=0.85, depthshade=True)
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_zlim(zfloor - 0.1, max(hi[2] + pad, zfloor + 1.9))
        ax.set_box_aspect((hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad, 1.8))
        ax.view_init(elev=a.elev, azim=a.azim)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title('SMPL 3D  fit 帧 %d / %d   （绿=pid0  橙=pid1，米制系）' % (fr, frames[-1]))
        fp = os.path.join(fr_dir, 'f%06d.png' % fr)
        fig.savefig(fp, dpi=a.dpi)
        plt.close(fig)
        written += 1
        if written % 50 == 0:
            print('  渲染 %d/%d' % (written, len(frames)), flush=True)
    print('帧渲染完成 %d' % written, flush=True)
    # ---- 合成 mp4 ----
    import cv2
    ps = sorted(glob.glob(os.path.join(fr_dir, '*.png')))
    im0 = cv2.imread(ps[0]); H, W = im0.shape[:2]
    vw = cv2.VideoWriter(os.path.join(a.out, 'render3d_mesh.mp4'),
                         cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (W, H))
    for p in ps:
        im = cv2.imread(p)
        vw.write(im if im.shape[:2] == (H, W) else cv2.resize(im, (W, H)))
    vw.release()
    print('[OK] %s（%d 帧 @ %.1f fps）' % (os.path.join(a.out, 'render3d_mesh.mp4'), len(ps), a.fps))


if __name__ == '__main__':
    main()
