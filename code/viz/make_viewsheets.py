#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_viewsheets.py — 把叠加图拼成"能扫"的图：多视图对照 + 连续帧接触表

用法（容器内）：
  python make_viewsheets.py --out <dir> --frames 1,5,10,20,40 --views 01,03,04,07,09,14
"""
import os, sys, json, argparse, subprocess, glob
import numpy as np
import cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
HERE = os.path.dirname(os.path.abspath(__file__))


def tile(paths, cols, out, cell_w=960, label_h=0):
    imgs = [cv2.imread(p) for p in paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return
    h0 = int(imgs[0].shape[0] * cell_w / imgs[0].shape[1])
    imgs = [cv2.resize(im, (cell_w, h0)) for im in imgs]
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.zeros((rows * h0, cols * cell_w, 3), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * h0:(r + 1) * h0, c * cell_w:(c + 1) * cell_w] = im
        cv2.line(canvas, (c * cell_w, r * h0), ((c + 1) * cell_w, r * h0), (255, 255, 255), 2)
        cv2.line(canvas, (c * cell_w, r * h0), (c * cell_w, (r + 1) * h0), (255, 255, 255), 2)
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('  -> %s  (%d 格 %dx%d)' % (out, len(imgs), cols, rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--frames', default='1,5,10,20,40')
    ap.add_argument('--seq', nargs='+', default=['01:1-20', '03:1-20', '07:1-20'],
                    help='连续帧接触表：view:起-止')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    PY = sys.executable
    V = [v.zfill(2) for v in a.views.split(',')]
    frames = [int(x) for x in a.frames.split(',')]

    # 1) 渲染每个视角的指定帧
    made = {}
    for v in V:
        for fr in frames:
            subprocess.run([PY, os.path.join(HERE, 'viz_overlay.py'), '--view', v,
                            '--frames', str(fr), '--out', a.out],
                           env=dict(os.environ), check=False, capture_output=True)
            made[(v, fr)] = os.path.join(a.out, 'overlay_%s_%06d.jpg' % (v, fr))
    # 2) 多视图对照（同一帧、6 个视角并排）
    for fr in frames:
        ps = [made[(v, fr)] for v in V if os.path.exists(made[(v, fr)])]
        if len(ps) >= 2:
            tile(ps, 3, os.path.join(a.out, 'SHEET_views_f%03d.jpg' % fr), cell_w=880)
    # 3) 连续帧接触表（看选人是否逐帧乱跳）
    for spec in a.seq:
        v, rng = spec.split(':')
        v = v.zfill(2)
        s, e = [int(x) for x in rng.split('-')]
        paths = []
        for fr in range(s, e + 1):
            p = os.path.join(a.out, 'overlay_%s_%06d.jpg' % (v, fr))
            if not os.path.exists(p):
                subprocess.run([PY, os.path.join(HERE, 'viz_overlay.py'), '--view', v,
                                '--frames', str(fr), '--out', a.out],
                               env=dict(os.environ), check=False, capture_output=True)
            if os.path.exists(p):
                paths.append(p)
        if paths:
            tile(paths, 4, os.path.join(a.out, 'SHEET_%s_f%03d-%03d.jpg' % (v, s, e)), cell_w=760)


if __name__ == '__main__':
    main()
