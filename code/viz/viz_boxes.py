#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""viz_boxes.py — **只画我们脚本输出的检测框**（不叠加真值/骨架/候选/任何其它层）

用户口径（2026-09-29）："给我的可视化图片我只要检测框的，不需要叠加，
就只要我们脚本跑出来的检测框"。本脚本严格遵守。

用法：
  python viz_boxes.py --det <检测输出树> --out <图片目录> --views 01,... --frames 5,20 \
                      [--tag 016_mma4] [--scale 0.55] [--title "v2 topk=8"]
输出：{out}/{标题}_{view}_{frame}.jpg
"""
import os, json, argparse
import numpy as np
import cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
COLORS = [(60, 220, 60), (60, 160, 255), (255, 120, 60)]      # gid0 绿 / gid1 橙 / gid2 蓝


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--tag', default='016_mma4')
    ap.add_argument('--title', default='')
    ap.add_argument('--scale', type=float, default=0.55)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for v in [x.zfill(2) for x in a.views.split(',')]:
        for fr in [int(x) for x in a.frames.split(',')]:
            fp = os.path.join(B, 'frames/15/4', v, '%06d.png' % fr)
            img = cv2.imread(fp)
            if img is None:
                continue
            dj = os.path.join(a.det, v, '%s_%06d.json' % (a.tag, fr))
            n = 0
            if os.path.exists(dj):
                for s in json.load(open(dj))['shapes']:
                    if s.get('label') != 'person':
                        continue
                    gid = int(s.get('group_id', 0))
                    p = s['points']
                    x1, y1 = int(min(p[0][0], p[1][0])), int(min(p[0][1], p[1][1]))
                    x2, y2 = int(max(p[0][0], p[1][0])), int(max(p[0][1], p[1][1]))
                    col = COLORS[gid % len(COLORS)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, 7, cv2.LINE_AA)
                    cv2.putText(img, 'gid%d' % gid, (x1, max(y1 - 14, 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4, cv2.LINE_AA)
                    n += 1
            head = '%s  view %s  f%03d  框数=%d' % (a.title or os.path.basename(a.det.rstrip('/')),
                                                    v, fr, n)
            cv2.putText(img, head, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(img, head, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 20), 2, cv2.LINE_AA)
            name = '%s_%s_%06d.jpg' % ((a.title or 'boxes').replace(' ', ''), v, fr)
            cv2.imwrite(os.path.join(a.out, name),
                        cv2.resize(img, None, fx=a.scale, fy=a.scale), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('-> %s' % a.out)


if __name__ == '__main__':
    main()
