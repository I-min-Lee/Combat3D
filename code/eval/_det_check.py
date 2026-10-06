#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_det_check.py <det_json_root> <frames_root> [--views 1,2,3,4,7,11] [--step 7]
检验检测框脚本产物：格式 / 数量 / 画幅一致性 / ROI 内 / 框尺寸 / 重复与重叠。
det_json_root 形如 /root/autodl-tmp/rtdetr_v2/输出/30/1/json
frames_root   形如 /root/autodl-tmp/frames/30/1
"""
import os, sys, json, glob
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith('--')]
DET, FRM = args[0], args[1]
def opt(k, d):
    for a in sys.argv[1:]:
        if a.startswith('--' + k + '='):
            return a.split('=', 1)[1]
    return d
VIEWS = [v.strip() for v in opt('views', '1,2,3,4,7,11').split(',') if v.strip()]
STEP = int(opt('step', '7'))
print('DET=%s\nFRM=%s\nVIEWS=%s STEP=%d' % (DET, FRM, VIEWS, STEP))

def rd(p):
    try:
        d = json.load(open(p))
    except Exception as e:
        return None, str(e)
    return d, None

# 实际抽帧画幅
print('\n--- 抽帧画幅 ---')
FSZ = {}
for v in VIEWS:
    fs = sorted(glob.glob('%s/%s/*.png' % (FRM, v))) or sorted(glob.glob('%s/%s/*.jpg' % (FRM, v)))
    if not fs:
        print('  v%s: 无帧' % v); continue
    from PIL import Image
    im = Image.open(fs[0])
    FSZ[v] = im.size
    print('  v%s: n=%d size=%s  样例=%s' % (v, len(fs), im.size, os.path.basename(fs[0])))

print('\n--- 检测框 ---')
allbad = 0
for v in VIEWS:
    fs = sorted(glob.glob('%s/%s/*.json' % (DET, v)))
    if not fs:
        print('  v%s: 无检测' % v); continue
    n2 = n1 = n0 = n3 = 0
    bad_fmt = bad_frm = bad_roi = 0
    areas = []          # 框面积 / 画幅面积
    whs = set()
    ious = []
    for i, f in enumerate(fs):
        if i % STEP: continue
        d, err = rd(f)
        if err or not isinstance(d, dict) or 'shapes' not in d:
            bad_fmt += 1; continue
        W, H = d.get('imageWidth'), d.get('imageHeight')
        whs.add((W, H))
        if v in FSZ and (W, H) != FSZ[v]:
            bad_frm += 1
        sh = [s for s in d['shapes'] if s.get('label') == 'person']
        gids = [s.get('group_id') for s in sh]
        if len(sh) == 0: n0 += 1
        elif len(sh) == 1: n1 += 1
        elif len(sh) == 2: n2 += 1
        else: n3 += 1
        boxes = []
        for s in sh:
            pts = np.array(s['points'], float)
            x1, y1 = pts.min(0); x2, y2 = pts.max(0)
            if x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1:
                bad_roi += 1
            boxes.append([x1, y1, x2, y2])
            areas.append((x2 - x1) * (y2 - y1) / float(W * H))
            if 'description' not in s or 'group_id' not in s:
                bad_fmt += 1
        if len(boxes) == 2:
            a, b = boxes
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
            ious.append(inter / ua if ua > 0 else 0)
    areas = np.array(areas)
    tot = n0 + n1 + n2 + n3
    print('  v%-2s 采样%-4d 双框%.1f%% 单框%.1f%% 零框%.1f%% 三框+%d | 画幅%s | 格式错%d 画幅不符%d 出界%d'
          % (v, tot, 100.0*n2/max(tot,1), 100.0*n1/max(tot,1), 100.0*n0/max(tot,1), n3,
             sorted(whs)[:2], bad_fmt, bad_frm, bad_roi))
    if len(areas):
        print('       框面积占画幅: 中位%.3f p05=%.4f p95=%.4f | 双框IoU中位=%.3f max=%.3f'
              % (np.median(areas), np.percentile(areas, 5), np.percentile(areas, 95),
                 np.median(ious) if ious else -1, np.max(ious) if ious else -1))
    allbad += bad_fmt + bad_frm + bad_roi
print('\n总异常项: %d' % allbad)
