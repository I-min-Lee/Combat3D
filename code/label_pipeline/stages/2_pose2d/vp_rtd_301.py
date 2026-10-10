#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vp_rtd_range.py — 与 rtd2/vp_rtd.py 完全同路径, 但支持 --views/--start/--end/断点续跑

按视角并行: 每个进程只跑一个视角 -> 5 进程把 GPU 喂满, 比 vp_rtd.py(单进程串 5 视角)快约 5 倍。
personID 仍写**原始 group_id**(与 vp_rtd.py 一致), idflip 由 conv_rtd.py 负责施加。
已存在的帧文件默认跳过, 可反复重跑。
"""
import os, json, glob, argparse
import numpy as np
import cv2
import torch

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
from mmpose.apis import inference_topdown, init_model

B = '/root/autodl-tmp'
MATCH, SEG = 30, 1
FRAMES_DIR = '%s/frames/%d/%d' % (B, MATCH, SEG)
CONFIG = ('/root/miniconda3/lib/python3.12/site-packages/mmpose/.mim/configs/'
          'body_2d_keypoint/topdown_heatmap/coco/'
          'td-hm_ViTPose-base_8xb64-210e_coco-256x192.py')
CKPT = B + '/vitpose_ft.pth'


def parse_detect(jf):
    j = json.load(open(jf))
    persons = {}
    for s in j['shapes']:
        if s['label'] == 'person':
            gid = s.get('group_id')
            if gid is not None:
                pts = s['points']
                x1, y1 = pts[0]; x2, y2 = pts[1]
                persons[int(gid)] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return persons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default=B + '/rtdetr_v2/输出/1/1/json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', nargs='+', default=['1', '3', '4', '7', '11'])
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    ap.add_argument('--no-skip', action='store_true')
    a = ap.parse_args()

    model = init_model(CONFIG, CKPT, device='cuda')
    n = 0; skip = 0
    for v in a.views:
        od = os.path.join(a.out, v)
        os.makedirs(od, exist_ok=True)
        for jf in sorted(glob.glob(os.path.join(a.det, v, '*.json'))):
            fidx = int(os.path.basename(jf).rsplit('_', 1)[-1].split('.')[0])
            if not (a.start <= fidx <= a.end):
                continue
            op = os.path.join(od, '%06d.json' % fidx)
            if os.path.exists(op) and not a.no_skip:
                skip += 1
                continue
            img = cv2.imread(os.path.join(FRAMES_DIR, v, '%06d.png' % fidx))
            if img is None:
                continue
            persons = parse_detect(jf)
            annots = []
            gids = [g for g in (0, 1) if g in persons]
            if gids:
                bboxes = np.array([persons[g] for g in gids], np.float32)
                res = inference_topdown(model, img, bboxes=bboxes)
                for idx, r in enumerate(res):
                    if idx >= len(gids):
                        break
                    k = r.pred_instances.keypoints[0].tolist()
                    s = (r.pred_instances.keypoints_scores[0].tolist()
                         if hasattr(r.pred_instances, 'keypoints_scores')
                         else r.pred_instances.keypoint_scores[0].tolist())
                    kp = [[k[j][0], k[j][1], s[j]] for j in range(17)]
                    annots.append({'personID': int(gids[idx]), 'keypoints': kp})
            json.dump({'image': '%06d' % fidx, 'height': int(img.shape[0]),
                       'width': int(img.shape[1]), 'annots': annots},
                      open(op, 'w'))
            n += 1
            if n % 1000 == 0:
                print('[vp v%s] +%d (skip %d)' % (v, n, skip), flush=True)
        print('[vp v%s] DONE +%d skip=%d' % (v, n, skip), flush=True)
    print('[vp] ALL_DONE views=%s +%d skip=%d' % (a.views, n, skip), flush=True)


if __name__ == '__main__':
    main()
