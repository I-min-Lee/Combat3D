#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  ✅  UNIVERSAL LAYER — REUSE AS-IS ACROSS SCENES
#
#  这是 Combat3D-Label 五层里【通用】的那几层之一（2D 提取 / 三角化 / SMPL 拟合 /
#  单目 lifter）。两个域上的实测：换场景时这几层【一行未改】。
#
#  与它对照的是【场景相关】的检测框层与身份层 —— 那两个必须换算法，
#  文件头带 "SCENE-SPECIFIC LAYER" 横幅。层契约见 code/adapters/README.md。
# =============================================================================
"""vp_h4d.py — stage2 ViTPose，Harmony4D 运行版

★ 本文件是 `stage2_pose2d/vp_rtd_batch.py` 的**副本**，只改 5 处（全部用 `★H4D` 标出）：
    1. 新增 `import mmpretrain.models` —— 注册 mmpretrain.* 供 mmpose registry 解析
       （222 上 mmpose 1.3.2 的 ViTPose 配置用 `mmpretrain.VisionTransformer`）
    2. CONFIG / CKPT 路径 → 环境变量可覆盖，默认指向 pose312 env 与本项目 port/
    3. B 根目录 → `VP_ROOT` 可覆盖（原写死 /root/autodl-tmp）
    4. `--det` / `--views` 默认值 → Harmony4D
    5. 批推理 / 攒批 / 输出格式 **一字未改**

用法：
  VP_ROOT=/workshop/Lym/combat3d VP_MATCH=15 VP_SEG=4 \
  python vp_h4d.py --det $VP_ROOT/det_h4d_016mma4 --out $VP_ROOT/vp_h4d_016mma4 \
                   --views 01 03 04 07 09 14 --start 1 --end 741
"""
import os, sys, json, glob, argparse, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

import mmpretrain.models                                    # ★H4D-1 注册 mmpretrain.* scope
from mmengine.dataset import Compose
try:
    from mmengine.dataset import pseudo_collate
except ImportError:
    from mmengine.dataset.utils import pseudo_collate
from mmpose.apis import init_model

B = os.environ.get('VP_ROOT', '/workshop/Lym/combat3d')      # ★H4D-3
MATCH = int(os.environ.get('VP_MATCH', '15'))
SEG   = int(os.environ.get('VP_SEG', '4'))
FRAMES_DIR = os.environ.get('VP_FRAMES_DIR', '%s/frames/%d/%d' % (B, MATCH, SEG))
CONFIG = os.environ.get(                                     # ★H4D-2
    'VP_CONFIG',
    '/workshop/Lym/combat3d/envs/miniconda3/envs/pose312/lib/python3.12/site-packages/mmpose/'
    '.mim/configs/body_2d_keypoint/topdown_heatmap/coco/'
    'td-hm_ViTPose-base_8xb64-210e_coco-256x192.py')
CKPT = os.environ.get('VP_CKPT', B + '/port/autodl-tmp/vitpose_ft.pth')   # ★H4D-2


def parse_detect(jf):
    j = json.load(open(jf))
    persons = {}
    for s in j['shapes']:
        if s['label'] == 'person':
            gid = s.get('group_id')
            if gid is not None:
                pts = s['points']
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                persons[int(gid)] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return persons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default=B + '/det_h4d_016mma4')          # ★H4D-4
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', nargs='+', default=['01', '03', '04', '07', '09', '14'])  # ★H4D-4
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    ap.add_argument('--no-skip', action='store_true')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()

    cv2.setNumThreads(1)
    model = init_model(CONFIG, CKPT, device='cuda')
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    meta = model.dataset_meta

    tasks, skip = [], 0
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
            tasks.append((v, fidx, op, jf))
    print('[batch] views=%s 待处理=%d 已跳过=%d batch=%d workers=%d'
          % (a.views, len(tasks), skip, a.batch, a.workers), flush=True)
    if not tasks:
        print('[batch] 无待处理帧，退出', flush=True)
        return

    def load_and_prepare(t):
        v, fidx, op, jf = t
        img = cv2.imread(os.path.join(FRAMES_DIR, v, '%06d.png' % fidx))
        if img is None:
            return t, None, None, []
        persons = parse_detect(jf)
        gids = [g for g in (0, 1) if g in persons]
        bbs = [np.array(persons[g], np.float32) for g in gids]
        return t, img, bbs, gids

    def prep_one(item):
        img, bbox = item
        d = dict(img=img)
        d['bbox'] = bbox[None]
        d['bbox_score'] = np.ones(1, dtype=np.float32)
        d.update(meta)
        return pipeline(d)

    t_start = time.time()
    n_done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        CHUNK = a.batch
        for ci in range(0, len(tasks), CHUNK):
            chunk = tasks[ci:ci + CHUNK]
            loaded = list(ex.map(load_and_prepare, chunk))

            flat, owner = [], []
            for k, (t, img, bbs, gids) in enumerate(loaded):
                if img is None:
                    continue
                for bi, bb in enumerate(bbs):
                    flat.append((img, bb))
                    owner.append((k, bi))

            if flat:
                data_list = list(ex.map(prep_one, flat))
                batch = pseudo_collate(data_list)
                with torch.no_grad():
                    results = model.test_step(batch)
            else:
                results = []

            per_frame = {}
            for (k, bi), r in zip(owner, results):
                per_frame.setdefault(k, []).append((bi, r))

            for k, (t, img, bbs, gids) in enumerate(loaded):
                v, fidx, op, jf = t
                if img is None:
                    continue
                annots = []
                for bi, r in sorted(per_frame.get(k, []), key=lambda x: x[0]):
                    if bi >= len(gids):
                        continue
                    kp_all = r.pred_instances.keypoints[0].tolist()
                    s_all = (r.pred_instances.keypoints_scores[0].tolist()
                             if hasattr(r.pred_instances, 'keypoints_scores')
                             else r.pred_instances.keypoint_scores[0].tolist())
                    kp = [[kp_all[j][0], kp_all[j][1], s_all[j]] for j in range(17)]
                    annots.append({'personID': int(gids[bi]), 'keypoints': kp})
                annots.sort(key=lambda x: gids.index(x['personID']))
                with open(op, 'w') as f:
                    json.dump({'image': '%06d' % fidx, 'height': int(img.shape[0]),
                               'width': int(img.shape[1]), 'annots': annots}, f)
                n_done += 1

            if n_done and n_done % 1000 < CHUNK:
                el = time.time() - t_start
                print('[batch] +%d (skip %d)  %.1f 帧/s'
                      % (n_done, skip, n_done / max(el, 1e-9)), flush=True)

    el = time.time() - t_start
    print('[batch] ALL_DONE +%d skip=%d  %.1fs (%.1f 帧/s)'
          % (n_done, skip, el, n_done / max(el, 1e-9)), flush=True)


if __name__ == '__main__':
    main()
