#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vp_rtd_batch.py — ViTPose 批量推理版（真正喂饱 GPU）

════════════════════════════════════════════════════════════════
为什么原版慢 —— 实测数据
════════════════════════════════════════════════════════════════
  inference_topdown 处理 1 个 bbox : 46.5 ms
  inference_topdown 处理 2 个 bbox : 46.4 ms      ← 几乎不变！

说明 46ms 里绝大部分【不是 GPU 计算】，而是每次调用的固定开销：
  · Compose(pipeline) 的 crop+resize（CPU）
  · 组 batch / H2D 传输
  · model.test_step 的框架开销
  · 后处理（heatmap → keypoints）

而 inference_topdown 的源码是：

    for bbox in bboxes:
        data_list.append(pipeline(data_info))   # ← 每个 bbox 跑一遍，且逐帧串行
    batch = pseudo_collate(data_list)
    results = model.test_step(batch)            # ← 一次前向

只要把【多帧】的 data_list 攒到一起再 test_step，
固定开销就被摊薄到整个 batch 上。

════════════════════════════════════════════════════════════════
本版做法
════════════════════════════════════════════════════════════════
① 多线程并行跑 pipeline（CPU 密集部分并行化）
② 攒够 --batch 个实例（默认 64）再一次性 test_step（GPU 一次吃饱）
③ 输出格式与 vp_rtd_range.py 【完全一致】，可逐字节比对验证

用法：
    vp_rtd_batch.py --det <框目录> --out <输出> --views 1 --start 0 --end 5000 \
                    [--batch 64] [--workers 16]
"""
import os, json, glob, argparse, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

from mmengine.dataset import Compose
try:
    from mmengine.dataset import pseudo_collate
except ImportError:
    from mmengine.dataset.utils import pseudo_collate
from mmpose.apis import init_model

B = '/root/autodl-tmp'
MATCH = int(os.environ.get('VP_MATCH', '30'))
SEG   = int(os.environ.get('VP_SEG', '1'))
FRAMES_DIR = os.environ.get('VP_FRAMES_DIR', '%s/frames/%d/%d' % (B, MATCH, SEG))
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
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                persons[int(gid)] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return persons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default=B + '/rtdetr_v2/输出/30/1/json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', nargs='+', default=['1', '3', '4', '7', '11'])
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

    # ---- 扫描任务 + 读图（按视角分组，组内有序）----
    # 每项: (view, fidx, out_path, img, bbox_or_None)
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
        """读图 + 解析框（线程池里跑）"""
        v, fidx, op, jf = t
        img = cv2.imread(os.path.join(FRAMES_DIR, v, '%06d.png' % fidx))
        if img is None:
            return t, None, None, []
        persons = parse_detect(jf)
        gids = [g for g in (0, 1) if g in persons]
        bbs = [np.array(persons[g], np.float32) for g in gids]
        return t, img, bbs, gids

    def prep_one(item):
        """单个实例跑 pipeline（线程池里跑，CPU 密集部分并行）"""
        img, bbox = item
        d = dict(img=img)
        d['bbox'] = bbox[None]
        d['bbox_score'] = np.ones(1, dtype=np.float32)
        d.update(meta)
        return pipeline(d)

    t_start = time.time()
    n_done = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        # 先批量读图+解析，一次一批 tasks
        CHUNK = a.batch
        for ci in range(0, len(tasks), CHUNK):
            chunk = tasks[ci:ci + CHUNK]
            loaded = list(ex.map(load_and_prepare, chunk))

            # 把这一批里所有人的 (img, bbox) 摊平，并记录归属
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

            # 按帧聚合结果（顺序与 owner 一致）
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
                # ⚠️ 原版按 gids 顺序 append；这里 bi 升序 == gids 顺序，一致
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
