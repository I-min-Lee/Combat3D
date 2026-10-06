#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""final13_from.py <src_base> <out_base>  — 从任意 emfit 树基目录生成 final13(13节点+conf+npz)"""
import os, sys, json, glob
import numpy as np, cv2
sys.path.insert(0, '/root/autodl-tmp/emoff/EasyMocap-master')
B = '/root/autodl-tmp'; VIEWS = ['1', '3', '4', '7', '11']
IDX13 = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
from easymocap.mytools.camera_utils import read_camera
cams = read_camera(B + '/intri.yml', B + '/extri_refined.yml'); cams.pop('basenames', None)
CAM = {}
for v in VIEWS:
    K = np.asarray(cams[v]['K'], np.float64).reshape(3, 3).copy(); K[2, 2] = 1.0
    R = np.asarray(cams[v]['R'], np.float64).reshape(3, 3)
    T = np.asarray(cams[v]['T'], np.float64).reshape(3, 1)
    CAM[v] = {'K': K, 'dist': np.asarray(cams[v]['dist'], np.float64).reshape(-1),
              'rvec': cv2.Rodrigues(np.ascontiguousarray(R))[0], 'T': T}
src_base, out_base = sys.argv[1], sys.argv[2]
for pid in (0, 1):
    src = '%s/pid%d/keypoints3d_b25' % (src_base, pid)
    trisrc = os.environ.get('TRI_SRC', '%s/em_rtd14k/pid%d/keypoints3d' % (B, pid))
    if not os.path.isdir(src):
        print('skip pid%d (无 %s)' % (pid, src)); continue
    out3 = '%s/pid%d/keypoints3d' % (out_base, pid); os.makedirs(out3, exist_ok=True)
    _N = int(os.environ.get('F13_N', '14017'))   # 帧数：B组 0.1 传 2428
    files = sorted(glob.glob(src + '/*.json'))[:_N]
    for fp in files:
        fr = int(os.path.basename(fp)[:-5])
        k = np.array(json.load(open(fp))[0]['keypoints3d']); k13 = k[IDX13, :3]
        conf = np.zeros(13)
        tfp = '%s/%06d.json' % (trisrc, fr)
        if os.path.exists(tfp):
            t = np.array(json.load(open(tfp))[0]['keypoints3d'])
            conf = np.where(np.abs(t[IDX13, :3]).sum(1) > 1e-6, t[IDX13, 3], 0.0)
        json.dump([{'id': pid, 'keypoints3d': [[float(k13[j, 0]), float(k13[j, 1]), float(k13[j, 2]), float(conf[j])]
                                               for j in range(13)]}], open('%s/%06d.json' % (out3, fr), 'w'))
    print('final13 %s pid%d -> %d' % (out_base, pid, len(files)), flush=True)
print('FINAL13_DONE', flush=True)
