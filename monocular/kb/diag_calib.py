#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：每个 (场次,视角) 在 两套标定 x 两种 pid<->personID 配对 下的 3D->2D 重投影中位。
判据（见 docs/METRICS.md 的标定一节）：中位最低的那套 = 正确标定。"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_common as K

TAKES = sys.argv[1:] or ['f01', 'f12', 'f101', 'f21', 'f51', 'f02']


def load_3d(tag, pid, step=8, nmax=600):
    d = f'{K.FINAL13.format(tag=tag)}/pid{pid}'
    out = []
    files = sorted(glob.glob(f'{d}/keypoints3d/*.json'))[::step][:nmax]
    for p in files:
        fr = int(os.path.basename(p)[:6])
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if not j:
            continue
        kp = np.asarray(j[0]['keypoints3d'], dtype=np.float64)
        out.append((fr, kp))
    return out


def load_2d(tag, view, nmax=600):
    f13, vp = K.take_paths(tag)
    files = sorted(glob.glob(f'{vp}/{view}/*.json'))
    out = {}
    for p in files[:nmax * 8]:
        try:
            j = json.load(open(p))
        except Exception:
            continue
        rec = {}
        for a in j.get('annots', []):
            rec[int(a.get('personID', -1))] = np.asarray(a['keypoints'], dtype=np.float64)
        out[int(os.path.basename(p)[:6])] = (rec, j.get('width'), j.get('height'))
    return out


def reproj_err(k3d, k2d, cal, view):
    """k3d: (13,4) 世界米, k2d: (17,3) COCO 像素。返回中位像素差。"""
    H3 = K.map_to_h36m(k3d[None], K.H36M_FROM_13)[0]      # (17,4)
    H2 = K.map_to_h36m(k2d[None], K.H36M_FROM_COCO)[0]    # (17,3)
    uv = cal.project(H3[None, :, :3], view)               # (17,2)
    m = (H3[:, 3] > 0) & (H2[:, 2] > 0.05)
    if m.sum() < 4:
        return None, 0
    return float(np.median(np.linalg.norm(uv[m] - H2[m, :2], axis=1))), int(m.sum())


CAL_960 = K.Calib(K.CALIB['960'])
CAL_1920 = K.Calib(K.CALIB['1920'])

print(f'{"take":6s} {"view":4s} {"2D wh":11s} {"calib":16s} {"pair":10s} {"med_px":>8s} {"n":>4s}')
for tag in TAKES:
    f13r, vpr = K.take_paths(tag)
    if not os.path.isdir(f13r):
        print(f'{tag}  无 final13'); continue
    for view in ['1', '4']:
        d2 = load_2d(tag, view)
        if not d2:
            continue
        wh = next(iter(d2.values()))[1:]
        for cname, cal in (('calib_idfix', CAL_960), ('calib_B_192', CAL_1920)):
            for pair in ('id', 'swap'):
                errs = []
                for pid in (0, 1):
                    for fr, k3d in load_3d(tag, pid, step=13, nmax=300):
                        if fr not in d2:
                            continue
                        rec, _, _ = d2[fr]
                        j = pid if pair == 'id' else 1 - pid
                        if j not in rec:
                            continue
                        e, n = reproj_err(k3d, rec[j], cal, view)
                        if e is not None:
                            errs.append(e)
                if errs:
                    print(f'{tag:6s} {view:4s} {str(wh):11s} {cname:16s} {pair:10s} '
                          f'{np.median(errs):8.2f} {len(errs):4d}')
    print()
