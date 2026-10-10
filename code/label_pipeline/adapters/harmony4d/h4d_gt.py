#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h4d_gt.py — Harmony4D 真值 → MPJPE 评测用的 GT 目录

数据源：
  processed_data/poses3d/%05d.npy  dict{subject: (17,4)}  ← COCO17 + conf，**米制世界系**
  processed_data/poses2d/cam{NN}/%05d.npy  dict{subject: (45,2)}  ← 数据集自带 2D（45 SMPL 关节）
  colmap/workplace/scale.npy       4×4 相似变换（1.375482×旋转 + 平移）

输出：
  {out}/gt3d_metric/%06d.json   {subject: [[x,y,z,conf]×17]}   ← 官方米制系（报 MPJPE 用）
  {out}/gt3d_colmap/%06d.json   同上但变换到 **COLMAP 世界系**（= 管线运行的世界系）
  {out}/ref2d/{view}/%06d.json  {subject: [[x,y]×45]}          ← 数据集自带 2D 参考
  {out}/scale_metric.npy, {out}/h4d_gt_meta.json

为什么给两套坐标：
  经实测，管线的 extri.yml 直接写 COLMAP 世界系（P=K[R|T]，与 read_camera 一致），
  所以管线输出的 3D 在 **COLMAP 系**；而官方 MPJPE 数字在**米制系**。
  两套都给，评测时二选一即可，避免任何隐式近似。
  换算：X_metric = S @ X_colmap ，X_colmap = S^-1 @ X_metric

用法：
  python h4d_gt.py --seq-root <...>/016_mma4 --out <gt 目录> \
                   --views 01,03,04,07,09,14 [--start 1 --end 741]
"""
import os, json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--no-ref2d', action='store_true')
    a = ap.parse_args()

    _sp = os.path.join(a.seq_root, 'colmap', 'workplace', 'scale.npy')
    S = np.load(_sp) if os.path.exists(_sp) else np.eye(4)   # ★无 scale.npy → 单位阵
    Rs, ts = S[:3, :3], S[:3, 3]
    s = float(np.linalg.norm(Rs[:, 0]))
    Rn = Rs / s
    Sinv = np.eye(4); Sinv[:3, :3] = Rn.T / s; Sinv[:3, 3] = -Rn.T @ ts / s
    def to_colmap(Xm):
        Xm = np.asarray(Xm, float)
        return (Sinv[:3, :3] @ Xm.T + Sinv[:3, 3:]).T

    for d in ('gt3d_metric', 'gt3d_colmap'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    np.save(os.path.join(a.out, 'scale_metric.npy'), S)

    p3dir = os.path.join(a.seq_root, 'processed_data', 'poses3d')
    frames = sorted(int(f.split('.')[0]) for f in os.listdir(p3dir) if f.endswith('.npy'))
    frames = [f for f in frames if a.start <= f <= a.end]

    confs, subj_hist, nconf_hist = [], {}, {}
    for fi in frames:
        d = np.load(os.path.join(p3dir, '%05d.npy' % fi), allow_pickle=True)
        d = d.item() if getattr(d, 'dtype', None) == object else d
        gm, gc = {}, {}
        for subj, arr in d.items():
            arr = np.asarray(arr, float)
            assert arr.shape == (17, 4), (subj, arr.shape)
            gm[subj] = arr.tolist()
            c = to_colmap(arr[:, :3])
            gc[subj] = np.hstack([c, arr[:, 3:4]]).tolist()
            confs.append(arr[:, 3])
            nconf_hist[int((arr[:, 3] < 0.3).sum())] = nconf_hist.get(int((arr[:, 3] < 0.3).sum()), 0) + 1
        subj_hist[len(d)] = subj_hist.get(len(d), 0) + 1
        for dd, val in (('gt3d_metric', gm), ('gt3d_colmap', gc)):
            with open(os.path.join(a.out, dd, '%06d.json' % fi), 'w') as f:
                json.dump(val, f)

    print('=== GT 导出 ===')
    print('  帧数 %d (范围 %d..%d)   每帧人数分布=%s'
          % (len(frames), frames[0] if frames else '-', frames[-1] if frames else '-',
             sorted(subj_hist.items())))
    if confs:
        c = np.concatenate(confs)
        print('  poses3d conf: min=%.3f  p05=%.3f  中位=%.3f  低置信(<0.3)关节占 %.2f%%'
              % (c.min(), np.percentile(c, 5), float(np.median(c)), 100.0 * (c < 0.3).mean()))
        print('  每帧低置信关节数分布 (top5)=%s'
              % sorted(nconf_hist.items())[:5])

    if not a.no_ref2d:
        views = [v.zfill(2) for v in a.views.split(',')]
        for v in views:
            src = os.path.join(a.seq_root, 'processed_data', 'poses2d', 'cam%s' % v)
            dst = os.path.join(a.out, 'ref2d', v)
            os.makedirs(dst, exist_ok=True)
            n = 0
            for fi in frames:
                p = os.path.join(src, '%05d.npy' % fi)
                if not os.path.exists(p):
                    continue
                d = np.load(p, allow_pickle=True)
                d = d.item() if getattr(d, 'dtype', None) == object else d
                with open(os.path.join(dst, '%06d.json' % fi), 'w') as f:
                    json.dump({k: np.asarray(vv, float).tolist() for k, vv in d.items()}, f)
                n += 1
            print('  ref2d view %s : %d 帧 (%d 关节)' % (v, n, 45))

    meta = dict(seq_root=os.path.abspath(a.seq_root), scale=s,
                n_frames=len(frames), frames=[frames[0], frames[-1]] if frames else None,
                metric_frame='官方米制（重力对齐, z 向上）',
                colmap_frame='管线运行系（extri.yml 所在系）',
                convert='X_metric = scale_metric.npy @ X_colmap')
    with open(os.path.join(a.out, 'h4d_gt_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print('[OK] %s/{gt3d_metric,gt3d_colmap,ref2d,scale_metric.npy}' % a.out)


if __name__ == '__main__':
    main()
