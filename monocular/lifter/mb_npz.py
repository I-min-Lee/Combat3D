#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mb_npz.py —— Harmony4D → MotionBERT 训练用 npz 适配器

把 combat3d 那边的 Harmony4D 产物转成 kendo MotionBERT 管线认的 npz：
    {OUT}/<tag>_v<view>_p<pid>.npz
        k2d   (T,17,3) float32   H36M17 像素(x,y)+conf
        k3d   (T,17,4) float32   H36M17 世界坐标(★gt3d_colmap 系，与 calib 的 extri 同系)+conf
        valid (T,)     bool
        meta  JSON 字符串 {group, take, view, pid, fps, n, src2d, src3d}

来源
    2D :  asm_gt_<tag>/annots/<view>/%06d.json    (COCO17, personID 0/1 已对齐 aria01/aria02)
    3D :  gt_<tag>/gt3d_colmap/%06d.json          ({'aria01':[(4,)]*17, 'aria02':[...]})
    pid:  personID 0 ↔ aria01, 1 ↔ aria02（h4d_boxes.py 的约定）

★★ 坐标系：k3d 必须用 **gt3d_colmap**，不是 gt3d_metric。
   2026-10-03 实测反投影残差：gt3d_colmap = 10.9/13.8/27.1 px（view 01/03/04）；
   gt3d_metric = 4158/5480/14130 px。说明 calib 的 extri 世界系就是 COLMAP 系。

★ fps：按数据集原生 20fps 写入 meta，**不做重采样**（用户要求"按人家的 fps 微调"）。

用法:
  python mb_npz.py --tag 16_mma5 --views 01,03,04,07,09,14 \
      --root /workshop/Lym/combat3d --out /workshop/Lym/combat3d/mb/data_h4d
"""
import os, sys, json, glob, argparse, time
import numpy as np

ROOT_DEF = os.environ.get('MB_ROOT_DIR', '/workshop/Lym/combat3d')
sys.path.insert(0, f'{ROOT_DEF}/mb/kb')
import kb_common as K

H4D_FPS = 20          # Harmony4D 序列原生帧率
MAP23D = {0: 'aria01', 1: 'aria02'}   # pid -> gt3d_colmap 里的键


def load_frames(annot_dir, gt_dir, view):
    """返回 (frames列表, 2d字典{fr: {pid: (17,3)}}, 3d字典{fr: {pid: (17,4)}})"""
    a2 = {}
    for p in sorted(glob.glob(f'{annot_dir}/{view}/*.json')):
        fr = int(os.path.basename(p)[:-5])
        try:
            d = json.load(open(p))
        except Exception:
            continue
        m = {}
        for it in d.get('annots', []):
            pid = it.get('personID', -1)
            if pid in (0, 1):
                kp = np.asarray(it['keypoints'], dtype=np.float32)
                if kp.shape[0] >= 17:
                    m[pid] = kp[:17, :3]
        a2[fr] = m
    a3 = {}
    for p in sorted(glob.glob(f'{gt_dir}/*.json')):
        fr = int(os.path.basename(p)[:-5])
        try:
            d = json.load(open(p))
        except Exception:
            continue
        m = {}
        for pid, key in MAP23D.items():
            if key in d:
                v = np.asarray(d[key], dtype=np.float32)
                if v.shape[0] >= 17:
                    m[pid] = v[:17, :4]
        a3[fr] = m
    frames = sorted(set(a2) | set(a3))
    return frames, a2, a3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='如 16_mma5')
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--root', default=ROOT_DEF)
    ap.add_argument('--out', default=None)
    ap.add_argument('--fps', type=float, default=H4D_FPS)
    ap.add_argument('--annots-dir', default=None,
                    help='2D 源目录；默认 {root}/asm_gt_{tag}/annots。'
                         '★自建版要传 {root}/asm_self_{tag}/annots')
    ap.add_argument('--gt-dir', default=None,
                    help='3D 真值目录；默认 {root}/gt_{tag}/gt3d_colmap')
    ap.add_argument('--src2d-tag', default=None, help='写进 meta 的 2D 来源标签')
    a = ap.parse_args()

    ann = a.annots_dir or f'{a.root}/asm_gt_{a.tag}/annots'
    gt = a.gt_dir or f'{a.root}/gt_{a.tag}/gt3d_colmap'
    src2d = a.src2d_tag or ('self' if 'asm_self' in ann else 'gt')
    out = a.out or f'{a.root}/mb/data_h4d'
    assert os.path.isdir(ann), f'缺 {ann}'
    assert os.path.isdir(gt), f'缺 {gt}'
    os.makedirs(out, exist_ok=True)
    views = [v.strip() for v in a.views.split(',') if v.strip()]

    print(f'[mb_npz] tag={a.tag}  2D={ann}  3D={gt}')
    print(f'[mb_npz] 输出 {out}  fps={a.fps}  views={views}')
    conf3d_all = []
    for v in views:
        frames, a2, a3 = load_frames(ann, gt, v)
        T = len(frames)
        if T == 0:
            print(f'  view {v}: 无数据，跳过'); continue
        for pid in (0, 1):
            p2 = np.zeros((T, 17, 3), np.float32)
            p3 = np.zeros((T, 17, 4), np.float32)
            ok = np.zeros(T, bool)
            for k, fr in enumerate(frames):
                m2 = a2.get(fr, {}).get(pid)
                m3 = a3.get(fr, {}).get(pid)
                if m2 is not None:
                    p2[k] = m2
                if m3 is not None:
                    p3[k] = m3
                    conf3d_all.append(m3[:, 3])
                ok[k] = (m2 is not None) and (m3 is not None)
            k2d = K.map_to_h36m(p2, K.H36M_FROM_COCO)
            k3d = K.map_to_h36m(p3, K.H36M_FROM_COCO)
            # ★ valid 必须是 (T,17) 逐关节 —— kb_train.calibrate() 用 seq['valid'][s:s+L]
            #   做布尔索引，形状 (T,) 会少一维导致 SVD 维度错（17 vs 7776）。
            valid = ok[:, None] & (k2d[:, :, 2] > 0.05)
            meta = dict(group=f'h4d_{a.tag}', take=a.tag, view=v, pid=pid,
                        fps=a.fps, n=int(T), n_ok=int(ok.sum()),
                        src2d=src2d, src3d='gt3d_colmap', frame0=int(frames[0]))
            dst = f'{out}/{a.tag}_v{v}_p{pid}.npz'
            np.savez_compressed(dst, k2d=k2d.astype(np.float32),
                                k3d=k3d.astype(np.float32),
                                valid=valid, meta=json.dumps(meta))
            print(f'  view {v} pid{pid}: T={T} 有效 {int(ok.sum())} -> {os.path.basename(dst)}')
    if conf3d_all:
        c = np.concatenate(conf3d_all)
        print(f'[mb_npz] 3D 真值第4通道范围: min={c.min():.3f} max={c.max():.3f} '
              f'中位={np.median(c):.3f}  (若是 0/1 表示可见性)')
    print('DONE', out)


if __name__ == '__main__':
    main()
