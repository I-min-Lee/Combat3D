#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mb_npz2.py —— 用【官方 2D + 自建三角化 3D】生成 MotionBERT 训练 npz。

与 mb_npz.py 的唯一区别：3D 标签不再来自数据集自带的 poses3d（gt3d_colmap），
而是来自 **我们自己跑 tri_h4d.py 的三角化产物**（em_off_<tag>/lam1.0/pid<p>/keypoints3d）。
这样 k2d / k3d 是**同一批 2D 观测**推出来的，天然同源。

    {OUT}/<tag>_v<view>_p<pid>.npz
        k2d   (T,17,3)  H36M17 像素(x,y)+conf      <- asm_off_<tag>/annots/<view>/%06d.json
        k3d   (T,17,4)  H36M17 COLMAP 世界系+conf   <- em_off_<tag>/lam1.0/pid<p>/keypoints3d/%06d.json
        valid (T,17)    bool
        meta  JSON {group, take, view, pid, fps, n, src2d, src3d}

关节链：em 输出是 OpenPose body25 (25,4)
        -> COCO17  取 COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
        -> H36M17  用 kb_common.H36M_FROM_COCO
       2D 侧：annots 是 COCO17 -> H36M17（同 mb_npz.py）

用法:
  python mb_npz2.py --tag 16_mma5 --views 01,03,04,07,09,14 --out .../mb/data_h4d_offtri
"""
import os, sys, json, glob, argparse
import numpy as np

ROOT_DEF = os.environ.get('MB_ROOT_DIR', '/workshop/Lym/combat3d')
sys.path.insert(0, f'{ROOT_DEF}/mb/kb')
import kb_common as K

H4D_FPS = 20
C17_IN_B25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
MAP23D = {0: 'aria01', 1: 'aria02'}


def load_2d(annot_dir, view):
    out = {}
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
        out[fr] = m
    return out


def load_3d(em_root, view):
    """em_<tag>/lam1.0/pid<p>/keypoints3d/%06d.json -> {fr: {pid: (17,4)}}"""
    out = {}
    for pid in (0, 1):
        d = f'{em_root}/lam1.0/pid{pid}/keypoints3d'
        for p in sorted(glob.glob(d + '/*.json')):
            fr = int(os.path.basename(p)[:-5])
            try:
                j = json.load(open(p))
            except Exception:
                continue
            if not j:
                continue
            k = np.asarray(j[0]['keypoints3d'], dtype=np.float32)
            if k.shape[0] < 25:
                continue
            out.setdefault(fr, {})[pid] = k[C17_IN_B25, :4]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--root', default=ROOT_DEF)
    ap.add_argument('--out', default=None)
    ap.add_argument('--fps', type=float, default=H4D_FPS)
    ap.add_argument('--annots-dir', default=None)
    ap.add_argument('--em-dir', default=None)
    ap.add_argument('--src2d-tag', default='off')
    a = ap.parse_args()

    ann = a.annots_dir or f'{a.root}/asm_off_{a.tag}/annots'
    em = a.em_dir or f'{a.root}/em_off_{a.tag}'
    out = a.out or f'{a.root}/mb/data_h4d_offtri'
    assert os.path.isdir(ann), f'缺 {ann}'
    assert os.path.isdir(em + '/lam1.0'), f'缺 {em}/lam1.0'
    os.makedirs(out, exist_ok=True)
    views = [v.strip() for v in a.views.split(',') if v.strip()]

    print(f'[mb_npz2] tag={a.tag}  2D={ann}  3D={em}  fps={a.fps}')
    for v in views:
        a2 = load_2d(ann, v)
        a3 = load_3d(em, v)
        frames = sorted(set(a2) | set(a3))
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
                ok[k] = (m2 is not None) and (m3 is not None)
            k2d = K.map_to_h36m(p2, K.H36M_FROM_COCO)
            k3d = K.map_to_h36m(p3, K.H36M_FROM_COCO)
            valid = ok[:, None] & (k2d[:, :, 2] > 0.05) & (k3d[:, :, 3] > 0.0)
            meta = dict(group=f'h4d_{a.tag}', take=a.tag, view=v, pid=pid,
                        fps=a.fps, n=int(T), n_ok=int(ok.sum()),
                        src2d=a.src2d_tag, src3d='em_off_tri', frame0=int(frames[0]))
            dst = f'{out}/{a.tag}_v{v}_p{pid}.npz'
            np.savez_compressed(dst, k2d=k2d.astype(np.float32),
                                k3d=k3d.astype(np.float32),
                                valid=valid, meta=json.dumps(meta))
            print(f'  view {v} pid{pid}: T={T} 有效帧 {int(ok.sum())} '
                  f'有效观测 {int(valid.sum())} -> {os.path.basename(dst)}')
    print('DONE', out)


if __name__ == '__main__':
    main()
