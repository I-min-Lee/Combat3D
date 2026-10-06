#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把单目 3D 的【骨架节点】用鱼眼投影叠回原视频（不画网格），出 mp4。

节点来源（按优先级）：
  1) --joints smpl25 : emfit/pid{p}/k3d/*.json（EasyMocap 拟合出的 SMPL 25 关节）
  2) --joints f13    : final13 目录的 13 节点（= MotionBERT 输出本身）
两种都能画；默认两个都画（f13 用实心点+粗线，smpl25 用细线），方便对比。

用法:
  v26_node_overlay.py --take 002_sword3 --view 04 --match 602 \
      --f13 <final13 目录> --out <输出目录> [--smpl <emfit 目录>] [--subsample 1]
"""
import os, sys, glob, argparse, json
import numpy as np, cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
COL = {0: (90, 230, 90), 1: (60, 170, 255)}          # pid0 绿 / pid1 橙
E13 = [(0, 1), (0, 4), (1, 4), (1, 2), (2, 3), (4, 5), (5, 6),
       (1, 7), (4, 10), (7, 10), (7, 8), (8, 9), (10, 11), (11, 12)]
# body25 (EasyMocap) 骨架
E25 = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
       (10, 11), (8, 12), (12, 13), (13, 14), (0, 15), (15, 17), (0, 16), (16, 18),
       (14, 19), (19, 20), (20, 21), (11, 22), (22, 23), (23, 24)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--take', required=True)
    ap.add_argument('--view', default='04')
    ap.add_argument('--match', default='602')
    ap.add_argument('--f13', default=None)
    ap.add_argument('--smpl', default=None)
    ap.add_argument('--calib', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--scale', type=float, default=0.5, help='输出缩放（原图 3840x2160 太大）')
    a = ap.parse_args()
    T, V = a.take, a.view.zfill(2)
    f13 = a.f13 or f'{B}/final13_mono_{T}'
    calib = a.calib or f'{B}/calib_gt_{T}'
    froot = f'{B}/frames/{a.match}/1/{V}'
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, os.environ.get('EMCORE', B + '/port/emcore'))
    from easymocap.mytools.camera_utils import read_camera
    C = read_camera(os.path.join(calib, 'intri.yml'), os.path.join(calib, 'extri.yml'))
    c = C[V]
    K = np.asarray(c['K'], float).reshape(3, 3)
    D = np.asarray(c['dist'], float).reshape(-1)[:4].reshape(4, 1)
    R = np.asarray(c['R'], float).reshape(3, 3)
    T_ = np.asarray(c['T'], float).reshape(3)

    def load_joints(pid, k):                     # k = 0-based 帧索引
        out = {}
        p = f'{f13}/pid{pid}/keypoints3d/{k:06d}.json'
        if os.path.exists(p):
            d = json.load(open(p))
            out['f13'] = np.array(d[0]['keypoints3d'], float)[:, :3]
        if a.smpl:
            for name, sub in (('k3d', 'k3d'), ('keypoints3d', 'keypoints3d')):
                q = f'{a.smpl}/pid{pid}/{sub}/{k:06d}.json'
                if os.path.exists(q):
                    d = json.load(open(q))
                    arr = np.array(d[0].get('keypoints3d', d[0].get('k3d')), float)
                    out['smpl'] = arr[:, :3]
                    break
        return out


    frames = sorted(int(os.path.basename(f).split('.')[0])
                    for f in glob.glob(f'{f13}/pid0/keypoints3d/*.json'))
    frames = [f for f in frames if a.start - 1 <= f <= a.end - 1]
    if not frames:
        print('★ 没有帧（检查 --f13）'); return
    img0 = cv2.imread(f'{froot}/{frames[0] + 1:06d}.png')
    if img0 is None:
        print('★ 找不到原图', froot); return
    H0, W0 = img0.shape[:2]
    W, H = int(W0 * a.scale), int(H0 * a.scale)
    print(f'{len(frames)} 帧，{W0}x{H0} -> {W}x{H}')

    vw = None
    for i, k in enumerate(frames):
        img = cv2.imread(f'{froot}/{k + 1:06d}.png')
        if img is None:
            continue
        img = cv2.resize(img, (W, H))
        sc = W / W0
        for pid in (0, 1):
            js = load_joints(pid, k)
            for kind, edges in (('smpl', E25), ('f13', E13)):
                if kind not in js:
                    continue
                X = js[kind]
                Xc = (R @ X.T + T_[:, None]).T
                okj = Xc[:, 2] > 0.05
                if okj.sum() < 4:
                    continue
                uv, _ = cv2.fisheye.projectPoints(Xc[okj].reshape(-1, 1, 3),
                                                  np.zeros(3), np.zeros((3, 1)), K, D)
                uv = uv.reshape(-1, 2) * sc
                idx = np.where(okj)[0]
                pos = {int(j): uv[t] for t, j in enumerate(idx)}
                thick = 3 if kind == 'f13' else 1
                rad = 4 if kind == 'f13' else 2
                for e0, e1 in edges:
                    if e0 in pos and e1 in pos:
                        cv2.line(img, tuple(pos[e0].astype(int)), tuple(pos[e1].astype(int)),
                                 COL[pid], thick, cv2.LINE_AA)
                for j, p in pos.items():
                    cv2.circle(img, tuple(p.astype(int)), rad, COL[pid], -1, cv2.LINE_AA)
        cv2.putText(img, 'view %s  f%d   (粗=MB 13节点, 细=SMPL25)' % (V, k + 1), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if vw is None:
            vw = cv2.VideoWriter(f'{a.out}/nodes_reproj_{T}_v{V}.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (W, H))
        vw.write(img)
        if (i + 1) % 100 == 0:
            print('  写出 %d/%d' % (i + 1, len(frames)), flush=True)
    vw.release()
    out = f'{a.out}/nodes_reproj_{T}_v{V}.mp4'
    print('[OK]', out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
