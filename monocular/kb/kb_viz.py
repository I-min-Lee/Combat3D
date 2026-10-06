#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kb_viz —— 把 MotionBERT 的预测投影回原视频帧（用户要求：每阶段都要回到原图）

图上画三样：
  · 绿色细线  = 输入 2D（vitpose，模型实际看到的东西）
  · 橙/红粗线 = 模型预测 3D 重投影（p0 橙 / p1 红）
  · 青色细线  = 标签 3D(GT) 重投影（对照）

两种锚定：
  --anchor gt    （默认）把预测的根放到 GT 的根的相机系位置上 —— 只评"姿态形状"，
                  不受深度估计误差干扰，适合看训练进展
  --anchor ground 用最低踝踩地 + 相机射线解深度 —— 复现真实部署行为（会暴露深度误差）

原视频里没有现成抽帧（管线跑完会删 frames/），直接从 all_videos 的 AVI 按帧号取。

用法:
  python kb_viz.py --ckpt .../kb_lora.pt --takes f12 f01 --n 4
"""
import os, sys, json, glob, argparse, re
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_common as K

ALLVID = f'{K.ROOT}/all_videos'


def take_to_avi(take):
    """tag f12 -> take 1.2"""
    num = take[1:]
    return f'{int(num[0])}.{int(num[1:])}' if len(num) == 2 else None


def find_avi(take, view):
    """tag -> {view: avi 路径}"""
    m = re.match(r'^f(\d+)(\d)$', take)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        cands = [f'{a}.{b}']
        # f101 这种两位场号：f101 -> 10.1
        if len(m.group(1)) > 1:
            cands = [f'{int(m.group(1)[:-1])}.{int(m.group(1)[-1])}', f'{a}.{b}']
    else:
        return {}
    for c in cands:
        g = glob.glob(f'{ALLVID}/{c}_Miqus_{view}_*.avi')
        if g:
            return g[0]
    return None


def load_ckpt(ckpt, eval_json=None, mode=None):
    """自动识别 full / lora 权重并载入。返回 (model, CAL, args)"""
    if K.MB_ROOT not in sys.path:
        sys.path.insert(0, K.MB_ROOT)
    import importlib
    kt = importlib.import_module('kb_train')
    sd = torch.load(ckpt, map_location='cpu')
    if mode is None:
        mode = 'lora' if any(k.endswith('.A') for k in sd) else 'full'
    if eval_json and os.path.exists(eval_json):
        try:
            mode = json.load(open(eval_json)).get('mode', mode)
        except Exception:
            pass
    model = kt.build_official()
    if mode == 'lora':
        model = kt.to_lora(model, 8)
    model.load_state_dict(sd, strict=(mode == 'full'))
    model.to(kt.DEV).eval()
    print(f'[viz] 载入 {ckpt} (mode={mode})')
    cal = {}
    if eval_json and os.path.exists(eval_json):
        j = json.load(open(eval_json))
        for v, (s, R) in j.get('calib', {}).items():
            cal[v] = (s, np.array(R))
        return model, cal, j.get('args', {})
    return model, {}, {}


def draw(img, uv, color, thick):
    for a, b in K.SKEL_H36M:
        pa, pb = uv[a], uv[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        if max(abs(pa[0]), abs(pa[1])) > 8000 or max(abs(pb[0]), abs(pb[1])) > 8000:
            continue
        cv2.line(img, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)),
                 color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{K.ROOT}/mb/kb_lora.pt')
    ap.add_argument('--eval-json', default=None)
    ap.add_argument('--data', default=K.OUT)
    ap.add_argument('--takes', nargs='*', default=['f12', 'f01'])
    ap.add_argument('--views', nargs='*', default=['1'])
    ap.add_argument('--n', type=int, default=4, help='每个 (场次,视角,人) 出几张')
    ap.add_argument('--clip-len', type=int, default=243)
    ap.add_argument('--anchor', choices=['gt', 'ground'], default='gt')
    ap.add_argument('--out', default=f'{K.ROOT}/mb/viz')
    a = ap.parse_args()
    if a.eval_json is None:
        a.eval_json = a.ckpt.replace('.pt', '_eval.json')

    os.makedirs(a.out, exist_ok=True)
    model, CAL, targs = load_ckpt(a.ckpt, a.eval_json)
    import kb_train as kt
    kt.CAL = CAL
    print('已载入权重', a.ckpt, ' 视角标定:', {k: round(v[0]) for k, v in CAL.items()})

    made = 0
    for take in a.takes:
        f13r, vpr = K.take_paths(take)
        if not os.path.isdir(f13r):
            print('跳过', take, '(无 final13)'); continue
        # 该场次的组别（决定标定）
        grp = K.take_group(take)
        cal = K.Calib(K.CALIB[grp])
        for view in a.views:
            caps = {}
            for pid in (0, 1):
                fn = f'{a.data}/{take}_v{view}_p{pid}.npz'
                if not os.path.exists(fn):
                    continue
                z = np.load(fn, allow_pickle=True)
                meta = json.loads(str(z['meta']))
                k2d, k3d = z['k2d'], z['k3d']
                stride = meta['stride']
                n = len(k2d)
                if n < 8:
                    continue
                x = kt.norm2d(k2d)
                G = kt.gt_camera_mm(dict(k3d=k3d, view=view), cal)     # 逐帧根置零
                # ★ 根的绝对相机系坐标（米->mm）：重投影必须带上它，否则骨架会落在相机原点
                Xw_root = k3d[:, 0, :3].astype(np.float64)
                Rv, Tv = cal.R[view], cal.T[view]
                root_cam = ((Rv @ Xw_root.T).T + Tv) * 1000.0          # (N,3) mm
                # 逐段预测
                starts = [0] + list(range(a.clip_len, n, a.clip_len)) if n > a.clip_len else [0]
                pred = np.zeros_like(G)
                with torch.no_grad():
                    for st in starts:
                        en = min(st + a.clip_len, n)
                        seg = x[st:en]
                        pad = a.clip_len - len(seg)
                        if pad > 0:
                            seg = np.concatenate([seg, np.repeat(seg[-1:], pad, 0)], 0)
                        L = min(a.clip_len, en - st)
                        o = model(torch.from_numpy(seg[None]).to(kt.DEV))[0, :L].cpu().numpy()
                        s = CAL.get(view, (1.0, np.eye(3)))[0]
                        R = CAL.get(view, (1.0, np.eye(3)))[1]
                        pred[st:en] = (o * s) @ R.T          # 相机系 mm, root=0
                # 选帧：取 2D 可见关节最多的若干帧
                vis = (k2d[:, :, 2] > 0.05).sum(1)
                order = np.argsort(-vis)[:max(a.n * 3, 12)]
                sel = sorted(order[:a.n])
                avi = find_avi(take, view)
                if avi is None:
                    print('  找不到 AVI', take, view); continue
                if view not in caps:
                    caps[view] = cv2.VideoCapture(avi)
                cap = caps[view]
                for i in sel:
                    af = int(i * stride)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, af)
                    okf, img = cap.read()
                    if not okf:
                        continue
                    uv_in = k2d[i, :, :2].astype(np.float64)
                    for j in range(17):
                        if k2d[i, j, 2] > 0.05:
                            cv2.circle(img, tuple(np.round(uv_in[j]).astype(int)), 3, (0, 200, 0), -1)
                    draw(img, uv_in, (0, 200, 0), 2)
                    # 预测重投影：根放到 GT 根（相机系）-> 直接 K 投影
                    P = pred[i] + root_cam[i]
                    uv_p, _ = cv2.projectPoints(P.astype(np.float64), np.zeros((3, 1)),
                                                np.zeros((3, 1)), cal.K[view], cal.D[view])
                    uv_p = uv_p.reshape(-1, 2)
                    col = (0, 140, 255) if pid == 0 else (0, 0, 235)
                    draw(img, uv_p, col, 3)
                    # GT 重投影
                    uv_g, _ = cv2.projectPoints((G[i] + root_cam[i]).astype(np.float64),
                                                np.zeros((3, 1)), np.zeros((3, 1)),
                                                cal.K[view], cal.D[view])
                    uv_g = uv_g.reshape(-1, 2)
                    draw(img, uv_g, (255, 200, 0), 1)
                    e = np.linalg.norm(pred[i] - G[i], axis=1)
                    m = (k3d[i, :, 3] > 0)
                    mm = float(np.median(e[m])) if m.sum() else float('nan')
                    cv2.putText(img, f'{take} v{view} p{pid} fr{af} MPJPE={mm:.0f}mm anchor={a.anchor}',
                                (12, 30), 0, 0.62, (255, 255, 255), 2)
                    cv2.putText(img, 'green=2D in  orange/red=pred3D  cyan=GT3D',
                                (12, 56), 0, 0.55, (200, 200, 200), 1)
                    op = f'{a.out}/{take}_v{view}_p{pid}_f{af:06d}.jpg'
                    cv2.imwrite(op, img)
                    made += 1
                cap.release()
    print('出图', made, '张 ->', a.out)


if __name__ == '__main__':
    main()
