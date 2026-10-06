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
"""tri_temporal_ab.py — 时序正则化三角化 (把时序放进求解器内部)

前端与 tri_ablate_id.py 完全一致: 观测构建 / 去畸变 / 边框权重 / 子集枚举 / 内点选取。
只换最后一步:
   原版 = 每帧对被选内点做一次加权 DLT(SVD), 帧间无耦合
   本版 = 所有帧的 DLT 线性方程 + 时序平滑项 拼成一个大稀疏最小二乘, 整条轨迹一次解出

线性化:  DLT 行 r = u*P2 - P0 (4维), r·X_h = 0; 取 X_h=[X,1] => r[:3]·X = -r[3]  (对 X 线性)

⚠️ 尺度陷阱(踩过, 记录备查):
   直接最小化 Σw(r·[X;1])² 会系统性把点【拉向原点】—— 因为 ||[X;1]|| 随离原点距离增大,
   而 DLT 最小化的是尺度无关的 Σw(r·X_h)²/||X_h||²。实测两者差 70-260mm。
   去掉"逐行归一化"后偏差压到 ~18mm。完全等价需要非线性求解器。
   18mm 相对本任务 150-1000mm 的效应量可接受; 且整个 λ 扫描共用同一求解器,
   故时序项的消融是干净的(λ=0 即该求解器的自身基线)。

时序项:  (lam * RN) * (X_t - 2X_{t-1} + X_{t-2})   二阶 = 加速度惩罚
        RN = 该关节观测行的中位范数, 用来自动定标, 使 lam 是无量纲的相对权重

用法: --lams 0,0.01,0.1,0.3,1,3  一次采集, 多 λ 求解 -> out/lam{L}/pid{p}/
"""
import os, json, glob, argparse, itertools
import numpy as np
import cv2
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import sys
sys.path.insert(0, os.environ.get('EMCORE', '/workshop/Lym/combat3d/port/emcore'))   # ★2026-10-03 修：原写死 westc 路径，222 上不存在
from easymocap.mytools.camera_utils import read_camera
from easymocap.dataset.config import coco17tobody25

B = '/root/autodl-tmp'
VIEWS = os.environ.get('AB_VIEWS', '1,2,3,4,7,11').split(',')
AB_ANNOTS = os.environ.get('AB_ANNOTS', f'{B}/easymocap_6v_2danchor_gated/1_1/annots')
W_IMG = int(os.environ.get('AB_W_IMG', '960'))   # B组 1920x1440 -> 1920/1440
H_IMG = int(os.environ.get('AB_H_IMG', '720'))
BORDER_PX = 12.0
W_CLIP = 0.15
W_BOXTOUCH = 0.5
MIN_VIEWS = 3
DET_DIR = os.environ.get('AB_DET_DIR', B + '/rtdetr_v2/输出/1/1/json')
OUT_JOINTS = 19


def _load_fisheye_flags(intri_path):
    """★H4D: 从我们写的 intri.yml 读 fisheye_<v>: 1（read_camera 不读自定义键）"""
    import re as _re
    flags = {}
    try:
        for _ln in open(intri_path):
            _m = _re.match(r'\s*fisheye_(\S+)\s*:\s*1', _ln)
            if _m:
                flags[_m.group(1)] = True
    except Exception:
        pass
    return flags


def build_cams(intri, extri):
    cams = read_camera(intri, extri)
    cams.pop('basenames', None)
    _FISHEYE = _load_fisheye_flags(intri)                      # ★H4D
    C = {}
    for v in VIEWS:
        K = np.asarray(cams[v]['K'], np.float64).reshape(3, 3).copy(); K[2, 2] = 1.0
        R = np.asarray(cams[v]['R'], np.float64).reshape(3, 3)
        T = np.asarray(cams[v]['T'], np.float64).reshape(3, 1)
        dist = np.asarray(cams[v]['dist'], np.float64).reshape(-1)
        C[v] = {'K': K, 'R': R, 'T': T, 'dist': dist, 'P': K @ np.hstack([R, T]),
                'fisheye': _FISHEYE.get(v, False)}                      # ★H4D
    return C


def undistort(uv, cam):
    # ★H4D: Harmony4D 是 OPENCV_FISHEYE（4 个径向参数）。OpenCV 标准 undistortPoints 会把
    # k1..k4 当成 [k1,k2,p1,p2]（2 径向 + 2 切向）—— 模型都不对，不是"精度差一点"。
    # 鱼眼分支同样用 P=K，输出"去畸变后的针孔像素坐标"，与管线其余部分
    # （proj_lin / dlt / 内点判据）所用 P=K[R|T] 完全同一口径。
    if cam.get('fisheye', False):
        _d = np.asarray(cam['dist'], np.float64).reshape(-1)[:4].reshape(4, 1)
        out = cv2.fisheye.undistortPoints(np.array([[uv]], np.float64), cam['K'], _d,
                                          None, None, cam['K'])
        return out.reshape(2)
    out = cv2.undistortPoints(np.array([[uv]], np.float64), cam['K'], cam['dist'],
                              None, cam['K'])
    return out.reshape(2)


def proj_lin(X, cam):
    x = cam['R'] @ np.asarray(X, np.float64) + cam['T'].reshape(3)
    u = cam['K'] @ x
    return u[:2] / u[2]


def dlt(uvs, cams, views, weights):
    A = []
    for uv, v, w in zip(uvs, views, weights):
        P = cams[v]['P']
        sw = np.sqrt(max(w, 1e-6))
        A.append(sw * (uv[0] * P[2] - P[0]))
        A.append(sw * (uv[1] * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.stack(A))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def border_weight(uv_obs, cam_bbox_touch):
    import os as _os
    if _os.environ.get('AB_NO_BORDER'):
        return 1.0
    w = 1.0
    if (uv_obs[0] < BORDER_PX or uv_obs[0] > W_IMG - BORDER_PX or
            uv_obs[1] < BORDER_PX or uv_obs[1] > H_IMG - BORDER_PX):
        w *= W_CLIP
    if cam_bbox_touch:
        w *= W_BOXTOUCH
    return w


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def index_det(v):
    m = {}
    for fp in glob.glob(os.path.join(DET_DIR, v, '*.json')):
        try:
            m[int(os.path.basename(fp).rsplit('_', 1)[1][:6])] = fp
        except Exception:
            pass
    return m


def boxes_touch(v, fr, detidx, obs_bbox):
    fp = detidx[v].get(fr)
    if not fp:
        return False
    try:
        d = json.load(open(fp))
    except Exception:
        return False
    best, touch = 0.0, False
    for s in d.get('shapes', []):
        if s.get('label') != 'person':
            continue
        p = s['points']
        b = (min(p[0][0], p[1][0]), min(p[0][1], p[1][1]),
             max(p[0][0], p[1][0]), max(p[0][1], p[1][1]))
        s_ = iou(obs_bbox, b)
        if s_ > best:
            best = s_
            touch = (b[0] <= 1 or b[1] <= 1 or b[2] >= W_IMG - 1 or b[3] >= H_IMG - 1)
    return touch if best > 0.05 else False


def collect(pid, start, end, min_conf, detidx, cams):
    """前端与 tri_ablate_id.py 一致; 产出每关节的 DLT 行 / 逐帧初值 / conf"""
    src, T = AB_ANNOTS, end - start
    rows = [[] for _ in range(OUT_JOINTS)]
    inits = [np.full((T, 3), np.nan) for _ in range(OUT_JOINTS)]
    confs = [np.zeros(T) for _ in range(OUT_JOINTS)]
    n_sub = {k: 0 for k in range(1, 8)}

    for fr in range(start, end):
        t = fr - start
        obs = {}
        for v in VIEWS:
            fp = f'{src}/{v}/{fr:06d}.json'
            if not os.path.exists(fp):
                continue
            try:
                d = json.load(open(fp))
                kp17 = None
                for a in d['annots']:
                    if a.get('personID') == pid:
                        kp17 = np.array(a['keypoints'], np.float64)
                        break
                if kp17 is None or kp17.shape != (17, 3):
                    continue
                k = coco17tobody25(kp17)
            except Exception:
                continue
            obs[v] = k
        if not obs:
            continue
        touch = {}
        for v, k in obs.items():
            m = k[:, 2] > min_conf
            if m.sum() >= 3:
                lo = k[m, :2].min(axis=0); hi = k[m, :2].max(axis=0)
                touch[v] = boxes_touch(v, fr, detidx, (lo[0], lo[1], hi[0], hi[1]))
            else:
                touch[v] = False

        for j in range(OUT_JOINTS):
            cand = []
            for v, k in obs.items():
                if k[j, 2] < min_conf:
                    continue
                uv_o = np.array(k[j, :2], np.float64)
                cand.append((v, undistort(uv_o, cams[v]), uv_o,
                             float(k[j, 2]) * border_weight(uv_o, touch[v])))
            if len(cand) < MIN_VIEWS:
                continue
            cand.sort(key=lambda z: -z[3])
            best = None
            import os as _os
            _noenum = bool(_os.environ.get('AB_NO_ENUM'))
            if _noenum:                      # ★消融：只解一次（全部可用视角），不做子集枚举
                _sub = tuple(range(len(cand)))
                _X = dlt([cand[i][1] for i in _sub], cams,
                         [cand[i][0] for i in _sub], [cand[i][3] for i in _sub])
                if _X is not None:
                    _w = sum(c[3] for c in cand)
                    _c = sum(c[3]*min(np.linalg.norm(proj_lin(_X, cams[c[0]])-c[1]),300.0)
                             for c in cand)/max(_w,1e-6)
                    best = (_c, _X, _sub)
            for r in (() if _noenum else range(len(cand), MIN_VIEWS - 1, -1)):
                for sub in itertools.combinations(range(len(cand)), r):
                    X = dlt([cand[i][1] for i in sub], cams,
                            [cand[i][0] for i in sub], [cand[i][3] for i in sub])
                    if X is None:
                        continue
                    cost = wtot = 0.0
                    for (vv, uu, uo, ww) in cand:
                        cost += ww * min(np.linalg.norm(proj_lin(X, cams[vv]) - uu), 300.0)
                        wtot += ww
                    cost /= max(wtot, 1e-6)
                    if best is None or cost < best[0]:
                        best = (cost, X, sub)
            if best is None:
                continue
            X0 = best[1]
            sub = best[2]                       # 枚举出的最优视角子集(索引)
            es = np.array([np.linalg.norm(proj_lin(X0, cams[c[0]]) - c[1]) for c in cand])
            scale = max(np.median(es) * 3.0, 20.0)
            inl = [i for i, e in enumerate(es) if e <= scale]
            # ⚠️ 与同事 tri_robust_14k_v2.py 逐字对齐: 内点够就用内点替换 sub,
            #    不够则【保留 sub】—— 不是"用全部候选"(那是我先前的实现偏差)
            if len(inl) >= MIN_VIEWS:
                Xr = dlt([cand[i][1] for i in inl], cams,
                         [cand[i][0] for i in inl], [cand[i][3] for i in inl])
                if Xr is not None:
                    X0 = Xr
                    sub = tuple(inl)
            use = [cand[i] for i in sub]
            inits[j][t] = X0
            n_sub[min(len(use), 6)] += 1
            confs[j][t] = float(np.mean([u[3] for u in use]))
            for (v, uv_u, uv_o, w) in use:
                P = cams[v]['P']
                for r4 in (uv_u[0] * P[2] - P[0], uv_u[1] * P[2] - P[1]):
                    rows[j].append((t, r4[:3], -r4[3], np.sqrt(max(w, 1e-6))))
        if t % 1000 == 0:
            print(f'  [collect] pid{pid} f{fr}', flush=True)
    return rows, inits, confs, n_sub


def solve_joint(R, init, lam, order, T):
    X = init.copy()
    ok = ~np.isnan(X[:, 0])
    if ok.sum() == 0:
        return None
    if ok.sum() < T:
        gi = np.where(ok)[0]
        for c in range(3):
            X[:, c] = np.interp(np.arange(T), gi, X[gi, c])
    RN = float(np.median([np.linalg.norm(coef) for (_, coef, _, _) in R]))
    nT = max(T - order, 0)
    ne = len(R) + 3 * nT
    ii, jj, vv0, bb0, sw0, tt0 = [], [], [], [], [], []
    for k, (t, coef, rhs, sw) in enumerate(R):
        for c in range(3):
            ii.append(k); jj.append(3 * t + c); vv0.append(coef[c])
        bb0.append(rhs); sw0.append(sw); tt0.append(t)
    for m in range(nT):
        t = m + order
        for c in range(3):
            r = len(R) + 3 * m + c
            if order == 2:
                vv0.extend([lam, -2.0 * lam, lam])
                jj.extend([3 * t + c, 3 * (t - 1) + c, 3 * (t - 2) + c])
            else:
                vv0.extend([lam, -lam])
                jj.extend([3 * t + c, 3 * (t - 1) + c])
            ii.extend([r] * (3 if order == 2 else 2))
            bb0.append(0.0)
    ii = np.array(ii); jj = np.array(jj)
    vv0 = np.array(vv0, np.float64); bb0 = np.array(bb0, np.float64)
    sw0 = np.array(sw0, np.float64); tt0 = np.array(tt0)
    # 用结果尺度 s_t = ||[X_t;1]|| 折算观测行, 复现 DLT 的尺度无关性
    st = np.sqrt((X * X).sum(1) + 1.0)
    inv = 1.0 / st
    vv = vv0.copy(); bb = bb0.copy()
    vv[:len(R) * 3] = vv0[:len(R) * 3] * np.repeat(sw0 * inv[tt0], 3)
    bb[:len(R)] = bb0[:len(R)] * sw0 * inv[tt0]
    if lam != 0:
        vv[len(R) * 3:] = vv0[len(R) * 3:] * (lam * RN)
    M = sp.coo_matrix((vv, (ii, jj)), shape=(ne, 3 * T)).tocsc()
    MtM = (M.T @ M).tocsc() + sp.identity(3 * T, format='csc') * 1e-9
    return spsolve(MtM, M.T @ bb).reshape(T, 3), RN


def run_pid(pid, out_root, start, end, min_conf, detidx, cams, lams, order):
    T = end - start
    rows, inits, confs, n_sub = collect(pid, start, end, min_conf, detidx, cams)
    print(f'[collect] pid{pid} 完成 子集分布={n_sub}', flush=True)
    for lam in lams:
        out = f'{out_root}/lam{lam}/pid{pid}/keypoints3d'
        os.makedirs(out, exist_ok=True)
        kp = np.zeros((T, 25, 4))
        for j in range(OUT_JOINTS):
            if len(rows[j]) == 0:
                continue
            r = solve_joint(rows[j], inits[j], lam, order, T)
            if r is None:
                continue
            X, _ = r
            kp[:, j, :3] = X
            kp[:, j, 3] = confs[j]
        nz = 0
        for t in range(T):
            nz += int((kp[t, :, 3] > 0).sum())
            json.dump([{'id': pid, 'keypoints3d': kp[t].tolist()}],
                      open(f'{out}/{start+t:06d}.json', 'w'))
        print(f'[temporal] pid{pid} lam={lam} -> {out} 有效关节={nz}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=B + '/em_temporal')
    ap.add_argument('--start', type=int, default=4500)
    ap.add_argument('--end', type=int, default=9500)
    ap.add_argument('--min-conf', type=float, default=0.3)
    ap.add_argument('--lams', default='0,0.01,0.1,0.3,1,3')
    ap.add_argument('--order', type=int, default=2)
    a = ap.parse_args()
    lams = [float(x) for x in a.lams.split(',')]
    CAL = os.environ.get('AB_CAL', f'{B}/calib_t11_原始数据求解未修改版')
    cams = build_cams(f'{CAL}/intri.yml', f'{CAL}/extri.yml')
    detidx = {v: index_det(v) for v in VIEWS}
    for pid in (0, 1):
        run_pid(pid, a.out, a.start, a.end, a.min_conf, detidx, cams, lams, a.order)
    print('[temporal] DONE', flush=True)


if __name__ == '__main__':
    main()
