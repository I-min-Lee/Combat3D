#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""viz_mono3d.py —— 双人【单目】MotionBERT 3D 可视化

★ 3D 从哪里来（严格复刻生产脚本 mb_to_final13.py 的定根方式）：
    cam17 = MB原生输出 * s          （相机系 mm，root 相对；s 来自标定）
    rel_w = cam17[:, IDX17_FOR_13] @ Rw2c        （转到世界方向，root 相对）
    pel   = 2D 的骨盆像素 -> 世界射线 d_world
    λ     = 根回归头 head_lambda(...)            ← 深度由【头】给，不是 GT
    W     = C_w + λ·d_world + rel_w              ← 单目 3D（世界 mm）
  再按生产脚本的做法做 despike + savgol 防抖。

输出左：原视频 + 单目 3D 投回原图（对比 GT）
     右：空间坐标系渲染（地面网格 + 两人骨架 + 原相机位置）

用法: viz_mono3d.py <take> <view> <ckpt.mp4 用 MB 权重> <out.mp4>
"""
import os, sys, json, glob
import numpy as np, cv2, torch
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_common as K, kb_train as kt
import mb_to_final13 as M13

TAKE, VIEW, CK, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
CLIP = 121
SG = 17                      # ★ H4D 是 20fps，生产脚本默认 21 是给 25fps 的：21*20/25 ≈ 17
EDGES13 = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
           (10, 11), (11, 12), (7, 12), (7, 9)]
PCOL = {0: (0, 235, 0), 1: (0, 170, 255)}
GCOL = {0: (255, 90, 0), 1: (200, 90, 160)}

ANN = f'{B}/asm_off_{TAKE}/annots'
CALDIR = f'{B}/calib_gt_{TAKE}'
meta = json.load(open(f'{CALDIR}/h4d_meta.json'))
SCALE = json.load(open(f'{B}/h4d_metric_scale.json'))
ms = SCALE[TAKE]
Rw2c, Tv, K1 = M13.load_calib(CALDIR, VIEW)
R_c2w = Rw2c.T
C_w = (-R_c2w @ Tv) * 1000.0 * ms                      # 相机中心(世界 mm，已米制化)
rh = M13.load_roothead(f'{B}/mb/ckpt/rh_h4d_v{VIEW}_w0.pt')
_fs = cv2.FileStorage(f'{CALDIR}/intri.yml', cv2.FILE_STORAGE_READ)
DIST = _fs.getNode(f'dist_{VIEW}').mat().astype(np.float64).reshape(-1)[:4].reshape(4, 1)
_fs.release()
print('take=%s view=%s  米制尺度 ms=%.5f  相机离地 %.0fmm'
      % (TAKE, VIEW, ms, 0))

mb = kt.build_official(); mb = kt.load_ckpt_weights(mb, CK, 'full'); mb.to(kt.DEV).eval()


D = {}
for pid in (0, 1):
    gtd = f'{B}/mb/data_h4d_test15/{TAKE}_v{VIEW}_p{pid}.npz'
    if not os.path.exists(gtd):
        gtd = f'{B}/mb/data_h4d_offtri/{TAKE}_v{VIEW}_p{pid}.npz'
    if not os.path.exists(gtd):
        continue
    gk = np.load(gtd, allow_pickle=True)
    x = gk['k2d']
    T = len(x)
    if T < 20:
        continue
    # ★ 标定必须用真实 k3d（喂全零会拟合出 s=0）
    seq = dict(k2d=x, k3d=gk['k3d'], valid=gk['valid'], view=VIEW,
               group=json.loads(str(gk['meta']))['group'], take=TAKE)
    kt.CAL.clear()
    try:
        kt.calibrate(mb, [seq], CLIP)
        s, Rfit = kt.CAL[kt.cal_key(seq)]
    except Exception as e:
        print('pid%d 标定失败 %s' % (pid, e)); continue
    if not s or s <= 0:
        print('pid%d 标定 s=%s 异常，跳过' % (pid, s)); continue
    # MB 推理（重叠窗）
    xn = kt.norm2d(x)
    cam17 = np.zeros((T, 17, 3)); cnt = np.zeros(T)
    with torch.no_grad():
        for st in range(0, max(1, T - CLIP + 1), CLIP // 2):
            seg = xn[st:st + CLIP]
            if len(seg) < CLIP:
                seg = np.concatenate([seg, np.repeat(seg[-1:], CLIP - len(seg), 0)], 0)
            # ★ cam = (native * s) @ R —— 生产脚本 mb_to_final13.predict() 的口径。
            #   丢 R 只乘 s 会让骨架被缩掉一半（实测 695mm vs GT 1334mm），
            #   观感就是"缩小版的人 + 没踩地面"。踩过这个坑。
            o = (mb(torch.from_numpy(seg[None]).to(kt.DEV)).cpu().numpy()[0] * s) @ Rfit
            cam17[st:st + CLIP] += o[:T - st]; cnt[st:st + CLIP] += 1
    cam17 /= np.maximum(cnt, 1)[:, None, None]
    cam13 = cam17[:, M13.IDX17_FOR_13, :]
    rel_w = cam13 @ Rw2c
    # 骨盆像素 -> 世界射线
    pel = x[:, 0, :2].copy()
    pel = M13.savgol(pel, SG, 3)
    dw = np.zeros((T, 3))
    for t in range(T):
        dc = np.linalg.inv(K1) @ np.array([pel[t, 0], pel[t, 1], 1.0])
        w = R_c2w @ dc
        dw[t] = w / max(np.linalg.norm(w), 1e-9)
    lam = M13.head_lambda(rh, x, Rw2c, Tv, K1, 3840, 2160)
    lam = np.clip(lam, 0.3, 60.0) * 1000.0
    lam = M13.savgol(lam, SG, 3)
    W = C_w[None, None, :] + lam[:, None, None] * dw[:, None, :] + rel_w
    # 防抖（生产脚本同款顺序：despike 再 savgol）
    base = np.stack([M13.savgol(W[:, j, :], SG, 3) for j in range(13)], 1)
    dev = np.linalg.norm(W - base, axis=2)
    thr = np.maximum(120.0, np.median(dev, axis=0) * 4.0)
    bad = dev > thr[None, :]
    W = np.where(bad[:, :, None], base, W)
    W = np.stack([M13.savgol(W[:, j, :], SG, 3) for j in range(13)], 1)
    Xc = (W - C_w) @ Rw2c.T                                   # 世界 -> 相机系 mm
    Xw = gk['k3d'][:, :, :3].astype(np.float64)
    Gcam = ((Rw2c @ Xw.reshape(-1, 3).T).T.reshape(Xw.shape) + Tv) * 1000.0 * ms
    D[pid] = dict(W=W, Xc=Xc, G=Gcam, s=s, T=T, valid=gk['valid'])
    print('pid%d  T=%d  s=%.1f  λ中位 %.2fm  despike %d' %
          (pid, T, s, np.median(lam) / 1000, int(bad.sum())), flush=True)
assert D, '没有可用数据'

# ---- 地面（★ GT 是 H36M17，脚踝 = 3 / 6；final13 的 9/12 不适用） ----
ank = []
for pid, v in D.items():
    for f in range(v['T']):
        q = [v['G'][f, j] for j in (3, 6) if v['valid'][f, j]]
        if q:
            ank.append(np.array(q))
P = np.concatenate(ank); c0 = P.mean(0)
_, _, vt = np.linalg.svd(P - c0, full_matrices=False)
nrm = vt[-1] / np.linalg.norm(vt[-1])
UP = nrm * np.sign(nrm @ C_w + (-float(nrm @ c0)))          # 指向相机一侧
UP /= np.linalg.norm(UP)
print('地面: 法向 %s  相机离地 %.0fmm' % (np.round(UP, 3), abs((C_w - c0) @ nrm)))

# ---- 虚拟相机 ----
CEN = np.mean([v['W'][:, 0, :].mean(0) for v in D.values()], 0)
fwd = np.cross(UP, [1.0, 0.0, 0.0]); fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, UP); right /= np.linalg.norm(right)
EYE = CEN + right * 1800.0 + UP * 1800.0 - fwd * 3600.0
zf = CEN - EYE; zf /= np.linalg.norm(zf)
xf = np.cross(UP, zf); xf /= np.linalg.norm(xf)
yf = np.cross(zf, xf)


def w2p(X, Wd, Hd, F3=900.0):
    X = np.asarray(X, float).reshape(-1, 3) - EYE
    zc = X @ zf
    zc = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
    return np.stack([Wd / 2 + F3 * (X @ xf) / zc, Hd / 2 - F3 * (X @ yf) / zc], 1), zc


e1 = np.cross(UP, [0, 1.0, 0])
e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.cross(UP, [1.0, 0, 0])
e2 = np.cross(UP, e1)
grd = []
for a in np.arange(-3000, 3001, 600):
    grd.append((CEN + e1 * a - e2 * 3000, CEN + e1 * a + e2 * 3000))
    grd.append((CEN + e2 * a - e1 * 3000, CEN + e2 * a + e1 * 3000))
gp = np.array([p for ab in grd for p in ab])

PW, PH = 960, 720
imgs = sorted(glob.glob(f'{meta["seq_root"]}/exo/cam{VIEW}/images/*.jpg'))
Tmax = min(min(v['T'] for v in D.values()), len(imgs))
print('渲染 %d 帧' % Tmax)
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), 20, (PW * 2, PH))
for f in range(Tmax):
    im = cv2.imread(imgs[f])
    left = cv2.resize(im, (PW, PH)) if im is not None else np.zeros((PH, PW, 3), np.uint8)
    for pid, v in D.items():
        for X, col, th in ((v['G'][f], GCOL[pid], 3), (v['Xc'][f], PCOL[pid], 6)):
            uv, _ = cv2.fisheye.projectPoints(
                np.ascontiguousarray(np.asarray(X, float)).reshape(-1, 1, 3),
                np.zeros((3, 1)), np.zeros((3, 1)), K1, DIST)
            uv = uv.reshape(-1, 2) * (PW / 3840.0)
            for a, b in EDGES13:
                cv2.line(left, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), col, th)
    cv2.putText(left, 'left: mono-3D reprojected  green/orange=MB(mono)  blue/purple=GT',
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    rt = np.zeros((PH, PW, 3), np.uint8)
    g2, _ = w2p(gp, PW, PH)
    for i in range(0, len(g2), 2):
        cv2.line(rt, tuple(np.int32(g2[i])), tuple(np.int32(g2[i + 1])), (55, 55, 55), 1)
    for pid, v in D.items():
        for X, col, th in ((v['G'][f], GCOL[pid], 1), (v['W'][f], PCOL[pid], 3)):
            uv, zc = w2p(X, PW, PH)
            for a, b in EDGES13:
                if zc[a] > 1 and zc[b] > 1:
                    cv2.line(rt, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), col, th)
    cuv, _ = w2p(C_w[None], PW, PH)
    if cuv[0][0] == cuv[0][0]:
        cv2.circle(rt, tuple(np.int32(cuv[0])), 7, (0, 255, 255), -1)
        cv2.putText(rt, 'CAM', tuple(np.int32(cuv[0] + [8, -8])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(rt, 'right: 3D space (ground grid)   thick=MB mono   thin=GT   %d/%d'
                % (f + 1, Tmax), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    vw.write(np.hstack([left, rt]))
vw.release()
print('写出', OUT)
