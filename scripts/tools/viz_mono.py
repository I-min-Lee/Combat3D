#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""viz_mono.py —— 单目效果可视化：把模型的 3D 预测投回原图。

每帧画三样：
  ● 红：输入 2D（模型看到的全部信息）
  ● 绿：模型预测的 3D 反投影（根挂在 GT 根位置，所以画的是【姿态本身】的质量）
  ● 蓝：GT 3D 反投影
并在角上打 PA 形状误差 / 端到端误差。

用法: viz_mono.py <take> <view> <pid> <ckpt> <out.mp4>
"""
import os, sys, json, glob
import numpy as np, cv2, torch
B = '/workshop/Lym/combat3d'
sys.path.insert(0, f'{B}/mb/kb'); sys.path.insert(0, B)
import kb_common as K, kb_train as kt

TAKE, VIEW, PID, CK, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
PID = int(PID)
CLIP = 121
SCALE = json.load(open(f'{B}/h4d_metric_scale.json'))
DATA = f'{B}/mb/data_h4d_offtri'
nf = glob.glob(f'{DATA}/{TAKE}_v{VIEW}_p{PID}.npz')
if not nf:
    DATA = f'{B}/mb/data_h4d_test15'
    nf = glob.glob(f'{DATA}/{TAKE}_v{VIEW}_p{PID}.npz')
assert nf, '找不到 npz'
d = np.load(nf[0], allow_pickle=True)
k2d, k3d, valid = d['k2d'], d['k3d'], d['valid']
meta = json.loads(str(d['meta']))
T = len(k2d)
print('take=%s view=%s pid=%d  T=%d  group=%s' % (TAKE, VIEW, PID, T, meta['group']))

cal = K.Calib(K.CALIB[meta['group']])
m = kt.build_official(); m = kt.load_ckpt_weights(m, CK, 'full'); m.to(kt.DEV).eval()
seq = dict(k2d=k2d, k3d=k3d, valid=valid, view=VIEW, group=meta['group'], take=TAKE)
kt.CAL.clear(); kt.calibrate(m, [seq], CLIP)
s = kt.CAL[kt.cal_key(seq)][0]
print('标定 s = %.1f mm/unit' % s)

# ---- 模型推理（整条序列，重叠窗口） ----
x = kt.norm2d(k2d)
pred = np.zeros((T, 17, 3), np.float32)
cnt = np.zeros(T, np.float32)
with torch.no_grad():
    for st in range(0, max(1, T - CLIP + 1), CLIP // 2):
        seg = x[st:st + CLIP]
        if len(seg) < CLIP:
            seg = np.concatenate([seg, np.repeat(seg[-1:], CLIP - len(seg), 0)], 0)
        o = m(torch.from_numpy(seg[None]).to(kt.DEV)).cpu().numpy()[0] * s
        pred[st:st + CLIP] += o[:T - st]
        cnt[st:st + CLIP] += 1
pred /= np.maximum(cnt, 1)[:, None, None]

# ---- GT（相机系 mm，绝对位置；不用 kb_train 的根相对版） ----
ms = SCALE.get(TAKE, 1.0)
Xw = k3d[:, :, :3].astype(np.float64)
Xc = (cal.R[VIEW] @ Xw.reshape(-1, 3).T).T.reshape(Xw.shape) + cal.T[VIEW]
Xc = Xc * 1000.0 * ms
gt = Xc - Xc[:, 0:1, :]                        # 根相对 GT
pr = pred - pred[:, 0:1, :]                    # 根相对 预测
gt_abs = Xc                                     # 绝对 GT
pr_abs = gt_abs[:, 0:1, :] + pr                 # 预测挂到 GT 根上


def proj(X, v):
    """★ 输入已经是【相机系 mm】——只做鱼眼投影，不要再乘一次外参（踩过这个坑）"""
    Kk = cal.K[v]; D = np.asarray(cal.D[v], float).reshape(-1)[:4].reshape(4, 1)
    uv, _ = cv2.fisheye.projectPoints(
        np.ascontiguousarray(np.asarray(X, float)).reshape(-1, 1, 3),
        np.zeros((3, 1)), np.zeros((3, 1)), Kk, D)
    return uv.reshape(-1, 2)


# ---- 误差数字 ----
ok = valid & (k2d[:, :, 2] > 0.05)
e2e = np.linalg.norm(pr - gt, axis=2)[ok]
pa = []
for f in range(T):
    m_ = ok[f]
    if m_.sum() < 8:
        continue
    A = pr[f][m_] - pr[f][m_].mean(0); Gg = gt[f][m_] - gt[f][m_].mean(0)
    H = A.T @ Gg; U, Dm, Vt = np.linalg.svd(H)
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ S @ U.T; sc = float(np.trace(np.diag(Dm) @ S) / max((A ** 2).sum(), 1e-9))
    pa.append(np.linalg.norm(A @ R.T * sc - Gg, axis=1))
pa_med = np.median(np.concatenate(pa)) if pa else float('nan')
e2e_med = np.median(e2e)
print('PA 形状 %.1f mm   端到端 %.1f mm' % (pa_med, e2e_med))

# ---- 帧来源 ----
seq_root = json.load(open(f'{B}/calib_gt_{TAKE}/h4d_meta.json'))['seq_root']
imgs = sorted(glob.glob(f'{seq_root}/exo/cam{VIEW}/images/*.jpg'))
print('原图 %d 张' % len(imgs))
BASE = 4                                     # 音序从 00001.jpg 开始
SK = K.SKEL_H36M if hasattr(K, 'SKEL_H36M') else None
EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]

vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))
if not vw.isOpened():
    vw = cv2.VideoWriter(OUT.replace('.mp4', '_x264.mp4'),
                         cv2.VideoWriter_fourcc(*'avc1'), 20, (1280, 720))
for f in range(T):
    ip = f'{seq_root}/exo/cam{VIEW}/images/{f + 1:05d}.jpg'
    if not os.path.exists(ip):
        continue
    im = cv2.imread(ip)
    if im is None:
        continue
    g = proj(gt_abs[f], VIEW)               # 蓝：GT
    p = proj(pr_abs[f], VIEW)               # 绿：预测
    for a, b in EDGES:
        cv2.line(im, tuple(np.int32(g[a])), tuple(np.int32(g[b])), (255, 80, 0), 6)
        cv2.line(im, tuple(np.int32(p[a])), tuple(np.int32(p[b])), (0, 230, 0), 6)
    for j in range(17):
        if k2d[f, j, 2] > 0.05:
            cv2.circle(im, tuple(np.int32(k2d[f, j, :2])), 9, (0, 0, 255), -1)   # 红：输入 2D
    cv2.putText(im, 'PA shape %.0fmm   end2end %.0fmm   frame %d/%d'
                % (pa_med, e2e_med, f + 1, T), (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(im, 'red=input 2D   green=MB predict   blue=GT', (30, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    vw.write(cv2.resize(im, (1280, 720)))
vw.release()
print('写出', OUT)
