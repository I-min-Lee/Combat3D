#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mb_to_final13.py —— 用微调后的 MotionBERT 替换三角化层

    单视角(默认 view1) 2D ──微调MotionBERT──> 13节点(相机系, root相对)
        └─ 位置锚定(腿长强制850mm + 最低踝踩地等式解深度) ──> 世界坐标
              └─ 写成 final13 同格式： <out>/pid{0,1}/keypoints3d/{f:06d}.json

输出可直接喂 rtd2/render3d_5000f13.py（它要求 conf>0 才画线）。

★ 时间基准：MotionBERT 是在 **25fps** 域训练的（原 200fps 数据按 stride=8 抽帧），
   所以这里也按 stride=8 喂 2D，输出 25fps 序列。

★ 2D 来源：easymocap_{tag}/<mx>_<sx>/annots/{v}/  —— 手册里 conv_rtd_v3.py + idflip_color
   跑完的**身份修正后** annots（personID 就是全局 pid），不是原始 vitpose。

用法:
  python mb_to_final13.py --take 1.1 --view 1 --tag idfix \
      --annots /workshop/Lym/combat3d/easymocap_idfix/1_1/annots \
      --calib /workshop/Lym/combat3d/calib_idfix \
      --ckpt /workshop/Lym/combat3d/mb/ckpt/kb_full_v3.pt \
      --out /workshop/Lym/combat3d/final13_mb1 --stride 8
"""
import os, sys, json, glob, argparse, time
import numpy as np
import torch
import cv2

sys.path.insert(0, '/workshop/Lym/combat3d/mb/kb')
import kb_common as K
import kb_train as kt
from scipy.signal import savgol_filter
sys.path.insert(0, '/workshop/Lym/combat3d')
import roothead as RH

B = '/workshop/Lym/combat3d'


def load_roothead(path):
    """加载第1视角专用的根回归头；返回 (net, mu, std) 或 None"""
    if not path or not os.path.exists(path):
        return None
    import torch.nn as nn
    ck = torch.load(path, map_location='cpu', weights_only=False)   # ckpt 里含 numpy 数组
    d = ck['feat_dim']
    win = int(ck.get('win', 0))          # ★ 0/缺失 = 逐帧头(v1)；>0 = 时序头
    in_dim = d * (2 * win + 1)
    h = 256
    net = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.LayerNorm(h),
                        nn.Linear(h, h), nn.ReLU(), nn.LayerNorm(h),
                        nn.Linear(h, h // 2), nn.ReLU(),
                        nn.Linear(h // 2, 1))
    # 训练时的容器是 Net.f（Sequential），键名带 "f." 前缀，这里剥掉
    net.load_state_dict({k[2:] if k.startswith('f.') else k: v for k, v in ck['sd'].items()})
    net.eval()
    kind = f'时序 ±{win} 帧' if win else '逐帧'
    print(f'[roothead] 载入 {path} (view={ck.get("view")}, {kind}, 输入 {in_dim} 维)')
    return net, ck['mu'], ck['fstd'], win


def head_lambda(rh, x2d, Rw2c, Tv, K1, W, H):
    """用根回归头预测每帧的深度 λ（米）。返回 (T,) 或 None"""
    if rh is None:
        return None
    net, mu, fstd, win = rh
    F = RH.person_feats(x2d, Rw2c, Tv, K1, W, H).astype(np.float64)
    L0 = F[:, 6]                                     # log λ₀（几何先验，网络是残差学习）
    if win > 0:
        # ★ 时序头：把 ±win 帧的特征拼成一维（边界复制，与训练时的 windowize 一致）
        off = np.arange(-win, win + 1)
        ii = np.clip(np.arange(len(F))[:, None] + off[None, :], 0, len(F) - 1)
        Fr = F[ii].reshape(len(F), -1)
    else:
        Fr = F
    Fn = (Fr - mu) / fstd
    with torch.no_grad():
        out = net(torch.from_numpy(Fn).float()).squeeze(-1).numpy()
    return np.exp(L0 + out)


def savgol(x, win, poly=3):
    """零相位 Savitzky-Golay 平滑；窗口自动调成奇数并夹到 <= T。
    对 (T,) 或 (T,C) 都可用。零相位 = 不会引入时间延迟（和管线的 mk_k100lp 同思路）。"""
    x = np.asarray(x, dtype=np.float64)
    w = int(win)
    if w % 2 == 0:
        w += 1
    w = min(w, (len(x) - 1) | 1)
    if w <= poly + 1:
        return x
    return savgol_filter(x, w, poly, axis=0, mode='interp')
ANK_Y = -120.0        # 踝骨离地高度 (mm)；世界 Y 向下为正，地面 Y=0
LEG_LEN = 850.0       # 腿链(髋→膝→踝)目标长度 (mm)，沿用上一轮 run_viz5.py

# H36M17 里对应我们 13 槽位的索引
# 13槽: 0鼻 1R肩 2R肘 3R腕 4L肩 5L肘 6L腕 7R髋 8R膝 9R踝 10L髋 11L膝 12L踝
IDX17_FOR_13 = [10, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6]


def load_calib(d, view):
    ex = f'{d}/extri_refined.yml'
    if not os.path.exists(ex):
        ex = f'{d}/extri.yml'
    ce = cv2.FileStorage(ex, cv2.FILE_STORAGE_READ)
    ci = cv2.FileStorage(f'{d}/intri.yml', cv2.FILE_STORAGE_READ)
    Rw2c = cv2.Rodrigues(ce.getNode(f'R_{view}').mat().astype(np.float64))[0]
    Tv = ce.getNode(f'T_{view}').mat().astype(np.float64).ravel()
    K1 = ci.getNode(f'K_{view}').mat().astype(np.float64)
    ce.release(); ci.release()
    return Rw2c, Tv, K1


def build_2d_vitpose(tag, view, pid, stride):
    """从 vitpose_rtd5000of/<mx>/<sx>/<view>/ + rtd2/idflip_color_<tag>.json 造 2D。

    这是手册里"身份层"那条路：vitpose 的 personID 是**每视角独立**的原始编号，
    要与全局 pid 对齐必须查 idflip：  personID_原始 = pid XOR flip[帧][视角]
    """
    f13, vp = K.take_paths(tag)
    if vp is None:
        raise SystemExit(f'找不到 {tag} 的 vitpose 目录')
    flip = None
    fp = f'{B}/rtd2/idflip_color_{tag}.json'
    if os.path.exists(fp):
        flip = {int(k): v for k, v in json.load(open(fp)).items()}
    else:
        print(f'!! 缺 {fp}，将按 personID==pid 直接配对（可能错配）')
    files = sorted(glob.glob(f'{vp}/{view}/*.json'))
    n = len(files)
    arr = np.zeros((n, 17, 3), dtype=np.float32)
    miss = 0
    for p in files:
        fr = int(os.path.basename(p)[:6])
        if flip is None:
            want = pid
        else:
            f0 = flip.get(fr)
            if f0 is None:
                miss += 1
                continue
            want = pid ^ int(f0.get(view, 0))
        try:
            j = json.load(open(p))
        except Exception:
            continue
        a = None
        for it in j.get('annots', []):
            if int(it.get('personID', -1)) == want:
                a = it
                break
        if a is None:
            continue
        kp = np.asarray(a['keypoints'], dtype=np.float32)
        if kp.shape[0] >= 17:
            arr[fr] = K.map_to_h36m(kp[None], K.H36M_FROM_COCO)[0]
    if miss:
        print(f'  [warn] {miss} 帧缺 idflip 记录，已跳过')
    return arr[::stride]


def build_2d(annots_root, view, idxs, n25, stride):
    """annots/{view}/{f}.json -> 两个 pid 的 (T,17,3) H36M17 像素序列（25fps 域）"""
    T = len(idxs)
    out = {}
    for pid in (0, 1):
        arr = np.zeros((T, 17, 3), dtype=np.float32)
        for k, f in enumerate(idxs):
            fp = f'{annots_root}/{view}/{f:06d}.json'
            if not os.path.exists(fp):
                continue
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            a = None
            for it in d.get('annots', []):
                if int(it.get('personID', -1)) == pid:
                    a = it
                    break
            if a is None:
                continue
            kp = np.asarray(a['keypoints'], dtype=np.float32)
            if kp.shape[0] >= 17:
                arr[k] = K.map_to_h36m(kp[None], K.H36M_FROM_COCO)[0]
        out[pid] = arr
    return out


def predict(mb, x2d, cal_s, cal_R):
    """x2d (T,17,3) -> 相机系 mm (T,13,3)，root 相对。重叠窗平均，减少拼接处抖动。

    模型输出的换算（与训练时 gt_native 严格互逆）：
        native = (G_cam @ Rᵀ) / s   =>   G_cam = (native * s) @ R
    """
    T = len(x2d)
    L, S = 243, 81
    acc = np.zeros((T, 13, 3), dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    starts = list(range(0, max(1, T - L + 1), S))
    if not starts or starts[-1] + L < T:
        starts.append(max(0, T - L))
    xs = []
    for s in starts:
        seg = x2d[s:s + L]
        if len(seg) < L:
            seg = np.concatenate([seg, np.repeat(seg[-1:], L - len(seg), 0)], 0)
        xs.append(kt.norm2d(seg))
    X = torch.from_numpy(np.stack(xs)).to(kt.DEV)
    with torch.no_grad():
        for b0 in range(0, len(xs), 8):
            o = mb(X[b0:b0 + 8]).cpu().numpy()                 # (b,L,17,3) 原生单位
            for bi, s in enumerate(starts[b0:b0 + 8]):
                cam = (o[bi] * cal_s) @ cal_R                  # (L,17,3) 相机系 mm
                cam = cam[:, IDX17_FOR_13, :]                  # (L,13,3)
                ln = min(L, T - s)
                acc[s:s + ln] += cam[:ln]
                cnt[s:s + ln] += 1
    return acc / np.maximum(cnt, 1)[:, None, None], cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--take', default='1.1')
    ap.add_argument('--view', default='1', help='只用这一个视角的 2D（单目）')
    ap.add_argument('--tag', default='idfix')
    ap.add_argument('--annots', default=f'{B}/easymocap_idfix/1_1/annots')
    ap.add_argument('--src', choices=['annots', 'vitpose'], default='annots',
                    help='annots=用 easymocap_*/annots（身份已修正）；'
                         'vitpose=用 vitpose_rtd5000of + rtd2/idflip_color_<tag>.json（手册身份层那条路）')
    ap.add_argument('--fps-src', type=int, default=200,
                    help='源视频帧率：960x720 组=200，1920x1440 组=25')
    ap.add_argument('--calib', default=f'{B}/calib_idfix')
    ap.add_argument('--ckpt', default=f'{B}/mb/ckpt/kb_full_v3.pt')
    ap.add_argument('--root-head', default=None,
                    help='第1视角根回归头 .pt；给了就用它定位，否则退回几何锚定')
    ap.add_argument('--out', default=f'{B}/final13_mb1')
    ap.add_argument('--stride', type=int, default=8, help='200fps -> 25fps')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    ap.add_argument('--leg-len', type=float, default=LEG_LEN,
                    help='腿链目标长度 mm；0 = 不做骨长强制')
    ap.add_argument('--despike', type=float, default=120.0,
                    help='野值判定阈值(mm)：偏离平滑轨迹超过它就替换成平滑值；0=关。'
                         '★不要用中值滤波替代，中值会造平台 -> 观感"卡壳"')
    ap.add_argument('--smooth', type=int, default=1, help='1=开时间平滑（λ/k/世界坐标 零相位 savgol）')
    ap.add_argument('--sg', type=int, default=21,
                    help='savgol 窗口（25fps 域，奇数）；0/1=关。'
                         '★2026-10-03 起默认 21(≈0.84s)，原先 11(≈0.44s)：'
                         '实测锚点抖动 8.9/11.9 -> 4.1/3.9mm（反超三角化 6.6/8.2）')
    ap.add_argument('--pix-sg', type=int, default=21,
                    help='★骨盆像素(u,v)的 savgol 窗口；0=关。'
                         '★2026-10-03 起默认 21（原先 0=关）：骨盆像素逐帧决定射线方向，'
                         '不平滑会把它被 λ(~6.7m) 放大成整帧空间抖')
    a = ap.parse_args()

    # ---- 标定 ----
    # ★2026-10-04 修 bug：extri.yml 是 **COLMAP 单位**，必须乘该场次的米制尺度 ms
    #   才是米。原写法只 *1000 就当成 mm，导致相机中心被放到 1/ms 倍远的地方
    #   （train01_hugging ms=0.543 -> 偏 1.84 倍），人也跟着被推远、投影变小。
    #   gt_camera_mm() 那边是乘了 ms 的，两边口径必须一致。
    _ms = 1.0
    try:
        _ms = float(json.load(open(f'{B}/h4d_metric_scale.json')).get(a.take, 1.0))
    except Exception:
        pass
    Rw2c, Tv, K1 = load_calib(a.calib, a.view)
    R_c2w = Rw2c.T
    C_w = (-R_c2w @ Tv) * 1000.0 * _ms           # 相机中心(世界 mm)
    print(f'[calib] 场次 {a.take} 米制尺度 ms={_ms:.4f}')


    # ---- 2D 源与图像尺寸 ----
    if a.src == 'vitpose':
        _f13, _vp = K.take_paths(a.tag)
        if _vp is None:
            raise SystemExit(f'--src vitpose 但找不到 {a.tag} 的 vitpose 目录')
        _g = sorted(glob.glob(f'{_vp}/{a.view}/*.json'))
        _d = json.load(open(_g[0]))
        IMG_W, IMG_H = int(_d.get('width', 960)), int(_d.get('height', 720))
        print(f'[src] vitpose {_vp}  共 {len(_g)} 帧')
    else:
        _g = sorted(glob.glob(f'{a.annots}/{a.view}/*.json'))
        _d = json.load(open(_g[0]))
        IMG_W, IMG_H = int(_d.get('width', 960)), int(_d.get('height', 720))
    print(f'[img] {IMG_W}x{IMG_H}')

    rh = load_roothead(a.root_head)

    # ---- 模型 + 训练时的标定 (s,R) ----
    ej = a.ckpt.replace('.pt', '_eval.json')
    cal = json.load(open(ej))['calib']
    if a.view in cal:                                   # kendo：键=视角
        s_v, R_v = cal[a.view]
    else:                                               # ★H4D_COMPAT：键=(take, view)
        _k = next((k for k in cal if a.take in k and a.view in k), None)
        if _k is None:
            raise KeyError(f'calib 里找不到 view={a.view} take={a.take}；现有键示例 '
                           f'{list(cal)[:3]}')
        s_v, R_v = cal[_k]
        print(f'[H4D_COMPAT] 用 (take,view) 键 {_k!r}')
    R_v = np.array(R_v)
    mb = kt.build_official()
    mb = kt.load_ckpt_weights(mb, a.ckpt, 'full')
    mb.to(kt.DEV).eval()
    print(f'[mb] {a.ckpt}  view{a.view}  s={s_v:.1f} mm/unit  calib={ej}', flush=True)

    # ---- 帧清单（25fps 域）----
    src_dir = _vp if a.src == 'vitpose' else a.annots
    n200 = len(glob.glob(f'{src_dir}/{a.view}/*.json'))
    end = n200 if a.end < 0 else min(a.end, n200)
    out_fps = max(1, a.fps_src // a.stride)
    if a.src == 'vitpose':
        # 抽帧后的下标 k 对应原帧 k*stride，所以按原帧区间换算
        k0 = a.start // a.stride
        k1 = -(-end // a.stride)
        d2 = {}
        for pid in (0, 1):
            d2[pid] = build_2d_vitpose(a.tag, a.view, pid, a.stride)[k0:k1]
        T = len(d2[0])
    else:
        idxs = list(range(a.start, end, a.stride))
        T = len(idxs)
        d2 = build_2d(a.annots, a.view, idxs, T, a.stride)
    print(f'[data] {a.take} view{a.view}  源帧 {a.start}..{end}  stride {a.stride} '
          f'-> {T} 帧  (输出 {out_fps}fps，逐帧对齐原片)', flush=True)

    for pid in (0, 1):
        x = d2[pid]
        n_ok = int((x[:, :, 2] > 0.05).sum())
        print(f'  pid{pid}: 有 2D 信号的关节观测 {n_ok}', flush=True)
        cam13, cnt = predict(mb, x, s_v, R_v)
        # 相机系 mm -> 世界相对偏移 mm
        rel_w = cam13 @ Rw2c                       # (T,13,3) 世界方向，root 相对
        # 骨长强制
        if a.leg_len > 0:
            hip = [7, 10]; knee = [8, 11]; ank = [9, 12]
            L = np.mean([np.linalg.norm(rel_w[:, hip[i]] - rel_w[:, knee[i]], axis=1) +
                         np.linalg.norm(rel_w[:, knee[i]] - rel_w[:, ank[i]], axis=1) for i in (0, 1)], axis=0)
            k = np.clip(a.leg_len / np.maximum(L, 1e-6), 0.6, 1.6)
            if a.smooth and a.sg > 2:                 # 逐帧的 k 也在变 -> 一起平滑，否则人会"呼吸"
                k = savgol(k, a.sg, 3)
            rel_w = rel_w * k[:, None, None]
        # 骨盆像素 -> 世界射线
        pel = x[:, 0, :2]                          # (T,2) H36M17 的 0 = 骨盆
        # ★★ 骨盆像素也要零相位平滑：d_world 逐帧由骨盆像素算出，
        #    亚像素抖动 -> 射线角度抖动 -> 被 λ(~6-7m) 放大成整帧空间抖动。
        #    实测：只平滑 λ 锚点抖动 20mm，加像素平滑后 6mm，且定位误差不升。
        if a.smooth and a.pix_sg > 2:
            pel = savgol(pel, a.pix_sg, 3)
        d_world = np.zeros((T, 3))
        for t in range(T):
            dc = np.linalg.inv(K1) @ np.array([pel[t, 0], pel[t, 1], 1.0])
            dw = R_c2w @ dc
            n = np.linalg.norm(dw)
            d_world[t] = dw / (n if n > 1e-9 else 1.0)
        # ---- 深度 λ：优先用【根回归头】，没有就退回几何锚定 ----
        lam_head = head_lambda(rh, x, Rw2c, Tv, K1, IMG_W, IMG_H)
        if lam_head is not None:
            lam = np.clip(lam_head, 0.3, 60.0) * 1000.0              # 米 -> mm
            lam_src = "roothead"
        else:
            min_ank = np.minimum(rel_w[:, 9, 1], rel_w[:, 12, 1])    # 最低踝的相对 Y
            lam = (ANK_Y - C_w[1] - min_ank) / np.where(np.abs(d_world[:, 1]) < 1e-6, 1e-6, d_world[:, 1])
            lam = np.clip(lam, 500.0, 60000.0)                       # 5cm ~ 60m 合理区间
            lam_src = "geometric"
        # 逐帧独立解深度是"不丝滑"的主因：骨盆像素的亚像素抖动会整帧变成前后跳。
        # λ 做零相位平滑（三角化侧管线也有 mk_k100lp.py 做同类后处理）。
        if a.smooth and a.sg > 2:
            lam = savgol(lam, a.sg, 3)
        W = C_w[None, None, :] + lam[:, None, None] * d_world[:, None, :] + rel_w   # 世界 mm
        # ★★ 时序处理（顺序很重要，这里是"卡壳"的修复）
        #   旧版用 5 帧中值滤波 -> 中值在连续轨迹上会造出**平台+跳变**（值卡住几帧再跳），
        #   观感就是"卡壳/一顿一顿"。改成：
        #     ① despike：只把偏离平滑轨迹过大的野值用平滑值替换，其余帧**原样保留**
        #     ② savgol：零相位磨平（不引入时间延迟）
        if a.smooth:
            if a.despike > 0:
                base = np.stack([savgol(W[:, j, :], a.sg or 11, 3) for j in range(13)], 1)
                dev = np.linalg.norm(W - base, axis=2)              # (T,13)
                thr = np.maximum(a.despike, np.median(dev, axis=0) * 4.0)   # 绝对阈值 + 自适应
                bad = dev > thr[None, :]
                if bad.any():
                    W = np.where(bad[:, :, None], base, W)
                    print(f'    despike: 替换 {int(bad.sum())} 个关节帧', flush=True)
            if a.sg > 2:
                W = np.stack([savgol(W[:, j, :], a.sg, 3) for j in range(13)], 1)
        Wm = W / 1000.0                                              # -> 米
        # conf: 用该槽位源关节的 2D conf（至少 0.05，保证渲染会画）
        src = {0: 0, 1: 6, 2: 8, 3: 10, 4: 5, 5: 7, 6: 9, 7: 12, 8: 14, 9: 16, 10: 11, 11: 13, 12: 15}
        c13 = np.stack([x[:, src[s], 2] for s in range(13)], 1)      # (T,13)
        c13 = np.clip(c13, 0.05, 1.0)
        od = f'{a.out}/pid{pid}/keypoints3d'
        os.makedirs(od, exist_ok=True)
        for k in range(T):
            rec = [{"id": pid, "keypoints3d":
                    [[float(Wm[k, j, 0]), float(Wm[k, j, 1]), float(Wm[k, j, 2]), float(c13[k, j])]
                     for j in range(13)]}]
            json.dump(rec, open(f'{od}/{k:06d}.json', 'w'))
        print(f'  pid{pid}: 写出 {T} 帧 -> {od}  深度源={lam_src}  λ中位 {np.median(lam)/1000:.2f} m', flush=True)

    print(f'完成 -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
