# -*- coding: utf-8 -*-
"""锚点口径的定位质量评估（含时间归一化，可跨帧率比较）

为什么不直接用 analyze_smooth/root_err：
  它们取 xyz[:, 0] 当 "root"，但 13 槽排布是 0头 1R肩...7R髋...，槽0 是【头】，
  13 个节点里根本没有骨盆。所以那两个脚本量的是【头的世界位置】。
  真正决定定位质量的是【锚点】= C_w + λ·d_world，即骨盆世界位置，
  这里用【髋中点】近似骨盆。

★ 跨帧率必须用时间归一化口径：
    帧抖  = |d2x|                (mm/帧^2)   <- 只在同帧率内可比，否则 200fps 必然"看起来"小
    时间抖 = |d2x| / dt^2        (mm/s^2)   <- 跨帧率可比

用法: python anchor_metric.py <真值目录> 名字=目录[@fps] ...
"""
import json, glob, sys
import numpy as np


def load(root, pid):
    fs = sorted(glob.glob(f"{root}/pid{pid}/keypoints3d/*.json"))
    W = []
    for f in fs:
        try:
            W.append(np.array(json.load(open(f))[0]['keypoints3d'], dtype=float)[:, :3])
        except Exception:
            W.append(np.full((13, 3), np.nan))
    return np.array(W) * 1000.0          # m -> mm


def metrics(A, fps):
    ok = np.isfinite(A).all(axis=2).all(axis=1)
    P = A[ok]
    if len(P) < 20:
        return None
    dt = 1.0 / fps
    hipm = 0.5 * (P[:, 7] + P[:, 10])
    head = P[:, 0]
    off = head - hipm
    f2 = lambda X: np.median(np.linalg.norm(np.diff(X, 2, axis=0), axis=-1))
    t2 = lambda X: np.median(np.linalg.norm(np.diff(X, 2, axis=0), axis=-1)) / dt ** 2
    vel = lambda X: np.median(np.linalg.norm(np.diff(X, axis=0), axis=-1)) / dt
    return dict(n=len(P), hipm=hipm, head=head,
                j_anchor=f2(hipm), j_head=f2(head), j_off=f2(off),
                t_anchor=t2(hipm), t_head=t2(head), t_off=t2(off),
                v_anchor=vel(hipm))


truth = sys.argv[1]
Ts = {pid: load(truth, pid) for pid in (0, 1)}

hdr = (f"{'序列':<20}{'fps':>5}{'帧数':>7}{'锚点抖':>8}{'时间抖':>9}{'锚速mm/s':>10}"
       f"{'头抖':>8}{'姿态抖':>8}{'锚误差':>8}{'p90':>7}{'姿态误差':>9}")
print(hdr); print('-' * len(hdr))
for spec in sys.argv[2:]:
    name, rest = spec.split('=', 1)
    if '@' in rest:
        root, fps = rest.rsplit('@', 1); fps = float(fps)
    else:
        root, fps = rest, 25.0
    for pid in (0, 1):
        A = load(root, pid)
        m = metrics(A, fps)
        if m is None:
            continue
        line = (f"{name + ' p' + str(pid):<20}{fps:>5.0f}{m['n']:>7}"
                f"{m['j_anchor']:>8.1f}{m['t_anchor']:>9.0f}{m['v_anchor']:>10.0f}"
                f"{m['j_head']:>8.1f}{m['j_off']:>8.1f}")
        T = Ts[pid]
        n = min(len(A), len(T))
        if abs(len(A) - len(T)) <= 0.1 * max(1, len(T)):      # 长度差>10% 说明帧率不同，不比误差
            mm = np.isfinite(A[:n]).all(axis=(1, 2)) & np.isfinite(T[:n]).all(axis=(1, 2))
            if mm.sum() >= 10:
                h1 = 0.5 * (A[:n, 7] + A[:n, 10]); h2 = 0.5 * (T[:n, 7] + T[:n, 10])
                d = np.linalg.norm(h1[mm] - h2[mm], axis=1)
                line += f"{np.median(d):>8.0f}{np.percentile(d, 90):>7.0f}"
                # 姿态误差：13 关节逐关节距离，再对关节取平均、对帧取中位（= MPJPE 口径）
                pe = np.linalg.norm(A[:n][mm] - T[:n][mm], axis=2).mean(axis=1)
                line += f"{np.median(pe):>9.0f}"
        print(line)
print()
print("锚点=髋中点(≈骨盆世界位置)。'帧抖'只在同帧率间可比；'时间抖'(mm/s^2)跨帧率可比。")
print("姿态误差 = 13 关节平均距离的时间中位（MPJPE 口径）；过大 = 平滑把快动作磨掉了。")
