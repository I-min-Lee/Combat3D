#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""roothead.py —— 单目根回归头（共享模块：特征/模型/几何先验）

任务：只回归 **骨盆沿相机射线的深度 λ**（一个标量）。
      X/Y 由"骨盆像素射线"决定，姿态朝向 MotionBERT 已给 —— 所以补上 λ 就有完整的世界坐标。

监督：**三角化的世界根**投影到该射线上  —— 5 视角几何当老师。

为什么这么设计：
  · 单目深度多解，直接回归 3D 根会把"像素→世界"的映射焊死在训练机位上；
    改成回归 **沿射线的深度**，相机外参始终以几何方式参与，换机位时只需换外参。
  · 几何先验 λ₀（最低踝像素射线 ∩ 地面）作为**输入特征**，头学的是它的**修正量**，
    起点就站在几何解上，不会比几何解更差。

相机地面几何（世界 Y 向下为正，地面 Y=0）：
    相机系下世界 Y 轴 n = R[1,:]；地面:  n·X_c = n·T
    射线 X_c = λ d  =>  λ₀ = (n·T) / (n·d)
"""
import numpy as np


FEAT_DIM = 12


def cam_ground_prior(Rw2c, Tv, K, uv):
    """最低可见关节像素 uv -> 地面交点沿射线的距离 λ₀（米，欧氏）。返回 None 表示无解。

    地面几何：世界点 X_w 满足 X_w[1]=0（Y 向下为正，地面 Y=0）。
        X_w = Rw2cᵀ (X_c - Tv)  =>  X_w[1] = Rw2c[:,1]·(X_c - Tv) = 0
        => n·X_c = n·Tv ,  其中 **n = Rw2c[:,1]**（R 的**第 2 列**，不是第 2 行！）
        射线 X_c = λ d（d 为单位向量）=> λ₀ = (n·Tv)/(n·d)
    """
    n = Rw2c[:, 1]                                  # ★ 列，不是行
    dc = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    d = Rw2c.T @ dc
    nn = np.linalg.norm(d)
    if nn < 1e-12:
        return None
    d = d / nn
    den = n @ d
    if abs(den) < 1e-9:
        return None
    return float((n @ Tv) / den)


def person_feats(k2d, Rw2c, Tv, K, W, H):
    """k2d (T,17,3) 像素+conf（H36M17）-> 特征 (T, FEAT_DIM)

    特征全部做**分辨率归一化**（除以 W/H），这样 960x720 与 1920x1440 两组可混训。
    """
    T = len(k2d)
    X = np.zeros((T, FEAT_DIM), np.float32)
    v = k2d[:, :, 2] > 0.05
    pel = k2d[:, 0, :2]                              # 骨盆像素（H36M17 的 0 = mid-hip）
    Rc2w = Rw2c.T
    C_w = -Rc2w @ Tv                                 # 相机中心（世界，米）
    cam_h = -C_w[1]                                  # 高于地面的高度（Y 向下为正）
    axis_w = Rc2w @ np.array([0.0, 0.0, 1.0])        # 光轴在世界系
    pitch = float(np.arcsin(np.clip(axis_w[1], -1, 1)))   # 俯角（向下看为负）
    for t in range(T):
        vis = v[t]
        pts = k2d[t, vis, :2]
        if len(pts) >= 4:
            span = float(max((pts.max(0) - pts.min(0)).max(), 1.0))
            low = pts[np.argmax(pts[:, 1])]           # 最低可见关节（地面线索）
        else:
            span, low = 0.0, pel[t]
        lam0 = cam_ground_prior(Rw2c, Tv, K, low)
        lam0 = lam0 if (lam0 is not None and lam0 > 0.05) else 1.0
        mc = float(k2d[t, vis, 2].mean()) if vis.any() else 0.0
        X[t] = [pel[t, 0] / W, pel[t, 1] / H,
                np.log(max(span, 1.0) / W),
                mc, float(vis.mean()),
                low[1] / H,
                np.log(max(lam0, 0.05)),             # ★ 几何先验
                cam_h, pitch,
                float(K[0, 0] / W),                  # 归一化焦距
                float(K[0, 2] / W), float(K[1, 2] / H)]
    return X


def true_lambda(k2d, k3d, Rw2c, Tv, K):
    """三角化世界根 -> 沿「骨盆像素射线」的深度 λ_true（米）。

    射线由 **2D 的骨盆像素** 决定（那是模型实际看到的东西），
    再取世界根在该射线上的投影长度作为真值。
    """
    T = len(k3d)
    out = np.full(T, np.nan, np.float64)
    Kinv = np.linalg.inv(K)
    C = -Rw2c.T @ Tv                                    # 相机中心（米）
    for t in range(T):
        if k3d[t, 0, 3] <= 0 or k2d[t, 0, 2] <= 0.05:
            continue
        uv = k2d[t, 0, :2]
        dw = Rw2c.T @ (Kinv @ np.array([uv[0], uv[1], 1.0]))
        n = np.linalg.norm(dw)
        if n < 1e-9:
            continue
        out[t] = float((k3d[t, 0, :3] - C) @ (dw / n))
    return out
