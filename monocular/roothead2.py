#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""roothead2.py —— 时序根回归头（在 roothead.py 基础上加【时间窗】）

为什么：roothead_v1 是**逐帧 MLP**(12维 -> logλ)，对 λ 完全没有时序上下文。
        单目深度回归的噪声只能靠事后 savgol 磨，而 savgol 是"一刀切"，
        磨 λ 的同时也磨掉了 λ 里真实的快变分量。
        把前后 K 帧的特征一起喂进去，网络自己就能区分
        "这一帧的抖动" 和 "人真的在快速前后移动" —— 从**源头**降 λ 噪声。

输入：(T, FEAT_DIM*(2K+1))  由 windowize() 把 ±K 帧特征拼接而成（边界复制）
输出：与 v1 同构 —— log λ = log λ₀(几何先验) + net(...)   仍是残差学习，退化为几何解
"""
import numpy as np
import torch.nn as nn

FEAT_DIM = 12


def windowize(F, K):
    """F (T,d) -> (T, d*(2K+1))；边界复制。K<=0 时原样返回。"""
    F = np.asarray(F)
    if K <= 0:
        return F
    T = len(F)
    off = np.arange(-K, K + 1)
    idx = np.clip(np.arange(T)[:, None] + off[None, :], 0, T - 1)
    return F[idx].reshape(T, -1)


def build_net(in_dim, h=256):
    net = nn.Sequential(
        nn.Linear(in_dim, h), nn.ReLU(), nn.LayerNorm(h),
        nn.Linear(h, h), nn.ReLU(), nn.LayerNorm(h),
        nn.Linear(h, h // 2), nn.ReLU(),
        nn.Linear(h // 2, 1))
    # 零初始化输出层 -> 起点 = 几何先验（与 v1 一致）
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class Net(nn.Module):
    """训练用容器：键带 f. 前缀（与 v1 的 ckpt 习惯一致）。"""

    def __init__(self, in_dim, h=256):
        super().__init__()
        self.f = build_net(in_dim, h)

    def forward(self, x, lam0_log):
        return lam0_log + self.f(x).squeeze(-1)
