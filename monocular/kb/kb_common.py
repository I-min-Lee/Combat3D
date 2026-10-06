#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kb_common —— 剑道 final13(13节点) + vitpose(COCO17) → MotionBERT(H36M17) 公共层

坐标/标定约定（沿用上一轮 mb_ft4.py，已验证）:
    X_cam = Rmw[v] @ X_world_m + T[v]        (Rmw = Rodrigues(R_v), T 单位米)
    C_cam = (-Rmw.T @ T) * 1000              (相机中心, 世界 mm)
    ray  = Rmw.T @ (K^-1 @ [u,v,1])  归一化   (世界方向)

3D 关节序: final13 = body25 的 [0,2,3,4,5,6,7,9,10,11,12,13,14]
   0鼻 1R肩 2R肘 3R腕 4L肩 5L肘 6L腕 7R髋 8R膝 9R踝 10L髋 11L膝 12L踝
"""
import os, json, glob, re
import numpy as np
import cv2

# ------------------------------------------------------------------ 路径
ROOT = '/workshop/Lym/combat3d'
FINAL13 = ROOT + '/final13_{tag}'          # FINAL13.format(tag=...)
VITPOSE = f'{ROOT}/vitpose_rtd5000of'
OUT     = f'{ROOT}/mb/data'
VIZDIR  = f'{ROOT}/mb/viz'

VIEWS = ['1', '3', '4', '7', '11']
MB_ROOT = f'{ROOT}/mb/MotionBERT'
CK_FT   = f'{ROOT}/mb/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin'

# 两组标定（手册 §一.2；B 组按记忆回退 calib_B_1920x1440，不用 calib_B_fromA）
CALIB = {
    '960':  f'{ROOT}/calib_idfix',          # 960x720 组（A+C 合并）
    '1920': f'{ROOT}/calib_B_1920x1440',    # 1920x1440 组（B）
}
# ★2026-10-03 增量：Harmony4D 各序列的标定。group 名 = 'h4d_<tag>'，目录 = {ROOT}/calib_gt_<tag>。
#   纯追加，kendo 的 960/1920 两项不受影响；glob 不匹配时字典保持原样。
for _d in sorted(glob.glob(f'{ROOT}/calib_gt_*')):
    CALIB['h4d_' + os.path.basename(_d)[len('calib_gt_'):]] = _d

# ------------------------------------------------------------------ 关节映射
# COCO17: 0鼻 1L眼 2R眼 3L耳 4R耳 5L肩 6R肩 7L肘 8R肘 9L腕 10R腕
#         11L髋 12R髋 13L膝 14R膝 15L踝 16R踝
# H36M17: 0Pel 1R髋 2R膝 3R踝 4L髋 5L膝 6L踝 7脊 8胸 9颈 10头 11L肩 12L肘 13L腕 14R肩 15R肘 16R腕
# 每项 = (源关节索引列表, 权重列表)；合成关节 conf 取 min
H36M_FROM_COCO = {
    0:  ([11, 12], [.5, .5]),
    1:  ([12], [1.]), 2: ([14], [1.]), 3: ([16], [1.]),
    4:  ([11], [1.]), 5: ([13], [1.]), 6: ([15], [1.]),
    7:  ([11, 12, 5, 6], [.25, .25, .25, .25]),   # 脊 = mid(髋, 胸)
    8:  ([5, 6], [.5, .5]),                        # 胸 = mid(双肩)
    9:  ([0], [1.]),                               # 颈 ≈ 鼻（COCO 无颈）
    10: ([0], [1.]),                               # 头 = 鼻
    11: ([5], [1.]), 12: ([7], [1.]), 13: ([9], [1.]),
    14: ([6], [1.]), 15: ([8], [1.]), 16: ([10], [1.]),
}

# final13 槽: 0鼻 1R肩 2R肘 3R腕 4L肩 5L肘 6L腕 7R髋 8R膝 9R踝 10L髋 11L膝 12L踝
H36M_FROM_13 = {
    0:  ([7, 10], [.5, .5]),                       # 骨盆 = mid(双髋)
    1:  ([7], [1.]), 2: ([8], [1.]), 3: ([9], [1.]),
    4:  ([10], [1.]), 5: ([11], [1.]), 6: ([12], [1.]),
    7:  ([7, 10, 1, 4], [.25, .25, .25, .25]),     # 脊 = mid(骨盆, 胸)
    8:  ([1, 4], [.5, .5]),                        # 胸 = mid(双肩)
    9:  ([0, 1, 4], [.5, .25, .25]),               # 颈 = mid(鼻, 胸)
    10: ([0], [1.]),                               # 头 = 鼻
    11: ([4], [1.]), 12: ([5], [1.]), 13: ([6], [1.]),
    14: ([1], [1.]), 15: ([2], [1.]), 16: ([3], [1.]),
}

# MotionBERT 的左右关节（flip 用）
LEFT_J  = [4, 5, 6, 11, 12, 13]
RIGHT_J = [1, 2, 3, 14, 15, 16]

# 各关节的"源证据"槽位（13 节点域）——用来判定该关节的 3D conf 是否有测量证据
VALID_SRC_13 = {j: v[0] for j, v in H36M_FROM_13.items()}
VALID_SRC_COCO = {j: v[0] for j, v in H36M_FROM_COCO.items()}

# 骨架连线（可视化）
SKEL_H36M = [(0, 7), (7, 8), (8, 14), (14, 15), (15, 16), (8, 11), (11, 12), (12, 13),
             (8, 9), (9, 10), (0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (0, 4)]


def map_to_h36m(P, table):
    """P: (T, S, C) 源关节（S 为源关节数，C>=3: x,y[,z],conf 在最后一维）
       返回 (T, 17, C)。合成关节 = 加权和；conf 取源关节 conf 的 min。"""
    T, S, C = P.shape
    out = np.zeros((T, 17, C), dtype=P.dtype)
    for j, (srcs, ws) in table.items():
        pos = np.zeros((T, C - 1), dtype=P.dtype)
        confs = []
        for s, w in zip(srcs, ws):
            pos += w * P[:, s, :C - 1]
            confs.append(P[:, s, C - 1])
        out[:, j, :C - 1] = pos
        out[:, j, C - 1] = np.min(np.stack(confs), axis=0)
    return out


# ------------------------------------------------------------------ 标定
class Calib:
    """一套（或两组）标定的读写。"""

    def __init__(self, calib_dir):
        self.dir = calib_dir
        ex = cv2.FileStorage(f'{calib_dir}/extri.yml', cv2.FILE_STORAGE_READ)
        self.have_refined = os.path.exists(f'{calib_dir}/extri_refined.yml')
        im = cv2.FileStorage(f'{calib_dir}/intri.yml', cv2.FILE_STORAGE_READ)
        assert ex.isOpened(), f'打不开 {calib_dir}/extri.yml'
        assert im.isOpened(), f'打不开 {calib_dir}/intri.yml'
        self.K, self.D, self.T, self.R = {}, {}, {}, {}
        # ★2026-10-03 增量：视角名优先从 yml 的 'names' 序列读（Harmony4D 是 '01','03',...），
        #   读不到才回退 VIEWS。kendo 的 yml 没有 'names' -> 行为与原来完全一致。
        _vs = []
        try:
            _nd = ex.getNode('names')
            if not _nd.empty():
                _vs = [_nd.at(_i).string() for _i in range(_nd.size())]
        except Exception:
            _vs = []
        for v in (_vs or VIEWS):
            _node = ex.getNode(f'R_{v}')
            if _node.empty():
                continue
            self.R[v] = cv2.Rodrigues(_node.mat().astype(np.float64))[0]
            self.T[v] = ex.getNode(f'T_{v}').mat().astype(np.float64).ravel()
            self.K[v] = im.getNode(f'K_{v}').mat().astype(np.float64)
            dn = im.getNode(f'dist_{v}')
            self.D[v] = dn.mat().astype(np.float64) if not dn.empty() else np.zeros((5, 1))
        ex.release(); im.release()

    def cam_center_mm(self, v):
        return (-self.R[v].T @ self.T[v]) * 1000.0

    def ray_world(self, v, uv):
        """像素 -> 世界方向（未归一化则先归一化）"""
        d = np.linalg.inv(self.K[v]) @ np.array([uv[0], uv[1], 1.0])
        d = self.R[v].T @ d
        return d / np.linalg.norm(d)

    def world_to_cam(self, Xw_m, v):
        """Xw_m: (T,17,3) 米 -> 相机系米"""
        return (self.R[v] @ Xw_m.reshape(-1, 3).T).T.reshape(Xw_m.shape) + self.T[v]

    def project(self, Xw_m, v):
        """世界米 -> 像素 (T,17,2)"""
        Xc = self.world_to_cam(Xw_m, v)
        uv, _ = cv2.projectPoints(Xc.reshape(-1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
                                  self.K[v], self.D[v])
        return uv.reshape(-1, 2)


def upright_matrix(cal, v):
    """相机 -> 『正立相机系』的 3x3（消 roll）。世界 up = (0,-1,0)（Y 轴向下为正）。"""
    u = cal.R[v] @ np.array([0.0, -1.0, 0.0])      # 世界 up 在相机系
    u /= np.linalg.norm(u)
    zc = cal.R[v] @ np.array([0.0, 0.0, 1.0])      # 光轴在相机系 = [0,0,1]
    zc /= np.linalg.norm(zc)
    xp = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), u) * u
    n = np.linalg.norm(xp)
    xp = xp / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    yp = np.cross(u, xp)
    return np.stack([xp, yp, u])                   # 输出系 = (x', y', up)


# ------------------------------------------------------------------ 场次清单
def load_takes(segqa_summary=f'{ROOT}/_segqa_summary.txt'):
    """解析 _segqa_summary.txt -> [(tag, n_frames, fps, group)]，group ∈ {'960','1920'}"""
    takes = []
    pat = re.compile(r'^(f\d+)\s+.*?帧=(\d+)\s+fps=(\d+)')
    with open(segqa_summary, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            m = pat.match(line.strip())
            if not m:
                continue
            tag, n, fps = m.group(1), int(m.group(2)), int(m.group(3))
            group = '1920' if fps == 25 else '960'
            takes.append((tag, n, fps, group))
    return takes


def load_quality(segqa_summary=f'{ROOT}/_segqa_summary.txt'):
    """解析 _segqa_summary.txt 的质量栏目 -> {tag: dict(cv_max, problems, frames, fps)}
    骨长CV 健康区 0.065~0.146；问题箱 = 帧间跳变超阈的箱数。
    """
    import re as _re
    q = {}
    pat = _re.compile(r'^(f\d+)\s+.*?帧=(\d+)\s+fps=(\d+)\s+骨长CV=([\d.]+)/([\d.]+)\s+问题箱=(\d+)')
    with open(segqa_summary, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            m = pat.match(line.strip())
            if not m:
                continue
            tag = m.group(1)
            cv0, cv1 = float(m.group(4)), float(m.group(5))
            q[tag] = dict(frames=int(m.group(2)), fps=int(m.group(3)),
                          cv0=cv0, cv1=cv1, cv_max=max(cv0, cv1),
                          problems=int(m.group(6)))
    return q


def take_weights(min_cv_ok=0.18, max_problems=3, segqa_summary=f'{ROOT}/_segqa_summary.txt'):
    """按质量报告给每个场次一个训练权重。
    返回 (weights: {tag: float}, excluded: set(tag))
      · 骨长CV 超 0.18 或 问题箱 > 3  -> 排除（标签本身不可信）
      · 其余按 1/(0.06+cv_max) 加权，越干净权重越高，均值归一到 1
    """
    import numpy as np
    q = load_quality(segqa_summary)
    w, bad = {}, set()
    for tag, d in q.items():
        if d['cv_max'] > min_cv_ok or d['problems'] > max_problems:
            bad.add(tag)
            continue
        w[tag] = 1.0 / (0.06 + d['cv_max'])
    if w:
        vals = np.array(list(w.values()))
        m = float(vals.mean())
        for k in w:
            w[k] /= m
    return w, bad


_TAKE_CACHE = None


def take_group(tag):
    """该场次属于哪组标定: '960'(calib_idfix) / '1920'(calib_B_1920x1440)。
    判据 = _segqa_summary.txt 里的 fps：25 -> 1920 组，200 -> 960 组。"""
    global _TAKE_CACHE
    if _TAKE_CACHE is None:
        _TAKE_CACHE = {t[0]: t[3] for t in load_takes()}
    return _TAKE_CACHE.get(tag, '960')


def take_paths(tag):
    """tag f<mx><sx> -> (final13 根, vitpose/<mx>/<sx> 根)；用目录存在消歧 mx/sx。"""
    num = tag[1:]
    for i in range(1, len(num)):
        mx, sx = num[:i], num[i:]
        p = f'{VITPOSE}/{mx}/{sx}'
        if os.path.isdir(p):
            return f'{FINAL13.format(tag=tag)}', p
    return f'{FINAL13.format(tag=tag)}', None
