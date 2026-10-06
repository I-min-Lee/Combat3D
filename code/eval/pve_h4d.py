#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pve_h4d.py — **PVE / PA-PVE**：与 Harmony4D 官方榜单同口径的网格误差

为什么值钱：Harmony4D 榜单上 LookMa* / CameraHMR / MAMMA / Multi-HMR2 报的就是
**MPJPE + PVE（+ PA 版）**，PVE 是能**直接并到同一张表**的硬指标。

数据来源（两边都是 6890 顶点、同拓扑 → 顶点一一对应，标准 PVE）：
  · 我方：`emfit_h4d/pid{p}/verts/{fr:06d}.npy`（stage6 拟合出的网格，**COLMAP 世界系**）
  · 真值：`processed_data/smpl/{fr:05d}.npy` 里的 **`vertices`** 字段（**米制系**）——数据集直接给了，
         不用重建。
  · 换算：`X_metric = scale_metric.npy @ X_colmap`（4×4 相似变换，实测 s=1.375482）

报什么：
  · **PVE(Abs)**：世界系直接比（不做任何对齐）= 榜单的 Abs-PVE
  · **PVE(root)**：两边各减去自身根关节（骨盆）后比 —— 排除整体平移漂移
  · **PA-PVE**：逐帧对两片网格做 **相似变换 Procrustes 对齐**（允许尺度）后再比 = 榜单的 PA-PVE
  · 中位 / 均值 / P95（mm）

用法：
  python pve_h4d.py --fit <emfit_h4d> --seq-root <.../016_mma4> --calib <calib 目录> \
                    [--start 1 --end 61] [--csv out.csv]
"""
import os, json, glob, argparse
import numpy as np

SUBJ = {0: 'aria01', 1: 'aria02'}          # 与 h4d_boxes.py 的身份映射一致


def umeyama_sim(A, B):
    """求把 A 对齐到 B 的**相似变换**（含尺度）：B ≈ s·R·A + t；返回 (s, R, t)

    标准 Umeyama(1991)：Σ = (B0ᵀ A0)/n，SVD(Σ)=U D Vᵀ，R = U·diag(1,1,det(U Vᵀ))·Vᵀ，
    c = trace(D·diag(1,1,d)) / var(A0)。（第一版把 Σ 写反、尺度项也取错 → 出了 2.6e6 mm）
    """
    n = len(A)
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    Sigma = (B0.T @ A0) / n
    U, D, Vt = np.linalg.svd(Sigma)
    d = np.sign(np.linalg.det(U @ Vt))
    Dm = np.diag([1.0, 1.0, d])
    R = U @ Dm @ Vt
    varA = (A0 ** 2).sum() / n
    c = float(np.trace(np.diag(D) @ Dm) / max(varA, 1e-12))
    t = muB - c * (R @ muA)
    return c, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', required=True, help='emfit_h4d 根目录（下含 pid{p}/verts）')
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--pids', default='0,1')
    ap.add_argument('--frameshift', type=int, default=0,
                    help='fit 输出索引 → Harmony4D 帧号 的偏移（EasyMocap 要求 0-based，'
                         '故 H4D 的 GT 帧 = 索引 + 1）')
    ap.add_argument('--fitglobsuffix', default='*.npy')
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()

    S = np.load(os.path.join(a.calib, 'scale_metric.npy'))
    Rs, ts = S[:3, :3], S[:3, 3]
    sc = float(np.linalg.norm(S[:3, 0]))
    print('米制化：scale=%.6f（X_metric = S @ X_colmap）' % sc)

    rows = []
    for pid in [int(x) for x in a.pids.split(',')]:
        vd = os.path.join(a.fit, 'pid%d' % pid, 'verts')
        fss = sorted(glob.glob(os.path.join(vd, '*.npy')))
        if not fss:
            print('  pid%d: 无顶点产物（%s）' % (pid, vd)); continue
        subj = SUBJ[pid]
        absd, rootd, pad = [], [], []
        for f in fss:
            fr = int(os.path.basename(f).split(".")[0]) + a.frameshift   # ★fit 0-based → GT 1-based
            if not (a.start <= fr <= a.end):
                continue
            gtf = os.path.join(a.seq_root, 'processed_data', 'smpl', '%05d.npy' % fr)
            if not os.path.exists(gtf):
                continue
            g = np.load(gtf, allow_pickle=True).item().get(subj)
            if g is None or 'vertices' not in g:
                continue
            Vg = np.asarray(g['vertices'], np.float64)                       # 米制
            Vo = np.load(f).astype(np.float64)                                # COLMAP 系
            Vo = Vo @ Rs.T + ts                                               # → 米制
            if Vo.shape != Vg.shape:
                continue
            # Abs-PVE
            absd.append(np.linalg.norm(Vo - Vg, axis=1))
            # root-PVE：各自减去根关节（用 mesh 顶点的质心作代理，避免引 body model）
            rootd.append(np.linalg.norm((Vo - Vo.mean(0)) - (Vg - Vg.mean(0)), axis=1))
            # PA-PVE：相似变换对齐后再比
            try:
                s, R, t = umeyama_sim(Vo, Vg)
                pa = np.linalg.norm(s * (Vo @ R.T) + t - Vg, axis=1)
                pad.append(pa)
            except np.linalg.LinAlgError:
                pass
        if not absd:
            print('  pid%d (%s): 无对齐帧' % (pid, subj)); continue
        A = np.concatenate(absd) * 1000
        Rr = np.concatenate(rootd) * 1000
        P = np.concatenate(pad) * 1000 if pad else np.array([np.nan])
        rows.append((pid, subj, len(absd), A, Rr, P))
        print('  pid%d %s：%d 帧 | PVE(Abs) 中位 %.1f / 均 %.1f / P95 %.1f mm | '
              'PVE(root) 中位 %.1f | PA-PVE 中位 %.1f / 均 %.1f mm'
              % (pid, subj, len(absd), np.median(A), A.mean(), np.percentile(A, 95),
                 np.median(Rr), np.nanmedian(P), np.nanmean(P)))

    if rows:
        A = np.concatenate([r[3] for r in rows]); Rr = np.concatenate([r[4] for r in rows])
        P = np.concatenate([r[5] for r in rows])
        print('\n=== 合计（%d 帧）===' % sum(r[2] for r in rows))
        print('  PVE(Abs)  中位 %6.1f  均 %6.1f  P95 %6.1f mm' % (np.median(A), A.mean(), np.percentile(A, 95)))
        print('  PVE(root) 中位 %6.1f  均 %6.1f  P95 %6.1f mm' % (np.median(Rr), Rr.mean(), np.percentile(Rr, 95)))
        print('  PA-PVE    中位 %6.1f  均 %6.1f  P95 %6.1f mm' % (np.nanmedian(P), np.nanmean(P), np.nanpercentile(P, 95)))
        print('\n参照（Harmony4D 公开榜单，单目/多目 HMR 方法，口径不完全相同，仅供量级对照）：'
              'LookMa* PVE 57.6 / CameraHMR 50.7 / MAMMA 31.5~34.0 mm')
        if a.csv:
            with open(a.csv, 'w') as fh:
                fh.write('pid,subject,frames,pve_abs_med,pve_abs_mean,pve_abs_p95,pve_root_med,pa_pve_med,pa_pve_mean\n')
                for pid, subj, n, A_, R_, P_ in rows:
                    fh.write('%d,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (
                        pid, subj, n, np.median(A_), A_.mean(), np.percentile(A_, 95),
                        np.median(R_), np.nanmedian(P_), np.nanmean(P_)))
            print('CSV -> %s' % a.csv)


if __name__ == '__main__':
    main()
