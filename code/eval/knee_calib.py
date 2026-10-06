#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""knee_calib.py — 时序参数标定判据（通用：换场次/分辨率/帧率都能跑）

对同一段观测、同一套标定下的多组 3D 结果（不同 λ 的三角化输出 / 不同 k 的 fit 输出），
同时给出四类可观测判据，用于选"knee 点"而不是靠帧率换算猜：

  A. 帧间抖动      jitter: 逐帧最大关节位移 的中位/p90/max (mm/帧)
  B. 位置偏置      pull : 相对参考组(默认 λ=0)的关节位移 中位/p90/max (mm)
  C. 几何自洽      boneCV: 13 骨长 CV 的中位/p90
  D. 2D 重投影残差 reproj: 3D 反投影到各视角 vs 观测 2D 的像素残差
                   中位/p90 px、内点率(err<=20px)、以及相对参考组的增幅

判据用法：D 是最硬的（"3D 还解不解释得动 2D"）。取 **重投影残差增幅仍小** 的最大 λ
（knee）；A/C 变好但 D 明显变差，说明已经在用位移偏置换平滑，不是去噪。

用法（示例，B组 0.1 的 λ 扫描）：
  PYTHONPATH=$MASTER $PYO knee_calib.py \
    --calib calib_B_1920x1440 --annots easymocap_b01/0_1/annots \
    --views 1,3,4,7,11 --n 2428 --ref 0.0 \
    --runs 0.0:em_b01_sweep/lam0.0 0.05:em_b01_sweep/lam0.05 0.2:em_b01_sweep/lam0.2 \
           1.0:em_b01_sweep/lam1.0 3.0:em_b01_sweep/lam3.0
"""
import os, sys, json, glob, argparse
import numpy as np

B = '/root/autodl-tmp'
MASTER = B + '/emoff/EasyMocap-master'

# ---- body25 -> COCO17 的对应（只保留两边都有的关节）----
B25_TO_COCO17 = {0: 0, 5: 5, 6: 7, 7: 9, 2: 6, 3: 8, 4: 10,
                 12: 11, 13: 13, 14: 15, 9: 12, 10: 14, 11: 16}
# final13(13节点) 的局部索引 -> b25 索引 -> coco17
IDX13_B25 = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
# 13 节点局部索引对的骨（用于骨长 CV）
BONES13 = [(0, 1), (0, 4), (1, 2), (4, 5), (2, 3), (5, 6),
           (1, 7), (4, 10), (7, 8), (10, 11), (8, 9), (11, 12)]
COCO_BONES = [(0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16)]


def to_coco17(P):
    """(...,J,C) -> (...,K,C) 与对应的 coco 索引列表；支持 25 / 17 / 13 节点输入"""
    J = P.shape[-2]
    if J == 17:
        return P, list(range(17))
    if J == 25:
        idx = sorted(B25_TO_COCO17)
        return P[..., idx, :], [B25_TO_COCO17[i] for i in idx]
    if J == 13:
        # 13 节点是 b25 的子集，其局部顺序 = IDX13_B25
        pairs = [(i, B25_TO_COCO17[b]) for i, b in enumerate(IDX13_B25) if b in B25_TO_COCO17]
        i13 = [p[0] for p in pairs]
        return P[..., i13, :], [p[1] for p in pairs]
    raise ValueError('不支持的关节数 %d' % J)


def load3d(run_dir, n):
    """返回 {pid: (N,J,4)}；自动识别 keypoints3d / keypoints3d_b25"""
    out = {}
    for pid in (0, 1):
        # ★ 优先 b25：fit 目录里的 keypoints3d 只是指向三角化输出的软链（会读成同一份数据）
        for name in ('keypoints3d_b25', 'keypoints3d'):
            d = os.path.join(run_dir, 'pid%d' % pid, name)
            fs = sorted(glob.glob(d + '/*.json'))[:n]
            if fs:
                break
        if not fs:
            out[pid] = None
            continue
        arrs = []
        for f in fs:
            j = json.load(open(f))
            a = j['annots'][0] if isinstance(j, dict) and 'annots' in j else j[0]
            arrs.append(np.array(a['keypoints3d'], float))
        out[pid] = np.stack(arrs)[:, :, :3]
    return out


def load2d(annots_dir, view, n):
    """返回 {pid: (N,17,3)}"""
    out = {}
    for pid in (0, 1):
        arr = np.zeros((n, 17, 3), float)
        for fr in range(n):
            f = os.path.join(annots_dir, view, '%06d.json' % fr)
            if not os.path.exists(f):
                continue
            for a in json.load(open(f)).get('annots', []):
                if int(a['personID']) == pid:
                    arr[fr] = np.array(a['keypoints'], float)
        out[pid] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True, help='含 intri.yml/extri.yml 的目录')
    ap.add_argument('--annots', required=True, help='观测 2D 目录（含 {view}/{fr:06d}.json）')
    ap.add_argument('--views', default='1,3,4,7,11')
    ap.add_argument('--n', type=int, default=2428)
    ap.add_argument('--ref', default=None, help='参考组标签（默认取第一个）')
    ap.add_argument('--fps', type=float, default=200.0)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--runs', nargs='+', required=True, help='标签:目录 ...')
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    views = a.views.split(',')
    sys.path.insert(0, MASTER)
    from easymocap.mytools.camera_utils import read_camera
    cams = read_camera(os.path.join(a.calib, 'intri.yml'), os.path.join(a.calib, 'extri.yml'))
    P = {}
    for v in views:
        c = cams[v]
        P[v] = np.array(c['P'], float) if 'P' in c else np.array(c['K'], float) @ np.hstack(
            [np.array(c['R'], float), np.array(c['T'], float).reshape(3, 1)])

    obs = {v: load2d(a.annots, v, a.n) for v in views}
    print('观测 2D: %d 帧 × %d 视角' % (a.n, len(views)))

    rows, ref3d = [], None
    for spec in a.runs:
        lab, dirp = spec.split(':', 1)
        if not os.path.isabs(dirp):
            dirp = os.path.join(B, dirp)
        K3 = load3d(dirp, a.n)
        if K3[0] is None:
            print('  跳过（无数据）%s' % lab); continue
        if ref3d is None:
            ref3d = K3
        rec = {'label': lab}
        # ---- A 抖动 / B 偏置（都在 coco17 公共关节上算）----
        jit, pull, bcv = [], [], []
        for pid in (0, 1):
            Pp, cidx = to_coco17(K3[pid])
            c2l = {c: i for i, c in enumerate(cidx)}
            bones = [(c2l[x], c2l[y]) for x, y in COCO_BONES if x in c2l and y in c2l]
            if ref3d[pid] is not None:
                Pr, _ = to_coco17(ref3d[pid])
                ok = (Pp.shape[:2] == Pr.shape[:2])
                if ok:
                    dd = np.linalg.norm(Pp - Pr, axis=-1) * 1000
                    jv = ((Pp[:, :, 0] == 0) & (Pp[:, :, 1] == 0))
                    dd[jv] = np.nan
                    pull.append(np.nanpercentile(dd, [50, 90, 100]))
            dv = np.linalg.norm(np.diff(Pp, axis=0), axis=-1) * 1000
            dv[(Pp[1:, :, 0] == 0) & (Pp[1:, :, 1] == 0)] = np.nan
            mx = np.nanmax(dv, axis=1)
            jit.append(np.nanpercentile(mx, [50, 90, 100]))
            bl = np.array([np.linalg.norm(Pp[:, b[0]] - Pp[:, b[1]], axis=-1) for b in bones])
            cvm = bl.std(axis=1) / np.maximum(bl.mean(axis=1), 1e-9)
            bcv.append(np.nanpercentile(cvm, [50, 90]))
        rec['jit'] = np.nanmedian(jit, axis=0)
        rec['pull'] = np.nanmedian(pull, axis=0) if pull else np.array([np.nan] * 3)
        rec['bone'] = np.nanmedian(bcv, axis=0)
        # ---- D 2D 重投影残差 ----
        errs = []
        for pid in (0, 1):
            Pp, cidx = to_coco17(K3[pid])
            for v in views:
                o = obs[v][pid][:, cidx, :]                      # (N,K,3)
                X = Pp                                            # (N,K,3)
                h = np.concatenate([X, np.ones_like(X[:, :, :1])], axis=-1)   # (N,K,4)
                uvw = h @ P[v].T                                  # (N,K,3)
                uv = uvw[:, :, :2] / np.where(np.abs(uvw[:, :, 2:3]) < 1e-9, np.nan, uvw[:, :, 2:3])
                e = np.linalg.norm(uv - o[:, :, :2], axis=-1)
                valid = (o[:, :, 2] > a.conf) & np.isfinite(e) & (X[:, :, 0] != 0) & (X[:, :, 1] != 0)
                errs.append(e[valid])
        E = np.concatenate(errs) if errs else np.array([np.nan])
        rec['reproj'] = np.array([np.nanmedian(E), np.nanpercentile(E, 90), 100.0 * np.mean(E <= 20)])
        rows.append(rec)

    # ---- 输出表 ----
    ref = a.ref if a.ref is not None else rows[0]['label']
    base = None
    for r in rows:
        if r['label'] == ref:
            base = r
    hdr = ('%-8s | %-22s | %-22s | %-14s | %s' %
           ('参数', '帧间抖动 中位/p90/max', '位置偏置 中位/p90/max', '骨长CV 中位/p90', '重投影 中位/p90/内点%(增幅)'))
    print('\n' + hdr); print('-' * len(hdr))
    for r in rows:
        gr = ''
        if base is not None and r is not base:
            gr = '  (+%.1f%%)' % (100 * (r['reproj'][0] / base['reproj'][0] - 1))
        print('%-8s | %7.2f/%7.2f/%8.1f mm | %7.2f/%7.2f/%8.1f mm | %6.4f/%6.4f | %6.2f/%7.2f/%5.1f%%%s'
              % (r['label'], r['jit'][0], r['jit'][1], r['jit'][2],
                 r['pull'][0], r['pull'][1], r['pull'][2],
                 r['bone'][0], r['bone'][1],
                 r['reproj'][0], r['reproj'][1], r['reproj'][2], gr))
    if a.csv:
        with open(a.csv, 'w') as f:
            f.write('label,jit_med,jit_p90,jit_max,pull_med,pull_p90,pull_max,bone_med,bone_p90,reproj_med,reproj_p90,inlier_pct\n')
            for r in rows:
                f.write('%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%.2f\n'
                        % (r['label'], r['jit'][0], r['jit'][1], r['jit'][2],
                           r['pull'][0], r['pull'][1], r['pull'][2],
                           r['bone'][0], r['bone'][1],
                           r['reproj'][0], r['reproj'][1], r['reproj'][2]))
        print('\nCSV -> %s' % a.csv)
    print('\n注：重投影残差是最硬判据 —— 取「残差增幅仍小」的最大 λ 为 knee；')


if __name__ == '__main__':
    main()
