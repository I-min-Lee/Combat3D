#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h4d_metrics.py — Harmony4D 评估：四类内部指标（与你 knee_calib.py 同口径）+ MPJPE

与 `eval/knee_calib.py` 的关系：
  A 帧间抖动 / B 位置偏置 / C 骨长CV —— **逐字镜像原文的实现**
     （同一 B25_TO_COCO17、同一 COCO_BONES、同一 nan 掩码与 percentile 口径）
  D 2D 重投影 —— ★**改为鱼眼投影**：把 3D 用 (K, k1..k4) 投回**原始（含畸变）像素坐标**
     再与 conv 输出（= 原始 2D）比。这比"先把观测去畸变再比"更直接，也不需要改观测。
     若仍想用针孔口径，加 `--pinhole-reproj`（会先把观测按针孔去畸变）。
  追加：**MPJPE / PA-MPJPE / P95 / 覆盖率**，直接对上 Harmony4D 的 17 点 COCO 真值。

★ 单位：本管线跑在 **COLMAP 世界系**（extri.yml 所在系），1 COLMAP 单位 = **scale 米**
   （scale 取自 calib_h4d_*/scale_metric.npy 的 4×4 相似变换 3×3 模长，实测 1.375482）。
   所有 mm 量已按此换算；MPJPE 同时给 COLMAP 系与米制系（后者 = 前者 × scale）。

用法：
  PYTHONPATH=<emcore> python h4d_metrics.py \
    --calib calib_h4d_016mma4 --annots conv_h4d_test/annots --gt gt_h4d_016mma4 \
    --views 01,03,04,07,09,14 --n 30 --ref 0 \
    --runs 0:em_h4d_test/lam0 0.2:em_h4d_test/lam0.2 ...
"""
import os, sys, json, glob, argparse
import numpy as np

B25_TO_COCO17 = {0: 0, 5: 5, 6: 7, 7: 9, 2: 6, 3: 8, 4: 10,
                 12: 11, 13: 13, 14: 15, 9: 12, 10: 14, 11: 16}
# ★ 补上 4 个脸点，才凑齐真正的 COCO17（B25 的 15/16/17/18 = R/L 眼、R/L 耳）
B25_TO_COCO17_FULL = dict(B25_TO_COCO17)
B25_TO_COCO17_FULL.update({15: 2, 16: 1, 17: 4, 18: 3})
COCO_BONES = [(0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16)]
IDX13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]          # 鼻 + 12 身体点
IDX12 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]              # 再去掉鼻


def fit_sim(X, Y):
    """Umeyama 相似变换（**含尺度**）：求 s,R,t 使 s·R·X+t ≈ Y。X,Y: (N,3)"""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / max(len(X), 1))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (D * np.diag(E)).sum() / max((Xc ** 2).sum() / max(len(X), 1), 1e-12)
    return s, R, my - s * R @ mx


def mpjpe_arrays(pairs, mode='none', sc=1.0):
    """把 {pid: (O,G,ok)} 变成 (逐关节误差mm, 有效掩码, 关节索引)。

    mode: none  = 原样比（= 官方"同一世界系无需对齐"口径，也是 h4d_metrics 的历史口径）
          seq   = **整段**共用一个相似变换后再比 ← 补偿"缺世界尺度变换"的序列（06/08/01）
          frame = **逐帧**相似变换后再比 ← 官方 PA-MPJPE 常用口径
    """
    E, M, J = [], [], []
    for pid in sorted(pairs):
        O, G, ok = pairs[pid]
        O = np.array(O, float)
        if mode == 'seq' and ok.sum() >= 5:
            s, R, t = fit_sim(O[ok], G[ok])
            O = (s * (R @ O.reshape(-1, 3).T).T).reshape(O.shape) + t
        elif mode == 'frame':
            for i in range(len(O)):
                if ok[i].sum() >= 5:
                    s, R, t = fit_sim(O[i][ok[i]], G[i][ok[i]])
                    O[i][ok[i]] = s * (R @ O[i][ok[i]].T).T + t
        E.append(np.linalg.norm(O - G, axis=-1)[ok] * sc * 1000.0)
        M.append(ok[ok])
        J.append(np.tile(np.arange(O.shape[1]), (O.shape[0], 1))[ok])
    if not E:
        e = np.array([np.nan])
        return e, np.array([True]), np.zeros(1, int)
    return np.concatenate(E), np.concatenate(M), np.concatenate(J)


def to_coco17(P, full=False):
    """(N,J,C) -> (N,K,C) + 对应的 coco 索引；支持 25 / 17 / 13 输入（镜像 knee_calib）"""
    J = P.shape[-2]
    if J == 17:
        return P, list(range(17))
    if J == 25:
        m = B25_TO_COCO17_FULL if full else B25_TO_COCO17
        idx = sorted(m)
        return P[..., idx, :], [m[i] for i in idx]
    if J == 13:
        raise ValueError('13 节点输入请直接用 final13 目录（本脚本按 b25 读）')
    raise ValueError('不支持的关节数 %d' % J)


def run_frames(run_dir):
    """三角化实际输出的帧号（可能少于 --n：order=2 的时序解会少头/尾帧）"""
    d = os.path.join(run_dir, 'pid0', 'keypoints3d')
    return sorted(int(os.path.basename(f).split('.')[0]) for f in glob.glob(d + '/*.json'))


def load3d(run_dir, frames):
    """★ 优先 keypoints3d_b25（镜像 knee_calib.load3d）：fit 目录里的 keypoints3d 只是
    指向三角化输出的软链，读它会拿到同一份数据从而"看不出 fit 的效果"。"""
    out = {}
    for pid in (0, 1):
        sub = None
        for name in ('keypoints3d_b25', 'keypoints3d'):
            if os.path.isdir(os.path.join(run_dir, 'pid%d' % pid, name)):
                sub = name
                break
        if sub is None:
            out[pid] = None; continue
        arrs = []
        ok = True
        for fr in frames:
            f = os.path.join(run_dir, 'pid%d' % pid, sub, '%06d.json' % fr)
            if not os.path.exists(f):
                ok = False; break
            j = json.load(open(f))
            a = j['annots'][0] if isinstance(j, dict) and 'annots' in j else j[0]
            arrs.append(np.array(a['keypoints3d'], float)[:, :4])
        out[pid] = np.stack(arrs) if ok and arrs else None      # (N,25,4)
    return out


def load2d(annots_dir, view, frames):
    """按**绝对帧号**读，返回 {pid: (N,17,3)}，行序与 frames 一致"""
    out = {}
    for pid in (0, 1):
        arr = np.zeros((len(frames), 17, 3), float)
        for i, fr in enumerate(frames):
            f = os.path.join(annots_dir, view, '%06d.json' % fr)
            if not os.path.exists(f):
                continue
            for a in json.load(open(f)).get('annots', []):
                if int(a['personID']) == pid:
                    arr[i] = np.array(a['keypoints'], float)
        out[pid] = arr
    return out


def proj_fisheye_world(Xw, R, T, K, dist):
    """世界系 (N,3) -> 原始（含畸变）像素 (N,2)。
    ★ 用 OpenCV 自带的 fisheye.projectPoints（与产出数据集的同一个库），
      不手写公式；内部自行完成 world→cam 的 R,t 变换。"""
    import cv2
    Xw = np.asarray(Xw, float).reshape(-1, 1, 3)
    rvec = cv2.Rodrigues(np.ascontiguousarray(np.asarray(R, float).reshape(3, 3)))[0]
    tvec = np.asarray(T, float).reshape(3, 1)
    uv, _ = cv2.fisheye.projectPoints(Xw, rvec, tvec,
                                      np.asarray(K, float).reshape(3, 3),
                                      np.asarray(dist, float).reshape(-1)[:4].reshape(4, 1))
    return uv.reshape(-1, 2)


def proj_pinhole_world(Xw, R, T, K, dist):
    """★针孔 + 多参数畸变（k1,k2,p1,p2,k3）的正向投影 —— Panoptic 用这套。
    与 `tri_h4d.py` 的 `undistort()` 分支对应：那里 fisheye 缺省就当针孔反解。"""
    import cv2
    Xw = np.ascontiguousarray(np.asarray(Xw, float).reshape(-1, 1, 3))
    rvec = cv2.Rodrigues(np.ascontiguousarray(np.asarray(R, float).reshape(3, 3)))[0].reshape(3, 1)
    tvec = np.asarray(T, float).reshape(3, 1)
    uv, _ = cv2.projectPoints(Xw, rvec, tvec, np.asarray(K, float).reshape(3, 3),
                              np.ascontiguousarray(np.asarray(dist, float).reshape(-1, 1)))
    return uv.reshape(-1, 2)


def load_fisheye_flags(intri_path):
    """读我们写出的 `fisheye_<v>: 1`（read_camera 不读自定义键）——
    Harmony4D 有、Panoptic 没有；缺省 False = 针孔。与 tri_h4d.py 同一套约定。"""
    import re
    flags = {}
    if not os.path.exists(intri_path):
        return flags
    for ln in open(intri_path):
        m = re.match(r'\s*fisheye_(\S+)\s*:\s*1', ln)
        if m:
            flags[m.group(1)] = True
    return flags


def load_gt(gt_dir, frames):
    """{fr: {subj: (17,4)}}，subj 名 → pid 映射由调用方给"""
    d = {}
    for fr in frames:
        f = os.path.join(gt_dir, '%06d.json' % fr)
        d[fr] = json.load(open(f)) if os.path.exists(f) else {}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True)
    ap.add_argument('--annots', required=True)
    ap.add_argument('--gt', default=None, help='gt_h4d_*/gt3d_colmap 目录（给了就算 MPJPE）')
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--ref', default=None)
    ap.add_argument('--runs', nargs='+', required=True, help='标签:目录 ...')
    ap.add_argument('--pinhole-reproj', action='store_true')
    ap.add_argument('--pa', default='none', choices=['none', 'seq', 'frame'],
                    help='MPJPE 对齐口径：none=原样(历史口径) / seq=整段共用一个相似变换 / '
                         'frame=逐帧相似变换(官方 PA-MPJPE)。'
                         '★06/08/01 缺世界尺度变换，只有 seq/frame 口径才可与真值框版同表比较')
    ap.add_argument('--match-perm', action='store_true',
                    help='★置换匹配：同时算 pid0↔aria01/pid1↔aria02 与互换两种全局配对，取更优者。'
                         '自建管线的 gid 编号与数据集的 aria01/aria02 之间**没有任何锚点**，'
                         '身份只能定义到"全局置换"这一层；真值框版则天然继承了真值身份（第 3 方数据不提供）')
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    views = a.views.split(',')
    sys.path.insert(0, os.environ.get('EMCORE', '/workshop/Lym/combat3d/port/emcore'))
    from easymocap.mytools.camera_utils import read_camera
    cams = read_camera(os.path.join(a.calib, 'intri.yml'), os.path.join(a.calib, 'extri.yml'))
    # 米制换算系数
    sc = 1.0
    sp = os.path.join(a.calib, 'scale_metric.npy')
    if os.path.exists(sp):
        S = np.load(sp); sc = float(np.linalg.norm(S[:3, 0]))
    P, KD = {}, {}
    FISHEYE = load_fisheye_flags(os.path.join(a.calib, 'intri.yml'))
    for v in views:
        c = cams[v]
        P[v] = np.array(c['P'], float) if 'P' in c else np.array(c['K'], float) @ np.hstack(
            [np.array(c['R'], float), np.array(c['T'], float).reshape(3, 1)])
        KD[v] = dict(K=np.array(c['K'], float),
                     dist=np.asarray(c['dist'], float).reshape(-1),      # ★保留全部系数（针孔要 5 个）
                     R=np.array(c['R'], float).reshape(3, 3),
                     T=np.array(c['T'], float).reshape(3))
    n_fish = sum(1 for v in views if FISHEYE.get(v, False))
    print('畸变模型: %d/%d 视角鱼眼%s' % (n_fish, len(views),
          '' if n_fish else '（其余按针孔+多参数畸变，Panoptic 即此类）'))
    # 帧号以第一个 run 的实际输出为准（order=2 的时序解会少头/尾帧）
    first_dir = a.runs[0].split(':', 1)[1]
    allf = run_frames(first_dir)
    frames = allf[:a.n] if a.n > 0 else allf
    obs = {v: load2d(a.annots, v, frames) for v in views}
    gtraw = load_gt(a.gt, frames) if a.gt else None
    print('实际帧 %d..%d (%d 帧) | 视角 %d | 米制换算 scale=%.6f'
          % (frames[0], frames[-1], len(frames), len(views), sc))
    print('视角 %s' % ','.join(views))

    # ---- ★ 预加载 GT（两个 subject 各一份）：MPJPE / PA口径 / 置换匹配 共用 ----
    # 用**两人都在场的共同帧索引**，这样"原样配对"与"互换配对"面对的是同一批帧，可直接比大小
    GT, GT_IDX = {}, []
    if gtraw is not None:
        GT_IDX = [i for i, fr in enumerate(frames)
                  if all(s in gtraw.get(fr, {}) for s in ('aria01', 'aria02'))]
        for sub in ('aria01', 'aria02'):
            arr = [np.array(gtraw[frames[i]][sub], float)[:, :3] for i in GT_IDX]
            GT[sub] = np.stack(arr) if arr else None

    rows, base = [], None
    for spec in a.runs:
        lab, dirp = spec.split(':', 1)
        K3 = load3d(dirp, frames)
        if K3[0] is None:
            print('  跳过（无数据）%s' % lab); continue
        rec = {'label': lab}
        jit, bcv, reproj = [], [], []
        OPID = {}           # {pid: (O, ok)} 本 run 的 3D 点；配对到哪个 subject 由 _pairs 决定
        for pid in (0, 1):
            Pp, cidx = to_coco17(K3[pid])                       # (N,13,4) 常用 13 视
            Pfull, cfull = to_coco17(K3[pid], full=True)        # (N,17,4) 真 COCO17
            c2l = {c: i for i, c in enumerate(cidx)}
            bones = [(c2l[x], c2l[y]) for x, y in COCO_BONES if x in c2l and y in c2l]
            # A 抖动
            dv = np.linalg.norm(np.diff(Pp[:, :, :3], axis=0), axis=-1) * sc * 1000
            dv[(Pp[1:, :, 0] == 0) & (Pp[1:, :, 1] == 0)] = np.nan
            jit.append(np.nanpercentile(np.nanmax(dv, axis=1), [50, 90, 100]))
            # C 骨长 CV
            bl = np.array([np.linalg.norm(Pp[:, b[0], :3] - Pp[:, b[1], :3], axis=-1) for b in bones])
            bl[:, (bl == 0).all(0)] = np.nan
            cvm = np.nanstd(bl, axis=1) / np.maximum(np.nanmean(bl, axis=1), 1e-9)
            bcv.append(np.nanpercentile(cvm, [50, 90]))
            # D 重投影（鱼眼，或针孔）
            for v in views:
                o = obs[v][pid][:, cidx, :]
                X = Pp[:, :, :3]
                if a.pinhole_reproj:
                    import cv2
                    kk, dd = KD[v]['K'], KD[v]['dist'].astype(np.float64).reshape(4, 1)
                    uv = np.array([cv2.fisheye.undistortPoints(
                        o[i, :, :2].astype(np.float64).reshape(-1, 1, 2), kk,
                        dd, None, None, kk).reshape(-1, 2) for i in range(len(o))])
                    uv3 = np.concatenate([X, np.ones_like(X[:, :, :1])], -1) @ P[v].T
                    e = np.linalg.norm(uv3[:, :, :2] / uv3[:, :, 2:3] - uv, axis=-1)
                else:
                    _proj = proj_fisheye_world if FISHEYE.get(v, False) else proj_pinhole_world
                    uv = _proj(X, KD[v]['R'], KD[v]['T'], KD[v]['K'],
                               KD[v]['dist']).reshape(X.shape[0], X.shape[1], 2)
                    e = np.linalg.norm(uv - o[:, :, :2], axis=-1)
                valid = (o[:, :, 2] > a.conf) & np.isfinite(e) & (X[:, :, 0] != 0) & (X[:, :, 1] != 0)
                reproj.append(e[valid])
            # MPJPE vs GT：只收集本 run 的 3D 点（统计在 pid 循环外统一算，
            # 因为 PA 口径与"置换匹配"都要同时看到两种配对）
            if GT_IDX:
                O = Pfull[GT_IDX, :, :3]                 # 按 sorted(b25) 顺序
                O = O[:, np.argsort(cfull), :]           # ★重排到 COCO 原生顺序
                OPID[pid] = (O, (O != 0).any(-1) & (np.abs(O).sum(-1) < 1e6))
        rec['jit'] = np.nanmedian(jit, axis=0)
        rec['bone'] = np.nanmedian(bcv, axis=0)
        E = np.concatenate(reproj) if reproj else np.array([np.nan])
        rec['reproj'] = np.array([np.nanmedian(E), np.nanpercentile(E, 90), 100.0 * np.mean(E <= 20)])

        def _pairs(mapping):
            """mapping: {pid: subject 名} -> {pid: (O,G,ok)}"""
            out, covs = {}, []
            for pid, sub in mapping.items():
                if pid not in OPID or GT.get(sub) is None:
                    continue
                O, ok = OPID[pid]
                G = GT[sub]
                okk = ok & (np.abs(G).sum(-1) > 0)
                out[pid] = (O, G, okk)
                covs.append(float(okk.mean()))
            return out, (float(np.mean(covs)) if covs else 0.0)

        def _fill(pr, sfx=''):
            """sfx='' → 写 mpj/mpj13/mpj14；sfx='_alt' → 写 mpj_alt/mpj_alt13/mpj_alt14"""
            e, _mk, jj = mpjpe_arrays(pr, a.pa, sc)
            s13, s12 = np.isin(jj, IDX13), np.isin(jj, IDX12)
            rec['mpj' + sfx] = np.array([np.nanmean(e), np.nanpercentile(e, 95), np.nanmedian(e)])
            rec['mpj' + sfx + '13'] = np.array([np.nanmean(e[s13]), np.nanpercentile(e[s13], 95),
                                                np.nanmedian(e[s13])])
            rec['mpj' + sfx + '14'] = np.array([np.nanmean(e[s12]), np.nanpercentile(e[s12], 95),
                                                np.nanmedian(e[s12])])
            return np.nanmedian(e)

        P_as, cov_as = _pairs({0: 'aria01', 1: 'aria02'})
        if P_as:
            med_as = _fill(P_as)
            rec['cov'] = cov_as
            rec['perm'] = 'as-is'
            if a.match_perm and 1 in OPID:
                P_sw, cov_sw = _pairs({0: 'aria02', 1: 'aria01'})   # ★互换全局配对
                med_sw = _fill(P_sw, '_alt')
                if med_sw < med_as:                                  # 互换更优 → 采用互换
                    for k in ('mpj_alt', 'mpj_alt13', 'mpj_alt14'):
                        rec[k.replace('_alt', '')] = rec.pop(k)
                    rec['perm'], rec['cov'] = 'swapped', cov_sw
                else:
                    rec['perm_alt'] = float(med_sw)
                    rec.pop('mpj_alt'); rec.pop('mpj_alt13'); rec.pop('mpj_alt14')
        rows.append(rec)

    ref = a.ref if a.ref is not None else rows[0]['label']
    base = next((r for r in rows if r['label'] == ref), None)
    print()
    print('%-6s | %-21s | %-13s | %-22s | %s' % ('参数', '帧间抖动 中/p90/max (mm)',
          '骨长CV 中/p90', '2D重投影 中/p90/内点%',
          'MPJPE 中/P95 mm (17点) [--pa %s%s]'
          % (a.pa, ' --match-perm' if a.match_perm else '')))
    print('-' * 118)
    for r in rows:
        gr = '' if base is None or r is base else '  (+%.1f%%)' % (100 * (r['reproj'][0] / base['reproj'][0] - 1))
        s = ('%-6s | %7.1f/%7.1f/%8.1f | %6.4f/%6.4f | %6.2f/%7.2f/%5.1f%%%s'
             % (r['label'], r['jit'][0], r['jit'][1], r['jit'][2], r['bone'][0], r['bone'][1],
                r['reproj'][0], r['reproj'][1], r['reproj'][2], gr))
        if 'mpj' in r:
            s += ' | 中%6.1f 均%7.1f P95%7.1f (13点中%6.1f | 无脸12点中%6.1f | 覆盖%.0f%%)' % (
                r['mpj'][2], r['mpj'][0], r['mpj'][1],
                r['mpj13'][2], r['mpj14'][2], 100 * r['cov'])
            if r.get('perm') == 'swapped':
                s += '  ★已按最优全局置换（pid0↔pid1 互换）'
            elif r.get('perm_alt') is not None:
                s += '  （编号正确；互换后 %.1f）' % r['perm_alt']
        print(s)
    if a.csv:
        with open(a.csv, 'w') as f:
            f.write('label,jit_med,jit_p90,jit_max,bone_med,bone_p90,reproj_med,reproj_p90,inlier_pct,'
                    'mpjpe17_med,mpjpe17_p95,mpjpe13,p95_13,mpjpe14,p95_14,coverage\n')
            for r in rows:
                f.write('%s,%.3f,%.3f,%.3f,%.5f,%.5f,%.3f,%.3f,%.2f,%s,%s,%s,%s,%s,%s,%.4f\n' % (
                    r['label'], r['jit'][0], r['jit'][1], r['jit'][2], r['bone'][0], r['bone'][1],
                    r['reproj'][0], r['reproj'][1], r['reproj'][2],
                    getattr(r.get('mpj'), '__getitem__', lambda i: float('nan'))(0) if 'mpj' in r else float('nan'),
                    r['mpj'][1] if 'mpj' in r else float('nan'),
                    r['mpj13'][0] if 'mpj' in r else float('nan'), r['mpj13'][1] if 'mpj' in r else float('nan'),
                    r['mpj14'][0] if 'mpj' in r else float('nan'), r['mpj14'][1] if 'mpj' in r else float('nan'),
                    r.get('cov', 0.0)))
        print('\nCSV -> %s' % a.csv)
    print('\n注：重投影为**鱼眼正向投影**（3D→含畸变像素）与 conv 原始 2D 比；'
          'MPJPE 在 COLMAP 世界系，已按 scale=%.4f 折成米制 mm。' % sc)


if __name__ == '__main__':
    main()
