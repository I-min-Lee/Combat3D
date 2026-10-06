#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kb_frameqa —— 逐帧的「3D 标签质量」度量（比 5 秒分箱细得多）

做法：把该 (take,view,pid) 的 3D 标签投回该视角，只与**可信的 2D 关节**比
      （3D conf>0 且 2D conf>=--c2d），取这些关节的中位像素残差。
      · 只剩极少可信关节的帧 -> qa_n 小，单独处理（不当成"坏"，也不当"好"）
      · 残差按 fx 归一化到 960 域，两组(A/C 与 B)阈值才可直接横比
        （手册：残差与 fx 成正比，1920 组数值天然是 960 组的 2 倍）

输出 data_qa/<tag>_v<view>_p<pid>.npz:
    qa      (N,) float32  中位残差(归一化到 960 域)
    qa_n    (N,) int16    参与比对的关节数
    qa_raw  (N,) float32  原始像素残差(未归一化)

用法:
  python kb_frameqa.py --stats                  # 只看分布、不下结论
  python kb_frameqa.py --workers 48             # 全量产出
"""
import os, sys, json, glob, argparse, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_common as K

FX_REF = 1020.747803          # 960x720 组主视角 fx（手册 §一.2）


def one(fn, outdir, c2d):
    base = os.path.basename(fn)[:-4]
    try:
        take, v, p = base.rsplit('_', 2)
        v = v[1:]
        p = int(p[1:])
    except Exception:
        return (base, 'name', 0)
    z = np.load(fn, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    k2d, k3d = z['k2d'], z['k3d']
    n = len(k2d)
    if n == 0:
        return (base, 'empty', 0)
    cal = K.Calib(K.CALIB[meta['group']])
    Rv, Tv, Kv, Dv = cal.R[v], cal.T[v], cal.K[v], cal.D[v]
    fx = float(Kv[0, 0])
    # 3D(世界米) -> 相机系米 -> 像素
    Xw = k3d[:, :, :3].astype(np.float64).reshape(-1, 3)
    Xc = (Rv @ Xw.T).T + Tv
    Z = Xc[:, 2].copy()
    Z[np.abs(Z) < 1e-6] = 1e-6
    xn = Xc[:, 0] / Z
    yn = Xc[:, 1] / Z
    # 畸变（cv2.projectPoints 会做，这里用 K 直接投 + 忽略畸变会造成小偏差；
    # 用 cv2 保证与管线一致）
    import cv2
    uv, _ = cv2.projectPoints(Xc, np.zeros((3, 1)), np.zeros((3, 1)), Kv, Dv)
    uv = uv.reshape(n, 17, 2)
    m = (k3d[:, :, 3] > 0) & (k2d[:, :, 2] >= c2d)
    d = np.linalg.norm(uv - k2d[:, :, :2], axis=2)
    d = np.where(m, d, np.nan)
    qa_n = m.sum(1).astype(np.int16)
    qa_raw = np.full(n, -1.0, dtype=np.float64)
    sel = qa_n > 0
    if sel.any():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')            # 全 NaN 行会发 RuntimeWarning，无妨
            qa_raw[sel] = np.nanmedian(d[sel], axis=1)
    qa = np.where(qa_raw >= 0, qa_raw * (FX_REF / fx), -1.0).astype(np.float32)
    qa_raw = qa_raw.astype(np.float32)
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(f'{outdir}/{base}.npz', qa=qa, qa_raw=qa_raw, qa_n=qa_n)
    return (base, 'ok', n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=K.OUT)
    ap.add_argument('--outdir', default='/workshop/Lym/combat3d/mb/data_qa')
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--c2d', type=float, default=0.5, help='判定"可信 2D 关节"的 conf 下限')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--min-n', type=int, default=5, help='统计时要求的最少可信关节数')
    a = ap.parse_args()

    files = sorted(glob.glob(f'{a.data}/*.npz'))
    files = [f for f in files if '_build_report' not in f and '_exclude' not in f]
    print(f'待处理 {len(files)} 个序列')

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, f, a.outdir, a.c2d): f for f in files}
        done = 0
        for fu in as_completed(futs):
            try:
                res.append(fu.result())
            except Exception as e:
                res.append((os.path.basename(futs[fu]), 'EXC:' + str(e)[:80], 0))
            done += 1
            if done % 150 == 0 or done == len(files):
                print(f'  {done}/{len(files)}', flush=True)
    bad = [r for r in res if r[1] != 'ok']
    print(f'完成 ok={len(res)-len(bad)} bad={len(bad)}')
    for r in bad[:10]:
        print('   ', r)

    if a.stats:
        allq = []
        tot_fr = 0
        for p in glob.glob(f'{a.outdir}/*.npz'):
            z = np.load(p)
            qa, qn = z['qa'], z['qa_n']
            ok = (qa >= 0) & (qn >= a.min_n)
            allq.append(qa[ok])
            tot_fr += len(qa)
        q = np.concatenate(allq) if allq else np.array([])
        print(f'\n=== 逐帧标签残差分布（归一化到 960 域；n={len(q)} / 总 {tot_fr} 帧）===')
        for p in (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100):
            print('  p%-5.1f = %8.2f px' % (p, float(np.percentile(q, p))))
        print('\n各阈值下"保留帧"比例:')
        for t in (8, 10, 12, 15, 20, 25, 30, 50):
            print('  qa <= %-4d px -> 保留 %6.2f%%' % (t, 100 * float((q <= t).mean())))
        json.dump(dict(percentiles={str(p): float(np.percentile(q, p)) for p in
                                    (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100)},
                       n=int(len(q)), total_frames=tot_fr),
                  open(f'{a.outdir}/_stats.json', 'w'), ensure_ascii=False, indent=1)
        print('->', f'{a.outdir}/_stats.json')


if __name__ == '__main__':
    main()
