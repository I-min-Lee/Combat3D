#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h4d_calib.py — Harmony4D(COLMAP) → EasyMocap 风格 intri.yml / extri.yml

═══════════════════════════════════════════════════════════════════
全部结论经 2026-09-29 实测验收（GT3D 反投影中位 9.2px / 门限 20px）
═══════════════════════════════════════════════════════════════════
· cameras.txt 23 台**全是 OPENCV_FISHEYE**（4 个径向参数 k1..k4）
· images.txt 是**稀疏 SfM 关键帧**：20 台固定机位各仅 ~5 个 →
  **四元数符号对齐后平均**得一组固定外参
    （实测 t_std ≤ 0.0044 m、R_std ≤ 0.11°，机架刚性成立）
· COLMAP id 1,2 = aria01/aria02（1408×1408，**移动**：R_std>100°）→ 剔除
  COLMAP id 3..22 = exo/cam01..cam20（3840×2160，固定）→ 保留
  COLMAP id 23 = mobile/（**手持移动机**，147 关键帧）→ 剔除
· scale.npy 是 **4×4 相似变换**（3×3 = 1.375482×旋转，末列平移）= 米制化矩阵。
  ★ 本脚本**不做**米制化：extri 直接写 COLMAP 世界系（与 read_camera 的
  P=K[R|T] 一致），米制化在**导出/评估**阶段用同一份 scale.npy 一次性施加。
  这样管线内部零改动，且验收链路与实测一致。
· R_<cam> 写**罗德里格斯向量**（read_camera 用 cv2.Rodrigues(Rvec) 反解矩阵）
  Rot_<cam> 写矩阵（read_camera 不读，仅为与 EasyMocap write_extri 习惯一致）
· 额外写 fisheye_<cam>=1 与 model_<cam>=OPENCV_FISHEYE，
  供三角化副本判断是否需要走 cv2.fisheye.undistortPoints

用法：
  python h4d_calib.py --seq-root <...>/016_mma4 --out <calib 目录> \
                      [--views 01,03,04,07,09,14] [--all-views]
"""
import os, sys, json, argparse, collections
import numpy as np
import cv2


def q2R(q):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵"""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def read_cameras_txt(path):
    cams = {}
    for l in open(path):
        if not l.strip() or l.startswith('#'):
            continue
        p = l.split()
        cams[int(p[0])] = dict(model=p[1], W=int(p[2]), H=int(p[3]),
                               fx=float(p[4]), fy=float(p[5]),
                               cx=float(p[6]), cy=float(p[7]),
                               dist=np.array([float(x) for x in p[8:]], float))
    return cams


def read_images_txt(path):
    lines = [l for l in open(path) if l.strip() and not l.startswith('#')]
    recs = []
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        recs.append(dict(img_id=int(p[0]),
                         q=np.array([float(x) for x in p[1:5]]),
                         t=np.array([float(x) for x in p[5:8]]),
                         cam_id=int(p[8]), name=p[9]))
    return recs


def average_pose(recs):
    """同一台相机的多个关键帧 -> 单组固定外参（四元数符号对齐后平均）"""
    qs = [r['q'] for r in recs]
    q0 = qs[0]
    qs = [q if np.dot(q, q0) > 0 else -q for q in qs]
    qm = np.mean(qs, 0); qm /= np.linalg.norm(qm)
    ts = np.array([r['t'] for r in recs])
    Rs = np.array([q2R(q) for q in qs])
    Rm = q2R(qm)
    return dict(q=qm, R=Rm, t=ts.mean(0),
                t_std=float(np.linalg.norm(ts.std(0))),
                R_std_deg=float(np.degrees(np.abs(Rs - Rm).max())),
                n=len(recs))


def _fs_write(fs, key, val):
    """复用 EasyMocap FileStorage.write 的格式（!!opencv-matrix）"""
    v = np.asarray(val, float)
    if v.ndim == 1:
        v = v[None, :]
    fs.write('{}: !!opencv-matrix\n'.format(key))
    fs.write('  rows: {}\n'.format(v.shape[0]))
    fs.write('  cols: {}\n'.format(v.shape[1]))
    fs.write('  dt: d\n')
    fs.write('  data: [{}]\n'.format(', '.join('{:.10f}'.format(x) for x in v.reshape(-1))))


def _fs_write_int(fs, key, val):
    fs.write('{}: {}\n'.format(key, int(val)))


def _fs_write_list(fs, key, vals):
    fs.write('{}:\n'.format(key))
    for v in vals:
        fs.write('  - "{}"\n'.format(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-root', required=True, help='如 <raw>/15_mma4/016_mma4')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14',
                    help='exo 机位号（两位零填充），默认实测推荐的 6 视角')
    ap.add_argument('--all-views', action='store_true')
    a = ap.parse_args()

    cm = os.path.join(a.seq_root, 'colmap', 'workplace')
    cams = read_cameras_txt(os.path.join(cm, 'cameras.txt'))
    recs = read_images_txt(os.path.join(cm, 'images.txt'))

    bycam = collections.defaultdict(list)
    for r in recs:
        bycam[r['cam_id']].append(r)

    # --- 机位筛选：只保留固定机架（id 3..22 = exo/cam01..cam20）---
    views = None if a.all_views else [v.zfill(2) for v in a.views.split(',')]
    report = []
    keep = {}
    for cid in sorted(bycam):
        name = bycam[cid][0]['name'].split('/')[0]      # 'cam01' / 'aria01' / 'mobile'
        if not (name.startswith('cam') and name[3:].isdigit()):
            report.append((cid, name, 'SKIP(非固定机架)'))
            continue
        idx = int(name[3:])
        vid = '%02d' % idx                     # ★ 机位键 = 两位零填充，与 frames/det/gt 目录名一致
        if idx != cid - 2:
            report.append((cid, name, 'WARN(id 与 exo 序号不符 id-2=%d)' % (cid - 2)))
        if views is not None and vid not in views:
            report.append((cid, name, 'SKIP(未选入)')); continue
        p = average_pose(bycam[cid])
        if p['t_std'] > 0.05 or p['R_std_deg'] > 1.0:
            report.append((cid, name, 'WARN(刚性存疑 t_std=%.4f R_std=%.3f)'
                           % (p['t_std'], p['R_std_deg'])))
        keep[vid] = dict(cam=cams[cid], pose=p)
        report.append((cid, name, 'KEEP n=%d t_std=%.5f R_std=%.4f'
                       % (p['n'], p['t_std'], p['R_std_deg'])))

    print('=== 机位处理报告 ===')
    for cid, name, msg in report:
        print('  colmap id %2d  %-8s %s' % (cid, name, msg))
    names = sorted(keep.keys())
    assert names, '没有保留任何机位'

    os.makedirs(a.out, exist_ok=True)
    fi = open(os.path.join(a.out, 'intri.yml'), 'w')
    fe = open(os.path.join(a.out, 'extri.yml'), 'w')
    _fs_write_list(fi, 'names', names)
    _fs_write_list(fe, 'names', names)
    for n in names:
        c = keep[n]['cam']; K = np.array([[c['fx'], 0, c['cx']],
                                          [0, c['fy'], c['cy']],
                                          [0, 0, 1]], float)
        _fs_write(fi, 'K_%s' % n, K)
        d = c['dist'][:4].reshape(1, 4)                 # 鱼眼 k1..k4，(1,4) 合法
        _fs_write(fi, 'dist_%s' % n, d)
        _fs_write_int(fi, 'H_%s' % n, c['H'])
        _fs_write_int(fi, 'W_%s' % n, c['W'])
        fi.write('fisheye_{}: 1\n'.format(n))
        fi.write('model_{}: "{}"\n'.format(n, c['model']))
        R = keep[n]['pose']['R']; t = keep[n]['pose']['t'].reshape(3, 1)
        rvec = cv2.Rodrigues(np.ascontiguousarray(R))[0].reshape(3, 1)
        _fs_write(fe, 'R_%s' % n, rvec)                 # ★ 罗德里格斯（read_camera 读它）
        _fs_write(fe, 'Rot_%s' % n, R)                  # 矩阵（read_camera 不读，留兼容）
        _fs_write(fe, 'T_%s' % n, t)
    fi.close(); fe.close()

    # --- 米制化矩阵与几何摘要（供导出/评估阶段使用）---
    _sp = os.path.join(cm, 'scale.npy')
    if os.path.exists(_sp):
        S = np.load(_sp)
    else:
        S = np.eye(4)          # ★非 exo_only 序列无 scale.npy：世界系已是米制 → 单位阵
        print('[WARN] 无 scale.npy → 按单位阵处理（假定世界系已米制）')
    np.save(os.path.join(a.out, 'scale_metric.npy'), S)
    cen = {}
    for n in names:
        R = keep[n]['pose']['R']; t = keep[n]['pose']['t']
        C = -R.T @ t                                  # COLMAP 系相机中心
        cen[n] = (S[:3, :3] @ C + S[:3, 3]).tolist()  # 米制系
    meta = dict(seq_root=os.path.abspath(a.seq_root), views=names,
                frame_glob='exo/<view>/images/%05d.jpg',
                n_fixed_rig=len(names), scale=float(np.linalg.norm(S[:3, 0])),
                camera_center_metric=cen,
                note='extri 为 COLMAP 世界系；米制化请用 scale_metric.npy (X_metric = S @ X_colmap)')
    with open(os.path.join(a.out, 'h4d_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print('\n[OK] 写出 %s/{intri.yml,extri.yml,scale_metric.npy,h4d_meta.json}' % a.out)
    print('     保留机位 %d 个: %s' % (len(names), ','.join(names)))


if __name__ == '__main__':
    main()
