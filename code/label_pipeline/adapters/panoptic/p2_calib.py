#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p2_calib.py — CMU Panoptic 标定 → EasyMocap 风格 intri.yml / extri.yml

Panoptic 的 `calibration_<seq>.json` 直接给每台相机的 K / distCoef(5) / R / t（**cm 世界系**），
不需要像 Harmony4D 那样从稀疏 SfM 关键帧平均。要点：

· **畸变模型是 OpenCV 针孔 + 5 参数**(k1,k2,p1,p2,k3)，**不是鱼眼**。
  → intri.yml 里**不写** `fisheye_<v>` 键；`tri_h4d.py` 的 `undistort()` 见 `fisheye` 缺省 False
    就走 `cv2.undistortPoints(K, dist)`（正确），鱼眼分支不会误用。
· `t` 的定义就是 X_cam = R·X_world + t，与 EasyMocap 的 P=K[R|T] 一致（和 Harmony4D 同）。
· ★单位是 **cm**：所以 scale_metric.npy = 0.01·I（cm→m）。
  `h4d_metrics.py` 取 sc=||S[:3,0]||=0.01，再 ×1000 得 mm → cm×10 = mm ✓
· GT 的 3D 与标定**同系**（都来自 Panoptic），所以 gt3d_colmap 就直接写原始坐标，不做变换。

用法：
  python p2_calib.py --calib <calibration_<seq>.json> --out <calib 目录> \
      --views 00_01,00_06,00_13,00_16,00_17
"""
import os, json, argparse
import numpy as np
import cv2


def _fs_write(fs, key, val):
    v = np.asarray(val, float)
    if v.ndim == 1:
        v = v[None, :]
    fs.write('{}: !!opencv-matrix\n'.format(key))
    fs.write('  rows: {}\n'.format(v.shape[0]))
    fs.write('  cols: {}\n'.format(v.shape[1]))
    fs.write('  dt: d\n')
    fs.write('  data: [{}]\n'.format(', '.join('{:.10f}'.format(x) for x in v.reshape(-1))))


def _fs_int(fs, key, val):
    fs.write('{}: {}\n'.format(key, int(val)))


def _fs_list(fs, key, vals):
    fs.write('{}:\n'.format(key))
    for v in vals:
        fs.write('  - "{}"\n'.format(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True, help='calibration_<seq>.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', required=True, help='如 00_01,00_06,00_13,00_16,00_17')
    a = ap.parse_args()

    calib = json.load(open(a.calib))
    cams = {c['name']: c for c in calib['cameras']}
    views = [v.strip() for v in a.views.split(',')]
    missing = [v for v in views if v not in cams]
    assert not missing, '标定里没有这些相机: %s' % missing
    for v in views:
        assert cams[v]['type'] == 'hd', '%s 不是 hd 相机' % v

    os.makedirs(a.out, exist_ok=True)
    fi = open(os.path.join(a.out, 'intri.yml'), 'w')
    fe = open(os.path.join(a.out, 'extri.yml'), 'w')
    _fs_list(fi, 'names', views)
    _fs_list(fe, 'names', views)
    print('=== Panoptic 标定 (%s) ===' % calib.get('calibDataSource', '?'))
    for v in views:
        c = cams[v]
        K = np.array(c['K'], float).reshape(3, 3)
        W, H = c['resolution']
        dist = np.array(c['distCoef'], float).reshape(1, -1)
        _fs_write(fi, 'K_%s' % v, K)
        _fs_write(fi, 'dist_%s' % v, dist)
        _fs_int(fi, 'H_%s' % v, H)
        _fs_int(fi, 'W_%s' % v, W)
        fi.write('model_{}: "OPENCV"\n'.format(v))            # ★不写 fisheye_ → 走针孔分支
        R = np.array(c['R'], float).reshape(3, 3)
        t = np.array(c['t'], float).reshape(3, 1)
        rvec = cv2.Rodrigues(np.ascontiguousarray(R))[0].reshape(3, 1)
        _fs_write(fe, 'R_%s' % v, rvec)
        _fs_write(fe, 'Rot_%s' % v, R)
        _fs_write(fe, 'T_%s' % v, t)
        C = -R.T @ t.reshape(3)
        print('  %-6s %dx%d fx=%.1f cx=%.1f cy=%.1f | dist=%s | 相机中心(cm)=%s'
              % (v, W, H, K[0, 0], K[0, 2], K[1, 2], np.round(dist.ravel(), 3), np.round(C, 1)))
    fi.close(); fe.close()

    # ★cm -> m 的相似变换（h4d_metrics 会读它的 3×3 模长当 met 换算系数）
    S = np.eye(4) * 1.0
    S[:3, :3] *= 0.01
    np.save(os.path.join(a.out, 'scale_metric.npy'), S)
    meta = dict(dataset='CMU Panoptic', calib=os.path.abspath(a.calib), views=views,
                unit='cm', scale_to_meter=0.01,
                note='intri/extri 直接来自 Panoptic；GT 3D 与标定同系，故 gt3d_colmap=原始坐标')
    json.dump(meta, open(os.path.join(a.out, 'p2_meta.json'), 'w'), indent=2)
    print('\n[OK] -> %s/{intri.yml,extri.yml,scale_metric.npy,p2_meta.json}（scale=0.01 cm→m）' % a.out)


if __name__ == '__main__':
    main()
