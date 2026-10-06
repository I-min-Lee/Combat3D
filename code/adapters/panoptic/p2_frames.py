#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p2_frames.py — Panoptic HD mp4 → 抽帧（%06d.png，**用绝对帧号**）

Panoptic 的 `body3DScene_%08d.json` 与 HD 视频帧**按索引一一对应**（官方 hdImgsExtractor 也是整段抽），
所以这里保持**绝对帧号**命名：帧 N 的 GT 就是 `body3DScene_%08d.json`(N)。
整条下游（--start/--end、gt json、bbox npy）都用同一个绝对帧号，不做任何重映射。

★这个容器（lym_isaac）**没有 ffmpeg**，所以默认走 cv2.VideoCapture（自带的 ffmpeg 后端）；
  宿主有 ffmpeg 时可用 --backend ffmpeg 换回命令行（更快）。

用法：
  python p2_frames.py --videos <dir> --out <frames 根> --views 00_01,... \
      --start 4256 --n 600 [--backend auto|ffmpeg|cv2] [--jobs 5]
输出：{out}/<view>/%06d.png（%06d 是绝对帧号）
"""
import os, argparse, subprocess, shutil, multiprocessing as mp


def by_ffmpeg(v, src, od, start, n, qscale):
    end = start + n - 1
    cmd = ['ffmpeg', '-v', 'error', '-i', src,
           '-vf', "select='between(n\\,%d\\,%d)'" % (start, end),
           '-vsync', '0', '-start_number', str(start), '-q:v', str(qscale),
           os.path.join(od, '%06d.png')]
    subprocess.run(cmd, check=True)


def by_cv2(v, src, od, start, n, qscale):
    import cv2
    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), '打不开 %s' % src
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    q = max(1, 31 - qscale * 3)          # qscale(1..5) -> JPEG 质量
    for k in range(n):
        ok, fr = cap.read()
        if not ok:
            print('    [warn] %s 在第 %d 帧提前结束' % (v, start + k)); break
        cv2.imwrite(os.path.join(od, '%06d.png' % (start + k)), fr,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cap.release()


def worker(args):
    return _do(*args)


def _do(v, src, out, start, n, backend, qscale):
    od = os.path.join(out, v)
    os.makedirs(od, exist_ok=True)
    if backend == 'ffmpeg':
        by_ffmpeg(v, src, od, start, n, qscale)
    else:
        by_cv2(v, src, od, start, n, qscale)
    k = len([f for f in os.listdir(od) if f.endswith('.png')])
    return v, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True, help='视频目录（里面是 hd_<view>.mp4）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', required=True)
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--backend', default='auto', choices=['auto', 'ffmpeg', 'cv2'])
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--qscale', type=int, default=3)
    a = ap.parse_args()
    views = [v.strip() for v in a.views.split(',')]

    backend = a.backend
    if backend == 'auto':
        backend = 'ffmpeg' if shutil.which('ffmpeg') else 'cv2'
    print('[p2_frames] backend=%s  start=%d n=%d views=%s' % (backend, a.start, a.n, views), flush=True)

    jobs = []
    for v in views:
        src = os.path.join(a.videos, 'hd_%s.mp4' % v)
        assert os.path.exists(src), '缺视频 %s' % src
        jobs.append((v, src, a.out, a.start, a.n, backend, a.qscale))
    with mp.Pool(min(a.jobs, len(jobs))) as pool:
        for v, k in pool.imap_unordered(worker, jobs):
            print('    %s -> %d 帧' % (v, k), flush=True)
    print('[p2_frames] 完成 -> %s' % a.out)


if __name__ == '__main__':
    main()
