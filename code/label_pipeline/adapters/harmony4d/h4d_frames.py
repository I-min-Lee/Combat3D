#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h4d_frames.py — Harmony4D 帧 → 管线要求的帧目录布局

管线约定（见 stage2_pose2d/vp_rtd_batch.py:59,121）：
    {FRAMES_DIR}/{view}/%06d.png
而 Harmony4D 给的是：{seq_root}/exo/cam{NN}/images/%05d.jpg（**已抽好的 jpg**）

本脚本把 jpg **软链**成管线要的 %06d.png（不复制、不占盘；cv2.imread 按内容
识别编码，扩展名不匹配无影响）。若需真实复制用 --copy。

★ 帧号映射：数据集 00001..00741 → 管线 fidx 1..741（与 bbox 文件名一致）

用法：
  python h4d_frames.py --seq-root <...>/016_mma4 --frames-root /workshop/Lym/combat3d/frames \
                       --match 15 --seg 1 --views 01,03,04,07,09,14
"""
import os, glob, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-root', required=True)
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--match', type=int, required=True)
    ap.add_argument('--seg', type=int, required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--copy', action='store_true', help='真复制而不是软链（占盘）')
    a = ap.parse_args()

    views = [v.zfill(2) for v in a.views.split(',')]
    print('=== 帧链接报告 (match=%d seg=%d) ===' % (a.match, a.seg))
    total = 0
    for v in views:
        src_dir = os.path.join(a.seq_root, 'exo', 'cam%s' % v, 'images')
        dst_dir = os.path.join(a.frames_root, str(a.match), str(a.seg), v)
        os.makedirs(dst_dir, exist_ok=True)
        srcs = sorted(glob.glob(os.path.join(src_dir, '*.jpg')))
        if not srcs:
            print('  view %s : ❌ 无源图 (%s)' % (v, src_dir)); continue
        n, miss = 0, 0
        for sp in srcs:
            fi = int(os.path.basename(sp).split('.')[0])       # 1..741
            dp = os.path.join(dst_dir, '%06d.png' % fi)
            if os.path.exists(dp) or os.path.islink(dp):
                n += 1; continue
            if a.copy:
                import shutil; shutil.copy2(sp, dp)
            else:
                try:
                    os.symlink(os.path.abspath(sp), dp)
                except OSError:
                    # 软链不可用（跨设备/无权限）→ 退回硬链，再退回复制
                    try:
                        os.link(sp, dp)
                    except OSError:
                        import shutil; shutil.copy2(sp, dp)
            n += 1
        # 连续性检查：帧号是否 1..N 无缺口
        idx = sorted(int(os.path.basename(p).split('.')[0]) for p in glob.glob(os.path.join(dst_dir, '*.png')))
        gaps = [i for i in range(idx[0], idx[-1] + 1) if i not in set(idx)] if idx else []
        print('  view %s : %d 帧  (范围 %s..%s)  缺口=%s  %s'
              % (v, n, idx[0] if idx else '-', idx[-1] if idx else '-',
                 len(gaps), ('⚠️ ' + str(gaps[:5])) if gaps else '✅'))
        total += n
    print('[OK] 共 %d 帧  根目录 %s/%d/%d' % (total, a.frames_root, a.match, a.seg))


if __name__ == '__main__':
    main()
