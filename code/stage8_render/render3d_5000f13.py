#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""render3d_5000f13.py — 场1_1 前5000帧「13 节点 smpl3D + 棍 3D 直线」场景渲染

与 render3d_5000.py 同一套画法(改编自 longweapon/run30/scripts/render3d_30.py)，
只把节点集从 body25 换成我们的最终标签 13 节点。

【13 节点集】IDX13(b25) = [0,2,3,4,5,6,7,9,10,11,12,13,14]
    = 鼻子 + 双肩/肘/腕 + 双髋/膝/踝
    头部只留鼻子(去眼/耳), 脚上只留脚踝(去脚尖/脚跟), 去 neck/pelvis, 双髋保留
  数据源 final13_rtd5000of/pid{X}/keypoints3d/{f:06d}.json  ([{"keypoints3d":(13,4)}])
  注: 这 13 个点的 XYZ 与 emfit 的 body25 同索引完全一致, conf 是三角化 kept 视角的 2D conf 均值。

【重要】世界坐标(标定系) Y 轴向下为正 (鼻子 Y≈-1.49, 地面 Y=0)。
  竖直方向 = -Y。显示映射 (x, y, z) -> (x, z, -y)。搞反会头朝下。

输入:
  smpl 13节点: final13_rtd5000of/pid{X}/keypoints3d/{f:06d}.json
  棍 3D 直线:  stick3d_5000.json
输出: <out>/frames/f{f:06d}.png + <out>/render3d.mp4 + <out>/grid.png
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.ticker import MaxNLocator

BASE = '/root/autodl-tmp'

# 颜色约定沿用我们管线的语义 (操作记录 §一⑨): 绿 = pid0(黑方), 橙 = pid1(红方)
COLOR = {'red': '#e8402a', 'black': '#111111'}
C_P1 = '#00a651'   # pid0 绿
C_P2 = '#f0a020'   # pid1 橙

N_BODY = 13        # 13 节点标签

# 13 节点骨架 (与 rtd3_viz_v3.py 的 BONES13 一致):
#   0 鼻 | 1 右肩 2 右肘 3 右腕 | 4 左肩 5 左肘 6 左腕 | 7 右髋 8 右膝 9 右踝 | 10 左髋 11 左膝 12 左踝
BONES13 = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
           (1, 7), (4, 10), (7, 10),
           (7, 8), (8, 9), (10, 11), (11, 12)]

G = {}             # fork 继承的全局配置


def to_disp(p):
    """世界坐标 -> 显示坐标。竖直 = -Y。对点和方向向量都适用(线性变换)。"""
    p = np.asarray(p, dtype=float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def draw(ax, people, stick, skel, lims, title='', lw=2.2, ms=22):
    """people: [ (13,4) ndarray, ... ] ; stick: {color:{point,dir,t_lo,t_hi}}"""
    ax.clear()

    # 地面参考面 (世界 Y=0 -> 显示竖直 0)
    _fl = FLOOR[0] if FLOOR[0] else lims[:2]
    gx = np.linspace(_fl[0][0], _fl[0][1], 2)
    gy = np.linspace(_fl[1][0], _fl[1][1], 2)
    GX, GY = np.meshgrid(gx, gy)
    ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.07, color='#888888',
                    linewidth=0, zorder=0)

    for n, k in enumerate(people):
        col = C_P1 if n == 0 else C_P2
        D = to_disp(k[:, :3])
        ok = k[:, 3] > 0.0          # 13 点全部画 (conf=0 是纯先验补全帧, 位置仍有效)
        for (i, j) in skel:
            if ok[i] and ok[j]:
                a, b = D[i], D[j]
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        '-', c=col, lw=lw, alpha=0.95, solid_capstyle='round')
        m = D[ok]
        if len(m):
            ax.scatter(m[:, 0], m[:, 1], m[:, 2], s=ms, c=col,
                       depthshade=False, edgecolors='white', linewidths=0.4)

    for color, key in [('red', '0'), ('black', '1')]:
        if not stick or key not in stick:
            continue
        s = stick[key]
        pt = to_disp(s['point'])
        d = to_disp(s['dir'])
        t0 = s.get('t_lo', -s['length'] / 2)
        t1 = s.get('t_hi', s['length'] / 2)
        a, b = pt + t0 * d, pt + t1 * d
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                '-', c=COLOR[color], lw=3.6, alpha=0.8, solid_capstyle='round')
        ax.scatter([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                   s=46, c=COLOR[color], depthshade=False,
                   marker='o', edgecolors='white', linewidths=0.6)

    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_xlabel('X (m)', fontsize=9, labelpad=-2)
    ax.set_ylabel('Z (m)', fontsize=9, labelpad=-2)
    ax.set_zlabel('height (m)', fontsize=9, labelpad=-2)
    ax.tick_params(labelsize=7, pad=0)
    # 场地是个细长条时 X 轴在屏幕上很短, 默认刻度会挤成一团 -> 限制刻度数
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.view_init(elev=14, azim=-66)
    sp = [(lims[i][1] - lims[i][0]) for i in range(3)]
    ax.set_box_aspect((sp[0], sp[1], sp[2])) if ASPECT[0] is None \
        else ax.set_box_aspect(ASPECT[0])
    if title:
        ax.set_title(title, fontsize=13, pad=2)


ASPECT = [None]
FLOOR = [None]


def auto_lims(allpts, pad=0.06):
    """allpts: list of (N,3) display-coord arrays。用分位数抗离群。"""
    A = np.vstack([p for p in allpts if len(p)])
    lo = np.percentile(A, 0.5, axis=0)
    hi = np.percentile(A, 99.5, axis=0)
    span_xy = np.maximum(hi[:2] - lo[:2], 0.5)
    lims = []
    for i in range(2):
        lims.append((lo[i] - span_xy[i] * pad, hi[i] + span_xy[i] * pad))
    h_hi = np.percentile(A[:, 2], 98.0)
    h_hi = max(h_hi * 1.08, 1.9)
    lims.append((-0.12, h_hi))
    return lims


def kfile(pid, fr):
    return os.path.join(G['smpl'], 'pid%d' % pid, 'keypoints3d', '%06d.json' % fr)


def load_k(pid, fr):
    fp = kfile(pid, fr)
    if not os.path.exists(fp):
        return None
    try:
        dd = json.load(open(fp))
    except Exception:
        return None
    if not dd:
        return None
    return np.array(dd[0]['keypoints3d'], dtype=float)[:N_BODY]


def collect_points(frs):
    """收集显示坐标点(13 节点 + 棍两端)用于定取景范围。"""
    pts = []
    for fr in frs:
        for pid in (0, 1):
            k = load_k(pid, fr)
            if k is not None and len(k) > 2:
                pts.append(to_disp(k[:, :3]))
        st = G['stick'].get(str(fr))
        if st:
            for v in st.values():
                pp = to_disp(v['point']); dv = to_disp(v['dir'])
                pts.append(np.stack([pp + v.get('t_lo', 0) * dv,
                                     pp + v.get('t_hi', 0) * dv]))
    return pts


def render_one(fr):
    people = []
    for pid in (0, 1):
        k = load_k(pid, fr)
        if k is not None:
            people.append(k)
    st = G['stick'].get(str(fr))
    if not people and not st:
        return None
    fig = plt.figure(figsize=(G['figw'], G['figh']), dpi=G['dpi'])
    ax = fig.add_subplot(111, projection='3d')
    draw(ax, people, st, G['skel'], G['lims'], title='frame %d' % fr)
    out = os.path.join(G['fdir'], 'f%06d.png' % fr)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)
    fig.savefig(out)
    plt.close(fig)
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smpl', default=BASE + '/final13_rtd5000of')
    ap.add_argument('--stick3d', default=BASE + '/stick3d_5000.json')
    ap.add_argument('--out', default=BASE + '/render3d_5000f13')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=4999)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--aspect', default=None,
                    help='显示盒比例 "ax,ay,az"（如 1,1,1=等比例）。默认按坐标轴跨度，'
                         '场地大时会把人压扁成小点')
    ap.add_argument('--floor', default=None,
                    help='x0,x1,z0,z1（世界 mm）地板参考面的【真实场地】范围；'
                         '不给则等于画布范围。与 --bounds 解耦后：画布可放大到人，'
                         '地板仍按真实场地画。')
    ap.add_argument('--bounds', default=None,
                    help='x0,x1,z0,z1（世界 mm）；给定则用它当【真实场地范围】，'
                         '替代按数据自动缩放（数据自动缩放会永远把框画成"围着人"）')
    ap.add_argument('--figw', type=float, default=8.0)
    ap.add_argument('--figh', type=float, default=5.6)
    ap.add_argument('--dpi', type=int, default=110)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    fdir = os.path.join(a.out, 'frames')
    os.makedirs(fdir, exist_ok=True)

    stick = {}
    if os.path.exists(a.stick3d):
        stick = json.load(open(a.stick3d))
    print('[render] stick3d 帧数 = %d' % len(stick), flush=True)

    G.update(smpl=a.smpl, stick=stick, fdir=fdir, skel=BONES13,
             figw=a.figw, figh=a.figh, dpi=a.dpi)

    # 先抽查一帧确认 13 节点读得到
    k0 = load_k(0, a.start)
    print('[render] 节点数抽查 pid0 f%d -> %s'
          % (a.start, None if k0 is None else k0.shape), flush=True)

    sample = list(range(a.start, a.end + 1, 25))
    lims = auto_lims(collect_points(sample))
    if a.floor:
        _f = [float(x) for x in a.floor.split(',')]
        FLOOR[0] = ((_f[0], _f[1]), (_f[2], _f[3]))
        print('[render] 地板范围(真实场地) X=(%.2f,%.2f) Y=(%.2f,%.2f) m'
              % (_f[0] / 1000, _f[1] / 1000, _f[2] / 1000, _f[3] / 1000), flush=True)
    if a.aspect:
        ASPECT[0] = tuple(float(x) for x in a.aspect.split(','))
        print('[render] 显示比例 %s' % (ASPECT[0],), flush=True)
    if a.bounds:
        _b = [float(x) for x in a.bounds.split(',')]
        lims[0] = (_b[0], _b[1])          # 世界 X
        lims[1] = (_b[2], _b[3])          # 世界 Z（显示映射 (x,y,z)->(x,z,-y)）
        print('[render] 用 --bounds 覆盖场地范围 X=(%.2f,%.2f) Y=(%.2f,%.2f) m'
              % (_b[0] / 1000, _b[1] / 1000, _b[2] / 1000, _b[3] / 1000), flush=True)
    G['lims'] = lims
    print('[render] 全局范围 X=%s Y=%s 竖直=%s'
          % (np.round(lims[0], 2), np.round(lims[1], 2), np.round(lims[2], 2)),
          flush=True)

    frs = list(range(a.start, a.end + 1))
    done = 0
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for r in pool.imap_unordered(render_one, frs, chunksize=8):
                done += 1
                if done % 1000 == 0:
                    print('[render] %d/%d' % (done, len(frs)), flush=True)
    else:
        for fr in frs:
            render_one(fr)
            done += 1
    print('[render] 渲染完成 %d 帧 -> %s' % (done, fdir), flush=True)

    saved = [fr for fr in frs if os.path.exists(os.path.join(fdir, 'f%06d.png' % fr))]
    lst = os.path.join(a.out, 'list.txt')
    with open(lst, 'w') as f:
        for fr in saved:
            f.write("file '%s'\n" % os.path.join(fdir, 'f%06d.png' % fr))
    mp4 = os.path.join(a.out, 'render3d.mp4')
    os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -r %d -f concat '
              '-safe 0 -i %s -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
              '-c:v libx264 -pix_fmt yuv420p %s' % (a.fps, lst, mp4))
    print('[render] 视频 -> %s (%s)' % (mp4, os.path.exists(mp4)), flush=True)

    if a.grid:
        k = a.grid
        step = max(len(saved) // (k * k), 1)
        pick = saved[::step][:k * k]
        fg, axes = plt.subplots(k, k, figsize=(4.0 * k, 3.5 * k), dpi=110,
                                subplot_kw={'projection': '3d'})
        for ax_, fr in zip(axes.ravel(), pick):
            people = [x for x in (load_k(p, fr) for p in (0, 1)) if x is not None]
            draw(ax_, people, stick.get(str(fr)), BONES13, lims,
                 title='f%d' % fr, lw=1.6, ms=13)
        for ax_ in axes.ravel()[len(pick):]:
            ax_.axis('off')
        fg.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01,
                           wspace=0.02, hspace=0.06)
        gp = os.path.join(a.out, 'grid.png')
        fg.savefig(gp)
        plt.close(fg)
        print('[render] 拼图 -> %s' % gp, flush=True)
    print('[render] ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
