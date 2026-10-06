#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充插图：① 端到端流程图 ② 三角化受控消融 ③ 帧级筛选负面结果"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

B = '/workshop/Lym/combat3d'
OUT = f'{B}/figs'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10.5, 'figure.dpi': 170, 'savefig.bbox': 'tight',
                     'axes.grid': True, 'grid.alpha': 0.3})

# ---------- Fig 7：端到端流程图 ----------
fig, ax = plt.subplots(figsize=(9.4, 5.6))
ax.axis('off')
C_UP = '#e8f0fe'; C_LAB = '#fff4e0'; C_TR = '#e6f4ea'; C_TX = '#fce8e6'; C_EV = '#f1e8fd'

def box(x, y, w, h, txt, col, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=col, ec='#555', lw=1.1, zorder=2))
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', zorder=3)

def arrow(x1, y1, x2, y2, style='-|>', col='#666'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=col, lw=1.3, zorder=1))

ax.text(4.7, 5.35, 'Combat3D: end-to-end pipeline', ha='center', fontsize=13, fontweight='bold')

# 上游（场景相关）
box(0.2, 4.05, 1.85, 0.85, 'Multi-view video\n(20 GoPro, in the wild)', C_UP, 8.6)
box(2.35, 4.05, 1.75, 0.85, 'Automatic SfM\n(COLMAP)\nno manual calibration', C_UP, 8.6)
box(4.40, 4.05, 1.70, 0.85, 'Detection boxes\n★ scene-specific', C_LAB, 8.6)
box(6.40, 4.05, 1.60, 0.85, 'Identity\n★ scene-specific', C_LAB, 8.6)
box(8.30, 4.05, 1.80, 0.85, '2D pose\n(ViTPose)', C_UP, 8.6)
for x in (2.05, 4.10, 6.10, 8.00):
    arrow(x, 4.475, x + 0.30, 4.475)
arrow(1.125, 4.05, 1.125, 3.45)
box(0.2, 2.72, 1.85, 0.73, '2D + camera params', C_UP, 8.6)
arrow(5.25, 4.05, 5.25, 3.45)
box(4.40, 2.72, 3.6, 0.73, '★ Combat3D-Label: triangulation (GENERAL)\n'
                          'view-subset enum. + border/box-touch weight. + λ temporal reg.',
    C_TR, 7.9)
arrow(7.0, 2.72, 7.0, 2.25)
box(5.30, 1.55, 3.20, 0.70, 'Zero-manual 3D labels', C_TR, 9, True)
arrow(4.40, 1.90, 3.10, 1.90)
box(0.90, 1.55, 2.20, 0.70, 'Fine-tune\nMotionBERT (mono)', C_TX, 8.6, True)
arrow(0.90, 1.55, 0.90, 1.05)
box(0.15, 0.22, 3.40, 0.78, 'Combat3D-Mono\nshape 32.6 mm  |  abs. loc. 9 mm', C_EV, 8.6, True)
arrow(6.60, 1.55, 6.60, 1.05)
box(3.95, 0.22, 2.30, 0.78, 'Evaluation\n(SA-Metric)', C_EV, 8.6)
box(6.70, 0.22, 3.35, 0.78, 'Independent mocap\n(CMU Panoptic) PA 76.5 mm', C_EV, 8.2)
arrow(8.40, 1.55, 8.40, 1.05)

ax.text(0.25, 3.66, '★ = scene-specific layer (pluggable)', fontsize=8.0,
        color='#b35c00', style='italic')
ax.text(8.15, 3.66, 'general layers', fontsize=8.0, color='#2a6f97', style='italic')
ax.set_xlim(-0.1, 10.3); ax.set_ylim(0, 5.7)
plt.savefig(f'{OUT}/fig7_pipeline.png'); plt.close()

# ---------- Fig 8：三角化受控消融 ----------
fig, ax = plt.subplots(figsize=(6.6, 4.2))
x = np.arange(2); w = 0.36
ours = [9.22, 20.47]; naive = [9.22, 36.75]
b1 = ax.bar(x - w/2, ours, w, label='Ours (subset enum + weighting)', color='#4c72b0')
b2 = ax.bar(x + w/2, naive, w, label='Naive (all views, equal weight)', color='#dd8452')
ax.bar_label(b1, fmt='%.2f', fontsize=9.5); ax.bar_label(b2, fmt='%.2f', fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels(['clean upstream\n(official SMPL-projection 2D)',
                    'dirty upstream\n(self-built detector 2D)'], fontsize=9)
ax.set_ylabel('median reprojection residual (px)')
ax.set_title('Triangulation ablation (same takes, same upstream 2D)', fontsize=11)
ax.legend(fontsize=8.5); ax.set_ylim(0, 44)
ax.annotate('', xy=(1 - w/2, 22.5), xytext=(1 + w/2, 22.5),
            arrowprops=dict(arrowstyle='<->', color='#c44'))
ax.text(1, 25, '−44%', ha='center', fontsize=11, color='#c44', fontweight='bold')
ax.text(0, 11.5, 'identical\n(nothing to fix)', ha='center', fontsize=8.5, color='#666')
plt.savefig(f'{OUT}/fig8_tri_ablation.png'); plt.close()

# ---------- Fig 9：帧级筛选负面结果 ----------
fig, ax = plt.subplots(figsize=(7.0, 4.2))
arms = ['A\nofficial GT', 'B\nour labels', 'C\nfully self-built']
off = [56.7, 18.4, 151.7]; on = [72.2, 56.4, 255.5]
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w/2, off, w, label='frame filter OFF', color='#55a868')
b2 = ax.bar(x + w/2, on, w, label='frame filter ON (10 px)', color='#c44e52')
ax.bar_label(b1, fmt='%.1f', fontsize=9.5); ax.bar_label(b2, fmt='%.1f', fontsize=9.5)
ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=9.5)
ax.set_ylabel('val MPJPE (mm)')
ax.set_title('Frame-level quality filtering is a NET LOSS at this data scale', fontsize=10.5)
ax.legend(fontsize=9); ax.set_ylim(0, 300)
ax.text(1, 200, 'every arm gets worse:\nthe marginal return of more data\nexceeds that of cleaner labels',
        ha='center', fontsize=8.7, color='#c44e52', style='italic')
plt.savefig(f'{OUT}/fig9_filter_negative.png'); plt.close()

for f in sorted(os.listdir(OUT)):
    if f.startswith(('fig7', 'fig8', 'fig9')):
        print('  ', f, os.path.getsize(os.path.join(OUT, f)) // 1024, 'KB')
