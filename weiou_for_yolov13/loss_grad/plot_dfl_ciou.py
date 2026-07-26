#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 DFL 与 Box 损失的梯度范数比及余弦相似度折线图
AAAI 会议论文风格
"""

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ======================== 全局样式设置 ========================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# ======================== 读取 CSV 数据 ========================
csv_path = r'E:\envs\for_sota\loss_grad\crowdhuman_dfl_ciou_grad_stats_rank0.csv'

steps = []
norm_ratios = []
cosine_sims = []

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row['step']))
        norm_ratios.append(float(row['dfl_box_norm_ratio']))
        cosine_sims.append(float(row['grad_cosine']))

steps = np.array(steps)
norm_ratios = np.array(norm_ratios)
cosine_sims = np.array(cosine_sims)

# ======================== 创建图表 ========================
fig, ax1 = plt.subplots(figsize=(10, 5.5))

# ---- 颜色方案 (AAAI 风格: 沉稳学术风) ----
color_ratio = '#2166AC'    # 深蓝 — 范数比
color_cosine = '#B2182B'   # 深红 — 余弦相似度

# ---- 左轴: 梯度范数比 (dfl / box) ----
line1, = ax1.plot(steps, norm_ratios,
                   color=color_ratio, linewidth=1.2, alpha=0.85,
                   label=r'$\|\nabla \mathcal{L}_{\mathrm{DFL}}\| \;/\; \|\nabla \mathcal{L}_{\mathrm{Box}}\|$')
ax1.set_xlabel('Training Step', fontweight='medium')
ax1.set_ylabel('Gradient Norm Ratio (DFL / Box)', color=color_ratio, fontweight='medium')
ax1.tick_params(axis='y', labelcolor=color_ratio, colors=color_ratio)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))

# ---- 右轴: 梯度余弦相似度 ----
ax2 = ax1.twinx()
line2, = ax2.plot(steps, cosine_sims,
                   color=color_cosine, linewidth=1.2, alpha=0.85,
                   label='Gradient Cosine Similarity')
ax2.set_ylabel('Gradient Cosine Similarity', color=color_cosine, fontweight='medium')
ax2.tick_params(axis='y', labelcolor=color_cosine, colors=color_cosine)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))

# ---- 网格 ----
ax1.grid(True, which='major', axis='both')
ax1.set_axisbelow(True)

# ---- 图例 ----
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', frameon=True,
           fancybox=True, framealpha=0.9, edgecolor='#cccccc')

# ---- 标题 ----
# ax1.set_title('Gradient Statistics of DFL and Box Losses During Training',
#               fontweight='bold', pad=12)

# ---- X 轴美化 ----
ax1.set_xlim(steps[0], steps[-1])
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=12))
ax1.tick_params(axis='x', length=4, width=0.8)

# ---- Y 轴范围微调 ----
y1_min, y1_max = norm_ratios.min(), norm_ratios.max()
y1_pad = (y1_max - y1_min) * 0.08
ax1.set_ylim(y1_min - y1_pad, y1_max + y1_pad)

y2_min, y2_max = cosine_sims.min(), cosine_sims.max()
y2_pad = (y2_max - y2_min) * 0.08
ax2.set_ylim(y2_min - y2_pad, y2_max + y2_pad)

# ---- 参考线: 余弦相似度 = 0 ----
# 如果余弦相似度接近 0 则不画，否则画一条淡灰色虚线
if y2_min < 0 < y2_max:
    ax2.axhline(y=0, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)

# ---- 添加统计注释 ----
stats_text = (
    f"Norm Ratio — mean: {norm_ratios.mean():.3f}  |  std: {norm_ratios.std():.3f}\n"
    f"Cosine Sim — mean: {cosine_sims.mean():.3f}  |  std: {cosine_sims.std():.3f}"
)
ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes,
         fontsize=8, fontfamily='monospace', color='#555555',
         verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#cccccc', alpha=0.85))

# ---- 保存 ----
fig.tight_layout()
output_path = r'E:\envs\for_sota\loss_grad\crowdhuman_dfl_ciou_grad_stats_rank0.png'
fig.savefig(output_path, format='png')
# 同时保存一份 PDF (矢量图，适合论文投稿)
pdf_path = r'E:\envs\for_sota\loss_grad\crowdhuman_dfl_ciou_grad_stats_rank0.pdf'
fig.savefig(pdf_path, format='pdf')

print(f'图片已保存至: {output_path}')
print(f'PDF 已保存至: {pdf_path}')
plt.close(fig)
