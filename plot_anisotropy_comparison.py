#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
各向异性力学响应对比图生成脚本
用于论文 Fig. 5: Haptic Feedback Force Analysis
"""

import matplotlib
matplotlib.use('Agg') # 使用非交互式后端，避免无GUI环境报错

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体和风格
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2
rcParams['lines.linewidth'] = 2.0
rcParams['figure.dpi'] = 300

def load_force_data(filename):
    times = []
    forces = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        f = float(parts[1])
                        times.append(t)
                        forces.append(f)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Error: 文件 {filename} 未找到！")
        return np.array([]), np.array([])
    return np.array(times), np.array(forces)

# 加载数据
t_x, f_x = load_force_data('TetgenFEM/force_data_x.txt')
t_y, f_y = load_force_data('TetgenFEM/force_data_y.txt')

if len(t_x) == 0 or len(t_y) == 0:
    print("数据不足，无法绘图。请先运行模拟程序生成 force_data_x.txt 和 force_data_y.txt")
    exit(1)

# 对齐时间轴
t_x = t_x - t_x[0]
t_y = t_y - t_y[0]

# 统计
peak_x = np.max(f_x)
peak_y = np.max(f_y)
peak_time_x = t_x[np.argmax(f_x)]
peak_time_y = t_y[np.argmax(f_y)]

print(f"X Peak: {peak_x:.2f} N")
print(f"Y Peak: {peak_y:.2f} N")
print(f"Ratio: {peak_x/peak_y:.2f}")

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t_x, f_x, 'r-o', label='X-Axis (Stiff, E_x=10e6)', 
        markersize=4, markevery=max(1, len(t_x)//20), linewidth=2, alpha=0.8)
ax.plot(t_y, f_y, 'b-s', label='Y-Axis (Soft, E_y=1e6)', 
        markersize=4, markevery=max(1, len(t_y)//20), linewidth=2, alpha=0.8)

# 标记峰值
ax.plot(peak_time_x, peak_x, 'ro', markersize=10, markerfacecolor='none', markeredgewidth=2)
ax.plot(peak_time_y, peak_y, 'bs', markersize=10, markerfacecolor='none', markeredgewidth=2)

ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Reaction Force (N)', fontweight='bold')
ax.set_title('Anisotropic Material Response', fontweight='bold', pad=15)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='best', framealpha=0.9)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# 添加文本框
stats = f'Peak Ratio: {peak_x/peak_y:.2f}x\nDiff: {(peak_x-peak_y)/peak_y*100:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, stats, transform=ax.transAxes, verticalalignment='top', bbox=props)

plt.tight_layout()
output_file = 'anisotropy_result.png'
plt.savefig(output_file)
print(f"图表已保存至: {output_file}")
