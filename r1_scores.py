import matplotlib.pyplot as plt
import numpy as np

# ================= 数据输入区 =================
k_values = [2, 4, 6, 8]

# 请将下面的数字替换为你真实的实验数据
# (87.03 和 88.67 是你 k=4 时的真实数据，其他的我先随便写了占位)
r1_scores = [77.03, 88.03, 81.46, 74.47]
ap_scores = [78.79, 85.45, 80.42, 76.69]
# ============================================

# 设置全局字体和大小
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# 创建画布
fig, ax = plt.subplots(figsize=(7, 5))

# 绘制折线：R@1 用实线+圆点，AP 用虚线+方块
ax.plot(k_values, r1_scores, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8, label='R@1')
ax.plot(k_values, ap_scores, marker='s', linestyle='--', color='#ff7f0e', linewidth=2.5, markersize=8, label='AP')

# 设置坐标轴标签 (英文标注更显学术专业)
ax.set_xlabel('Number of LPN Partitions ($k$)', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')

# 设置 x 轴刻度，严格对齐 2, 4, 6, 8
ax.set_xticks(k_values)

# 增加网格线背景，辅助读数
ax.grid(True, linestyle='--', alpha=0.6)

# 添加图例，放置在右上角或右下角
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

# 调整布局以防边缘被裁剪
plt.tight_layout()

# 保存为 300dpi 的高清图片，直接可用作论文插图
plt.savefig('lpn_ablation_line_chart.png', dpi=300, bbox_inches='tight')
plt.show()