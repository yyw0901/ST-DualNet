import matplotlib.pyplot as plt

# ==========================================
# 1. 全局字体与样式设置 (学术规范)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.5

# ==========================================
# 2. 准备数据 (仅包含 Rank-1, Rank-5, Rank-10)
# ==========================================
# 横坐标：严格对应你拥有的数据点
ranks = [1, 5, 10]

# ⚠️ 注意：列表中第一个数值是你论文里真实的 R@1 数据。
# 请将列表中第二、第三个数值（目前是随便写的占位符），替换为你测试跑出来的真实的 Rank-5 和 Rank-10 数据！
baseline_data   = [72.37, 87.56, 91.29]  # ResNet50
dualbranch_data = [80.80, 93.64, 96.26]  # ResNet50 + DINOv2
multiscale_data = [82.55, 94.65, 96.91]  # + Multi-scale
final_data      = [83.39, 93.25, 95.30]  # + Re-ranking (最终版本)

# ==========================================
# 3. 绘制曲线
# ==========================================
fig, ax = plt.subplots(figsize=(7, 5))

# 只有三个点，所以我稍微把 marker (标记) 调大了一点，让图表显得不那么空
ax.plot(ranks, baseline_data, marker='v', color='#377EB8', linestyle='--', linewidth=2, markersize=9, label='Baseline (ResNet50)')
ax.plot(ranks, dualbranch_data, marker='s', color='#4DAF4A', linestyle='-.', linewidth=2, markersize=9, label='+ DINOv2 (Dual-branch)')
ax.plot(ranks, multiscale_data, marker='^', color='#FF7F00', linestyle='-', linewidth=2, markersize=9, label='+ Multi-scale')
ax.plot(ranks, final_data, marker='o', color='#E41A1C', linestyle='-', linewidth=3, markersize=11, label='Ours Final (+ Re-ranking)')

# ==========================================
# 4. 坐标轴与图例美化
# ==========================================
ax.set_xlabel('Rank', fontsize=16, fontweight='bold')
ax.set_ylabel('Matching Rate (%)', fontsize=16, fontweight='bold')

# 核心修改：X 轴刻度严格限制为 1, 5, 10
ax.set_xlim(0.5, 10.5)
ax.set_xticks([1, 5, 10])

# Y 轴范围根据你的实际情况调整，一般设 70-100 即可
ax.set_ylim(70, 100)

ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, linestyle='--', alpha=0.5)

ax.legend(loc='lower right', fontsize=12, framealpha=0.9, edgecolor='black')

# ==========================================
# 5. 保存并展示
# ==========================================
plt.tight_layout()
save_path = "cmc_ablation_rank_1_5_10.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"🎉 包含 Rank-1, 5, 10 的 CMC 曲线图已成功保存为: {save_path}")