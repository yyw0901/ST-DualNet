import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备 (完美换行，告别拥挤)
# ==========================================
# 使用 \n 进行换行，彻底解决 X 轴标签挤在一起的问题
# labels = ['λ = 0.0', 'λ = 0.5', 'λ = 1.0', 'λ = 2.0']
labels = ['0.0', '0.5', '1.0', '2.0']
# Drone -> Satellite
ds_rank1 = [83.89, 84.50, 87.03, 82.77]
ds_ap = [86.07, 86.58, 88.67, 85.34]

# Satellite -> Drone
sd_rank1 = [87.19, 87.36, 88.03, 84.31]
sd_ap = [85.22, 85.24, 85.45, 83.78]

# ==========================================
# 2. 全局样式设置 (高级学术风)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.2

# 采用更柔和的高级莫兰迪学术配色
color_rank1 = '#4C72B0'  # 学术蓝
color_ap = '#C44E52'  # 学术红

# ==========================================
# 3. 绘图逻辑
# ==========================================
x = np.arange(len(labels))
width = 0.28  # 让柱子更修长
gap = 0.03  # 蓝红柱子之间增加一点微小的间隙，防止顶部数字打架

# 加大画布尺寸，给予充足的留白
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)


# def plot_beautiful_bar(ax, rank_data, ap_data, title):
#     """封装的极简风绘图函数"""
#     rects1 = ax.bar(x - width / 2 - gap / 2, rank_data, width, label='R@1', color=color_rank1, edgecolor='black',
#                     zorder=3)
#     rects2 = ax.bar(x + width / 2 + gap / 2, ap_data, width, label='AP', color=color_ap, edgecolor='black', zorder=3)
#
#     # 标题和轴标签
#     ax.set_title(title, fontweight='bold', fontsize=18, pad=15)
#     ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=14)
#
#     # X 轴设置
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, fontweight='bold', fontsize=13)
#
#     # Y 轴留白：上限提高到 92，给数字留足空间
#     ax.set_ylim(70, 92)
def plot_beautiful_bar(ax, rank_data, ap_data, title):
    """封装的极简风绘图函数"""
    rects1 = ax.bar(x - width / 2 - gap / 2, rank_data, width, label='R@1', color=color_rank1, edgecolor='black',
                    zorder=3)
    rects2 = ax.bar(x + width / 2 + gap / 2, ap_data, width, label='AP', color=color_ap, edgecolor='black', zorder=3)

    # 标题和轴标签
    ax.set_title(title, fontweight='bold', fontsize=18, pad=15)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=14)

    # X 轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=13)

    # ✨ 新增：横轴标题 (labelpad 控制标题和刻度数字的间距)
    ax.set_xlabel(r'Mutual learning weight $\lambda$', fontweight='bold', fontsize=14, labelpad=8)

    # Y 轴留白：上限提高到 92，给数字留足空间
    ax.set_ylim(70, 92)

    # 背景网格
    # ...后续代码保持不变...

    # 背景网格
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    # 🔥 顶刊排版秘籍：去掉顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例设置
    ax.legend(loc='upper left', framealpha=0.9, fontsize=12)
    return rects1, rects2


# 绘制两个子图
rects1_ds, rects2_ds = plot_beautiful_bar(ax1, ds_rank1, ds_ap, 'Drone $\\rightarrow$ Satellite')
rects1_sd, rects2_sd = plot_beautiful_bar(ax2, sd_rank1, sd_ap, 'Satellite $\\rightarrow$ Drone')


# ==========================================
# 4. 精致的数值标注
# ==========================================
def autolabel_elegant(rects, ax):
    """统一黑色，最优值可加粗"""
    for rect in rects:
        height = rect.get_height()
        if height in [87.03, 88.67, 88.03, 85.45]:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color='black')
        else:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11, color='black')

# 应用数值标注
autolabel_elegant(rects1_ds, ax1)
autolabel_elegant(rects2_ds, ax1)
autolabel_elegant(rects1_sd, ax2)
autolabel_elegant(rects2_sd, ax2)

# ==========================================
# 5. 渲染并保存
# ==========================================
# 增加子图之间的水平间距 (w_pad)
plt.tight_layout(w_pad=4.0)
plt.savefig('KD_Ablation_BarChart_Elegant.png', format='png', bbox_inches='tight')
plt.show()