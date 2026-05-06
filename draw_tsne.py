import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.manifold import TSNE

# 导入你的网络模型
from train_dual_mutual import DualStreamLPN

# ==========================================
# 0. 核心配置：参数与路径
# ==========================================
# 选取前 10 个类别进行可视化（10个类别颜色最好分，图不会太乱）
CLASS_NUM_TO_VISUALIZE = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 权重路径与数据根目录 (请根据你的实际路径微调)
weight_folder = r"E:\Desktop\LPN-main\model\LPN_Dual_Ultimate_v5"
data_root = r"E:\Desktop\LPN-main\University-1652\test"
drone_dir = os.path.join(data_root, "query_drone")
sat_dir = os.path.join(data_root, "gallery_satellite")

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 1. 自动加载权重并初始化模型
# ==========================================
print("📥 正在初始化模型并加载权重...")
model = DualStreamLPN(class_num=701, droprate=0.5).to(device)

files = [f for f in os.listdir(weight_folder) if f.endswith('.pth')]
target_file = next((f for f in files if '119' in f), files[-1])
weight_path = os.path.join(weight_folder, target_file)

checkpoint = torch.load(weight_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# ==========================================
# 2. 批量提取真实特征向量
# ==========================================
# ==========================================
# 2. 批量提取真实特征向量 (适配 University-1652)
# ==========================================
print(f"🚀 开始在 University-1652 数据集上提取前 {CLASS_NUM_TO_VISUALIZE} 个类别的真实特征...")

all_features = []
all_labels = []
all_domains = []

# 获取所有类别文件夹（比如 '0001', '0002' 等），并取前 CLASS_NUM_TO_VISUALIZE 个
class_folders = sorted(os.listdir(sat_dir))[:CLASS_NUM_TO_VISUALIZE]


def extract_feat(img_path, label, domain):
    if not os.path.exists(img_path): return
    img = cv2.imread(img_path)
    if img is None: return
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (252, 252))
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        feat = out[0] if isinstance(out, (tuple, list)) else out
        # 【极其重要】：务必保留 L2 归一化
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        all_features.append(feat.cpu().numpy().flatten())
        all_labels.append(label)
        all_domains.append(domain)


valid_exts = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')

for label, folder in enumerate(class_folders):
    # --- 1. 提取卫星特征 ---
    # 根据你的截图，路径是：.../train/satellite/0001/xxxx.jpg
    sat_class_dir = os.path.join(sat_dir, folder)
    if os.path.exists(sat_class_dir):
        sat_files = [f for f in os.listdir(sat_class_dir) if f.lower().endswith(valid_exts)]
        if sat_files:
            # 提取该类别下的第一张（也是唯一一张）卫星图
            extract_feat(os.path.join(sat_class_dir, sat_files[0]), label, 'sat')

    # --- 2. 提取无人机特征 ---
    # 根据你的截图，路径是：.../train/drone/0001/xxxx.jpg
    drone_class_dir = os.path.join(drone_dir, folder)
    if os.path.exists(drone_class_dir):
        d_files = [f for f in os.listdir(drone_class_dir) if f.lower().endswith(valid_exts)]
        # 提取该地点所有的无人机图片 (每个地点有 54 张)
        for df in d_files:
            extract_feat(os.path.join(drone_class_dir, df), label, 'drone')

# ==========================================
# 3. t-SNE 降维计算
# ==========================================
print(f"📉 提取完毕 (共 {len(all_features)} 个特征点)，正在进行 t-SNE 降维运算...")
X = np.array(all_features)
y = np.array(all_labels)
domains = np.array(all_domains)

# 确保 metric='cosine'，这是解决“卫星图抱团”的关键
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42, init='random')
X_2d = tsne.fit_transform(X)

# ==========================================
# 4. 学术级散点图绘制
# ==========================================
print("🎨 正在绘制高逼格聚类散点图...")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.5

fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.tab20(np.linspace(0, 1, CLASS_NUM_TO_VISUALIZE))

for i in range(CLASS_NUM_TO_VISUALIZE):
    idx_drone = (y == i) & (domains == 'drone')
    idx_sat = (y == i) & (domains == 'sat')

    ax.scatter(X_2d[idx_drone, 0], X_2d[idx_drone, 1],
               c=[colors[i]], marker='o', s=20, alpha=0.6, edgecolors='none')

    ax.scatter(X_2d[idx_sat, 0], X_2d[idx_sat, 1],
               c=[colors[i]], marker='^', s=150, alpha=1.0, edgecolors='white', linewidths=1.5)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("t-SNE Visualization of Feature Representations", fontsize=16, fontweight='bold', pad=15)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Drone View', markerfacecolor='gray', markersize=10, alpha=0.6),
    Line2D([0], [0], marker='^', color='w', label='Satellite View', markerfacecolor='gray', markersize=12,
           markeredgecolor='white', markeredgewidth=1.5)
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=14, framealpha=0.9, edgecolor='black')

plt.tight_layout()
save_path = "tsne_features_st_dualnet.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"🎉 真实数据的 t-SNE 图已成功保存至: {save_path}")