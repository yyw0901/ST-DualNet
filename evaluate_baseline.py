import torch
import scipy.io
import numpy as np
import os
from model import TwoStreamNet

# ================= 配置区 =================
# 请修改为您的实际路径
model_path = './model/dual_stream_v1/net_159.pth'
# 如果找不到 net_159，试试 last.pth
if not os.path.exists(model_path):
    model_path = './model/dual_stream_v1/last.pth'

mat_path = 'pytorch_result.mat'
# ==========================================

print("=" * 40)
print(">>> 步骤 1: 检查特征文件 (Result Mat Analysis)")
print("=" * 40)

if os.path.exists(mat_path):
    res = scipy.io.loadmat(mat_path)
    query_f = res['query_f']
    print(f"特征形状: {query_f.shape}")

    # 假设特征是拼接的：[DINO (前一半) | ResNet (后一半)]
    # 或者 [ResNet | DINO]，取决于 concat 顺序
    # 您的代码是: torch.cat((feat_dino, feat_cnn), 1)

    # 我们通过方差来判断特征是否正常
    # 训练好的特征，方差通常会有一定波动；随机初始化的特征，方差可能很奇怪

    dim = query_f.shape[1]
    # 假设 DINO 维度和 ResNet 维度可能不同，这里粗略对半切看一眼
    mid = dim // 2

    part1 = query_f[:, :mid]
    part2 = query_f[:, mid:]

    print(f"前半部分 (可能是DINO) 均值: {np.mean(part1):.6f}, 方差: {np.var(part1):.6f}")
    print(f"后半部分 (可能是CNN)  均值: {np.mean(part2):.6f}, 方差: {np.var(part2):.6f}")

    if np.var(part1) < 1e-5 or np.var(part2) < 1e-5:
        print("\n>>> ⚠️ 警告：某一部分特征方差极低，可能全是 0 或 1！")
    else:
        print("\n>>> 特征数值看起来基本正常（不是全0）。")

else:
    print("找不到 pytorch_result.mat，跳过此步。")

print("\n" + "=" * 40)
print(">>> 步骤 2: 检查权重加载 (Weight Loading Check)")
print("=" * 40)

# 初始化模型
# 注意：这里参数要和 test.py 里一致
try:
    model = TwoStreamNet(701, droprate=0.5, stride=2, pool='avg', block=4)
    print("模型初始化成功。")
except Exception as e:
    print(f"模型初始化失败: {e}")
    exit()

if os.path.exists(model_path):
    print(f"正在尝试加载: {model_path}")
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # 核心逻辑：对比 Key
    pretrained_keys = set(pretrained_dict.keys())
    model_keys = set(model_dict.keys())

    # 计算交集
    loaded_keys = pretrained_keys.intersection(model_keys)
    missing_in_pth = model_keys - pretrained_keys
    unused_in_pth = pretrained_keys - model_keys

    print(f"\n模型总参数量: {len(model_keys)} 个 Tensor")
    print(f"权重文件包含: {len(pretrained_keys)} 个 Tensor")
    print(f"成功匹配加载: {len(loaded_keys)} 个 Tensor")

    ratio = len(loaded_keys) / len(model_keys)
    print(f"\n>>> 加载成功率: {ratio * 100:.2f}%")

    if ratio < 0.1:
        print("\n>>> ❌ 严重错误：几乎没有加载任何权重！")
        print("原因可能是：模型变量名改了（比如 'model_1' 变成了 'cnn'），导致名字对不上。")
    elif ratio < 0.9:
        print("\n>>> ⚠️ 警告：有部分权重没加载进去。")
        print(f"未加载的层示例 (Model里有，pth里没有): {list(missing_in_pth)[:5]}")
    else:
        print("\n>>> ✅ 权重加载看起来很完美。")

else:
    print(f"找不到权重文件: {model_path}")