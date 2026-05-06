import torch
import os

# 您的目标文件 (刚刚跑出 45% 的那个)
target_path = r"E:\Desktop\LPN-main\model\dual_stream_v1\net_159.pth"

print(f"🕵️‍♂️ 正在分析文件成分: {target_path}")

if not os.path.exists(target_path):
    print("❌ 找不到文件！")
    exit()

try:
    state_dict = torch.load(target_path, map_location='cpu')
    keys = list(state_dict.keys())
    total_keys = len(keys)
    print(f"📦 参数总数: {total_keys}")

    # --- 1. 检测 ResNet (CNN) 成分 ---
    # 通常叫 model_1, cnn_backbone, 或包含 resnet 层的名字
    cnn_keys = [k for k in keys if 'model_1' in k or 'cnn_backbone' in k or 'layer1' in k]
    cnn_count = len(cnn_keys)

    # --- 2. 检测 DINO (Transformer) 成分 ---
    # 通常叫 dino_branch, transformer, blocks
    dino_keys = [k for k in keys if 'dino' in k or 'transformer' in k or 'blocks' in k]
    dino_count = len(dino_keys)

    print("\n" + "=" * 30)
    print("📊 成分分析报告")
    print("=" * 30)
    print(f"🔵 ResNet (CNN) 参数量: {cnn_count} 个")
    print(f"🟢 DINO (ViT) 参数量:   {dino_count} 个")
    print("-" * 30)

    # --- 3. 最终判决 ---
    if cnn_count > 100 and dino_count > 100:
        print("✅ 结论: 这是一个【双分支 (Dual-Stream)】模型！")
        print("   (结构没问题，分数低是因为缺位置编码)")
    elif cnn_count > 100 and dino_count < 10:
        print("⚠️ 结论: 这是一个【单流 ResNet】模型！")
        print("   (难怪分数只有 40-70%，因为它根本没有 DINO 部分)")
    elif cnn_count < 10 and dino_count > 100:
        print("⚠️ 结论: 这是一个【单流 DINO】模型！")
    else:
        print("❌ 结论: 模型结构异常，可能是空的或损坏的。")

    print("=" * 30)

except Exception as e:
    print(f"❌ 读取失败: {e}")