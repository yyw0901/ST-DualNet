import torch
import torchvision.models as models
from thop import profile
from thop import clever_format

# 定义统一的输入尺寸 (保证对比的绝对公平)
dummy_input = torch.randn(1, 3, 252, 252)

print("="*50)
print("开始计算基线模型的参数量和计算量...")
print("="*50)

# ==========================================
# 1. 计算单分支 CNN 基线 (ResNet50)
# ==========================================
print("\n[1/2] 正在加载 ResNet50...")
resnet50 = models.resnet50(pretrained=False)
resnet50.eval()

flops_res, params_res = profile(resnet50, inputs=(dummy_input, ), verbose=False)
f_res, p_res = clever_format([flops_res, params_res], "%.2f")
print(f"👉 ResNet50 Params: {p_res}")
print(f"👉 ResNet50 FLOPs : {f_res}")

# ==========================================
# 2. 计算单分支 Transformer 基线 (DINOv2-ViT-S/14)
# ==========================================
print("\n[2/2] 正在加载 DINOv2-ViT-S/14...")
# 直接从官方 hub 加载你论文里用的那个小模型
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14.eval()

flops_dino, params_dino = profile(dinov2_vits14, inputs=(dummy_input, ), verbose=False)
f_dino, p_dino = clever_format([flops_dino, params_dino], "%.2f")
print(f"👉 DINOv2-ViT-S Params: {p_dino}")
print(f"👉 DINOv2-ViT-S FLOPs : {f_dino}")

print("\n" + "="*50)
print("计算完成！你可以将这些数据填入论文表格中了。")
print("="*50)