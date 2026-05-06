import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 导入你的双分支网络
from train_dual_mutual import DualStreamLPN


# ==========================================
# 0. 定义双分支网络包装器 (兼容多输出)
# ==========================================
class ResNetBranch(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class DINOv2Branch(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[-1] if len(out) > 1 else out[0]
        return out


# ==========================================
# 1. 初始化模型并加载权重
# ==========================================
print("📥 正在初始化模型与加载权重...")
model = DualStreamLPN(class_num=701, droprate=0.5)

# 【务必核对】你的终极权重路径
# weight_path = "E:/Desktop/LPN-main/model/LPN_Dual_Ultimate_v5/net_120.pth"
weight_path = "E:/Desktop/LPN-main/model/sues200_training_v2/net_039.pth"
checkpoint = torch.load(weight_path, map_location='cpu')

try:
    model.load_state_dict(checkpoint, strict=False)
except RuntimeError:
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=False)

model.eval()

# 实例化 ResNet 包装器用于 GradCAM
model_resnet = ResNetBranch(model)
# 用索引 7 获取 layer4，并取最后一个 bottleneck
target_layer_resnet = [model.cnn_backbone[7][-1]]

# ==========================================
# 2. 读取并预处理【卫星图】
# ==========================================
print("🖼️ 正在处理卫星图像...")
# 【务必核对】替换为你的卫星图真实路径，这里假设是 0001 类的卫星图
image_path =r"E:\Desktop\LPN-main\SUES-200-512x512\satellite-view\0001\0.png"
rgb_img = cv2.imread(image_path, 1)

# 如果路径错误或图片不存在，防止 OpenCV 报错隐式崩溃
if rgb_img is None:
    raise FileNotFoundError(f"找不到图片，请检查路径是否正确: {image_path}")

rgb_img = rgb_img[:, :, ::-1]  # BGR to RGB
rgb_img = cv2.resize(rgb_img, (252, 252))
rgb_img_float = np.float32(rgb_img) / 255.0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(rgb_img).unsqueeze(0)

# ==========================================
# 3. 计算 ResNet50 (局部纹理) 热力图
# ==========================================
print("🔥 正在计算 ResNet50 局部纹理热力图...")
cam_resnet = GradCAM(model=model_resnet, target_layers=target_layer_resnet)
grayscale_cam_resnet = cam_resnet(input_tensor=input_tensor, targets=None)[0, :]
vis_resnet = show_cam_on_image(rgb_img_float, grayscale_cam_resnet, use_rgb=True)

# ==========================================
# 4. 计算 DINOv2 (全局语义) 注意力热力图 (Hook拦截法)
# ==========================================
print("🕸️ 正在提取 DINOv2 全局注意力矩阵...")
attention_maps = []


def get_attention_hook(module, input, output):
    B, N, C3 = output.shape
    C = C3 // 3
    num_heads = model.dino_backbone.blocks[-1].attn.num_heads
    head_dim = C // num_heads

    qkv = output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]

    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    attention_maps.append(attn.detach().cpu())


# 挂载钩子到最后一个 Block 的 qkv 层
hook_handle = model.dino_backbone.blocks[-1].attn.qkv.register_forward_hook(get_attention_hook)

with torch.no_grad():
    _ = model(input_tensor)

hook_handle.remove()

# 处理拦截到的注意力数据
attentions = attention_maps[0]
cls_attn = attentions[0, :, 0, 1:]  # 提取 [CLS] 对 Patch 的注意力
mean_attn = torch.mean(cls_attn, dim=0)  # 多头求平均
attn_map = mean_attn.reshape(18, 18).numpy()
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
attn_map_resized = cv2.resize(attn_map, (252, 252))
vis_dinov2 = show_cam_on_image(rgb_img_float, attn_map_resized, use_rgb=True)

# ==========================================
# 5. 融合双分支并绘制 1x4 对比图
# ==========================================
print("✨ 正在生成终极融合热力图...")
grayscale_cam_fused = np.maximum(grayscale_cam_resnet, attn_map_resized)
vis_fused = show_cam_on_image(rgb_img_float, grayscale_cam_fused, use_rgb=True)

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Satellite Image", fontsize=13)
plt.imshow(rgb_img)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("DINOv2 (Global Semantic)", fontsize=13)
plt.imshow(vis_dinov2)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("ResNet50-LPN (Local Texture)", fontsize=13)
plt.imshow(vis_resnet)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Fused Response", fontsize=13)
plt.imshow(vis_fused)
plt.axis('off')

plt.tight_layout()

# 保存高清图到本地
save_path = "E:/Desktop/LPN-main/cam_result_0001_satellite.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"🎉 卫星图硬核对比图已成功保存至: {save_path}")