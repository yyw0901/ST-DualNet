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
# 0. 定义双分支网络包装器 (核心修改点)
# ==========================================
# 为了让 GradCAM 正常工作，我们需要把双分支网络的输出拆开。
# 这里假设你的 DualStreamLPN 的 forward 返回的是两个 logits (例如: return out_resnet, out_dinov2)
# 如果你的 forward 返回的是拼接后的一个 tensor，请告诉我，这里的逻辑需要微调。

class ResNetBranch(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        # 如果模型返回的是元组或列表，就取第一个元素；如果是单个张量，直接返回
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

class DINOv2Branch(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        # 同理，为了代码健壮性，兼容单输出和多输出
        if isinstance(out, (tuple, list)):
            # 如果碰巧有多个输出，DINOv2 通常在后面，取最后一个；如果只有一个就取第一个
            return out[-1] if len(out) > 1 else out[0]
        return out

# ==========================================
# 1. 初始化模型并加载权重
# ==========================================
# 补全缺失的参数：SUES-200 的 701 个类别，droprate 设为默认 0.5
model = DualStreamLPN(class_num=701, droprate=0.5)

# 【务必修改】填入你最新的权重路径
# weight_path = "E:/Desktop/LPN-main/model/LPN_Dual_Ultimate_v5/net_120.pth"
weight_path = "E:/Desktop/LPN-main/model/sues200_training_v2/net_039.pth"
print("正在加载训练好的权重...")
checkpoint = torch.load(weight_path, map_location='cpu')

# try:
#     model.load_state_dict(checkpoint)
# except RuntimeError:
#     model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
try:
    model.load_state_dict(checkpoint, strict=False)
except RuntimeError:
    # 万一走到这里，也记得加上 strict=False
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=False)
model.eval()

# 实例化包装后的模型
model_resnet = ResNetBranch(model)
model_dinov2 = DINOv2Branch(model)

# ==========================================
# 2. 锁定提取特征的目标层 (Target Layers)
# ==========================================
# 【务必检查】你需要确认你的类里面 ResNet 和 DINOv2 的变量名叫什么
# 这里假设你的 ResNet 叫 cnn_backbone，DINOv2 叫 dinov2_backbone
target_layer_resnet = [model.cnn_backbone[7][-1]]

# 对于 DINOv2 (Vision Transformer)，通常取最后一个 Block 的 LayerNorm
target_layer_dinov2 = [model.dino_backbone.blocks[-1].norm1]

# ==========================================
# 3. 读取并预处理图片
# ==========================================
image_path = "E:/Desktop/LPN-main/SUES-200-512x512/drone_view_512/0001/150/0.jpg"
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (252, 252)) # 确保尺寸与你训练时一致
rgb_img_float = np.float32(rgb_img) / 255.0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(rgb_img).unsqueeze(0)

# ==========================================
# 4. 生成两组热力图并融合
# ==========================================
print("正在计算 ResNet50 (局部纹理) 热力图...")
cam_resnet = GradCAM(model=model_resnet, target_layers=target_layer_resnet)
grayscale_cam_resnet = cam_resnet(input_tensor=input_tensor, targets=None)[0, :]

print("正在计算 DINOv2 (全局语义) 注意力热力图...")
# 准备一个空列表，用来装我们“偷”出来的注意力矩阵
attention_maps = []


# 定义神级钩子：拦截 qkv 输出，我们自己还原注意力矩阵！
def get_attention_hook(module, input, output):
    # output 是 qkv 线性层的输出，形状为 [B, N, 3*C]
    B, N, C3 = output.shape
    C = C3 // 3
    # 获取 DINOv2 当前注意力头的数量
    num_heads = model.dino_backbone.blocks[-1].attn.num_heads
    head_dim = C // num_heads

    # 按照 DINOv2 的内部逻辑，还原 Query 和 Key
    qkv = output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]  # 形状都是: [B, num_heads, N, head_dim]

    # 自己动手，丰衣足食：矩阵乘法 + Scale + Softmax 得到 Attention Map
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)

    attention_maps.append(attn.detach().cpu())


# 这次我们把钩子挂在必经的 qkv (nn.Linear) 层上，它 100% 是个 Module，绝不会报错！
hook_handle = model.dino_backbone.blocks[-1].attn.qkv.register_forward_hook(get_attention_hook)

# 跑一次前向传播，让图像数据流过网络，触发钩子
with torch.no_grad():
    _ = model(input_tensor)

# 数据到手，把钩子拆掉，做到“神不知鬼不觉”
hook_handle.remove()

# 处理偷出来的注意力数据
attentions = attention_maps[0]

# 提取 [CLS] token (索引为 0) 对所有图像 Patch (索引 1 到最后) 的关注度
cls_attn = attentions[0, :, 0, 1:]

# 将所有注意力头 (heads) 的权重求平均，得出最稳健的全局响应
mean_attn = torch.mean(cls_attn, dim=0)

# 重塑为 18x18 的二维网格
attn_map = mean_attn.reshape(18, 18).numpy()

# 进行 Min-Max 归一化 (加上 1e-8 防止除以 0 报错)
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

# 平滑放大回 252x252 的原图尺寸
attn_map_resized = cv2.resize(attn_map, (252, 252))

# ==========================================
# 融合部分微调
# ==========================================
print("正在融合双分支响应图...")
# 取 ResNet 的局部纹理和 DINOv2 的全局语义的最大值进行硬核融合
grayscale_cam_fused = np.maximum(grayscale_cam_resnet, attn_map_resized)

# 统一将灰度图转为彩色热力图并叠加到原图上
vis_resnet = show_cam_on_image(rgb_img_float, grayscale_cam_resnet, use_rgb=True)
vis_dinov2 = show_cam_on_image(rgb_img_float, attn_map_resized, use_rgb=True)
vis_fused = show_cam_on_image(rgb_img_float, grayscale_cam_fused, use_rgb=True)
# ==========================================
# 5. 绘制 1x4 硬核对比图
# ==========================================
plt.figure(figsize=(16, 4)) # 调整画布比例适应 1x4

plt.subplot(1, 4, 1)
plt.title("UAV Image", fontsize=13)
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
save_path = "E:/Desktop/LPN-main/cam_result_0001_150.png" # 你可以随便改保存位置和名字
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"🎉 硬核对比图已成功保存至: {save_path}")