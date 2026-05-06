import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 导入你的网络类
from train_dual_mutual import DualStreamLPN

# ==========================================
# 0. 核心配置：路径与参数
# ==========================================
# ==========================================
# 0. 核心配置：路径与参数 (全自动查找版)
# ==========================================
import os

# 1. 权重文件夹
weight_folder = r"E:\Desktop\LPN-main\model\sues200_training_v3"
files = [f for f in os.listdir(weight_folder) if f.endswith('.pth')]
# 自动找序号最大的权重，或者包含 '119' 的
target_file = next((f for f in files if '119' in f), files[-1])
weight_path = os.path.join(weight_folder, target_file)
print(f"✅ 自动匹配权重: {weight_path}")

# 2. 无人机图路径 (这个你之前跑通了，保持不动)
drone_path = r"E:\Desktop\LPN-main\SUES-200-512x512\drone_view_512\0001\150\0.jpg"

# 3. 卫星图【全自动查找逻辑】
# 指向 0001 文件夹即可，让代码自己去里面抓唯一的图
sat_folder = r"E:\Desktop\LPN-main\SUES-200-512x512\satellite-view\0001"

if os.path.exists(sat_folder):
    # 查找文件夹下任何常见的图片格式
    valid_exts = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG')
    sat_files = [f for f in os.listdir(sat_folder) if f.lower().endswith(valid_exts)]

    if not sat_files:
        raise FileNotFoundError(f"❌ 0001 文件夹里竟然没看到任何图片！请核对路径: {sat_folder}")

    # 自动取该文件夹下的第一张图
    sat_path = os.path.join(sat_folder, sat_files[0])
    print(f"✅ 自动匹配卫星图: {sat_path}")
else:
    # 如果 satellite-view 不对，尝试换成 satellite_view
    raise FileNotFoundError(f"❌ 找不到卫星图文件夹，请核对拼写是否为 satellite-view: {sat_folder}")

class_num = 701
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 模型包装器 (用于 GradCAM)
# ==========================================
class SimpleWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


# ==========================================
# 2. 图像处理函数
# ==========================================
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"无法读取: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (252, 252))
    img_float = np.float32(img) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return img, img_float, tensor


# ==========================================
# 3. 主程序：计算并绘图
# ==========================================
def main():
    # --- 加载模型 ---
    print("正在初始化模型...")
    model = DualStreamLPN(class_num=class_num, droprate=0.5).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    wrapper = SimpleWrapper(model)
    # 锁定 ResNet50 的层
    target_resnet = [model.cnn_backbone[7][-1]]
    cam_extractor = GradCAM(model=wrapper, target_layers=target_resnet)

    # --- 准备画布 ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    rows = [('Drone', drone_path), ('Satellite', sat_path)]

    for row_idx, (label, path) in enumerate(rows):
        print(f"正在处理 {label} 视角...")
        img_orig, img_float, input_tensor = process_image(path)

        # A. 计算 ResNet (Local)
        grayscale_resnet = cam_extractor(input_tensor=input_tensor, targets=None)[0, :]
        vis_resnet = show_cam_on_image(img_float, grayscale_resnet, use_rgb=True)

        # B. 计算 DINOv2 (Global) - 使用 Hook 拦截
        attention_maps = []

        def hook(module, input, output):
            B, N, C3 = output.shape
            num_heads = model.dino_backbone.blocks[-1].attn.num_heads
            head_dim = (C3 // 3) // num_heads
            qkv = output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            attention_maps.append(attn.softmax(dim=-1).detach().cpu())

        handle = model.dino_backbone.blocks[-1].attn.qkv.register_forward_hook(hook)
        with torch.no_grad(): _ = model(input_tensor)
        handle.remove()

        attn = torch.mean(attention_maps[0][0, :, 0, 1:], dim=0).reshape(18, 18).numpy()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        attn_resized = cv2.resize(attn, (252, 252))
        vis_dinov2 = show_cam_on_image(img_float, attn_resized, use_rgb=True)

        # C. 计算融合图
        fused_gray = np.maximum(grayscale_resnet, attn_resized)
        vis_fused = show_cam_on_image(img_float, fused_gray, use_rgb=True)

        # --- 填充画布 ---
        axes[row_idx, 0].imshow(img_orig)
        axes[row_idx, 0].set_ylabel(label, fontsize=18, fontweight='bold')
        axes[row_idx, 1].imshow(vis_dinov2)
        axes[row_idx, 2].imshow(vis_resnet)
        axes[row_idx, 3].imshow(vis_fused)

    # 设置标题
    col_titles = ["Original Image", "DINOv2 (Global)", "ResNet50 (Local)", "Fused Response"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=16, pad=10)

    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    save_path = "E:/Desktop/LPN-main/Paper_Final_Figuresues.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 论文大图已保存至: {save_path}")


if __name__ == "__main__":
    main()