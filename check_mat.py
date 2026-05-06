import torch
import torch.nn as nn
from model import two_view_net
from torchvision import transforms
from PIL import Image
import os
import glob

# ================= 配置区 =================
# 1. 模型路径
model_path = r'model/LPN_New_Run/net_119.pth'

# 2. 数据集根目录 (只需要填到 train 这一层)
data_root = r'E:\Desktop\LPN-main\University-1652\train'


# ==========================================

def find_first_image(root_dir, sub_dir):
    """自动寻找文件夹下的第一张图片"""
    search_path = os.path.join(root_dir, sub_dir)
    # 遍历 ID 文件夹 (例如 0001, 0002...)
    for id_folder in os.listdir(search_path):
        full_id_path = os.path.join(search_path, id_folder)
        if os.path.isdir(full_id_path):
            # 找里面的 jpg 或 png
            images = glob.glob(os.path.join(full_id_path, '*.jpg')) + \
                     glob.glob(os.path.join(full_id_path, '*.png')) + \
                     glob.glob(os.path.join(full_id_path, '*.jpeg'))
            if len(images) > 0:
                return images[0], id_folder  # 返回图片路径和 ID
    return None, None


def load_model():
    print(f">>> 加载模型: {model_path}")
    # 强制使用 LPN=True, block=2, 补上 droprate
    model = two_view_net(701, 0.5, block=2, LPN=True)

    if not os.path.exists(model_path):
        print(f"❌ 致命错误：找不到模型文件 {model_path}")
        exit()

    state_dict = torch.load(model_path, map_location='cpu')

    # 过滤不匹配的层
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_feature(model, img_path, view):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(img_path).convert('RGB')
    except:
        print(f"❌ 无法打开图片: {img_path}")
        return None

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        if view == 'satellite':
            output, _ = model(img, None)
        else:
            _, output = model(None, img)

    # LPN 输出处理
    if isinstance(output, tuple) or isinstance(output, list):
        output = torch.cat(output, dim=1)

    # ----------------------------------------------------
    # 🛠️ 【关键修复】: 强制把 3D/4D 张量拍扁成 2D [Batch, Feature]
    output = output.view(output.size(0), -1)
    # ----------------------------------------------------

    # 归一化
    fnorm = torch.norm(output, p=2, dim=1, keepdim=True)
    output = output.div(fnorm.expand_as(output))
    return output

def run_test():
    # 1. 自动寻找图片
    print(f">>> 正在从 {data_root} 寻找测试图片...")

    # 找一张卫星图
    img_sat_path, sat_id = find_first_image(data_root, 'satellite')
    if img_sat_path is None:
        print("❌ 找不到卫星图片 (satellite文件夹为空或不存在)")
        return
    print(f"✅ 找到卫星图 (ID: {sat_id}): {img_sat_path}")

    # 找同 ID 的无人机图
    drone_path_same = os.path.join(data_root, 'drone', sat_id)
    images_same = glob.glob(os.path.join(drone_path_same, '*.jpg')) + glob.glob(os.path.join(drone_path_same, '*.jpeg'))

    if len(images_same) == 0:
        print(f"❌ ID {sat_id} 下找不到对应的无人机图片，尝试随机找一张...")
        img_drone_same_path, _ = find_first_image(data_root, 'drone')  # 降级策略
    else:
        img_drone_same_path = images_same[0]
    print(f"✅ 找到同类无人机图: {img_drone_same_path}")

    # 2. 加载模型提取特征
    model = load_model()

    print("\n>>> 开始提取特征并计算相似度...")
    feat_sat = get_feature(model, img_sat_path, 'satellite')
    feat_drone = get_feature(model, img_drone_same_path, 'drone')

    if feat_sat is None or feat_drone is None:
        print("❌ 特征提取失败")
        return

    # 3. 计算相似度
    sim = torch.mm(feat_sat, feat_drone.t()).item()

    print("\n" + "=" * 40)
    print(f"       🔍 最终判决 (ID: {sat_id})")
    print("=" * 40)
    print(f"卫星图 vs 无人机图 相似度: {sim:.4f}")
    print("-" * 40)

    if sim > 0.4:
        print("🎉 结论：【模型是好的！】")
        print("解释：相似度很高，说明模型认出了这是同一个地方。")
        print("下一步：这就证明一定是 test.py 代码写错了，我们专注改代码就行！")
    elif sim < 0.15:
        print("💀 结论：【模型没练好】")
        print("解释：相似度太低，模型就像在瞎猜。")
        print("原因：可能是学习率(lr)设置太大导致梯度爆炸，或者 batchsize 太小导致不收敛。")
        print("下一步：需要重新训练，我们可以微调一下参数。")
    else:
        print("⚠️ 结论：【模型一般】")
        print("解释：处于及格线边缘，可能训练还不够充分，或者参数不是最优。")


if __name__ == '__main__':
    run_test()