# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
import torch.nn.functional as F

# =============================================================================
# 1. 参数配置 (与训练保持一致)
# =============================================================================
parser = argparse.ArgumentParser(description='Test Drone -> Satellite (SOTA Version)')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
# parser.add_argument('--name', default='LPN_Dual_Mutual_v1', type=str, help='训练时保存的文件夹名字')
# parser.add_argument('--name', default='LPN_Dual_Ultimate_v6', type=str, help='训练时保存的文件夹名字')
parser.add_argument('--name', default='LPN_Dual_Ultimate_v7', type=str, help='保存文件夹的名字')
parser.add_argument('--test_dir', default=r"E:\Desktop\LPN-main\University-1652\test", type=str, help='测试集路径')
parser.add_argument('--which_epoch', default='120', type=str, help='加载第几轮的权重')
parser.add_argument('--ms', default='1,1.1,0.9', type=str, help='多尺度测试: 1,1.1,0.9')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--h', default=252, type=int, help='高度 (必须与训练一致)')
parser.add_argument('--w', default=252, type=int, help='宽度 (必须与训练一致)')
parser.add_argument('--block', default=1, type=int, help='LPN 切分数量 (必须与训练一致)')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# 解析多尺度
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    ms.append(float(s))


# =============================================================================
# 2. 模型定义 (直接复制训练代码，保证结构一致)
# =============================================================================
class DualStreamLPN(nn.Module):
    def __init__(self, class_num, droprate, block=4):
        super(DualStreamLPN, self).__init__()
        self.block = block

        print("📥 加载 ResNet50...")
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])

        print("📥 加载 DINOv2 (Small)...")
        # ✅ 修正 1: 必须加载 Small 版本，和训练保持一致
        self.dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.avgpool = nn.AdaptiveAvgPool2d((1, block))
        self.maxpool = nn.AdaptiveMaxPool2d((1, block))

        # ✅ 修正 2: Small 版本的维度是 384 (Base 才是 768)
        self.dino_proj = nn.Linear(384, 2048)
        self.num_ft = 2048

        for i in range(self.block):
            setattr(self, 'classifier_cnn' + str(i), nn.Linear(self.num_ft, class_num))
            setattr(self, 'classifier_dino' + str(i), nn.Linear(self.num_ft, class_num))

    def forward(self, x):
        # CNN Stream
        x_cnn = self.cnn_backbone(x)
        # 🔥 关键修改：训练时用了 avg+max，这里必须同步！
        x_cnn = 0.5 * (self.avgpool(x_cnn) + self.maxpool(x_cnn))
        x_cnn = x_cnn.view(x_cnn.size(0), x_cnn.size(1), -1)

        # # DINO Stream
        # dino_out = self.dino_backbone.forward_features(x)
        # patch_tokens = dino_out['x_norm_patchtokens']
        # B, N, C = patch_tokens.shape
        # grid = int(N ** 0.5)
        # x_dino = patch_tokens.permute(0, 2, 1).reshape(B, C, grid, grid)
        # DINO Stream
        dino_out = self.dino_backbone.forward_features(x)
        patch_tokens = dino_out['x_norm_patchtokens']
        B, N, C = patch_tokens.shape

        # 🔥 修改开始：动态计算高度和宽度的 patch 数，不再假设是正方形
        # DINOv2 patch_size 固定为 14
        h_grid = x.shape[2] // 14
        w_grid = x.shape[3] // 14

        # 容错：万一 patch 数对不上（虽然 extract_feature 里处理了，但防一手）
        if h_grid * w_grid != N:
            grid = int(N ** 0.5)  # 回退到正方形逻辑
            h_grid, w_grid = grid, grid

        x_dino = patch_tokens.permute(0, 2, 1).reshape(B, C, h_grid, w_grid)
        # 🔥 修改结束

        x_dino = F.adaptive_avg_pool2d(x_dino, (1, self.block)).squeeze(2).permute(0, 2, 1)
        x_dino = self.dino_proj(x_dino).permute(0, 2, 1)

        if self.training:
            predict = {}
            features = {}
            for i in range(self.block):
                features[i] = x_cnn[:, :, i]
                features[i + self.block] = x_dino[:, :, i]
                predict[i] = getattr(self, 'classifier_cnn' + str(i))(features[i])
                predict[i + self.block] = getattr(self, 'classifier_dino' + str(i))(features[i + self.block])
            return predict, features
        else:
            # 🔥 测试逻辑：直接拼接两个流的归一化特征
            f_cnn = x_cnn.permute(0, 2, 1).contiguous().view(x_cnn.size(0), -1)
            f_dino = x_dino.permute(0, 2, 1).contiguous().view(x_dino.size(0), -1)
            return torch.cat((F.normalize(f_cnn), F.normalize(f_dino)), dim=1)


# =============================================================================
# 3. 数据加载
# =============================================================================
data_transforms = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = opt.test_dir
image_datasets = {}
# 只跑 Drone -> Satellite (如需跑 Satellite -> Drone，请互换 query 和 gallery)
target_dirs = ['gallery_satellite', 'query_drone']
# 修改这里：加载 query_satellite 和 gallery_drone
# target_dirs = ['gallery_drone', 'query_satellite']
for x in target_dirs:
    dir_path = os.path.join(data_dir, x)
    if not os.path.exists(dir_path):
        print(f"⚠️ Warning: {dir_path} not found.")
        continue
    image_datasets[x] = datasets.ImageFolder(dir_path, data_transforms)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=0) for x in image_datasets.keys()}


# =============================================================================
# 4. 辅助函数
# =============================================================================
def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def get_id(img_path):
    labels = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        # 假设文件夹名即 ID (University-1652 格式)
        try:
            labels.append(int(folder_name))
        except:
            labels.append(-1)
    return labels


# =============================================================================
# 5. 核心：多尺度特征提取
# =============================================================================
def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0

    # DINOv2 的 Patch Size
    patch_size = 14

    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        if count % 100 == 0:
            print(f'   - 已处理 {count} 张图片...', end='\r')

        # 初始化特征容器 (根据您的模型输出维度，这里假设是 8192 或 16384)
        # 既然我们会在下面累加，这里先不指定具体维度，用 None 占位
        ff = None

        # 多尺度遍历
        for scale in ms:
            for i in range(2):  # 0: 原图, 1: 翻转
                if scale != 1:
                    # ✅ 修复逻辑：计算缩放后的尺寸，并强制对齐到 14 的倍数
                    new_h = int(round((h * scale) / patch_size) * patch_size)
                    new_w = int(round((w * scale) / patch_size) * patch_size)

                    # 使用 size 参数而不是 scale_factor
                    input_img = nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear',
                                                          align_corners=False)
                else:
                    input_img = img

                if i == 1:
                    input_img = fliplr(input_img)

                input_img = input_img.cuda()

                # 模型推理
                outputs = model(input_img)

                # 特征累加
                if ff is None:
                    ff = outputs
                else:
                    ff += outputs

        # 对多尺度融合后的特征再次归一化
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features

# =============================================================================
# 6. 主程序
# =============================================================================
if __name__ == '__main__':
    # 类别数随意，因为我们只用特征提取层，不加载分类层权重
    # 但为了对应 keys，建议还是填 701
    model = DualStreamLPN(class_num=701, droprate=0.5, block=opt.block)

    # 加载权重
    model_path = os.path.join('./model', opt.name, f'net_{opt.which_epoch}.pth')
    print(f"🔥 正在加载模型权重: {model_path}")
    if os.path.exists(model_path):
        try:
            # 1. 加载保存的权重
            state_dict = torch.load(model_path)

            # 2. 获取当前模型的字典
            model_dict = model.state_dict()

            # 3. ♻️ 智能过滤：只加载那些“名字存在”且“形状完全一致”的层
            # 这样就能自动忽略掉形状不匹配的分类器层 (classifier_cnn0 等)
            pretrained_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and v.size() == model_dict[k].size()}

            # 4. 更新并加载
            if len(pretrained_dict) == 0:
                print("❌ 警告：没有匹配的参数被加载！请检查权重文件是否正确。")
            else:
                print(f"✅ 成功匹配并加载了 {len(pretrained_dict)}/{len(model_dict)} 层参数")
                # 打印被忽略的层（通常是分类器）以供确认
                ignored = [k for k in state_dict.keys() if k not in pretrained_dict]
                if len(ignored) > 0:
                    print(f"   ℹ️ 已自动忽略 {len(ignored)} 个不匹配层 (主要是 classifiers，不影响测试)")

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            exit()
    else:
        print(f"❌ 错误: 找不到权重文件 {model_path}")
        exit()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    query_name = 'query_drone'
    gallery_name = 'gallery_satellite'
    # 修改这里：卫星搜无人机
    # query_name = 'query_satellite'
    # gallery_name = 'gallery_drone'

    print(f"\n>>> [开始提取] Query: {query_name} -> Gallery: {gallery_name}")

    if gallery_name in image_datasets and query_name in image_datasets:
        with torch.no_grad():
            print(">>> 正在提取 Query 特征...")
            query_feature = extract_feature(model, dataloaders[query_name])
            print("\n>>> 正在提取 Gallery 特征...")
            gallery_feature = extract_feature(model, dataloaders[gallery_name])

        print("\n>>> 正在保存 pytorch_result.mat ...")
        gallery_label = get_id(image_datasets[gallery_name].imgs)
        query_label = get_id(image_datasets[query_name].imgs)

        result = {
            'gallery_f': query_feature.numpy(),  # 注意：University-1652 这里的命名有点反直觉，通常 gallery 是卫星
            'gallery_label': query_label,
            'query_f': gallery_feature.numpy(),
            'query_label': gallery_label
        }

        # 修正：根据 University-1652 的习惯，Drone 是 Query, Satellite 是 Gallery
        # 但 evaluate_rerank.py 里的逻辑是 query 查 gallery
        # 确保保存的字典 key 和 evaluate 代码对应
        result = {
            'query_f': query_feature.numpy(),
            'query_label': query_label,
            'gallery_f': gallery_feature.numpy(),
            'gallery_label': gallery_label
        }

        scipy.io.savemat('pytorch_result.mat', result)
        print("✅ 特征提取完成！请运行 evaluate_rerank.py 查看最终分数！")
    else:
        print("❌ 错误: 未找到数据集路径，请检查 test_dir 设置。")