# -*- coding: utf-8 -*-
import argparse
import os
import scipy.io
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

# =============================================================================
# 1. 参数设置
# =============================================================================
parser = argparse.ArgumentParser(description='Test Single-Branch ResNet+LPN with Multi-Scale')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
# parser.add_argument('--name', default='LPN_ResNet_Baseline_Retrain', type=str, help='您单分支ResNet模型保存的文件夹名')
parser.add_argument('--name', default='LPN_ResNet_Full_Baseline', type=str, help='您单分支ResNet模型保存的文件夹名')
parser.add_argument('--test_dir', default=r'E:\Desktop\LPN-main\University-1652\test', type=str, help='./test_data')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--ms', default='1,1.1,0.9', type=str, help='多尺度评估，例如 1,1.1,0.9')
parser.add_argument('--block', default=4, type=int, help='LPN切分数量')  # 假设单分支LPN用了切分4
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
ms = list(map(float, opt.ms.split(',')))


# =============================================================================
# 2. 模型定义 (纯 ResNet50 LPN)
# =============================================================================
class SingleBranchLPN(nn.Module):
    def __init__(self, class_num, block=4):
        super(SingleBranchLPN, self).__init__()
        self.block = block
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, block))
        self.maxpool = nn.AdaptiveMaxPool2d((1, block))
        self.num_ft = 2048

    def forward(self, x):
        x = self.cnn_backbone(x)
        x = 0.5 * (self.avgpool(x) + self.maxpool(x))
        x = x.view(x.size(0), x.size(1), -1)
        # 测试时展平特征
        f = x.permute(0, 2, 1).contiguous().view(x.size(0), -1)
        return F.normalize(f, p=2, dim=1)


# =============================================================================
# 3. 提取特征函数 (带多尺度)
# =============================================================================
# def extract_feature(model, dataloaders):
#     features = torch.FloatTensor()
#     count = 0
#     for data in dataloaders:
#         img, label = data
#         n, c, h, w = img.size()
#         count += n
#         print(f'正在提取特征: {count}')
#
#         ff = torch.FloatTensor(n, model.num_ft * model.block).zero_().cuda()
#
#         for scale in ms:
#             if scale != 1:
#                 # 按照尺度缩放图像
#                 img_scaled = F.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
#             else:
#                 img_scaled = img
#
#             input_img = img_scaled.cuda()
#             with torch.no_grad():
#                 outputs = model(input_img)
#                 ff += outputs
#
#         # 翻转图像再提取一次 (常规的测试增强)
#         for scale in ms:
#             if scale != 1:
#                 img_scaled = F.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
#             else:
#                 img_scaled = img
#
#             # 水平翻转
#             inv_idx = torch.arange(img_scaled.size(3) - 1, -1, -1).long()
#             img_flip = img_scaled.index_select(3, inv_idx)
#             input_img = img_flip.cuda()
#             with torch.no_grad():
#                 outputs = model(input_img)
#                 ff += outputs
#
#         # L2归一化
#         fnorm = F.normalize(ff, p=2, dim=1)
#         features = torch.cat((features, fnorm.data.cpu()), 0)
#     return features
# =============================================================================
# 3. 提取特征函数 (非对称多尺度)
# =============================================================================
def extract_feature(model, dataloaders, is_satellite=False):
    features = torch.FloatTensor()
    count = 0

    # 🔥 核心逻辑：如果是卫星图，强行只用原图尺度 [1.0]；如果是无人机，用多尺度 [1, 1.1, 0.9]
    # current_ms = [1.0] if is_satellite else ms
    current_ms = ms

    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(f'正在提取特征: {count}')

        ff = torch.FloatTensor(n, model.num_ft * model.block).zero_().cuda()

        for scale in current_ms:
            if scale != 1:
                img_scaled = F.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
            else:
                img_scaled = img

            input_img = img_scaled.cuda()
            with torch.no_grad():
                outputs = model(input_img)
                ff += outputs

        # 翻转图像再提取一次
        for scale in current_ms:
            if scale != 1:
                img_scaled = F.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
            else:
                img_scaled = img

            inv_idx = torch.arange(img_scaled.size(3) - 1, -1, -1).long()
            img_flip = img_scaled.index_select(3, inv_idx)
            input_img = img_flip.cuda()
            with torch.no_grad():
                outputs = model(input_img)
                ff += outputs

        fnorm = F.normalize(ff, p=2, dim=1)
        features = torch.cat((features, fnorm.data.cpu()), 0)
    return features


# =============================================================================
# 4. 运行主干
# =============================================================================
if __name__ == '__main__':
    # 基础 transform
    data_transforms = transforms.Compose([
        transforms.Resize((252, 252), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # # 加载数据
    # query_dir = os.path.join(opt.test_dir, 'query_drone')
    # gallery_dir = os.path.join(opt.test_dir, 'gallery_satellite')
    #
    # image_datasets = {x: datasets.ImageFolder(os.path.join(opt.test_dir, x), data_transforms) for x in
    #                   ['query_drone', 'gallery_satellite']}
    # dataloaders = {
    #     x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=0) for x
    #     in ['query_drone', 'gallery_satellite']}
    #
    # # 加载模型 (请确保这里的 class_num 与您单分支训练时一致，通常是 701)
    # model = SingleBranchLPN(class_num=701, block=opt.block)
    #
    # # 找到您保存的单分支模型权重
    # model_path = os.path.join('./model', opt.name, 'net_120.pth')
    # if not os.path.exists(model_path):
    #     print(f"❌ 找不到单分支模型权重: {model_path} \n请修改 --name 参数！")
    #     exit()
    #
    # model.load_state_dict(torch.load(model_path), strict=False)
    # model = model.cuda()
    # model.eval()
    #
    # # 提取特征
    # print(">>> 开始提取 Query 特征 (带多尺度)...")
    # query_feature = extract_feature(model, dataloaders['query_drone'])
    # print(">>> 开始提取 Gallery 特征 (带多尺度)...")
    # gallery_feature = extract_feature(model, dataloaders['gallery_satellite'])
    #
    # # 保存 mat
    # result = {
    #     'query_f': query_feature.numpy(),
    #     'query_label': [label for _, label in image_datasets['query_drone']],
    #     'gallery_f': gallery_feature.numpy(),
    #     'gallery_label': [label for _, label in image_datasets['gallery_satellite']],
    # }
    # scipy.io.savemat('pytorch_result_single_ms.mat', result)
    # print("✅ 特征已保存至 pytorch_result_single_ms.mat")
    # === 核心修改点 1：把列表里的文件夹名字互换 ===
    target_folders = ['query_satellite', 'gallery_drone']

    image_datasets = {x: datasets.ImageFolder(os.path.join(opt.test_dir, x), data_transforms) for x in target_folders}
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=0) for x
        in target_folders}

    # 加载模型 (请确保这里的 class_num 与您单分支训练时一致，通常是 701)
    model = SingleBranchLPN(class_num=701, block=opt.block)

    # 找到您保存的单分支模型权重
    model_path = os.path.join('./model', opt.name, 'net_120.pth')
    if not os.path.exists(model_path):
        print(f"❌ 找不到单分支模型权重: {model_path} \n请修改 --name 参数！")
        exit()

    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.cuda()
    model.eval()

    # === 核心修改点 2：提取特征时保持不变 ===
    # === 核心修改点 2：调用时传入 is_satellite 参数 ===
    print(">>> 开始提取 Query 特征 (卫星 Sat) - 强行单尺度...")
    # 卫星图是 Query，所以 is_satellite=True
    query_feature = extract_feature(model, dataloaders['query_satellite'], is_satellite=True)

    print(">>> 开始提取 Gallery 特征 (无人机 Drone) - 开启多尺度...")
    # 无人机是 Gallery，所以 is_satellite=False
    gallery_feature = extract_feature(model, dataloaders['gallery_drone'], is_satellite=False)

    # 🔥🔥🔥 救命神药：直接从文件路径中提取真实的文件夹名字作为 Label 🔥🔥🔥
    def get_real_labels(dataset):
        labels = []
        for path, _ in dataset.samples:
            folder_name = os.path.basename(os.path.dirname(path))
            try:
                labels.append(int(folder_name))  # 把文件夹名字(如 0702)转成数字
            except ValueError:
                labels.append(-1)  # 如果有叫 'extra' 等非数字的干扰文件夹，标记为-1 (算分时会自动忽略)
        return labels


    query_real_labels = get_real_labels(image_datasets['query_satellite'])
    gallery_real_labels = get_real_labels(image_datasets['gallery_drone'])

    # === 核心修改点 3：使用真实标签保存 ===
    result = {
        'query_f': query_feature.numpy(),
        'query_label': query_real_labels,  # <--- 用了真实的标签
        'gallery_f': gallery_feature.numpy(),
        'gallery_label': gallery_real_labels,  # <--- 用了真实的标签
    }
    scipy.io.savemat('pytorch_result_single_ms_sat2drone.mat', result)
    print("✅ 真实标签特征已保存至 pytorch_result_single_ms_sat2drone.mat")