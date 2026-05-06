# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import os
import scipy.io
import torch.nn.functional as F

# =============================================================================
# 1. 参数配置 (专门用于提取 95% SOTA 的完美版)
# =============================================================================
parser = argparse.ArgumentParser(description='Test ST-DualNet (Multi-scale Feature Extraction)')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='Ablation_KD0.5', type=str, help='训练时保存的权重文件夹名字')
parser.add_argument('--test_dir', default=r"E:\Desktop\LPN-main\University-1652\test", type=str, help='测试集父级路径')
parser.add_argument('--which_epoch', default='120', type=str, help='加载第几轮的权重')
parser.add_argument('--ms', default='1,1.1,0.9', type=str, help='多尺度测试: 1,1.1,0.9')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize (测试时可以设大点，如32或64)')
parser.add_argument('--h', default=252, type=int, help='高度 (必须与训练一致)')
parser.add_argument('--w', default=252, type=int, help='宽度 (必须与训练一致)')
parser.add_argument('--block', default=4, type=int, help='LPN 切分数量 (必须与训练一致，跑消融时记得改)')

opt = parser.parse_args()

# GPU 设置
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
cudnn.benchmark = True

# 解析多尺度
ms = [float(s) for s in opt.ms.split(',')]


# =============================================================================
# 2. 模型定义 (绝对对齐训练时的双分支架构)
# =============================================================================
class DualStreamLPN(nn.Module):
    def __init__(self, class_num, block=4):
        super(DualStreamLPN, self).__init__()
        self.block = block

        print("📥 初始化 ResNet50 & DINOv2_vits14 (用于测试)...")
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)  # 测试时设为False加快加载
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.avgpool = nn.AdaptiveAvgPool2d((1, block))
        self.maxpool = nn.AdaptiveMaxPool2d((1, block))
        self.dino_proj = nn.Linear(384, 2048)
        self.num_ft = 2048

        for i in range(self.block):
            setattr(self, 'classifier_cnn' + str(i), nn.Linear(self.num_ft, class_num))
            setattr(self, 'classifier_dino' + str(i), nn.Linear(self.num_ft, class_num))

    def forward(self, x):
        # CNN 分支
        x_cnn = self.cnn_backbone(x)
        x_cnn = 0.5 * (self.avgpool(x_cnn) + self.maxpool(x_cnn))
        x_cnn = x_cnn.view(x_cnn.size(0), x_cnn.size(1), -1)

        # DINO 分支
        dino_out = self.dino_backbone.forward_features(x)
        patch_tokens = dino_out['x_norm_patchtokens']
        B, N, C = patch_tokens.shape

        # 动态计算高宽网格，自适应多尺度输入
        h_grid = x.shape[2] // 14
        w_grid = x.shape[3] // 14
        if h_grid * w_grid != N:
            grid = int(N ** 0.5)
            h_grid, w_grid = grid, grid

        x_dino = patch_tokens.permute(0, 2, 1).reshape(B, C, h_grid, w_grid)
        x_dino = F.adaptive_avg_pool2d(x_dino, (1, self.block)).squeeze(2).permute(0, 2, 1)
        x_dino = self.dino_proj(x_dino).permute(0, 2, 1)

        # 仅测试逻辑：直接返回 L2 归一化后的拼接特征 (1D)
        f_cnn = x_cnn.permute(0, 2, 1).contiguous().view(x_cnn.size(0), -1)
        f_dino = x_dino.permute(0, 2, 1).contiguous().view(x_dino.size(0), -1)
        # return torch.cat((F.normalize(f_cnn), F.normalize(f_dino)), dim=1)

        f_cnn_norm = F.normalize(f_cnn)
        f_dino_norm = F.normalize(f_dino)
        # 比如给 DINOv2 1.5 倍的权重，CNN 保持 1.0
        return torch.cat((f_cnn_norm * 1.0, f_dino_norm * 1.5), dim=1)


# =============================================================================
# 3. 数据加载 (🔥 彻底修复版：绝对安全的独立路径加载逻辑)
# =============================================================================
data_transforms = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\n>>> 正在加载测试集图片...")
# query_dir = os.path.join(opt.test_dir, 'query_drone')
# gallery_dir = os.path.join(opt.test_dir, 'gallery_satellite')
# 🚨 路径反转：Query 变成卫星，Gallery 变成无人机
query_dir = os.path.join(opt.test_dir, 'query_satellite')
gallery_dir = os.path.join(opt.test_dir, 'gallery_drone')
image_datasets = {}
dataloaders = {}

# # 1. 加载 Drone (无人机)
# if os.path.exists(query_dir):
#     image_datasets['query_drone'] = datasets.ImageFolder(query_dir, data_transforms)
#     dataloaders['query_drone'] = torch.utils.data.DataLoader(image_datasets['query_drone'], batch_size=opt.batchsize,
#                                                              shuffle=False, num_workers=0)
#     print(f"   🚁 Query (无人机) 图片数: {len(image_datasets['query_drone'])}")
# else:
#     raise RuntimeError(f"❌ 找不到无人机测试集路径: {query_dir}")
#
# # 2. 加载 Satellite (卫星)
# if os.path.exists(gallery_dir):
#     image_datasets['gallery_satellite'] = datasets.ImageFolder(gallery_dir, data_transforms)
#     dataloaders['gallery_satellite'] = torch.utils.data.DataLoader(image_datasets['gallery_satellite'],
#                                                                    batch_size=opt.batchsize, shuffle=False,
#                                                                    num_workers=0)
#     print(f"   🛰️ Gallery (卫星) 图片数: {len(image_datasets['gallery_satellite'])}")
# else:
#     raise RuntimeError(f"❌ 找不到卫星测试集路径: {gallery_dir}")


# # 1. 加载 Query (卫星)
if os.path.exists(query_dir):
    image_datasets['query_satellite'] = datasets.ImageFolder(query_dir, data_transforms)
    dataloaders['query_satellite'] = torch.utils.data.DataLoader(image_datasets['query_satellite'], batch_size=opt.batchsize, shuffle=False, num_workers=0)
    print(f"   🛰️ Query (卫星) 图片数: {len(image_datasets['query_satellite'])}")
else:
    raise RuntimeError(f"❌ 找不到卫星测试集路径: {query_dir}")

# 2. 加载 Gallery (无人机)
if os.path.exists(gallery_dir):
    image_datasets['gallery_drone'] = datasets.ImageFolder(gallery_dir, data_transforms)
    dataloaders['gallery_drone'] = torch.utils.data.DataLoader(image_datasets['gallery_drone'], batch_size=opt.batchsize, shuffle=False, num_workers=0)
    print(f"   🚁 Gallery (无人机) 图片数: {len(image_datasets['gallery_drone'])}")
else:
    raise RuntimeError(f"❌ 找不到无人机测试集路径: {gallery_dir}")

# =============================================================================
# 4. 辅助函数
# =============================================================================
def fliplr(img):
    """ 水平翻转张量 """
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def get_id(img_path):
    """ 获取 University-1652 的建筑 ID """
    labels = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        try:
            labels.append(int(folder_name))
        except:
            labels.append(-1)
    return labels


# =============================================================================
# 5. 核心：多尺度特征提取 (🔥 彻底修复版：变量不被污染)
# =============================================================================
def extract_feature(model, dataloader):
    features = torch.FloatTensor()
    count = 0
    patch_size = 14  # DINOv2 固定的 patch 大小

    for data in dataloader:
        img, label = data
        n, c, h, w = img.size()
        count += n
        if count % 100 == 0 or count == len(dataloader.dataset):
            print(f'   - 提取进度: [{count}/{len(dataloader.dataset)}] 张', end='\r')

        ff = None  # 初始化特征累加器

        for scale in ms:
            for i in range(2):  # 0: 原图, 1: 水平翻转
                # 🔥 绝杀修复：保证每次操作都是从最原始的 img 出发，使用 clone 防污染
                if scale != 1:
                    new_h = int(round((h * scale) / patch_size) * patch_size)
                    new_w = int(round((w * scale) / patch_size) * patch_size)
                    current_img = nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear',
                                                            align_corners=False)
                else:
                    current_img = img.clone()

                if i == 1:
                    current_img = fliplr(current_img)

                current_img = current_img.cuda()

                # 前向传播提取 1D 特征
                outputs = model(current_img)

                # 特征累加 (多尺度的精髓)
                if ff is None:
                    ff = outputs
                else:
                    ff += outputs

        # 多尺度融合完毕后，必须对这个终极特征向量再次进行严格的 L2 归一化
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    print()  # 换行
    return features


# =============================================================================
# 6. 主程序执行逻辑
# =============================================================================
if __name__ == '__main__':
    # 类别数必须与训练时一致 (University-1652 为 701)，否则由于 Linear 层尺寸不对会报错
    model = DualStreamLPN(class_num=701, block=opt.block)

    # 加载 15 小时炼出的神级权重
    model_path = os.path.join('./model', opt.name, f'net_{opt.which_epoch}.pth')
    print(f"\n🔥 正在加载模型权重: {model_path}")

    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model_dict = model.state_dict()

        # 智能剥离不匹配的层 (主要应对可能存在的分类器尺寸问题)
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        if len(pretrained_dict) == 0:
            raise RuntimeError("❌ 警告：没有匹配的参数被加载！请检查 block 数是否与训练时完全一致。")
        else:
            print(f"✅ 成功加载了 {len(pretrained_dict)}/{len(model_dict)} 层参数 (仅用于提特征，未加载的分类层无影响)")

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise RuntimeError(f"❌ 找不到权重文件 {model_path}，请先运行 train_dual_ablation.py")

    model = model.eval().cuda()

    # query_name = 'query_drone'
    # gallery_name = 'gallery_satellite'

    # print(f"\n>>> [任务启动] 无人机(Query) 检索 卫星(Gallery)")
    # 🚨 将检索方向彻底反转
    query_name = 'query_satellite'
    gallery_name = 'gallery_drone'

    print(f"\n>>> [任务启动] 卫星(Query) 检索 无人机(Gallery)")
    with torch.no_grad():
        print(">>> [1/2] 正在抽取 Query 特征 (Drone)...")
        query_feature = extract_feature(model, dataloaders[query_name])

        print(">>> [2/2] 正在抽取 Gallery 特征 (Satellite)...")
        gallery_feature = extract_feature(model, dataloaders[gallery_name])

    # print("\n>>> 正在将结果保存至 pytorch_result.mat ...")
    # gallery_label = get_id(image_datasets[gallery_name].imgs)
    # query_label = get_id(image_datasets[query_name].imgs)
    #
    # # 严格对齐 evaluate_rerank.py 的键名需求
    # result = {
    #     'query_f': query_feature.numpy(),
    #     'query_label': query_label,
    #     'gallery_f': gallery_feature.numpy(),
    #     'gallery_label': gallery_label
    # }
    #
    # scipy.io.savemat('pytorch_result.mat', result)
    print("\n>>> 正在将结果保存至 pytorch_result.pt ...")
    gallery_label = get_id(image_datasets[gallery_name].imgs)
    query_label = get_id(image_datasets[query_name].imgs)

    # 直接保存 PyTorch Tensor 和 Python List，不再转 numpy
    result = {
        'query_f': query_feature.cpu(),
        'query_label': query_label,
        'gallery_f': gallery_feature.cpu(),
        'gallery_label': gallery_label
    }

    # 使用 PyTorch 原生保存方法，完美支持超大文件
    torch.save(result, 'pytorch_result.pt')
    print("🎯 特征提取大功告成！文件已保存为 pytorch_result.pt")
    print("🎯 特征提取大功告成！")
    print("======================================================")
    print("下一步：请立即运行 evaluate_rerank.py 来见证您的终极分数！")
    print("======================================================")