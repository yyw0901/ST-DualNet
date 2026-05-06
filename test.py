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
from model import ft_net, two_view_net, three_view_net, TwoStreamNet
from utils import load_network
from image_folder import customData, customData_one

# fp16
try:
    from apex.fp16_utils import *
except ImportError:
    pass

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='dual_stream_v1', type=str, help='output model name')
parser.add_argument('--test_dir', default=r"E:\Desktop\LPN-main\University-1652\test", type=str, help='./test_data')
parser.add_argument('--which_epoch', default='159', type=str, help='0,1,2,3...or last')

# 多尺度测试：原图 + 1.1倍 + 0.9倍
parser.add_argument('--ms', default='1,1.1,0.9', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--h', default=224, type=int, help='height')
parser.add_argument('--w', default=224, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--pad', default=0, type=int, help='padding')
parser.add_argument('--block', default=4, type=int, help='block')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')

opt = parser.parse_args()

opt.nclasses = 701
opt.LPN = True
opt.block = 4
opt.h = 224
opt.w = 224
opt.stride = 2

# Load Config
config_path = os.path.join('./model', opt.name, 'opts.yaml')
if os.path.exists(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    opt.fp16 = config.get('fp16', False)
    opt.stride = config.get('stride', 2)

str_ids = opt.gpu_ids.split(',')
gpu_ids = [int(x) for x in str_ids if int(x) >= 0]

print('>>> [System] 正在启用多尺度测试，Scale List: %s' % opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(s_f)

if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = opt.test_dir
image_datasets = {}
target_dirs = ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']

for x in target_dirs:
    dir_path = os.path.join(data_dir, x)
    if not os.path.exists(dir_path):
        print(f"Warning: {dir_path} not found.")
        continue
    if 'query' in x:
        image_datasets[x] = customData(dir_path, data_transforms, rotate=0)
    else:
        image_datasets[x] = datasets.ImageFolder(dir_path, data_transforms)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=0) for x in image_datasets.keys()}
use_gpu = torch.cuda.is_available()


######################################################################
# Extract feature (修复版：DINO 智能取整)
# ----------------------
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders, view_index=1):
    features = torch.FloatTensor()
    count = 0
    # DINOv2 的 Patch Size
    patch_size = 14

    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        if count % 100 == 0:
            print(f'Detected {count} images...')

        ff = None

        for scale in ms:
            for i in range(2):  # 翻转循环
                # 1. 处理缩放 (关键修复！)
                if scale != 1:
                    # 计算目标尺寸
                    target_h = int(h * scale)
                    target_w = int(w * scale)

                    # 【核心】强制凑整为 14 的倍数
                    # 比如算出来 246，除以14四舍五入是18，再乘14等于252
                    target_h = int(round(target_h / patch_size) * patch_size)
                    target_w = int(round(target_w / patch_size) * patch_size)

                    input_img = nn.functional.interpolate(img, size=(target_h, target_w), mode='bilinear',
                                                          align_corners=False)
                else:
                    input_img = img

                # 2. 处理翻转
                if i == 1:
                    input_img = fliplr(input_img)

                input_img = Variable(input_img.cuda())

                # 3. DINO 特征提取
                out_dino_list = model.dino_branch(input_img)
                if isinstance(out_dino_list, list):
                    # DINO 输出的是 patch tokens
                    # 这里 LPN 依赖 AdaptiveAvgPool2d((block, 1))
                    # 所以只要输入是 14 的倍数，特征图尺寸不管是多少，都能被自适应池化成 block x 1
                    # 完美兼容！
                    feat_dino = torch.stack(out_dino_list, dim=2).view(n, -1)
                else:
                    feat_dino = out_dino_list.view(n, -1)

                # 4. CNN 特征提取
                feat_cnn = model.cnn_backbone(input_img)
                feat_cnn = model.avgpool(feat_cnn).view(n, -1)

                # 5. 归一化 + 融合
                feat_dino = F.normalize(feat_dino, p=2, dim=1)
                feat_cnn = F.normalize(feat_cnn, p=2, dim=1)
                outputs = torch.cat((feat_dino, feat_cnn), 1)

                # 6. 累加特征
                if ff is None:
                    ff = outputs
                else:
                    ff += outputs

        # 最后做一次总的归一化
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths


######################################################################
# Main Execution
print('------- Final Test (Multi-Scale Safe Mode) -----------')
model = TwoStreamNet(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool)

# 自动寻找权重
model_path = os.path.join('./model', opt.name, f'net_{opt.which_epoch}.pth')
if not os.path.exists(model_path):
    print(f"Warning: net_{opt.which_epoch}.pth not found. Trying last.pth")
    model_path = os.path.join('./model', opt.name, 'last.pth')

print(f"Loading model from: {model_path}")

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path), strict=False)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
else:
    print("Fatal Error: No model weights found!")

# 移除分类头
if hasattr(model, 'dino_branch') and hasattr(model.dino_branch, 'classifier'):
    model.dino_branch.classifier.classifier = nn.Sequential()
if hasattr(model, 'cnn_classifier'):
    model.cnn_classifier = nn.Sequential()

model = model.eval()
if use_gpu:
    model = model.cuda()

# 设置模式: 无人机搜卫星
# gallery_name = 'gallery_satellite'
# query_name = 'query_drone'
# 设置模式: 卫星搜无人机
gallery_name = 'gallery_drone'      # 被搜的库变成无人机
query_name = 'query_satellite'      # 搜图的人变成卫星

print(f"Mode: {query_name} -> {gallery_name}")

if gallery_name in image_datasets and query_name in image_datasets:
    gallery_path = image_datasets[gallery_name].imgs
    query_path = image_datasets[query_name].imgs

    gallery_label, gallery_path = get_id(gallery_path)
    query_label, query_path = get_id(query_path)

    with torch.no_grad():
        print("Extracting Query Features...")
        query_feature = extract_feature(model, dataloaders[query_name])
        print("Extracting Gallery Features...")
        gallery_feature = extract_feature(model, dataloaders[gallery_name])

    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': gallery_path,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
    scipy.io.savemat('pytorch_result.mat', result)

    print("Evaluation Start...")
    os.system('python evaluate_gpu.py')
else:
    print("Dataset paths error.")