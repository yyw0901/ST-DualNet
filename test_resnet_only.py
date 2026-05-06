# -*- coding: utf-8 -*-
# test_resnet_only.py
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import os
import scipy.io
import yaml
import math
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn.functional as F
from model import ft_net  # <--- 导入单分支模型
from image_folder import customData

# Options
parser = argparse.ArgumentParser(description='Test Ablation')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='ablation_resnet_stride1', type=str, help='output model name')
parser.add_argument('--test_dir', default=r"E:\Desktop\LPN-main\University-1652\test", type=str, help='./test_data')
parser.add_argument('--which_epoch', default='119', type=str, help='0,1,2,3...or last')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--h', default=224, type=int, help='height')
parser.add_argument('--w', default=224, type=int, help='width')
parser.add_argument('--block', default=1, type=int, help='block')

opt = parser.parse_args()
opt.nclasses = 701
opt.stride = 1

# GPU
str_ids = opt.gpu_ids.split(',')
gpu_ids = [int(x) for x in str_ids if int(x) >= 0]
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

# Data
data_transforms = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Extract Feature (单分支版)
# test_resnet_only.py 中的 extract_feature 函数
# test_resnet_only.py

# test_resnet_only.py

def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        input_img = Variable(img.cuda())

        # 因为训练时 LPN=False，模型现在直接输出 [B, 2048]
        # 不需要 view，也不需要 mean/max
        outputs = model(input_img)

        ff = outputs.data.cpu()

        # 归一化
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features
def get_id(img_path):
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths


# Load Model
print('------- Ablation Test (ResNet Only) -----------')
print("Loading Standard ResNet50 (No LPN) for testing...")
model = ft_net(opt.nclasses, stride=opt.stride, LPN=False, block=1)
model_path = os.path.join('./model', opt.name, f'net_{opt.which_epoch}.pth')

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path), strict=False)
    print("Weights loaded!")
else:
    print("Model not found!")

# 移除分类头 (Class Block)
# ft_net 里面通常叫 classifier
model.classifier = nn.Sequential()
model = model.eval().cuda()

# Dataset
image_datasets = {}
# 还是测卫星搜无人机
# image_datasets['gallery_drone'] = datasets.ImageFolder(os.path.join(opt.test_dir, 'gallery_drone'), data_transforms)
# image_datasets['query_satellite'] = customData(os.path.join(opt.test_dir, 'query_satellite'), data_transforms)
#
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=0)
#                for x in image_datasets.keys()}

# --- 模式 B：无人机搜卫星 (Drone -> Sat) ---
print("Mode: Drone query Satellite (Localization)")
# Gallery 是被搜的库 (卫星)
image_datasets['gallery_satellite'] = datasets.ImageFolder(os.path.join(opt.test_dir, 'gallery_satellite'), data_transforms)
# Query 是输入的查询 (无人机)
image_datasets['query_drone'] = customData(os.path.join(opt.test_dir, 'query_drone'), data_transforms)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=0)
               for x in ['gallery_satellite', 'query_drone']}


with torch.no_grad():
    # query_feature = extract_feature(model, dataloaders['query_satellite'])
    # gallery_feature = extract_feature(model, dataloaders['gallery_drone'])
    # 模式 B 对应写法 (如果上面选了模式B，这里也要改回来)：
    query_feature = extract_feature(model, dataloaders['query_drone'])
    gallery_feature = extract_feature(model, dataloaders['gallery_satellite'])
# Save and Eval
# query_label, query_path = get_id(image_datasets['query_satellite'].imgs)
# gallery_label, gallery_path = get_id(image_datasets['gallery_drone'].imgs)
query_label, query_path = get_id(image_datasets['query_drone'].imgs)
gallery_label, gallery_path = get_id(image_datasets['gallery_satellite'].imgs)
result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': gallery_path,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
scipy.io.savemat('pytorch_result.mat', result)
os.system('python evaluate_gpu.py')