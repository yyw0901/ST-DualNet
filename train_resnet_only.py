# -*- coding: utf-8 -*-
# train_resnet_only.py (消融实验专用)
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
from model import ft_net  # <--- 注意这里导入的是单分支网络
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy
import yaml
from shutil import copyfile
from utils import update_average, save_network
from itertools import cycle
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# --- Options ---
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='ablation_resnet_stride1', type=str, help='output model name')
parser.add_argument('--data_dir', default=r"E:\Desktop\LPN-main\University-1652\train", type=str,
                    help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=1, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=224, type=int, help='height')
parser.add_argument('--w', default=224, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_parsing', action='store_true', help='use parsing')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # ResNet通常用0.01或者0.05
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--block', default=1, type=int, help='LPN block' ) # 改为 1
parser.add_argument('--fp16', action='store_true', help='use float16')  # 开启混合精度加速

opt = parser.parse_args()
opt.fp16 = True

# 强制设置
opt.nclasses = 701

# GPU Config
str_ids = opt.gpu_ids.split(',')
gpu_ids = [int(x) for x in str_ids if int(x) >= 0]
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# Data Transforms
transform_train_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
transform_satellite_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomAffine(90),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'satellite': transforms.Compose(transform_satellite_list)
}

image_datasets = {}
image_datasets['satellite'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'satellite'),
                                                   data_transforms['satellite'])
image_datasets['drone'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'drone'), data_transforms['train'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
               for x in ['satellite', 'drone']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}

# --- Model Initialization (单分支) ---
print("Initializing Single Branch ResNet+LPN Model...")
# 注意：这里使用的是 ft_net，不是 TwoStreamNet
# 如果你的 model.py 里叫 ft_net_LPN，请改成 ft_net_LPN
print("Initializing Standard ResNet50 (No LPN)...")
model = ft_net(opt.nclasses, droprate=opt.droprate, stride=opt.stride, LPN=False, block=1)
# model = ft_net_LPN(opt.nclasses, droprate=opt.droprate, stride=opt.stride, pool='avg', block=opt.block) # 备选写法

model = model.cuda()

# Optimization
optimizer_ft = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)  # 120 epoch 衰减
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()


# Training Function
def train_model(model, optimizer, scheduler, num_epochs=120):
    # --- 新增：打印数据量检查 ---
    print(f"Dataset Sizes: Satellite={dataset_sizes['satellite']}, Drone={dataset_sizes['drone']}")
    print(f"Batch Count: Satellite={len(dataloaders['satellite'])}, Drone={len(dataloaders['drone'])}")
    # --------------------------
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train(True)

        running_loss = 0.0
        running_corrects = 0.0

        # 同时迭代卫星和无人机数据
        for data_sat, data_drn in zip(cycle(dataloaders['satellite']), dataloaders['drone']):
            inputs_sat, labels_sat = data_sat
            inputs_drn, labels_drn = data_drn

            if opt.batchsize > inputs_drn.size(0): continue  # Skip incomplete batch

            # 放到 GPU
            inputs_sat = Variable(inputs_sat.cuda())
            inputs_drn = Variable(inputs_drn.cuda())
            labels_sat = Variable(labels_sat.cuda())
            labels_drn = Variable(labels_drn.cuda())

            optimizer.zero_grad()

            with autocast(enabled=opt.fp16):
                # ResNet 共享权重，分别跑两次前向传播
                outputs_sat = model(inputs_sat)
                outputs_drn = model(inputs_drn)

                # 计算 LPN Loss
                # outputs 应该是一个 list (包含 block 个切片输出)
                _, loss_sat = one_LPN_output(outputs_sat, labels_sat, criterion)
                preds, loss_drn = one_LPN_output(outputs_drn, labels_drn, criterion)

                loss = loss_sat + loss_drn

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs_drn.size(0)
            running_corrects += float(torch.sum(preds == labels_drn.data))

        epoch_loss = running_loss / dataset_sizes['drone']
        epoch_acc = running_corrects / dataset_sizes['drone']

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        scheduler.step()

        # Save model
        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            save_network(model, opt.name, epoch)


def one_LPN_output(outputs, labels, criterion):
    # LPN 专用的 Loss 计算，把 4 个块的 loss 加起来
    if isinstance(outputs, list):  # 如果是 LPN 输出 list
        score = 0
        loss = 0
        for part in outputs:
            loss += criterion(part, labels)
            score += part
        _, preds = torch.max(score.data, 1)
        return preds, loss
    else:  # 如果不是 LPN (只是普通的 Global Pooling)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        return preds, loss


# Start Training
dir_name = os.path.join('./model', opt.name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# 只需要跑 60-80 轮消融实验就足够看趋势了，跑满 120 轮更好
train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=120)