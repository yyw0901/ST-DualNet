# -*- coding: utf-8 -*-
# train_sues.py
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
from model import TwoStreamNet
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy
import yaml
from shutil import copyfile
from utils import update_average, save_network
from itertools import cycle
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# --- 配置部分 ---
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='sues200_transfer', type=str, help='output model name')
# ⚠️修改这里：指向你的 SUES-200 数据集路径
parser.add_argument('--data_dir', default=r"E:\Desktop\LPN-main\SUES-200\train", type=str, help='training dir path')
parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')  # 迁移学习可以用稍大一点的LR
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--stride', default=1, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=224, type=int, help='height')
parser.add_argument('--w', default=224, type=int, help='width')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--block', default=4, type=int, help='the num of block')
parser.add_argument('--fp16', action='store_true', help='use float16')
parser.add_argument('--pool', default='avg', type=str, help='pool avg')

opt = parser.parse_args()
opt.fp16 = True
opt.block = 4

# SUES-200 的训练类别数通常是 120 (请根据你的数据集实际情况确认)
NUM_CLASSES_SUES = 120

# 配置 GPU
str_ids = opt.gpu_ids.split(',')
gpu_ids = [int(x) for x in str_ids if int(x) >= 0]
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# --- 数据加载 (这里偷个懒，用 street 占位，实际上只训练 satellite 和 drone) ---
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
# 这里的 street 只是为了不报错，你可以复制一份 drone 到 street 文件夹
image_datasets['street'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'street'), data_transforms['train'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
               for x in ['satellite', 'street', 'drone']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'street', 'drone']}

# --- 核心：模型初始化与权重迁移 ---
print(f">>> [System] 初始化 SUES-200 模型 (Classes: {NUM_CLASSES_SUES})...")
model = TwoStreamNet(NUM_CLASSES_SUES, droprate=opt.droprate, stride=opt.stride, pool=opt.pool, block=opt.block)

# 加载 University-1652 的神级权重
# ⚠️ 请确保路径正确指向你刚练好的那个 90% 的模型
uni1652_weights_path = './model/dual_stream_v1/net_159.pth'

if os.path.exists(uni1652_weights_path):
    print(f">>> [System] 正在加载 University-1652 预训练权重: {uni1652_weights_path}")
    pretrained_dict = torch.load(uni1652_weights_path)
    model_dict = model.state_dict()

    # 【关键步骤】过滤掉形状不匹配的层 (也就是分类头 classifier)
    # 因为 1652 是 701 类，SUES 是 120 类，全连接层形状不一样，必须扔掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    # 更新权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f">>> [System] 成功加载 Backbone 权重! 分类头已重置为随机初始化。")
else:
    print(">>> [Warning] 没找到 1652 权重，将使用 ImageNet 权重从头训练 (效果会差一点)")

# --- 训练策略：先冻结 DINO，只训新分类头，然后再解冻 ---
# 这里我们直接采用微调策略：ResNet 全开，DINO 稍微给点学习率
# 或者为了稳妥，你可以前 10 轮冻结 DINO，后面解冻。这里为了代码简单，直接用差异化学习率。

model = model.cuda()
optimizer_ft = optim.SGD([
    {'params': model.parameters(), 'lr': opt.lr * 0.1},  # 整体给小学习率
    {'params': model.cnn_classifier.parameters(), 'lr': opt.lr},  # 新分类头给大学习率
    # 如果你有 LPN 的 classifier，也可以在这里单独设大
], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()


# --- 简化版训练循环 (只关注 Satellite 和 Drone) ---
def train_sues(model, optimizer, scheduler, num_epochs=60):
    accumulation_steps = 8

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train(True)
        running_loss = 0.0
        running_corrects = 0.0

        # 我们这里忽略 street 的数据，只用 satellite 和 drone
        for i, (data_sat, data_drn) in enumerate(zip(cycle(dataloaders['satellite']), dataloaders['drone'])):
            inputs_sat, labels_sat = data_sat
            inputs_drn, labels_drn = data_drn

            # 为了配合 TwoStreamNet 的 forward(x1, x2, x3)，我们需要传三个参数
            # x1=Sat, x2=Street(这里用Drone顶替一下或者None), x3=Drone
            # 你的 TwoStreamNet 逻辑里，如果有 x2 会算 x2 的 loss。
            # 为了简单，我们把 x2 设为 inputs_drn，但 Loss 权重给 0 或者不管它

            if opt.batchsize > inputs_drn.size(0): continue

            inputs_sat = Variable(inputs_sat.cuda())
            inputs_drn = Variable(inputs_drn.cuda())
            labels_sat = Variable(labels_sat.cuda())
            labels_drn = Variable(labels_drn.cuda())

            with autocast(enabled=opt.fp16):
                # 传入 (Sat, Drone, Drone) -> 假装 Street 也是 Drone
                (ret_dino, ret_cnn) = model(inputs_sat, inputs_drn, inputs_drn)

                # 解包 DINO (Sat, Str, Drn)
                out_sat_d, _, out_drn_d = ret_dino
                # 解包 CNN
                out_sat_c, _, out_drn_c = ret_cnn

                # 计算 Loss (只算 Sat 和 Drone)
                # DINO Loss
                _, loss_sat_d = one_LPN_output(out_sat_d, labels_sat, criterion)
                preds, loss_drn_d = one_LPN_output(out_drn_d, labels_drn, criterion)

                # CNN Loss
                _, loss_sat_c = one_LPN_output(out_sat_c, labels_sat, criterion)
                _, loss_drn_c = one_LPN_output(out_drn_c, labels_drn, criterion)

                loss = loss_sat_d + loss_drn_d + loss_sat_c + loss_drn_c
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * inputs_drn.size(0) * accumulation_steps
            running_corrects += float(torch.sum(preds == labels_drn.data))

        epoch_loss = running_loss / dataset_sizes['drone']
        epoch_acc = running_corrects / dataset_sizes['drone']

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            save_network(model, opt.name, epoch)


def one_LPN_output(outputs, labels, criterion):
    # 简化的 LPN Loss 计算
    sm = nn.Softmax(dim=1)
    score = 0
    loss = 0
    # 假设 outputs 是 list
    for part in outputs:
        score += sm(part)
        loss += criterion(part, labels)
    _, preds = torch.max(score.data, 1)
    return preds, loss


# 开始训练
dir_name = os.path.join('./model', opt.name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

train_sues(model, optimizer_ft, exp_lr_scheduler, num_epochs=60)