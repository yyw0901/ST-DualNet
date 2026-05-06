# -*- coding: utf-8 -*-
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
import copy
import time
import os
from model import two_view_net, three_view_net, TwoStreamNet
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy
import yaml
import math
from shutil import copyfile
from utils import update_average, get_model_list, load_network, save_network, make_weights_for_balanced_classes
from itertools import cycle
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# --- 新增函数：计算两个分支之间的 KL 散度 (互学习) ---
def compute_kl_loss(p_list, q_list, temp=1.0):
    loss = 0
    for p, q in zip(p_list, q_list):
        p_s = F.log_softmax(p / temp, dim=1)
        q_s = F.softmax(q / temp, dim=1)
        loss += F.kl_div(p_s, q_s, reduction='batchmean')
    return loss

version =  torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='sues200_training_v4', type=str, help='output model name')
parser.add_argument('--data_dir', default=r"E:\Desktop\LPN-main\SUES-200-Splitted\train", type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
# 【BatchSize 设置】为了防止解冻后爆显存，建议保持 2。如果显存够大可改回 4。
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--stride', default=1, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=252, type=int, help='height')
parser.add_argument('--w', default=252, type=int, help='width')
parser.add_argument('--views', default=3, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', default=True, help='use Color Data Augmentation')
parser.add_argument('--resume', action='store_true', help='use resume trainning' )
parser.add_argument('--share', default=True,action='store_true', help='share weight between different view' )
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google' )
parser.add_argument('--LPN', default=True,action='store_true', help='use LPN' )
parser.add_argument('--block', default=4, type=int, help='the num of block' )
parser.add_argument('--fp16', action='store_true', default=True, help='use fp16 training')
parser.add_argument('--pool',default='avg', type=str, help='pool avg')

opt = parser.parse_args()

# ==========================================
# 【重点修改区】强制参数配置
# ==========================================
opt.fp16 = True
opt.LPN = True
opt.use_vgg16 = False
opt.block = 4
# 【重要】我们要手动加载，所以这里把 resume 设为 False，避免 load_network 报错
opt.resume = False

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
transform_train_list = [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad( opt.pad, padding_mode='edge'),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_satellite_list = [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad( opt.pad, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
    'satellite': transforms.Compose(transform_satellite_list) }

image_datasets = {}
image_datasets['satellite'] = datasets.ImageFolder(os.path.join(data_dir, 'satellite'), data_transforms['satellite'])
# image_datasets['street'] = datasets.ImageFolder(os.path.join(data_dir, 'street'), data_transforms['train'])
# SUES-200 hack: 这里的 os.path.join 里填 'drone'
image_datasets['street'] = datasets.ImageFolder(os.path.join(data_dir, 'drone'), data_transforms['train'])
image_datasets['drone'] = datasets.ImageFolder(os.path.join(data_dir, 'drone'), data_transforms['train'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
              for x in ['satellite', 'street', 'drone']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'street', 'drone']}
class_names = image_datasets['street'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Training Function
# ------------------
y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def one_LPN_output(outputs, labels, criterion, block):
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)
    _, preds = torch.max(score.data, 1)
    return preds, loss

def train_model(model, model_test, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    scaler = GradScaler()
    accumulation_steps = 8
    warm_up = 0.1
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch

    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            model.train(True)
            running_loss = 0.0
            running_corrects = 0.0

            for i, (data, data2, data3) in enumerate(
                    zip(cycle(dataloaders['satellite']), cycle(dataloaders['street']), dataloaders['drone'])):
                inputs, labels = data
                inputs2, labels2 = data2
                inputs3, labels3 = data3

                now_batch_size, c, h, w = inputs3.shape
                if now_batch_size < opt.batchsize:
                    continue

                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    inputs2 = Variable(inputs2.cuda().detach())
                    inputs3 = Variable(inputs3.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    labels2 = Variable(labels2.cuda().detach())
                    labels3 = Variable(labels3.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                with autocast(enabled=opt.fp16):
                    (ret_dino, ret_cnn) = model(inputs, inputs2, inputs3)
                    out_sat_d, out_str_d, out_drn_d = ret_dino
                    out_sat_c, out_str_c, out_drn_c = ret_cnn

                    _, loss_sat_d = one_LPN_output(out_sat_d, labels, criterion, opt.block)
                    _, loss_str_d = one_LPN_output(out_str_d, labels2, criterion, opt.block)
                    preds, loss_drn_d = one_LPN_output(out_drn_d, labels3, criterion, opt.block)
                    loss_dino = loss_sat_d + loss_str_d + loss_drn_d

                    _, loss_sat_c = one_LPN_output(out_sat_c, labels, criterion, opt.block)
                    _, loss_str_c = one_LPN_output(out_str_c, labels2, criterion, opt.block)
                    _, loss_drn_c = one_LPN_output(out_drn_c, labels3, criterion, opt.block)
                    loss_cnn = loss_sat_c + loss_str_c + loss_drn_c

                    loss_kl = compute_kl_loss(out_sat_d, out_sat_c) + \
                              compute_kl_loss(out_str_d, out_str_c) + \
                              compute_kl_loss(out_drn_d, out_drn_c)

                    loss = loss_dino + loss_cnn + 1.0 * loss_kl
                    loss = loss / accumulation_steps

                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    scaler.scale(loss).backward()

                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if opt.moving_avg < 1.0:
                            update_average(model_test, model, opt.moving_avg)

                running_loss += loss.item() * now_batch_size * accumulation_steps
                running_corrects += float(torch.sum(preds == labels3.data))

            epoch_loss = running_loss / dataset_sizes['drone']
            epoch_acc = running_corrects / dataset_sizes['drone']

            print('{} Loss: {:.4f} Drone_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)

            if phase == 'train':
                scheduler.step()
            if epoch % 20 == 19:
                save_network(model, opt.name, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    return model

######################################################################
# Main Execution
# ----------------------

print(">>> [System] 正在初始化双流网络 (TwoStreamNet)...")
model = TwoStreamNet(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, block=opt.block)
opt.nclasses = len(class_names)

# =====================================================================
# 【核心修正】手动加载权重 (绕过 opts.yaml 报错)
# =====================================================================
# 寻找权重文件
# weights_path = os.path.join('./model', opt.name, 'net_119.pth')
# if not os.path.exists(weights_path):
#     print(f">>> ⚠️ 警告：找不到 net_119.pth，尝试寻找 last.pth 或 net_last.pth...")
#     weights_path = os.path.join('./model', opt.name, 'last.pth')
#
# if os.path.exists(weights_path):
#     print(f">>> [System] 正在从 {weights_path} 恢复训练...")
#     # 允许 strict=False 以应对分类头维度变化等微小差异
#     model.load_state_dict(torch.load(weights_path), strict=False)
#     # 既然加载了 119 轮的权重，我们就从 120 轮开始算
#     start_epoch = 120
# else:
#     print(f">>> ❌ 致命错误：找不到任何权重文件！请检查 model/{opt.name} 文件夹。")
#     print(">>> 正在退出...")
#     exit()
start_epoch = 0
# =====================================================================
# 【部分解冻】配置 DINOv2
# =====================================================================
print(">>> [System] 正在配置 DINOv2 参数状态 (部分解冻模式)...")

# 1. 先全冻住
for name, param in model.named_parameters():
    if 'dino' in name:
        param.requires_grad = False

# 2. 解冻 DINO 的最后几个 Block (9, 10, 11) 和 Norm 层
blocks_to_unfreeze = ['blocks.9', 'blocks.10', 'blocks.11', 'norm.']

for name, param in model.named_parameters():
    if 'dino' in name:
        if 'classifier' in name or 'projector' in name:
            param.requires_grad = True
        else:
            for keyword in blocks_to_unfreeze:
                if keyword in name:
                    param.requires_grad = True
                    break

# 统计
count_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f">>> 当前可训练参数量: {count_grad / 1e6:.2f} M (DINO 部分已激活)")

# =====================================================================
# 优化器设置
# =====================================================================
dino_params = []
dino_ids = []
for name, param in model.named_parameters():
    if 'dino' in name and param.requires_grad:
        dino_params.append(param)
        dino_ids.append(id(param))

base_params = filter(lambda p: id(p) not in dino_ids and p.requires_grad, model.parameters())

optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': opt.lr},           # ResNet: 0.0001
    {'params': dino_params, 'lr': opt.lr * 0.1}      # DINO: 0.00001
], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Start Training
# ----------------------
# 强制多跑 40 轮
target_epochs = start_epoch + 40
print(f">>> 目标训练轮数: {target_epochs}")

model = model.cuda()
criterion = nn.CrossEntropyLoss()

model = train_model(model, None, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=target_epochs)