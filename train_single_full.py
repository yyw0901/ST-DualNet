# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import time
import os
import random
import math
import numpy as np
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# =============================================================================
# 1. 参数配置
# =============================================================================
parser = argparse.ArgumentParser(description='Full Single-Stream ResNet LPN with Triplet Loss')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='LPN_ResNet_Full_Baseline', type=str, help='保存文件夹的名字')
parser.add_argument('--data_dir', default=r"E:\Desktop\LPN-main\University-1652", type=str, help='数据集路径')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--num_instances', default=4, type=int, help='每个类别在batch中抽取的图片数 (K)')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=120, type=int, help='训练多少轮')
parser.add_argument('--img_h', default=252, type=int, help='统一高度')
parser.add_argument('--img_w', default=252, type=int, help='统一宽度')
parser.add_argument('--block', default=4, type=int, help='LPN 切分数量')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
cudnn.benchmark = True


# =============================================================================
# 2. PK-Sampler (难样本采样器) & Triplet Loss (三元组损失)
# =============================================================================
class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs_dict[pid] = idxs

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs_dict[pid][:self.num_instances])
                batch_idxs_dict[pid] = batch_idxs_dict[pid][self.num_instances:]
                if len(batch_idxs_dict[pid]) < self.num_instances:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return len(self.data_source)


import copy


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # 🔥 核心修复点 1：强行转为 float32，完美解决 Half 和 Float 冲突，还能防止平方溢出！
        inputs = inputs.float()

        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()

        # 🔥 核心修复点 2：修复 addmm_ 警告，使用官方推荐的最新传参格式
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)

        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

# =============================================================================
# 3. 数据处理与加载
# =============================================================================
transform_train = transforms.Compose([
    transforms.Resize((opt.img_h, opt.img_w), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((opt.img_h, opt.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(">>> 正在加载训练数据 (启动 PK-Sampler)...")
ds_drone = datasets.ImageFolder(os.path.join(opt.data_dir, 'train', 'drone'), transform_train)
ds_sat = datasets.ImageFolder(os.path.join(opt.data_dir, 'train', 'satellite'), transform_train)
full_dataset = torch.utils.data.ConcatDataset([ds_drone, ds_sat])
class_names = ds_drone.classes

# 重点：使用 PK Sampler
sampler = RandomIdentitySampler(full_dataset, opt.batchsize, opt.num_instances)
dataloaders = torch.utils.data.DataLoader(full_dataset, batch_size=opt.batchsize, sampler=sampler, num_workers=0,
                                          drop_last=True)
print(f"训练集总图片数: {len(full_dataset)}, 类别数: {len(class_names)}")


# =============================================================================
# 4. 模型定义 (完整 LPN，训练时返回特征用于计算 Triplet Loss)
# =============================================================================
class SingleStreamLPNFull(nn.Module):
    def __init__(self, class_num, block=4):
        super(SingleStreamLPNFull, self).__init__()
        self.block = block
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, block))
        self.maxpool = nn.AdaptiveMaxPool2d((1, block))
        self.num_ft = 2048
        for i in range(self.block):
            setattr(self, 'classifier' + str(i), nn.Linear(self.num_ft, class_num))

    def forward(self, x):
        x = self.cnn_backbone(x)
        x = 0.5 * (self.avgpool(x) + self.maxpool(x))
        x = x.view(x.size(0), x.size(1), -1)
        if self.training:
            logits_list = []
            features_list = []
            for i in range(self.block):
                f_i = x[:, :, i]
                logits_i = getattr(self, 'classifier' + str(i))(f_i)
                logits_list.append(logits_i)
                features_list.append(f_i)  # 收集特征用于 Triplet Loss
            return logits_list, features_list
        else:
            return x.permute(0, 2, 1).contiguous().view(x.size(0), -1)


# =============================================================================
# 5. 训练主循环 (ID Loss + Triplet Loss)
# =============================================================================
def train_model(model, criterion_id, criterion_triplet, optimizer, scheduler, num_epochs=120):
    since = time.time()
    scaler = GradScaler()
    save_dir = os.path.join('./model', opt.name)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloaders:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            with autocast():
                logits_list, features_list = model(inputs)
                loss = 0
                for i in range(opt.block):
                    # 1. 交叉熵损失 (分类)
                    loss += criterion_id(logits_list[i], labels)
                    # 2. 三元组损失 (特征拉近拉远)
                    loss += criterion_triplet(features_list[i], labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            with torch.no_grad():
                _, preds = torch.max(logits_list[0].data, 1)
                corrects += torch.sum(preds == labels.data).item()

        scheduler.step()
        # 注意：这里算出的 acc 只是单个块的粗略估计，作为参考即可
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = corrects / (len(dataloaders) * opt.batchsize)

        print(
            f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2%} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'net_{epoch + 1}.pth'))

    time_elapsed = time.time() - since
    print(f'✅ 完整版训练完成！耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model


if __name__ == '__main__':
    model = SingleStreamLPNFull(len(class_names), block=opt.block).cuda()
    ignored_params = list(map(id, model.cnn_backbone.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': opt.lr},
        {'params': model.cnn_backbone.parameters(), 'lr': 0.1 * opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

    # 双重损失函数
    criterion_id = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=0.3)

    train_model(model, criterion_id, criterion_triplet, optimizer_ft, exp_lr_scheduler, num_epochs=opt.epochs)