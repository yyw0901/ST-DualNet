# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
import math
import random
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# =============================================================================
# 1. 参数配置
# =============================================================================
parser = argparse.ArgumentParser(description='Training Dual-Stream LPN (Fixed Data Loading)')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='Ablation_OneWay_CNN2Dino', type=str, help='保存文件夹的名字')
parser.add_argument('--data_dir', default=r"E:\Desktop\LPN-main\University-1652", type=str, help='数据及路径')
parser.add_argument('--train_all', action='store_true', help='是否使用所有数据训练', default=True)
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--epochs', default=120, type=int, help='训练多少轮')
parser.add_argument('--img_h', default=252, type=int, help='统一高度')
parser.add_argument('--img_w', default=252, type=int, help='统一宽度')
parser.add_argument('--block', default=4, type=int, help='LPN 切分数量')

opt = parser.parse_args()

# GPU 设置
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True


# =============================================================================
# 2. 数据处理增强
# =============================================================================
class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability: return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                if img.size()[0] == 3:
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                return img
        return img


# ----------------- 复制开始 -----------------
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # 🔥 核心修复第一步：接收到特征后，立刻转成 float32
        # 这样能解决 "got Float and Half" 的报错，也能保证计算精度
        inputs = inputs.float()

        n = inputs.size(0)

        # 计算成对距离矩阵 (a^2 + b^2)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()

        # 🔥 核心修复第二步：更新 addmm_ 写法，解决 UserWarning
        # 旧写法: dist.addmm_(1, -2, inputs, inputs.t())
        # 新写法: 指定 beta 和 alpha，逻辑是 dist = 1*dist - 2*(inputs @ inputs.t())
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)

        dist = dist.clamp(min=1e-12).sqrt()  # numerical stability

        # 构造正负样本掩码 (这部分不用动)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # y=1 表示 dist_an 应该大于 dist_ap
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


transform_train_list = [
    transforms.Resize((opt.img_h, opt.img_w), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((opt.img_h, opt.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
if opt.erasing_p > 0:
    transform_train_list.append(RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0]))
transform_train = transforms.Compose(transform_train_list)


# 🔥 核心修正：包装类，给数据打上 "Drone" 或 "Sat" 的标签
class SourceTagDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, is_satellite):
        self.dataset = dataset
        self.is_satellite = is_satellite  # 0=Drone, 1=Sat

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, self.is_satellite

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes


print("🔥 正在加载数据集 (修正版 - 深入子文件夹)...")

# 1. 加载 Drone 数据
drone_path = os.path.join(opt.data_dir, 'train', 'drone')
if os.path.exists(drone_path):
    ds_drone = datasets.ImageFolder(drone_path, transform_train)
    tagged_drone = SourceTagDataset(ds_drone, is_satellite=0)
    print(f"   -> 载入无人机数据: {len(ds_drone)} 张 (类别数: {len(ds_drone.classes)})")
else:
    raise RuntimeError(f"❌ 找不到路径: {drone_path}")

# 2. 加载 Satellite 数据
sat_path = os.path.join(opt.data_dir, 'train', 'satellite')
if os.path.exists(sat_path):
    ds_sat = datasets.ImageFolder(sat_path, transform_train)
    tagged_sat = SourceTagDataset(ds_sat, is_satellite=1)
    print(f"   -> 载入卫星数据: {len(ds_sat)} 张 (类别数: {len(ds_sat.classes)})")
else:
    raise RuntimeError(f"❌ 找不到路径: {sat_path}")

# 3. (可选) 加载 Street/Google 数据以增加鲁棒性，这里为了对齐 Rank 暂时只用 Sat+Drone
# 如果您想训练得更猛，可以把下面两行取消注释
# street_path = os.path.join(opt.data_dir, 'train', 'street')
# ds_street = datasets.ImageFolder(street_path, transform_train)
# tagged_street = SourceTagDataset(ds_street, is_satellite=0) # 街景也算非卫星

# 合并数据集
full_dataset = torch.utils.data.ConcatDataset([tagged_drone, tagged_sat])
dataloaders = torch.utils.data.DataLoader(full_dataset, batch_size=opt.batchsize,
                                          shuffle=True, num_workers=0, pin_memory=True)

dataset_sizes = len(full_dataset)
# 获取真实的类别列表 (从 drone 数据集中获取)
class_names = ds_drone.classes
print(f"🔥 最终训练集: {dataset_sizes} 张图片, {len(class_names)} 个真实地点类别")
print("   (注意：这次是从 '0001' 开始分类，不再是 'drone/sat' 分类了！)")


# =============================================================================
# 3. 模型定义
# =============================================================================
class DualStreamLPN(nn.Module):
    def __init__(self, class_num, droprate, block=4):
        super(DualStreamLPN, self).__init__()
        self.block = block

        print("📥 加载 ResNet50...")
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])

        print("📥 加载 DINOv2...")
        self.dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.avgpool = nn.AdaptiveAvgPool2d((1, block))
        self.maxpool = nn.AdaptiveMaxPool2d((1, block))
        self.dino_proj = nn.Linear(384, 2048)
        self.num_ft = 2048

        for i in range(self.block):
            setattr(self, 'classifier_cnn' + str(i), nn.Linear(self.num_ft, class_num))
            setattr(self, 'classifier_dino' + str(i), nn.Linear(self.num_ft, class_num))

    def forward(self, x):
        # CNN
        x_cnn = self.cnn_backbone(x)
        x_cnn = 0.5 * (self.avgpool(x_cnn) + self.maxpool(x_cnn))
        x_cnn = x_cnn.view(x_cnn.size(0), x_cnn.size(1), -1)

        # DINO
        dino_out = self.dino_backbone.forward_features(x)
        patch_tokens = dino_out['x_norm_patchtokens']
        B, N, C = patch_tokens.shape
        grid = int(N ** 0.5)
        x_dino = patch_tokens.permute(0, 2, 1).reshape(B, C, grid, grid)
        x_dino = F.adaptive_avg_pool2d(x_dino, (1, self.block)).squeeze(2).permute(0, 2, 1)
        x_dino = self.dino_proj(x_dino).permute(0, 2, 1)

        if self.training:
            # --- 修改重点：同时返回 预测值 和 特征 ---
            predict = {}
            features = {}
            for i in range(self.block):
                # 提取特征用于 Triplet Loss
                features[i] = x_cnn[:, :, i]
                features[i + self.block] = x_dino[:, :, i]

                # 提取预测用于 ID Loss
                predict[i] = getattr(self, 'classifier_cnn' + str(i))(features[i])
                predict[i + self.block] = getattr(self, 'classifier_dino' + str(i))(features[i + self.block])
            return predict, features  # <--- 返回两个
        else:
            f_cnn = x_cnn.permute(0, 2, 1).contiguous().view(x_cnn.size(0), -1)
            f_dino = x_dino.permute(0, 2, 1).contiguous().view(x_dino.size(0), -1)
            return torch.cat((F.normalize(f_cnn), F.normalize(f_dino)), dim=1)


# =============================================================================
# 4. 训练主循环
# =============================================================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    criterion_triplet = TripletLoss(margin=0.3)  # 新增
    since = time.time()
    scaler = GradScaler()

    save_dir = os.path.join('./model', opt.name)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 20)

        model.train()
        running_loss = 0.0

        # 统计变量
        correct_drone = 0
        total_drone = 0
        correct_sat = 0
        total_sat = 0

        for data in dataloaders:
            # inputs: 图片, labels: 标签, is_sat: 来源(0=无人机, 1=卫星)
            inputs, labels, is_sat = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            is_sat = is_sat.cuda()

            optimizer.zero_grad()

            # with autocast():
            #     outputs, features = model(inputs)
            #     loss = 0
            #
            #     # 累加 Loss 并计算准确率
            #     # -------------------- 互学习核心代码 (替换原循环) --------------------
            #     # 遍历每一对切片 (Block=1 时只循环一次)
            #     for i in range(opt.block):
            #         # 0~block-1 是 CNN 的输出
            #         # block~2*block-1 是 DINO 的输出
            #         out_cnn = outputs[i]
            #         out_dino = outputs[i + opt.block]
            #
            #         # 1. 基础分类 Loss (CrossEntropy): 保证两个分支自己都能分类对
            #         loss_id_cnn = criterion(out_cnn, labels)
            #         loss_id_dino = criterion(out_dino, labels)
            #
            #         # 2. 互学习 Loss (KL Divergence): 核心！
            #         # 让 CNN 的预测分布去逼近 DINO (DINO 教 CNN)
            #         loss_kl_cnn_learn = F.kl_div(
            #             F.log_softmax(out_cnn, dim=1),  # 学生用 log_softmax
            #             F.softmax(out_dino, dim=1),  # 老师用 softmax
            #             reduction='batchmean'
            #         )
            #
            #         # 让 DINO 的预测分布去逼近 CNN (CNN 教 DINO)
            #         loss_kl_dino_learn = F.kl_div(
            #             F.log_softmax(out_dino, dim=1),  # 学生用 log_softmax
            #             F.softmax(out_cnn, dim=1),  # 老师用 softmax
            #             reduction='batchmean'
            #         )
            #
            #         # 🔥 总 Loss = 两个 ID Loss + 1.0 * 两个 KL Loss
            #         # 1.0 是经验权重，效果通常最好
            #         loss += loss_id_cnn + loss_id_dino + 1.0 * (loss_kl_cnn_learn + loss_kl_dino_learn)
            #
            #         # -------------------- 统计准确率 (保持不变) --------------------
            #         with torch.no_grad():
            #             # 我们主要看 DINO 的准确率，因为它通常更准
            #             _, preds = torch.max(out_dino.data, 1)
            #             match_mask = (preds == labels.data)
            #
            #             # 统计卫星
            #             sat_mask = (is_sat == 1)
            #             if sat_mask.sum() > 0:
            #                 correct_sat += match_mask[sat_mask].sum().item()
            #                 total_sat += sat_mask.sum().item()
            #
            #             # 统计无人机
            #             drone_mask = (is_sat == 0)
            #             if drone_mask.sum() > 0:
            #                 correct_drone += match_mask[drone_mask].sum().item()
            #                 total_drone += drone_mask.sum().item()
            #     # -------------------- 修改开始：ID + 互学习 + Triplet 全都要 --------------------
            #     # for i in range(opt.block):
            #     #     # 0~block-1 是 CNN 的输出
            #     #     # block~2*block-1 是 DINO 的输出
            #     #     out_cnn = outputs[i]
            #     #     out_dino = outputs[i + opt.block]
            #     #
            #     #     # 1. 基础分类 Loss (CrossEntropy)
            #     #     loss_id_cnn = criterion(out_cnn, labels)
            #     #     loss_id_dino = criterion(out_dino, labels)
            #     #
            #     #     # 2. 互学习 Loss (KL Divergence)
            #     #     loss_kl_cnn_learn = F.kl_div(
            #     #         F.log_softmax(out_cnn, dim=1),  # CNN 学 DINO
            #     #         F.softmax(out_dino, dim=1),
            #     #         reduction='batchmean'
            #     #     )
            #     #     loss_kl_dino_learn = F.kl_div(
            #     #         F.log_softmax(out_dino, dim=1),  # DINO 学 CNN
            #     #         F.softmax(out_cnn, dim=1),
            #     #         reduction='batchmean'
            #     #     )
            #     #
            #     #     # 3. 🔥 新增：三元组损失 (Triplet Loss) 🔥
            #     #     # features[i] 是 CNN 的特征
            #     #     loss_tri_cnn = criterion_triplet(features[i], labels)
            #     #     # features[i + opt.block] 是 DINO 的特征
            #     #     loss_tri_dino = criterion_triplet(features[i + opt.block], labels)
            #     #
            #     #     # 🔥 终极 Loss 公式 = ID + 1.0*互学习 + 1.0*三元组
            #     #     loss += loss_id_cnn + loss_id_dino + \
            #     #             1.0 * (loss_kl_cnn_learn + loss_kl_dino_learn) + \
            #     #             1.0 * (loss_tri_cnn + loss_tri_dino)
            #     #
            #     #     # -------------------- 统计准确率 (保持不变) --------------------
            #         with torch.no_grad():
            #             # 我们主要看 DINO 的准确率，因为它通常更准
            #             _, preds = torch.max(out_dino.data, 1)
            #             match_mask = (preds == labels.data)
            #
            #             # 统计卫星
            #             sat_mask = (is_sat == 1)
            #             if sat_mask.sum() > 0:
            #                 correct_sat += match_mask[sat_mask].sum().item()
            #                 total_sat += sat_mask.sum().item()
            #
            #             # 统计无人机
            #             drone_mask = (is_sat == 0)
            #             if drone_mask.sum() > 0:
            #                 correct_drone += match_mask[drone_mask].sum().item()
            #                 total_drone += drone_mask.sum().item()
            #     # -------------------- 修改结束 --------------------
            with autocast():
                outputs, features = model(inputs)
                loss = 0

                # -------------------- 🔥 终极形态：ID + Triplet + 互学习 全都要 --------------------
                # for i in range(opt.block):
                #     # 分别取出 CNN 和 DINO 的预测输出和特征
                #     out_cnn = outputs[i]
                #     out_dino = outputs[i + opt.block]
                #     feat_cnn = features[i]
                #     feat_dino = features[i + opt.block]
                #
                #     # 1. 基础分类 Loss (CrossEntropy)
                #     loss_id_cnn = criterion(out_cnn, labels)
                #     loss_id_dino = criterion(out_dino, labels)
                #
                #     # 2. 互学习 Loss (KL Divergence) - 提供超高 mAP
                #     loss_kl_cnn_learn = F.kl_div(
                #         F.log_softmax(out_cnn, dim=1),  # CNN 学 DINO
                #         F.softmax(out_dino, dim=1),
                #         reduction='batchmean'
                #     )
                #     loss_kl_dino_learn = F.kl_div(
                #         F.log_softmax(out_dino, dim=1),  # DINO 学 CNN
                #         F.softmax(out_cnn, dim=1),
                #         reduction='batchmean'
                #     )
                #
                #     # # 3. 三元组 Loss (Triplet) - 提供极致 Rank-1 爆发力
                #     # loss_tri_cnn = criterion_triplet(feat_cnn, labels)
                #     # loss_tri_dino = criterion_triplet(feat_dino, labels)
                #
                #     # 🔥 终极 Loss 融合：(ID + 互学习 + Triplet) 全都要！
                #     # loss += loss_id_cnn + loss_id_dino + \
                #     #         1.0 * (loss_kl_cnn_learn + loss_kl_dino_learn) + \
                #     #         1.0 * (loss_tri_cnn + loss_t 吧ri_dino)
                #     # loss += loss_id_cnn + loss_id_dino + \
                #     #         0.1 * (loss_kl_cnn_learn + loss_kl_dino_learn) + \
                #     #         1.0 * (loss_tri_cnn + loss_tri_dino)
                #     loss += loss_id_cnn + loss_id_dino + 1.0 * (loss_kl_cnn_learn + loss_kl_dino_learn)
                #
                #     # -------------------- 统计准确率 --------------------
                #     with torch.no_grad():
                #         # 使用 DINO 分支的准确率作为监控指标
                #         _, preds = torch.max(out_dino.data, 1)
                #         match_mask = (preds == labels.data)
                #
                #         sat_mask = (is_sat == 1)
                #         if sat_mask.sum() > 0:
                #             correct_sat += match_mask[sat_mask].sum().item()
                #             total_sat += sat_mask.sum().item()
                #
                #         drone_mask = (is_sat == 0)
                #         if drone_mask.sum() > 0:
                #             correct_drone += match_mask[drone_mask].sum().item()
                #             total_drone += drone_mask.sum().item()
                # # -------------------- 终极形态修改结束 --------------------
                # -------------------- 🔥 单向互学习：CNN 教 DINO --------------------
                for i in range(opt.block):
                    # 分别取出 CNN 和 DINO 的预测输出
                    out_cnn = outputs[i]
                    out_dino = outputs[i + opt.block]

                    # 1. 基础分类 Loss (CrossEntropy)
                    loss_id_cnn = criterion(out_cnn, labels)
                    loss_id_dino = criterion(out_dino, labels)

                    # 2. 单向互学习 Loss：CNN 教 DINO
                    # 注意：out_cnn 需要 .detach()，让 CNN 专心做老师，切断这部分梯度回传
                    loss_kl_dino_learn = F.kl_div(
                        F.log_softmax(out_dino, dim=1),  # 学生 DINO
                        F.softmax(out_cnn.detach(), dim=1),  # 老师 CNN (截断梯度)
                        reduction='batchmean'
                    )

                    # 🔥 终极 Loss 融合：(双分支 ID + 单向 KL)
                    loss += loss_id_cnn + loss_id_dino + 1.0 * loss_kl_dino_learn

                    # -------------------- 统计准确率 --------------------
                    with torch.no_grad():
                        # 使用 DINO 分支的准确率作为监控指标
                        _, preds = torch.max(out_dino.data, 1)
                        match_mask = (preds == labels.data)

                        sat_mask = (is_sat == 1)
                        if sat_mask.sum() > 0:
                            correct_sat += match_mask[sat_mask].sum().item()
                            total_sat += sat_mask.sum().item()

                        drone_mask = (is_sat == 0)
                        if drone_mask.sum() > 0:
                            correct_drone += match_mask[drone_mask].sum().item()
                            total_drone += drone_mask.sum().item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / dataset_sizes

        acc_drone = (correct_drone / total_drone) if total_drone > 0 else 0.0
        acc_sat = (correct_sat / total_sat) if total_sat > 0 else 0.0

        print(f'Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'🚁 Drone Acc: {acc_drone:.2%}  (无人机)')
        print(f'🛰️ Sat   Acc: {acc_sat:.2%}  (卫星)')

        # 预警：如果 Acc 又变成 99%，说明哪里不对；这次刚开始应该比较低
        if epoch == 0 and acc_drone > 0.9:
            print("⚠️ 警告：准确率异常高！请确认是否分类数正确！")

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'net_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'✅ 模型已保存: {save_path}')

    time_elapsed = time.time() - since
    print(f'训练完成！总耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model


# =============================================================================
# 5. 运行入口
# =============================================================================
if __name__ == '__main__':
    # 动态获取类别数
    # 如果数据集加载正确，len(class_names) 应该是 701 左右
    num_classes = len(class_names)
    print(f"🛠️ 模型初始化类别数: {num_classes}")

    model = DualStreamLPN(num_classes, opt.droprate, block=opt.block)
    model = model.cuda()

    ignored_params = list(map(id, model.cnn_backbone.parameters())) + list(map(id, model.dino_backbone.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': opt.lr},
        {'params': model.cnn_backbone.parameters(), 'lr': 0.1 * opt.lr},
        {'params': model.dino_backbone.parameters(), 'lr': 0.01 * opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=opt.epochs)