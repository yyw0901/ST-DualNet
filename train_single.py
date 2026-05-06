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
from torch.cuda.amp import autocast, GradScaler

# =============================================================================
# 1. 参数配置
# =============================================================================
parser = argparse.ArgumentParser(description='Training Single-Stream ResNet LPN')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='LPN_ResNet_Baseline_Retrain', type=str, help='保存文件夹的名字')
parser.add_argument('--data_dir', default=r"E:\Desktop\LPN-main\University-1652", type=str, help='数据及路径')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=120, type=int, help='训练多少轮')
parser.add_argument('--img_h', default=252, type=int, help='统一高度')
parser.add_argument('--img_w', default=252, type=int, help='统一宽度')
parser.add_argument('--block', default=4, type=int, help='LPN 切分数量 (单分支通常是4或6)')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
cudnn.benchmark = True


# =============================================================================
# 2. 数据处理增强 (保持与双分支绝对公平一致)
# =============================================================================
class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl;
        self.sh = sh;
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


transform_train = transforms.Compose([
    transforms.Resize((opt.img_h, opt.img_w), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((opt.img_h, opt.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
])

print(">>> 正在加载训练数据...")
ds_drone = datasets.ImageFolder(os.path.join(opt.data_dir, 'train', 'drone'), transform_train)
ds_sat = datasets.ImageFolder(os.path.join(opt.data_dir, 'train', 'satellite'), transform_train)
full_dataset = torch.utils.data.ConcatDataset([ds_drone, ds_sat])
dataloaders = torch.utils.data.DataLoader(full_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=0)
class_names = ds_drone.classes
print(f"训练集总图片数: {len(full_dataset)}, 类别数: {len(class_names)}")


# =============================================================================
# 3. 模型定义 (纯 ResNet50 + LPN)
# =============================================================================
class SingleStreamLPN(nn.Module):
    def __init__(self, class_num, block=4):
        super(SingleStreamLPN, self).__init__()
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
            predict = {}
            for i in range(self.block):
                predict[i] = getattr(self, 'classifier' + str(i))(x[:, :, i])
            return predict
        else:
            return x.permute(0, 2, 1).contiguous().view(x.size(0), -1)


# =============================================================================
# 4. 训练主循环
# =============================================================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=120):
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
                outputs = model(inputs)
                loss = 0
                for i in range(opt.block):
                    loss += criterion(outputs[i], labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            with torch.no_grad():
                _, preds = torch.max(outputs[0].data, 1)  # 以第一个块的准确率作为参考
                corrects += torch.sum(preds == labels.data).item()

        scheduler.step()
        epoch_loss = running_loss / len(full_dataset)
        epoch_acc = corrects / len(full_dataset)

        print(
            f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2%} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'net_{epoch + 1}.pth'))

    time_elapsed = time.time() - since
    print(f'✅ 训练完成！耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model


if __name__ == '__main__':
    model = SingleStreamLPN(len(class_names), block=opt.block).cuda()
    ignored_params = list(map(id, model.cnn_backbone.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': opt.lr},
        {'params': model.cnn_backbone.parameters(), 'lr': 0.1 * opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=opt.epochs)