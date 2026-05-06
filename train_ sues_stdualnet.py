# -*- coding: utf-8 -*-
"""
train_sues_stdualnet_v2.py

SUES-200 training code aligned with the paper:
ST-DualNet: DINOv2-ViT-S/14 semantic branch + ResNet50-LPN texture branch
with bidirectional mutual learning.

Expected dataset structure, either:
  --data_dir E:\\...\\SUES-200
      train/satellite/<class_id>/...
      train/drone/<class_id>/<height>/...

or:
  --data_dir E:\\...\\SUES-200\\train
      satellite/<class_id>/...
      drone/<class_id>/<height>/...
"""
from __future__ import print_function, division

import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms


# =============================================================================
# 1. Parameters
# =============================================================================
parser = argparse.ArgumentParser(description='Training ST-DualNet on SUES-200')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='sues200_stdualnet_ml', type=str, help='folder name for saving checkpoints')
parser.add_argument('--data_dir', default=r'E:\Desktop\LPN-main\SUES-200-Splitted', type=str,
                    help='SUES-200 root path or SUES-200/train path')
parser.add_argument('--batchsize', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='base learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='dropout rate, reserved for compatibility')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--epochs', default=120, type=int, help='number of training epochs')
parser.add_argument('--img_h', default=224, type=int, help='input height')
parser.add_argument('--img_w', default=224, type=int, help='input width')
parser.add_argument('--pad', default=10, type=int, help='random crop padding')
parser.add_argument('--block', default=4, type=int, help='number of LPN partitions')
parser.add_argument('--lambda_ml', default=1.0, type=float, help='weight for bidirectional mutual learning loss')
parser.add_argument('--kl_temp', default=1.0, type=float, help='temperature for KL-divergence loss')
parser.add_argument('--save_every', default=10, type=int, help='save checkpoint every N epochs')
parser.add_argument('--fp16', action='store_true', help='use mixed precision')
parser.add_argument('--num_workers', default=0, type=int, help='dataloader workers')
parser.add_argument('--pretrained_path', default=r'E:\Desktop\LPN-main\model\Ablation_LPN4\net_120.pth', type=str,
                    help='optional University-1652 checkpoint path for transfer learning')
opt = parser.parse_args()

# Keep paper setting fixed by default.
opt.block = 4 if opt.block is None else opt.block

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
str_ids = opt.gpu_ids.split(',')
gpu_ids = [int(x) for x in str_ids if int(x) >= 0]
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True


# =============================================================================
# 2. Dataset and transforms
# =============================================================================
class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.0, 0.0, 0.0)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for _ in range(100):
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


class SourceTagDataset(torch.utils.data.Dataset):
    """Wrap ImageFolder and return an extra source tag: 0=drone, 1=satellite."""
    def __init__(self, dataset, is_satellite):
        self.dataset = dataset
        self.is_satellite = is_satellite

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, target, self.is_satellite

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx


def resolve_train_root(data_dir):
    """Allow --data_dir to be either SUES-200 or SUES-200/train."""
    direct_sat = os.path.join(data_dir, 'satellite')
    direct_drn = os.path.join(data_dir, 'drone')
    nested_sat = os.path.join(data_dir, 'train', 'satellite')
    nested_drn = os.path.join(data_dir, 'train', 'drone')
    if os.path.isdir(direct_sat) and os.path.isdir(direct_drn):
        return data_dir
    if os.path.isdir(nested_sat) and os.path.isdir(nested_drn):
        return os.path.join(data_dir, 'train')
    raise RuntimeError(
        'Cannot find SUES-200 train folders. Expected satellite/ and drone/ under either '\
        f'{data_dir} or {os.path.join(data_dir, "train")}'
    )


transform_train = transforms.Compose([
    transforms.Resize((opt.img_h, opt.img_w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.img_h, opt.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    RandomErasing(probability=opt.erasing_p, mean=(0.0, 0.0, 0.0)) if opt.erasing_p > 0 else nn.Identity(),
])

train_root = resolve_train_root(opt.data_dir)
print(f'>>> Using SUES-200 training root: {train_root}')

drone_path = os.path.join(train_root, 'drone')
sat_path = os.path.join(train_root, 'satellite')

ds_drone = datasets.ImageFolder(drone_path, transform_train)
ds_sat = datasets.ImageFolder(sat_path, transform_train)

if ds_drone.classes != ds_sat.classes:
    print('WARNING: drone and satellite class folders are not exactly identical.')
    print(f'  drone classes: {len(ds_drone.classes)}, satellite classes: {len(ds_sat.classes)}')
    # ImageFolder uses sorted folder names. If names differ, labels may be inconsistent.

full_dataset = torch.utils.data.ConcatDataset([
    SourceTagDataset(ds_drone, is_satellite=0),
    SourceTagDataset(ds_sat, is_satellite=1)
])

dataloader = torch.utils.data.DataLoader(
    full_dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    num_workers=opt.num_workers,
    pin_memory=True,
    drop_last=True
)

class_names = ds_drone.classes
num_classes = len(class_names)
print(f'>>> Drone images: {len(ds_drone)}, Satellite images: {len(ds_sat)}')
print(f'>>> Number of SUES-200 training classes: {num_classes}')


# =============================================================================
# 3. Model: ST-DualNet
# =============================================================================
class STDualNet(nn.Module):
    """
    DINOv2-ViT-S/14 semantic branch + ResNet50-LPN texture branch.

    Training returns logits and part-level features.
    Evaluation returns weighted concatenated retrieval features.
    """
    def __init__(self, class_num, block=4, alpha=1.0, beta=1.5):
        super(STDualNet, self).__init__()
        self.block = block
        self.alpha = alpha  # texture weight
        self.beta = beta    # semantic weight
        self.num_ft = 2048

        print('>>> Loading ResNet50 texture branch...')
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])

        print('>>> Loading DINOv2-ViT-S/14 semantic branch...')
        self.dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.avgpool = nn.AdaptiveAvgPool2d((1, block))
        self.maxpool = nn.AdaptiveMaxPool2d((1, block))

        # Project 384-D DINO tokens to 2048-D per part, producing 2048*4=8192 DINO feature.
        # This follows your University-1652 training code. If your final paper strictly reports
        # 384-D semantic feature, remove this projection and adapt classifiers accordingly.
        self.dino_proj = nn.Linear(384, self.num_ft)

        for i in range(self.block):
            setattr(self, 'classifier_cnn' + str(i), nn.Linear(self.num_ft, class_num))
            setattr(self, 'classifier_dino' + str(i), nn.Linear(self.num_ft, class_num))

    def forward(self, x):
        # CNN texture branch: B x 2048 x H x W -> B x 2048 x block
        x_cnn = self.cnn_backbone(x)
        x_cnn = 0.5 * (self.avgpool(x_cnn) + self.maxpool(x_cnn))
        x_cnn = x_cnn.view(x_cnn.size(0), x_cnn.size(1), -1)

        # DINO semantic branch: patch tokens -> pseudo feature map -> part pooling
        dino_out = self.dino_backbone.forward_features(x)
        patch_tokens = dino_out['x_norm_patchtokens']  # B x N x 384
        B, N, C = patch_tokens.shape

        h_grid = x.shape[2] // 14
        w_grid = x.shape[3] // 14
        if h_grid * w_grid != N:
            grid = int(N ** 0.5)
            h_grid, w_grid = grid, grid

        x_dino = patch_tokens.permute(0, 2, 1).reshape(B, C, h_grid, w_grid)
        x_dino = F.adaptive_avg_pool2d(x_dino, (1, self.block)).squeeze(2).permute(0, 2, 1)
        x_dino = self.dino_proj(x_dino).permute(0, 2, 1)  # B x 2048 x block

        if self.training:
            logits = {}
            features = {}
            for i in range(self.block):
                features[i] = x_cnn[:, :, i]
                features[i + self.block] = x_dino[:, :, i]
                logits[i] = getattr(self, 'classifier_cnn' + str(i))(features[i])
                logits[i + self.block] = getattr(self, 'classifier_dino' + str(i))(features[i + self.block])
            return logits, features

        f_cnn = x_cnn.permute(0, 2, 1).contiguous().view(x_cnn.size(0), -1)
        f_dino = x_dino.permute(0, 2, 1).contiguous().view(x_dino.size(0), -1)

        f_cnn = F.normalize(f_cnn, p=2, dim=1)
        f_dino = F.normalize(f_dino, p=2, dim=1)
        fused = torch.cat((self.alpha * f_cnn, self.beta * f_dino), dim=1)
        return F.normalize(fused, p=2, dim=1)


# =============================================================================
# 4. Loss and transfer loading
# =============================================================================
def kl_mutual_loss(logits_a, logits_b, temp=1.0):
    """Symmetric KL divergence with temperature."""
    log_pa = F.log_softmax(logits_a / temp, dim=1)
    log_pb = F.log_softmax(logits_b / temp, dim=1)
    pa = F.softmax(logits_a / temp, dim=1)
    pb = F.softmax(logits_b / temp, dim=1)
    loss_ab = F.kl_div(log_pa, pb.detach(), reduction='batchmean')
    loss_ba = F.kl_div(log_pb, pa.detach(), reduction='batchmean')
    return (temp ** 2) * (loss_ab + loss_ba)


def load_pretrained_backbone(model, pretrained_path):
    if not pretrained_path:
        return model
    if not os.path.isfile(pretrained_path):
        print(f'WARNING: pretrained checkpoint not found: {pretrained_path}')
        return model

    print(f'>>> Loading transfer checkpoint: {pretrained_path}')
    pretrained = torch.load(pretrained_path, map_location='cpu')
    if isinstance(pretrained, dict) and 'state_dict' in pretrained:
        pretrained = pretrained['state_dict']

    model_dict = model.state_dict()
    matched = {}
    skipped = []
    for k, v in pretrained.items():
        key = k.replace('module.', '')
        if key in model_dict and model_dict[key].shape == v.shape:
            matched[key] = v
        else:
            skipped.append(key)
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    print(f'>>> Loaded matched layers: {len(matched)}; skipped layers: {len(skipped)}')
    return model


def build_optimizer(model):
    backbone_ids = set(map(id, model.cnn_backbone.parameters())) | set(map(id, model.dino_backbone.parameters()))
    base_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return optim.SGD([
        {'params': base_params, 'lr': opt.lr},
        {'params': model.cnn_backbone.parameters(), 'lr': 0.1 * opt.lr},
        {'params': model.dino_backbone.parameters(), 'lr': 0.01 * opt.lr},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)


# =============================================================================
# 5. Training loop
# =============================================================================
def train_model(model, criterion, optimizer, scheduler):
    since = time.time()
    scaler = GradScaler(enabled=opt.fp16)
    save_dir = os.path.join('./model', opt.name)
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')

    for epoch in range(opt.epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, opt.epochs))
        print('-' * 30)
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_ml = 0.0

        correct_drone, total_drone = 0, 0
        correct_sat, total_sat = 0, 0

        for inputs, labels, is_sat in dataloader:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            is_sat = is_sat.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=opt.fp16):
                logits, _ = model(inputs)
                loss = 0.0
                ce_loss_value = 0.0
                ml_loss_value = 0.0

                for i in range(opt.block):
                    out_cnn = logits[i]
                    out_dino = logits[i + opt.block]

                    loss_id_cnn = criterion(out_cnn, labels)
                    loss_id_dino = criterion(out_dino, labels)
                    loss_ml = kl_mutual_loss(out_cnn, out_dino, temp=opt.kl_temp)

                    loss = loss + loss_id_cnn + loss_id_dino + opt.lambda_ml * loss_ml
                    ce_loss_value = ce_loss_value + loss_id_cnn.detach() + loss_id_dino.detach()
                    ml_loss_value = ml_loss_value + loss_ml.detach()

                    with torch.no_grad():
                        _, preds = torch.max(out_dino.data, 1)
                        match = preds.eq(labels.data)
                        sat_mask = is_sat.eq(1)
                        drn_mask = is_sat.eq(0)
                        if sat_mask.sum() > 0:
                            correct_sat += match[sat_mask].sum().item()
                            total_sat += sat_mask.sum().item()
                        if drn_mask.sum() > 0:
                            correct_drone += match[drn_mask].sum().item()
                            total_drone += drn_mask.sum().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = inputs.size(0)
            running_loss += loss.item() * bs
            running_ce += ce_loss_value.item() * bs
            running_ml += ml_loss_value.item() * bs

        scheduler.step()

        epoch_loss = running_loss / len(full_dataset)
        epoch_ce = running_ce / len(full_dataset)
        epoch_ml = running_ml / len(full_dataset)
        acc_drone = correct_drone / total_drone if total_drone > 0 else 0.0
        acc_sat = correct_sat / total_sat if total_sat > 0 else 0.0

        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f'Loss: {epoch_loss:.4f} | CE: {epoch_ce:.4f} | ML: {epoch_ml:.4f}')
        print(f'LR(base/cnn/dino): {current_lrs[0]:.6g} / {current_lrs[1]:.6g} / {current_lrs[2]:.6g}')
        print(f'Drone Acc: {acc_drone:.2%} | Satellite Acc: {acc_sat:.2%}')

        if (epoch + 1) % opt.save_every == 0:
            ckpt = os.path.join(save_dir, f'net_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt)
            print(f'>>> Saved checkpoint: {ckpt}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(save_dir, 'net_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f'>>> Saved best checkpoint: {best_path}')

    elapsed = time.time() - since
    print(f'Finished training in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')


if __name__ == '__main__':
    model = STDualNet(num_classes, block=opt.block, alpha=1.0, beta=1.5)
    if opt.pretrained_path and os.path.exists(opt.pretrained_path):
        print(f">>> Loading University-1652 pretrained weights from: {opt.pretrained_path}")
        pretrained_dict = torch.load(opt.pretrained_path, map_location='cpu')
        model_dict = model.state_dict()

        matched_dict = {}
        skipped_keys = []

        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                matched_dict[k] = v
            else:
                skipped_keys.append(k)

        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)

        print(f">>> Loaded {len(matched_dict)} layers from University-1652 pretrained model.")
        print(f">>> Skipped {len(skipped_keys)} layers, usually classification heads due to class number mismatch.")
    else:
        print(">>> No valid University-1652 pretrained weights found. Training from ImageNet/DINOv2 initialization.")
    model = load_pretrained_backbone(model, opt.pretrained_path)
    model = model.cuda()

    optimizer_ft = build_optimizer(model)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    criterion_ce = nn.CrossEntropyLoss()

    train_model(model, criterion_ce, optimizer_ft, scheduler)
