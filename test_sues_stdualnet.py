# -*- coding: utf-8 -*-
"""
test_sues_stdualnet_v2.py

SUES-200 testing code aligned with the paper:
- ST-DualNet feature extraction
- weighted concatenation: texture weight alpha=1.0, semantic weight beta=1.5
- multi-scale inference: 0.9, 1.0, 1.1 by default
- Drone->Satellite and Satellite->Drone evaluation
- height-wise evaluation for 150m/200m/250m/300m
- optional k-reciprocal re-ranking

Expected test structure:
  --test_dir E:\\...\\SUES-200-Splitted\\test
      satellite/<class_id>/...
      drone/<class_id>/<height>/...
"""
from __future__ import print_function, division

import argparse
import gc
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm


# =============================================================================
# 1. Parameters
# =============================================================================
parser = argparse.ArgumentParser(description='Test ST-DualNet on SUES-200')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='sues200_stdualnet_ml', type=str, help='checkpoint folder name')
parser.add_argument('--test_dir', default=r'E:\Desktop\LPN-main\SUES-200-Splitted\test', type=str, help='SUES-200 test path')
parser.add_argument('--which_epoch', default='best', type=str, help='best, last, or epoch number, e.g. 120')
parser.add_argument('--batchsize', default=32, type=int, help='batch size')
parser.add_argument('--h', default=224, type=int, help='input height')
parser.add_argument('--w', default=224, type=int, help='input width')
parser.add_argument('--block', default=4, type=int, help='number of LPN partitions')
parser.add_argument('--num_classes', default=120, type=int, help='number of SUES-200 train classes')
parser.add_argument('--ms', default='0.9,1,1.1', type=str, help='multi-scale testing, e.g. 0.9,1,1.1')
parser.add_argument('--rerank', default=True, action='store_true', help='enable k-reciprocal re-ranking')
parser.add_argument('--k1', default=75, type=int, help='re-ranking k1')
parser.add_argument('--k2', default=25, type=int, help='re-ranking k2')
parser.add_argument('--lambda_value', default=0.7, type=float, help='re-ranking lambda')
parser.add_argument('--num_workers', default=0, type=int, help='dataloader workers')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
cudnn.benchmark = True
ms = [float(s) for s in opt.ms.split(',')]


# =============================================================================
# 2. Model: must match train_sues_stdualnet_v2.py
# =============================================================================
class STDualNet(nn.Module):
    def __init__(self, class_num, block=4, alpha=1.0, beta=1.5):
        super(STDualNet, self).__init__()
        self.block = block
        self.alpha = alpha
        self.beta = beta
        self.num_ft = 2048

        print('>>> Initializing ResNet50 and DINOv2-ViT-S/14 for testing...')
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.avgpool = nn.AdaptiveAvgPool2d((1, block))
        self.maxpool = nn.AdaptiveMaxPool2d((1, block))
        self.dino_proj = nn.Linear(384, self.num_ft)

        for i in range(self.block):
            setattr(self, 'classifier_cnn' + str(i), nn.Linear(self.num_ft, class_num))
            setattr(self, 'classifier_dino' + str(i), nn.Linear(self.num_ft, class_num))

    def forward(self, x):
        x_cnn = self.cnn_backbone(x)
        x_cnn = 0.5 * (self.avgpool(x_cnn) + self.maxpool(x_cnn))
        x_cnn = x_cnn.view(x_cnn.size(0), x_cnn.size(1), -1)

        dino_out = self.dino_backbone.forward_features(x)
        patch_tokens = dino_out['x_norm_patchtokens']
        B, N, C = patch_tokens.shape
        h_grid = x.shape[2] // 14
        w_grid = x.shape[3] // 14
        if h_grid * w_grid != N:
            grid = int(N ** 0.5)
            h_grid, w_grid = grid, grid

        x_dino = patch_tokens.permute(0, 2, 1).reshape(B, C, h_grid, w_grid)
        x_dino = F.adaptive_avg_pool2d(x_dino, (1, self.block)).squeeze(2).permute(0, 2, 1)
        x_dino = self.dino_proj(x_dino).permute(0, 2, 1)

        f_cnn = x_cnn.permute(0, 2, 1).contiguous().view(x_cnn.size(0), -1)
        f_dino = x_dino.permute(0, 2, 1).contiguous().view(x_dino.size(0), -1)

        f_cnn = F.normalize(f_cnn, p=2, dim=1)
        f_dino = F.normalize(f_dino, p=2, dim=1)
        fused = torch.cat((self.alpha * f_cnn, self.beta * f_dino), dim=1)
        return F.normalize(fused, p=2, dim=1)


# =============================================================================
# 3. Utility functions
# =============================================================================
def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    return img.index_select(3, inv_idx)


def load_network(network):
    model_dir = os.path.join('./model', opt.name)
    if opt.which_epoch.lower() == 'best':
        save_path = os.path.join(model_dir, 'net_best.pth')
    elif opt.which_epoch.lower() == 'last':
        candidates = []
        if os.path.isdir(model_dir):
            for f in os.listdir(model_dir):
                if f.startswith('net_') and f.endswith('.pth') and f[4:-4].isdigit():
                    candidates.append((int(f[4:-4]), os.path.join(model_dir, f)))
        if not candidates:
            raise RuntimeError(f'No epoch checkpoints found in {model_dir}')
        save_path = sorted(candidates)[-1][1]
    else:
        epoch = opt.which_epoch
        save_path = os.path.join(model_dir, f'net_{epoch}.pth')

    if not os.path.isfile(save_path):
        raise RuntimeError(f'Checkpoint not found: {save_path}')

    print(f'>>> Loading checkpoint: {save_path}')
    state = torch.load(save_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    network.load_state_dict(state, strict=True)
    return network


def get_labels(dataset):
    return np.array(dataset.targets)


def get_drone_heights(dataset):
    heights = []
    for path, _ in dataset.samples:
        path = path.replace('\\', '/')
        parts = path.split('/')
        tag = 'unknown'
        for h in ['150', '200', '250', '300']:
            if h in parts:
                tag = h
                break
        heights.append(tag)
    return np.array(heights)


@torch.no_grad()
def extract_feature(model, dataloader):
    features = torch.FloatTensor()
    patch_size = 14
    total = len(dataloader.dataset)
    count = 0

    model.eval()
    for img, _ in tqdm(dataloader, desc='Extracting features'):
        n, c, h, w = img.size()
        count += n
        ff = None

        for scale in ms:
            for flip in [False, True]:
                if scale != 1:
                    new_h = int(round((h * scale) / patch_size) * patch_size)
                    new_w = int(round((w * scale) / patch_size) * patch_size)
                    current_img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
                else:
                    current_img = img.clone()

                if flip:
                    current_img = fliplr(current_img)

                current_img = current_img.cuda(non_blocking=True)
                outputs = model(current_img)
                ff = outputs if ff is None else ff + outputs

        ff = F.normalize(ff, p=2, dim=1)
        features = torch.cat((features, ff.cpu()), dim=0)

    return features


def compute_ap_cmc(sorted_index, query_label, gallery_label):
    good_index = np.argwhere(gallery_label == query_label).flatten()
    junk_index = np.argwhere(gallery_label == -1).flatten()

    if good_index.size == 0:
        return 0.0, torch.IntTensor(len(gallery_label)).zero_() - 1

    mask = np.in1d(sorted_index, junk_index, invert=True)
    sorted_index = sorted_index[mask]

    mask = np.in1d(sorted_index, good_index)
    rows_good = np.argwhere(mask).flatten()
    if rows_good.size == 0:
        return 0.0, torch.IntTensor(len(gallery_label)).zero_() - 1

    cmc = torch.IntTensor(len(gallery_label)).zero_()
    cmc[rows_good[0]:] = 1

    ap = 0.0
    num_good = len(good_index)
    for i in range(num_good):
        d_recall = 1.0 / num_good
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        old_precision = i * 1.0 / rows_good[i] if rows_good[i] != 0 else 1.0
        ap += d_recall * (old_precision + precision) / 2
    return ap, cmc


def evaluate_by_similarity(query_feature, query_label, gallery_feature, gallery_label):
    query_feature = F.normalize(query_feature, p=2, dim=1)
    gallery_feature = F.normalize(gallery_feature, p=2, dim=1)

    score = torch.mm(gallery_feature, query_feature.t()).numpy()
    index = np.argsort(score, axis=0)[::-1]

    cmc = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    valid_queries = 0

    for i in range(len(query_label)):
        ap_tmp, cmc_tmp = compute_ap_cmc(index[:, i], query_label[i], gallery_label)
        if cmc_tmp[0] == -1:
            continue
        cmc += cmc_tmp
        ap += ap_tmp
        valid_queries += 1

    if valid_queries == 0:
        return None, 0.0, 0
    cmc = cmc.float() / valid_queries
    ap = ap / valid_queries
    return cmc, ap, valid_queries


def re_ranking_optimized(probFea, galFea, k1=75, k2=25, lambda_value=0.7):
    """Memory-optimized k-reciprocal re-ranking adapted from your evaluate_rerank.py."""
    print('>>> [Re-ranking] Running memory-optimized k-reciprocal re-ranking...')
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    feat = torch.cat([probFea, galFea], dim=0).cuda()

    original_dist = np.zeros((all_num, all_num), dtype=np.float16)
    chunk_size = 2000
    for i in range(0, all_num, chunk_size):
        end = min(i + chunk_size, all_num)
        feat_batch = feat[i:end]
        x2 = torch.pow(feat_batch, 2).sum(dim=1, keepdim=True)
        y2 = torch.pow(feat, 2).sum(dim=1, keepdim=True).t()
        dist_batch = x2 + y2
        dist_batch.addmm_(feat_batch, feat.t(), beta=1, alpha=-2)
        dist_batch = torch.clamp(dist_batch, min=0)
        original_dist[i:end] = dist_batch.cpu().numpy().astype(np.float16)
        del dist_batch, x2, y2
        torch.cuda.empty_cache()
    del feat
    torch.cuda.empty_cache()

    k1 = min(k1, all_num - 1)
    k2 = min(k2, all_num - 1)
    initial_rank = np.zeros((all_num, k1 + 1), dtype=np.int32)
    for i in range(all_num):
        row_dist = original_dist[i]
        idx = np.argpartition(row_dist, k1 + 1)[:k1 + 1]
        initial_rank[i] = idx[np.argsort(row_dist[idx])]

    original_dist = np.power(original_dist, 2).astype(np.float16)
    original_dist = original_dist / (np.max(original_dist, axis=1, keepdims=True) + 1e-10)

    V = np.zeros((all_num, all_num), dtype=np.float16)
    for i in range(all_num):
        forward = initial_rank[i]
        backward = initial_rank[forward, :k1 + 1]
        fi = np.where(backward == i)[0]
        reciprocal = forward[fi]
        reciprocal_expansion = reciprocal
        for j in range(len(reciprocal)):
            candidate = reciprocal[j]
            cand_forward = initial_rank[candidate, :int(np.round(k1 / 2)) + 1]
            cand_backward = initial_rank[cand_forward, :int(np.round(k1 / 2)) + 1]
            fi_candidate = np.where(cand_backward == candidate)[0]
            cand_reciprocal = cand_forward[fi_candidate]
            if len(cand_reciprocal) > 0 and len(np.intersect1d(cand_reciprocal, reciprocal)) > 2 / 3 * len(cand_reciprocal):
                reciprocal_expansion = np.append(reciprocal_expansion, cand_reciprocal)
        reciprocal_expansion = np.unique(reciprocal_expansion)
        weight = np.exp(-original_dist[i, reciprocal_expansion])
        V[i, reciprocal_expansion] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank
    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        ind_nonzero = np.where(V[i, :] != 0)[0]
        ind_images = [inv_index[ind] for ind in ind_nonzero]
        for j in range(len(ind_nonzero)):
            temp_min[0, ind_images[j]] += np.minimum(V[i, ind_nonzero[j]], V[ind_images[j], ind_nonzero[j]])
        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist[:query_num, :] * lambda_value
    del original_dist, V, jaccard_dist
    gc.collect()
    return final_dist[:, query_num:]


def evaluate_by_rerank(query_feature, query_label, gallery_feature, gallery_label):
    query_feature = F.normalize(query_feature, p=2, dim=1)
    gallery_feature = F.normalize(gallery_feature, p=2, dim=1)
    distmat = re_ranking_optimized(query_feature, gallery_feature, k1=opt.k1, k2=opt.k2, lambda_value=opt.lambda_value)

    cmc = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    valid_queries = 0
    for i in range(len(query_label)):
        index = np.argsort(distmat[i])  # smaller distance is better
        ap_tmp, cmc_tmp = compute_ap_cmc(index, query_label[i], gallery_label)
        if cmc_tmp[0] == -1:
            continue
        cmc += cmc_tmp
        ap += ap_tmp
        valid_queries += 1

    if valid_queries == 0:
        return None, 0.0, 0
    cmc = cmc.float() / valid_queries
    ap = ap / valid_queries
    return cmc, ap, valid_queries


def evaluate(query_feature, query_label, gallery_feature, gallery_label):
    if opt.rerank:
        return evaluate_by_rerank(query_feature, query_label, gallery_feature, gallery_label)
    return evaluate_by_similarity(query_feature, query_label, gallery_feature, gallery_label)


def print_metrics(title, cmc, ap, valid_queries):
    print('\n' + '=' * 70)
    print(title)
    if cmc is None:
        print('No valid queries.')
        print('=' * 70)
        return
    r1 = cmc[0].item() * 100
    r5 = cmc[4].item() * 100 if len(cmc) > 4 else 0
    r10 = cmc[9].item() * 100 if len(cmc) > 9 else 0
    print(f'Valid queries: {valid_queries}')
    print(f'Rank-1 : {r1:.2f}%')
    print(f'Rank-5 : {r5:.2f}%')
    print(f'Rank-10: {r10:.2f}%')
    print(f'AP     : {ap * 100:.2f}%')
    print('=' * 70)


def subset_by_indices(features, labels, indices):
    return features[indices], labels[indices]


# =============================================================================
# 4. Main
# =============================================================================
def main():
    transform_test = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    sat_dir = os.path.join(opt.test_dir, 'satellite')
    drone_dir = os.path.join(opt.test_dir, 'drone')
    if not os.path.isdir(sat_dir):
        raise RuntimeError(f'Cannot find satellite test folder: {sat_dir}')
    if not os.path.isdir(drone_dir):
        raise RuntimeError(f'Cannot find drone test folder: {drone_dir}')

    sat_dataset = datasets.ImageFolder(sat_dir, transform_test)
    drone_dataset = datasets.ImageFolder(drone_dir, transform_test)

    if sat_dataset.classes != drone_dataset.classes:
        print('WARNING: satellite and drone class folders are not exactly identical.')
        print(f'  satellite classes: {len(sat_dataset.classes)}, drone classes: {len(drone_dataset.classes)}')

    sat_loader = torch.utils.data.DataLoader(sat_dataset, batch_size=opt.batchsize, shuffle=False,
                                             num_workers=opt.num_workers, pin_memory=True)
    drone_loader = torch.utils.data.DataLoader(drone_dataset, batch_size=opt.batchsize, shuffle=False,
                                               num_workers=opt.num_workers, pin_memory=True)

    print(f'>>> Satellite test images: {len(sat_dataset)}')
    print(f'>>> Drone test images: {len(drone_dataset)}')
    print(f'>>> Multi-scale setting: {ms}')
    print(f'>>> Re-ranking: {opt.rerank}')

    model = STDualNet(opt.num_classes, block=opt.block, alpha=1.0, beta=1.5)
    model = load_network(model)
    model = model.cuda()
    model.eval()

    print('\n>>> Extracting satellite features...')
    sat_feature = extract_feature(model, sat_loader)
    sat_label = get_labels(sat_dataset)

    print('\n>>> Extracting drone features...')
    drone_feature = extract_feature(model, drone_loader)
    drone_label = get_labels(drone_dataset)
    drone_heights = get_drone_heights(drone_dataset)

    # Overall two retrieval settings
    cmc, ap, valid = evaluate(drone_feature, drone_label, sat_feature, sat_label)
    print_metrics('SUES-200 Overall: Drone→Satellite', cmc, ap, valid)

    cmc, ap, valid = evaluate(sat_feature, sat_label, drone_feature, drone_label)
    print_metrics('SUES-200 Overall: Satellite→Drone', cmc, ap, valid)

    # Height-wise evaluation
    for h in ['150', '200', '250', '300']:
        idx = np.where(drone_heights == h)[0]
        if len(idx) == 0:
            print(f'\nHeight {h}m: no drone images found.')
            continue

        # Drone->Satellite at a certain height: query is drone subset, gallery is all satellites
        qf, ql = subset_by_indices(drone_feature, drone_label, idx)
        cmc, ap, valid = evaluate(qf, ql, sat_feature, sat_label)
        print_metrics(f'SUES-200 Height {h}m: Drone→Satellite', cmc, ap, valid)

        # Satellite->Drone at a certain height: query is all satellites, gallery is drone subset at this height
        gf, gl = subset_by_indices(drone_feature, drone_label, idx)
        cmc, ap, valid = evaluate(sat_feature, sat_label, gf, gl)
        print_metrics(f'SUES-200 Height {h}m: Satellite→Drone', cmc, ap, valid)


if __name__ == '__main__':
    main()
