# -*- coding: utf-8 -*-
import scipy.io
import torch
import numpy as np
import os
import gc
import sys


# =============================================================================
#  内存优化版完整 Re-ranking (支持低内存运行 Jaccard)
# =============================================================================
def re_ranking_optimized(probFea, galFea, k1=75, k2=25, lambda_value=0.7):
    print(">>> [Re-ranking] 正在启动内存优化版 Jaccard 重排序...")

    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    feat = torch.cat([probFea, galFea])

    # 1. 计算原始欧氏距离 (使用 float16 节省一半内存)
    print(f"   [1/5] 计算全量距离矩阵 ({all_num}x{all_num})...")

    # 转换为 GPU 计算，分块进行以防显存不足
    # 最终结果存为 numpy float16
    original_dist = np.zeros((all_num, all_num), dtype=np.float16)


    feat = feat.cuda()

    chunk_size = 2000
    for i in range(0, all_num, chunk_size):
        end = min(i + chunk_size, all_num)
        feat_batch = feat[i:end]

        # dist = x^2 + y^2 - 2xy
        x2 = torch.pow(feat_batch, 2).sum(dim=1, keepdim=True)
        y2 = torch.pow(feat, 2).sum(dim=1, keepdim=True).t()

        dist_batch = x2 + y2
        dist_batch.addmm_(feat_batch, feat.t(), beta=1, alpha=-2)

        # 转回 CPU 存入大矩阵
        original_dist[i:end] = dist_batch.cpu().numpy().astype(np.float16)

        del dist_batch, x2, y2
        torch.cuda.empty_cache()
        print(f"      - 距离计算进度: {i}/{all_num}", end='\r')
    # print(f"   [1/5] 计算全量距离矩阵 ({all_num}x{all_num})...")
    #
    # # 转换为 GPU 计算，分块进行以防显存不足
    # # 使用 float32，避免 float16 精度过低影响 re-ranking
    # original_dist = np.zeros((all_num, all_num), dtype=np.float32)
    #
    # feat = feat.cuda()
    #
    # chunk_size = 2000
    # for i in range(0, all_num, chunk_size):
    #     end = min(i + chunk_size, all_num)
    #     feat_batch = feat[i:end]
    #
    #     # dist = x^2 + y^2 - 2xy
    #     x2 = torch.pow(feat_batch, 2).sum(dim=1, keepdim=True)
    #     y2 = torch.pow(feat, 2).sum(dim=1, keepdim=True).t()
    #
    #     dist_batch = x2 + y2
    #     dist_batch.addmm_(feat_batch, feat.t(), beta=1, alpha=-2)
    #
    #     # 防止数值误差导致极小负数
    #     dist_batch = torch.clamp(dist_batch, min=0)
    #
    #     # 转回 CPU 存入大矩阵
    #     original_dist[i:end] = dist_batch.cpu().numpy().astype(np.float32)
    #
    #     del dist_batch, x2, y2
    #     torch.cuda.empty_cache()
    #     print(f"      - 距离计算进度: {i}/{all_num}", end='\r')

    del feat
    torch.cuda.empty_cache()
    print("\n   [2/5] 获取 k-reciprocal 邻居 (逐行排序，防OOM)...")

    # 关键优化：不要对整个 NxN 矩阵做 argsort，那是 20GB 内存！
    # 我们逐行处理，只保留前 k1+1 个索引

    initial_rank = np.zeros((all_num, k1 + 1), dtype=np.int32)

    for i in range(all_num):
        # 只要前 k1+1 个最小的
        # argpartition 比 argsort 快且省内存
        row_dist = original_dist[i]
        indices = np.argpartition(row_dist, k1 + 1)[:k1 + 1]
        # 对这小部分做个排序
        sorted_indices = indices[np.argsort(row_dist[indices])]
        initial_rank[i] = sorted_indices

        if i % 5000 == 0:
            print(f"      - 排序进度: {i}/{all_num}", end='\r')

    # 将距离平方 (标准操作)
    original_dist = np.power(original_dist, 2).astype(np.float16)

    # 归一化 (使用 float16 兼容)
    max_val = np.max(original_dist, axis=1, keepdims=True)
    original_dist = original_dist / (max_val + 1e-10)  # 防除零

    # =======================================
    # 标准 Jaccard 逻辑 (完整版)
    # =======================================
    print("\n   [3/5] 计算 Jaccard 权重 (k-reciprocal expansion)...")

    # 这里依然需要 V 矩阵，我们用稀疏思想或者直接计算
    # 为防内存不够，我们只给 query 算
    # 但 Re-ranking 需要全局关系。我们尝试构建 V。

    # 也就是论文里的 V 矩阵
    V = np.zeros((all_num, all_num), dtype=np.float16)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]

        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]

        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.round(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.round(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)

        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

        if i % 2000 == 0:
            print(f"      - Expansion进度: {i}/{all_num}", end='\r')

    # Query Expansion
    if k2 != 1:
        print("\n   [4/5] Query Expansion...")
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    print("\n   [5/5] 计算最终 Jaccard 距离 (Query only)...")
    # 只需要计算 Query 到 Gallery 的距离，不需要算 Gallery 之间的
    # 这能极大节省时间

    # 倒排索引加速
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]

        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)

        if i % 100 == 0:
            print(f"      - Jaccard进度: {i}/{query_num}", end='\r')

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist[:query_num, :] * lambda_value

    # 释放大内存
    del original_dist, V, jaccard_dist
    gc.collect()

    return final_dist[:, query_num:]


# =============================================================================
#  评估函数
# =============================================================================
def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc


# =============================================================================
#  主程序
# =============================================================================
# if __name__ == '__main__':
#     result_path = 'pytorch_result.mat'
#     if not os.path.exists(result_path):
#         print("Error: 找不到 pytorch_result.mat，请先运行 test_dual_advanced.py")
#         exit()
#
#     print(f"Loading {result_path}...")
#     result = scipy.io.loadmat(result_path)
#
#     query_feature = torch.FloatTensor(result['query_f'])
#     query_label = result['query_label'][0]
#     gallery_feature = torch.FloatTensor(result['gallery_f'])
#     gallery_label = result['gallery_label'][0]
#
#     # print(f"Query: {query_feature.shape}, Gallery: {gallery_feature.shape}")
#     #
#     # # 运行终极版重排序
#     # distmat = re_ranking_optimized(query_feature, gallery_feature, k1=20, k2=6, lambda_value=0.3)
#     print(f"Query: {query_feature.shape}, Gallery: {gallery_feature.shape}")
if __name__ == '__main__':
    result_path = 'pytorch_result.pt'  # 更改文件后缀
    if not os.path.exists(result_path):
        print("Error: 找不到 pytorch_result.pt，请先运行 test_dual_advanced2.py")
        exit()

    print(f"Loading {result_path}...")
    # 使用 torch.load 加载
    result = torch.load(result_path)

    # 直接提取 Tensor 和标签 (不需要像 mat 文件那样加 [0] 了)
    query_feature = result['query_f']
    query_label = np.array(result['query_label'])
    gallery_feature = result['gallery_f']
    gallery_label = np.array(result['gallery_label'])

    print(f"Query: {query_feature.shape}, Gallery: {gallery_feature.shape}")

        # ======== 下面的 L2 归一化和重排序代码保持不变 ========
        # ...
    # ================= 修复开始 =================
    print(">>> [Fix] 正在对特征进行强制 L2 归一化...")
    # 这一步是 Re-ranking 生效的绝对前提！
    # 即使训练时归一化了，测试时的 Flip 操作也会破坏它，必须重新归一化。
    query_feature = torch.nn.functional.normalize(query_feature, p=2, dim=1)
    gallery_feature = torch.nn.functional.normalize(gallery_feature, p=2, dim=1)
    # ================= 修复结束 =================

    # 运行终极版重排序
    # 建议针对 Cross-View 任务微调 k1，默认 20 可能太大，建议尝试 10
    distmat = re_ranking_optimized(query_feature, gallery_feature, k1=95, k2=25, lambda_value=0.8)

    print("\n>>> 计算 mAP (Full Re-ranking)...")
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        dist_vec = distmat[i]
        # 逐个 argsort，避免 OOM
        index = np.argsort(dist_vec)

        query_index = np.argwhere(gallery_label == query_label[i])
        junk_index = np.argwhere(gallery_label == -1)

        ap_tmp, CMC_tmp = compute_mAP(index, query_index, junk_index)

        if CMC_tmp[0] == -1: continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)

    print('\n' + '=' * 40)
    print('University-1652 (Full Re-ranking) 结果:')
    print(f'Rank-1  : {CMC[0] * 100:.2f}%')
    print(f'Rank-5  : {CMC[4] * 100:.2f}%')
    print(f'Rank-10 : {CMC[9] * 100:.2f}%')
    print(f'mAP     : {ap / len(query_label) * 100:.2f}%')
    print('=' * 40)
