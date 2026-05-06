import scipy.io
import torch
import numpy as np
import os


#######################################################################
# Evaluate
def evaluate(qf, ql, gf, gl):
    # 将 query 特征转为 (D, 1)
    query = qf.reshape(-1, 1)

    # 计算余弦相似度 (假设特征已经归一化)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # 排序：从大到小
    index = np.argsort(score)  # 从小到大
    index = index[::-1]  # 翻转为从大到小

    # 获取正确答案的索引位置
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # 获取垃圾图像索引（通常是 -1）
    junk_index = np.argwhere(gl == -1)

    # 计算 AP 和 CMC
    ap, cmc = compute_mAP(index, good_index, junk_index)

    return ap, cmc, index


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()

    if good_index.size == 0:  # 如果库里没有正确答案
        cmc[0] = -1
        return ap, cmc

    # 移除垃圾图像 (junk_index)
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # 寻找正确匹配在排序中的位置
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    # 更新 CMC
    # rows_good[0] 是第一个正确答案的排名（从 0 开始计数）
    cmc[rows_good[0]:] = 1

    # 计算 AP
    for i in range(ngood):
        d_recall = 1.0 / ngood
        # precision = (相关文档数) / (检索到的文档总数)
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)

        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0

        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


######################################################################
# 主程序
print("正在加载结果文件...")
result = scipy.io.loadmat('pytorch_result.mat')

query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

# 转移到 GPU 加速计算
query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(f"Query Shape: {query_feature.shape}")
print(f"Gallery Shape: {gallery_feature.shape}")

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0

print("\n>>> 开始评估 (显示前 5 个查询的详细情况) <<<")

for i in range(len(query_label)):
    ap_tmp, CMC_tmp, sorted_index = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)

    if CMC_tmp[0] == -1:
        continue

    CMC = CMC + CMC_tmp
    ap += ap_tmp

    # --- DEBUG 核心代码：打印前 10 个查询的详细排位 ---
    if i < 10:
        # 获取第一名预测的 label
        top1_index = sorted_index[0]
        top1_label = gallery_label[top1_index]
        is_correct = (top1_label == query_label[i])

        print(
            f"Query[{i}] ID: {query_label[i]} | Top-1 Predict ID: {top1_label} | {'Correct' if is_correct else 'Wrong'}")
        if not is_correct:
            # 如果错了，看看正确答案排第几
            correct_rank = np.where(gallery_label[sorted_index] == query_label[i])[0]
            if len(correct_rank) > 0:
                print(f"   -> 真实答案排在第: {correct_rank[0] + 1} 名")
            else:
                print(f"   -> 库里找不到 ID 为 {query_label[i]} 的图")

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC

print('\n' + '-' * 40)
print('University-1652 最终结果:')
print(f'Rank-1  : {CMC[0] * 100:.2f}%')
print(f'Rank-5  : {CMC[4] * 100:.2f}%')
print(f'Rank-10 : {CMC[9] * 100:.2f}%')
print(f'mAP     : {ap / len(query_label) * 100:.2f}%')
print('-' * 40)