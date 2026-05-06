import scipy.io
import torch
import numpy as np
import os


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


if __name__ == '__main__':
    # result_path = 'pytorch_result.mat'
    result_path = 'pytorch_result_single_ms_sat2drone.mat'
    if not os.path.exists(result_path):
        print("Error: 找不到 pytorch_result.mat")
        exit()

    print(">>> 正在加载特征...")
    result = scipy.io.loadmat(result_path)
    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]

    print(">>> 正在进行 L2 归一化...")
    query_feature = torch.nn.functional.normalize(query_feature, p=2, dim=1)
    gallery_feature = torch.nn.functional.normalize(gallery_feature, p=2, dim=1)

    print(">>> 正在计算真实原始距离 (不带重排序)...")
    original_dist = 1.0 - torch.mm(query_feature, gallery_feature.t()).numpy()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        index = np.argsort(original_dist[i])
        query_index = np.argwhere(gallery_label == query_label[i])
        junk_index = np.argwhere(gallery_label == -1)
        ap_tmp, CMC_tmp = compute_mAP(index, query_index, junk_index)
        if CMC_tmp[0] == -1: continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float() / len(query_label)
    print('\n' + '🔥' * 20)
    print('🏆 最终版 v5 真实硬实力 (Drone -> Sat) 结果:')
    print(f'Rank-1  : {CMC[0] * 100:.2f}%  <--- 看这里！')
    print(f'Rank-5  : {CMC[4] * 100:.2f}%')
    print(f'Rank-10 : {CMC[9] * 100:.2f}%')
    print(f'mAP     : {ap / len(query_label) * 100:.2f}%')
    print('🔥' * 20)