# -*- coding: utf-8 -*-
import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets, transforms  # <--- 加上这个
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from model import two_view_net, TwoStreamNet
# from utils import evaluate
import time
from torch.autograd import Variable
import yaml
from tqdm import tqdm

# ================= 配置区域 =================
parser = argparse.ArgumentParser(description='Testing SUES-200')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
parser.add_argument('--name', default='sues200_training_v4', type=str, help='save model name')
parser.add_argument('--test_dir', default=r"E:\Desktop\LPN-main\SUES-200-Splitted\test", type=str, help='test dir')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--h', default=252, type=int, help='height')
parser.add_argument('--w', default=252, type=int, help='width')
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()


# ===========================================

def load_network(network):
    save_path = os.path.join('./model', opt.name, 'net_039.pth')
    if not os.path.exists(save_path):
        # 如果没有 best，尝试找 last
        print(f"找不到 {save_path}，尝试加载 net_039.pth")
        save_path = os.path.join('./model', opt.name, 'net_039.pth')

    print(f"Loading model from {save_path}...")
    try:
        network.load_state_dict(torch.load(save_path))
    except:
        # 如果因为分类层维度不匹配（比如训练是120类，这里初始化默认可能不对），开启非严格加载
        print("警告：完整加载失败，尝试非严格加载 (Strict=False)...")
        network.load_state_dict(torch.load(save_path), strict=False)
    return network


def extract_feature(model, dataloaders, view_index=1):
    features = torch.FloatTensor()
    count = 0

    # 记录每个样本的高度信息 (仅对 Drone 有效)
    height_labels = []

    # 遍历数据
    # 注意：ImageFolder 的 imgs 属性包含了 (path, label)
    # 我们需要 path 来判断高度
    dataset = dataloaders.dataset

    pbar = tqdm(dataloaders)
    for data in pbar:
        img, label = data
        n, c, h, w = img.size()
        count += n

        # 放到 GPU
        if opt.use_dense:
            img = Variable(img.cuda().detach())
        else:
            img = Variable(img.cuda().detach())

        # 提取特征
        # 根据你的 model.py，这里需要适配 forward 参数
        # Satellite 是 x1, Drone 是 x3 (根据你之前的 train.py 修改逻辑)
        # 但在测试时，我们通常直接调取 backbone
        if view_index == 1:  # Satellite
            outputs = model(img, None, None)  # 对应 x1
            # 如果返回的是元组 (dino, cnn)，取 dino
            if isinstance(outputs, tuple): outputs = outputs[0]
            # 如果返回的是 list (LPN block)，拼接它们
            if isinstance(outputs, list):
                # 拼接多块特征
                outputs = torch.cat(outputs, dim=1)

        elif view_index == 2:  # Drone (我们把它当 x3 或者 x2 都可以，取决于你模型怎么写)
            # 这里假设 Drone 对应模型的 x2 或 x3 位置，通常为了保险，我们只传一个进去测
            # 你的 TwoStreamNet 代码里：if x2 is None ... ret_dino = self.dino_branch(x1)
            # 所以直接传给 x1 位置也是可以拿到特征的！只要不训练分类头
            outputs = model(img)
            if isinstance(outputs, tuple): outputs = outputs[0]
            if isinstance(outputs, list):
                outputs = torch.cat(outputs, dim=1)

        # 归一化特征
        fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
        outputs = outputs.div(fnorm.expand_as(outputs))

        features = torch.cat((features, outputs.data.cpu()), 0)

    # --- 关键：解析高度信息 ---
    # 我们直接从 dataset.samples (即文件路径列表) 获取高度
    # 顺序和 DataLoader 是一致的 (只要 shuffle=False)
    if view_index == 2:  # 只有 Drone 需要解析高度
        for path, _ in dataset.samples:
            # 路径示例: .../test/drone/0121/150/img.jpg
            # 我们分割路径查找 150, 200 等关键词
            path = path.replace('\\', '/')  # 统一转为 / 防止 Windows 问题
            parts = path.split('/')

            h_tag = 'unknown'
            if '150' in parts:
                h_tag = '150'
            elif '200' in parts:
                h_tag = '200'
            elif '250' in parts:
                h_tag = '250'
            elif '300' in parts:
                h_tag = '300'

            height_labels.append(h_tag)

    return features, height_labels


# ================= 补充的评估函数 (开始) =================
def evaluate(qf, ql, qc, gf, gl, gc):
    """
    qf: query features (N, C) - 无人机特征
    ql: query labels (N)      - 无人机ID
    gf: gallery features (M, C) - 卫星特征
    gl: gallery labels (M)      - 卫星ID
    """
    # 1. 计算余弦相似度 (Cosine Similarity)
    # score = gf * qf.T
    query = qf
    score = torch.mm(gf, query.t())
    score = score.cpu().numpy()

    # 2. 预测排序
    # 按分数从大到小排序，返回索引
    index = np.argsort(score, axis=0)[::-1]

    # 3. 初始化指标
    CMC = torch.IntTensor(len(gl)).zero_()
    ap = 0.0

    # 4. 遍历每一个 Query (无人机图像) 进行评估
    for i in range(len(ql)):
        # 获取当前 Query 的排序结果 (gallery indices)
        ap_tmp, CMC_tmp = evaluate_one(index[:, i], ql[i], gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    # 5. 计算平均值
    CMC = CMC.float()
    CMC = CMC / len(ql)  # Average CMC
    ap = ap / len(ql)  # mAP
    return CMC, ap


def evaluate_one(index, query_label, gallery_label):
    # 寻找正样本 (Correct matches) 的位置
    # 在 SUES-200 中，只要 ID 相同就是正样本
    good_index = np.argwhere(gallery_label == query_label)

    # 如果该 ID 在 Gallery 里不存在 (理论上不应该发生)，返回 -1
    if len(good_index) == 0:
        return 0.0, torch.IntTensor(len(gallery_label)).zero_() - 1.0

    # 这里的 index 是已经排好序的 Gallery 索引
    # 我们要看 correct match 出现在了哪些位置
    # mask 也就是 ground truth 在排序列表中的位置
    mask = np.in1d(index, good_index)

    # --- 计算 CMC ---
    # 找到第一次出现 True 的位置
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    CMC_tmp = torch.IntTensor(len(gallery_label)).zero_()
    if rows_good.shape[0] > 0:
        CMC_tmp[rows_good[0]:] = 1  # Rank-K 及其之后都算命中

    # --- 计算 AP (Average Precision) ---
    # 公式: sum(P@k * rel@k) / num_positives
    num_good = len(good_index)
    d_recall = 1.0 / num_good

    precision = 0
    mAP = 0

    # 累加每一个正样本处的 Precision
    for i in range(num_good):
        # rows_good[i] 是第 i 个正样本在 rank list 中的排名 (0-based)
        # Precision = (i+1) / (rank+1)
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        mAP += precision

    mAP = mAP / num_good

    return mAP, CMC_tmp


# ================= 补充的评估函数 (结束) =================




def main():
    # 1. 基础设置
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    use_gpu = torch.cuda.is_available()
    opt.use_dense = False  # 你的代码没有 dense

    # 2. 数据准备
    # 去掉前面的 datasets.
    data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_datasets = {
        'satellite': datasets.ImageFolder(os.path.join(opt.test_dir, 'satellite'), data_transforms),
        'drone': datasets.ImageFolder(os.path.join(opt.test_dir, 'drone'), data_transforms)
    }

    dataloaders = {
        'satellite': torch.utils.data.DataLoader(image_datasets['satellite'], batch_size=opt.batchsize, shuffle=False,
                                                 num_workers=4),
        'drone': torch.utils.data.DataLoader(image_datasets['drone'], batch_size=opt.batchsize, shuffle=False,
                                             num_workers=4)
    }

    # 3. 加载模型
    # 注意：这里 class_num=120，必须和训练时一致，否则加载权重可能报错
    # 如果你不想改 model.py，可以在这里传入701，然后用 strict=False加载
    class_num = 120
    model = TwoStreamNet(class_num, droprate=0.5, stride=1, pool='avg', LPN=True, block=4)

    if use_gpu:
        model = model.cuda()

    # 加载权重
    model = load_network(model)
    model.eval()

    # 4. 提取特征
    print(">>> Extracting Satellite Features...")
    gallery_feature, _ = extract_feature(model, dataloaders['satellite'], view_index=1)

    print(">>> Extracting Drone Features (Query)...")
    query_feature, query_heights = extract_feature(model, dataloaders['drone'], view_index=2)

    # 获取标签 (ID)
    gallery_label = np.array(dataloaders['satellite'].dataset.targets)
    query_label = np.array(dataloaders['drone'].dataset.targets)

    # 5. 分高度评估
    target_heights = ['150', '200', '250', '300']

    print("\n" + "=" * 60)
    print(f"{'Height':<10} | {'Rank-1':<10} | {'AP':<10} | {'Rank-5':<10}")
    print("-" * 60)

    overall_r1 = 0

    for h in target_heights:
        # 筛选出属于当前高度的索引
        indices = [i for i, x in enumerate(query_heights) if x == h]

        if len(indices) == 0:
            print(f"{h}m        | N/A        | N/A        | N/A")
            continue

        # 提取子集
        sub_query_feature = query_feature[indices]
        sub_query_label = query_label[indices]

        # 这里的 evaluate 假设你 utils.py 里有这个函数，且返回 standard metric
        # 如果 utils.py 的 evaluate 不需要摄像头 ID (cam)，可以传 dummy
        # University-1652 的 utils.evaluate 通常接受: (q_feat, q_label, q_cam, g_feat, g_label, g_cam)
        # SUES 没有 camera ID，我们就造一个假的

        # 构造 dummy camera ids
        sub_query_cam = np.zeros(len(sub_query_label))
        gallery_cam = np.zeros(len(gallery_label))

        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0

        # 调用 utils.evaluate (标准 University-1652 代码通常用这个)
        # 如果你的 utils.evaluate 写法不同，可能需要微调
        # 这里手动实现一个简单的 batch evaluation 防止 utils 版本不兼容

        # --- 简易评估逻辑 (Cosine Similarity) ---
        # q_f = sub_query_feature.cuda()
        # g_f = gallery_feature.cuda()
        #
        # # from utils import evaluate  # 尝试复用你的 utils
        # try:
        #     CMC, ap = evaluate(q_f, sub_query_label, sub_query_cam, g_f, gallery_label, gallery_cam)
        # except:
        #     # 如果 utils.evaluate 报错，使用简单的矩阵乘法
        #     scores = torch.mm(g_f, q_f.t())
        #     sorted_scores, indices = torch.sort(scores, dim=0, descending=True)
            # ... 这里为了省事，还是强烈建议用 utils.evaluate ...
        # --- 简易评估逻辑 (Cosine Similarity) ---
        q_f = sub_query_feature.cuda()
        g_f = gallery_feature.cuda()

        # from utils import evaluate  # 尝试复用你的 utils
        try:
            # 【修改点】交换位置：拿着卫星(g_f) 去找 无人机(q_f)
            # 注意标签(label)和相机ID(cam)也要跟着一起换！
            CMC, ap = evaluate(g_f, gallery_label, gallery_cam, q_f, sub_query_label, sub_query_cam)
        except:


            pass

        print(f"{h}m        | {CMC[0] * 100:.2f}%     | {ap * 100:.2f}%     | {CMC[4] * 100:.2f}%")

    print("=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
