# ST-DualNet / LPN Cross-view Geo-localization

本项目用于跨视角地理定位检索实验，实验数据集主要包括：

- **SUES-200**
- **University-1652**

项目基于原始 LPN（Local Pattern Network）进行扩展，在局部特征建模的基础上加入 DINOv2 与 ResNet50-LPN 双分支结构。DINOv2 分支用于提取语义特征，ResNet50-LPN 分支用于提取局部纹理特征，并结合多尺度测试、特征融合、重排序、互学习/知识蒸馏等策略进行跨视角图像匹配。

![Overview](docs/index_files/visual.jpg)

## 项目特点

- 支持 **SUES-200** 与 **University-1652** 跨视角地理定位实验。
- 保留原始 LPN 训练与测试流程。
- 提供 DINOv2 + ResNet50-LPN 双分支模型实验代码。
- 支持无人机到卫星、卫星到无人机的双向检索。
- 支持 SUES-200 不同飞行高度下的分组评估。
- 支持多尺度测试、水平翻转增强与 k-reciprocal re-ranking。
- 提供 Grad-CAM、t-SNE、CMC 曲线、消融实验图表等可视化脚本。

## 目录结构

```text
.
├── README.md                         # 原始 LPN 项目说明
├── README_GitHub.md                  # 当前项目 GitHub 说明文档
├── model.py                          # LPN、TwoStreamNet、DINOv2 等模型定义
├── train.py / test.py                # 原始/改进版主训练与测试脚本
├── train_dual*.py                    # University-1652 双分支训练脚本
├── test_dual*.py                     # University-1652 双分支测试脚本
├── train_sues*.py                    # SUES-200 训练脚本
├── test_sues*.py                     # SUES-200 测试脚本
├── evaluate_*.py                     # 评估脚本
├── re_ranking.py                     # 重排序算法
├── prepare_sues.py                   # SUES-200 数据划分脚本
├── draw_*.py / heatmap.py            # 可视化脚本
├── docs/                             # 项目图片资源
├── model/                            # 训练得到的模型权重
├── Paper_Experiments/                # 论文实验与消融结果
├── University-1652/                  # University-1652 数据集
├── SUES-200-512x512/                 # SUES-200 原始格式数据
└── SUES-200-Splitted/                # SUES-200 划分后的训练/测试数据
```

## 环境依赖

建议使用 Python 3.8+ 与 CUDA GPU 环境。DINOv2 与 ResNet50 通过 `torch.hub` 加载，首次运行时需要联网下载模型，或提前准备好本地缓存。

```bash
pip install torch torchvision
pip install numpy scipy matplotlib pillow pyyaml tqdm thop
```

部分旧脚本支持 `apex` FP16 训练，但当前主要脚本可以使用 PyTorch AMP，不强制安装 `apex`。

## 数据集准备

### University-1652

请按照 University-1652 官方说明下载数据集，并保持如下结构：

```text
University-1652/
├── train/
│   ├── drone/
│   ├── satellite/
│   ├── street/
│   └── google/
└── test/
    ├── query_drone/
    ├── gallery_drone/
    ├── query_satellite/
    ├── gallery_satellite/
    ├── query_street/
    └── gallery_street/
```

### SUES-200

如果已有原始格式数据 `SUES-200-512x512`，可运行：

```bash
python prepare_sues.py
```

脚本会生成：

```text
SUES-200-Splitted/
├── train/
│   ├── drone/
│   └── satellite/
└── test/
    ├── drone/
    └── satellite/
```

默认划分方式：

- 训练集：ID 0001-0120
- 测试集：ID 0121-0200

## 训练与测试

### University-1652 / 原始 LPN

可以直接运行原始脚本：

```bash
sh run.sh
```

也可以手动指定参数：

```bash
python train.py \
  --name final_three_view_lpn \
  --data_dir ./University-1652/train \
  --views 3 \
  --share \
  --LPN \
  --block 4 \
  --gpu_ids 0

python test.py \
  --name final_three_view_lpn \
  --test_dir ./University-1652/test \
  --batchsize 128 \
  --gpu_ids 0
```

### University-1652 / 双分支实验

仓库中包含多组 DINOv2 + ResNet50-LPN 双分支实验脚本：

```bash
python train_dual_ablation.py
python train_dual_mutual.py
python train_dual_sota.py
python train_dual_CNNDino2.py
python train_dual_Dino2CNN.py
```

对应测试脚本：

```bash
python test_dual_advanced.py
python test_dual_advanced2.py
```

这些脚本主要用于 University-1652 上的双分支结构、LPN 分块数、互学习、单向蒸馏、多尺度推理等消融实验。运行前请确认脚本中的 `--name`、`--data_dir`、`--test_dir`、`--block` 与当前模型权重一致。

### SUES-200 / ST-DualNet-Light

轻量版训练脚本为 `train_sues_stdualnet_noml.py`。该版本移除了互学习、KL 与 KD 损失，保留 DINOv2 + ResNet50-LPN 双分支和 CE 监督。

```bash
python train_sues_stdualnet_noml.py \
  --name sues200_stdualnet_noml \
  --data_dir ./SUES-200-Splitted \
  --batchsize 32 \
  --epochs 120 \
  --img_h 224 \
  --img_w 224 \
  --block 4 \
  --gpu_ids 0
```

测试并输出整体指标与高度分组指标：

```bash
python test_sues_stdualnet_noml.py \
  --name sues200_stdualnet_noml \
  --which_epoch 120 \
  --test_dir ./SUES-200-Splitted/test \
  --class_num 80 \
  --batchsize 32 \
  --h 224 \
  --w 224 \
  --block 4 \
  --ms 1 \
  --direction both \
  --gpu_ids 0
```

启用多尺度、翻转和重排序：

```bash
python test_sues_stdualnet_noml.py \
  --name sues200_stdualnet_noml \
  --which_epoch best \
  --test_dir ./SUES-200-Splitted/test \
  --class_num 80 \
  --ms 0.9,1,1.1 \
  --use_flip \
  --rerank \
  --direction both \
  --gpu_ids 0
```

## 评估指标

主要评估指标：

- Rank-1
- Rank-5
- Rank-10
- mAP / AP
- SUES-200 不同无人机高度下的分组检索精度

常用评估脚本：

```bash
python evaluate_gpu.py
python evaluate_gpu1.py
python evaluate_rerank.py
python evaluate_baseline.py
```

部分测试脚本会先保存 `pytorch_result.mat` 或 `pytorch_result.pt`，再由评估脚本读取特征并计算指标。

## 可视化与分析

仓库中提供了多种可视化脚本：

```bash
python draw_cam.py
python draw_cam1.py
python draw_cam_satellite.py
python draw_tsne.py
python draw_cmc.py
python draw_paper_figure.py
python heatmap.py
python calc_params.py
python FPS.py
```

已有示例结果：

```text
cam_result_0001_150.png
cam_result_0001_satellite.png
tsne_features_st_dualnet.png
lpn_ablation_line_chart.png
KD_Ablation_BarChart_Elegant.png
Paper_Final_Figure.png
Paper_Final_Figuresues.png
```

## GitHub 上传注意事项

当前文件夹中包含数据集、模型权重和中间结果文件，例如：

```text
University-1652/
SUES-200-512x512/
SUES-200-Splitted/
model/
*.pth
*.pt
*.mat
```

这些文件通常非常大，不建议直接提交到 GitHub。建议添加 `.gitignore`：

```gitignore
University-1652/
SUES-200-512x512/
SUES-200-Splitted/
model/
__pycache__/
*.pyc
*.pth
*.pt
*.mat
*.pkl
*.npy
*.npz
.DS_Store
.idea/
```

如果需要发布模型权重，建议使用 GitHub Releases、Google Drive、Baidu Netdisk 或 Hugging Face，并在 README 中提供下载链接。

## 引用

如果本项目对你的研究有帮助，请引用原始 LPN 与 University-1652 论文：

```bibtex
@ARTICLE{wang2021LPN,
  title={Each Part Matters: Local Patterns Facilitate Cross-View Geo-Localization},
  author={Wang, Tingyu and Zheng, Zhedong and Yan, Chenggang and Zhang, Jiyong and Sun, Yaoqi and Zheng, Bolun and Yang, Yi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  volume={32},
  number={2},
  pages={867-879},
  doi={10.1109/TCSVT.2021.3061265}
}
```

```bibtex
@article{zheng2020university,
  title={University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization},
  author={Zheng, Zhedong and Wei, Yunchao and Yang, Yi},
  journal={ACM Multimedia},
  year={2020}
}
```

## License

本仓库继承原始项目的 MIT License。具体内容见 `LICENSE` 文件。
