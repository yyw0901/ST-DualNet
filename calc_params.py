import torch
from thop import profile
from thop import clever_format

# 1. 导入你的模型 (请将 your_model_file 替换为你真实的 Python 文件名)
from train_dual_ablation import DualStreamLPN

# 2. 实例化你的网络
model = DualStreamLPN(class_num=701, droprate=0.5)
model.eval() # 切换到评估模式

# 3. 构造一个“假输入” (Dummy Input)
# 根据你论文 4.2 节的设置，输入尺寸是 252x252
# 张量维度：(Batch_Size, Channels, Height, Width)
dummy_input = torch.randn(1, 3, 252, 252)

# 4. 开始计算
print("正在扫描网络结构并计算参数量，请稍候...")
flops, params = profile(model, inputs=(dummy_input, ), verbose=False)

# 5. 格式化输出为学术界常用的 M (兆) 和 G (吉)
flops_str, params_str = clever_format([flops, params], "%.2f")

print("\n" + "="*40)
print(f"ST-DualNet 模型参数量 (Params): {params_str}")
print(f"ST-DualNet 模型计算量 (FLOPs) : {flops_str}")
print("="*40 + "\n")