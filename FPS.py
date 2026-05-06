import torch
import time
from train_dual_ablation import DualStreamLPN
# 1. 准备你的模型并设置为推理模式
# model = ST_DualNet(...) # 实例化你的模型
# model.load_state_dict(...) # 加载训练好的权重（其实测速度不加载权重也行，只要结构对就行）
model = DualStreamLPN(class_num=701, droprate=0.5, block=4)
model.load_state_dict(torch.load('./model/Ablation_LPN4/net_120.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 构造一个随机的输入张量 (Batch Size=1, 模拟单张图片推理)
# 根据你论文 4.2 节，输入尺寸是 252x252
dummy_input = torch.randn(1, 3, 252, 252).to(device)

# 3. GPU 预热 (Warm-up)
print("开始 GPU 预热...")
with torch.no_grad():
    for _ in range(50):
        _ = model(dummy_input)

# 4. 正式测速
print("开始正式测速...")
repetitions = 1000 # 跑 1000 次求平均，结果更稳定
total_time = 0

# 确保在开始计时前，所有之前的 CUDA 任务都已完成
torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    for _ in range(repetitions):
        _ = model(dummy_input)

# 确保在结束计时前，当前所有的 CUDA 任务都已完成
torch.cuda.synchronize()
end_time = time.time()

# 5. 计算指标
total_time = end_time - start_time
avg_time_per_image_ms = (total_time / repetitions) * 1000 # 转换为毫秒(ms)
fps = 1000 / avg_time_per_image_ms # Frames Per Second

print(f"总耗时: {total_time:.2f} 秒")
print(f"单张图像推理延迟 (Latency): {avg_time_per_image_ms:.2f} ms")
print(f"FPS: {fps:.2f} 帧/秒")