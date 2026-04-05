import torch
import torch.nn as nn
import numpy as np

# 1. 定义与 C++ 对应的网络结构
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 对应 C++: Linear(784, 128) -> ReLU -> Linear(128, 10)
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def export_to_bin(tensor, filename):
    # 将 Tensor 转为 float32 的 numpy 数组，并存为二进制
    data = tensor.detach().cpu().numpy().astype(np.float32)
    data.tofile(filename)
    print(f"Exported: {filename}, Shape: {data.shape}")

# 2. 实例化并导出（这里你可以加载预训练好的，也可以随机初始化演示）
model = SimpleMLP()

# 注意：PyTorch 的 Linear 层存储权重是 (out_features, in_features)
# 而我们的 C++ matmul 是 (in, out)，所以需要转置 .T
export_to_bin(model.fc1.weight.T, "fc1_w.bin")
export_to_bin(model.fc1.bias, "fc1_b.bin")
export_to_bin(model.fc2.weight.T, "fc2_w.bin")
export_to_bin(model.fc2.bias, "fc2_b.bin")