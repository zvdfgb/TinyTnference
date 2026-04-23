import torch
import torchvision
import numpy as np

# 加载测试集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=False, download=True,
    transform=torchvision.transforms.ToTensor()), batch_size=10)

# 取出第一批（10张图）
images, labels = next(iter(test_loader))

# 展平并转成 float32
# 形状从 [10, 1, 28, 28] 变成 [10, 784]
batch_data = images.view(10, 784).numpy().astype(np.float32)

# 保存到二进制文件
batch_data.tofile('batch_test.bin')
# 同时也保存一下真实的标签，方便 C++ 验证对不对
labels.numpy().astype(np.int32).tofile('batch_labels.bin')