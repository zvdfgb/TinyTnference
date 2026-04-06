import torch
from torchvision import datasets, transforms
import numpy as np

# 1. 下载并加载一个测试样本
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, 
                   transform=transforms.ToTensor()),
    batch_size=1, shuffle=True)

# 2. 取出一张图及其标签
image, label = next(iter(test_loader))

# 3. 将这张 28x28 的图存为二进制文件
# image 的 shape 是 [1, 1, 28, 28]，我们需要把它展平
image_data = image.numpy().astype(np.float32)
image_data.tofile("test_digit.bin")

print(f"导出成功！这张图真实的数字是: {label.item()}")
print("文件已保存为: test_digit.bin")