import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# 1. 定义更深、更专业的模型
class ProfessionalMLP(nn.Module):
    def __init__(self):
        super(ProfessionalMLP, self).__init__()
        # 784 -> 256 -> 128 -> 10
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        
        # 专业的权重初始化 (Kaiming Initialization)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, 784) # 展平
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# 2. 数据准备：增加标准化 (Normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# 3. 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProfessionalMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
model.train()
for epoch in range(5): # 训练5轮
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed.")

# 4. 导出权重 (记得 .T 转置)
def save_bin(tensor, name):
    tensor.detach().cpu().numpy().astype(np.float32).tofile(f"{name}.bin")

save_bin(model.fc1.weight.T, "fc1_w")
save_bin(model.fc1.bias, "fc1_b")
save_bin(model.fc2.weight.T, "fc2_w")
save_bin(model.fc2.bias, "fc2_b")
save_bin(model.fc3.weight.T, "fc3_w")
save_bin(model.fc3.bias, "fc3_b")
print("All weights exported to .bin files.")