import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



# 定义修改后的CNN模型，减少卷积核数量
class CNN_Modified(nn.Module):
    def __init__(self):
        super(CNN_Modified, self).__init__()
        # 每层卷积核数量减半
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 从32减少到16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 从64减少到32
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # 设置设备、模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_Modified().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型并记录损失
    num_epochs = 5
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录每个 batch 的损失
            train_losses.append(loss.item())
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 保存模型和训练损失
    torch.save(model.state_dict(), 'mnist_cnn_TASK2.pth')
    torch.save(train_losses, 'train_losses_TASK2.pth')
    print("模型和损失已保存。")
