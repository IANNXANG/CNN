import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 定义带有 Dropout 的 CNN 模型
class CNN_WithDropout(nn.Module):
    def __init__(self, dropout_prob):
        super(CNN_WithDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_prob)  # 定义Dropout层

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(nn.functional.relu(self.fc1(x)))  # 应用 Dropout
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # 设置设备、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    dropout_probs = [0.01, 0.05, 0.1]  # 不同的Dropout概率

    # 逐个训练模型并保存
    for dropout_prob in dropout_probs:
        model = CNN_WithDropout(dropout_prob).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 5
        train_losses = []

        print(f"\nTraining model with Dropout probability: {dropout_prob * 100}%")
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
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 保存模型和训练损失
        model_filename = f'mnist_cnn_dropout_Task3_{int(dropout_prob * 100)}.pth'
        losses_filename = f'train_losses_dropout_Task3_{int(dropout_prob * 100)}.pth'
        torch.save(model.state_dict(), model_filename)
        torch.save(train_losses, losses_filename)
        print(f"模型和损失已保存，文件名：{model_filename}，{losses_filename}")
