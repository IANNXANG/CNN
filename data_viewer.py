import torchvision
import torchvision.transforms as transforms
import torch

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 读取训练集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 读取测试集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 打印数据集形状
print(f'训练集大小: {len(train_dataset)}')
print(f'测试集大小: {len(test_dataset)}')

# 获取一个批次的数据
images, labels = next(iter(train_loader))

# 打印批次数据的形状
print(f'批次图像形状: {images.shape}')  # 应为 (64, 1, 28, 28)
print(f'批次标签形状: {labels.shape}')  # 应为 (64,)

# 打印训练集中前10个数据
for i in range(10):
    image, label = train_dataset[i]
    print(f'图像 {i + 1} 的标签: {label}')
    print(f'图像 {i + 1} 的形状: {image.shape}')
    print(f'图像 {i + 1} 的数据类型: {image.dtype}')
    print('-' * 50)
    print(f'图像 {i + 1} 的像素值:')
    # 将图像转换为28x28的矩阵并打印
    print(image.reshape(28, 28))

