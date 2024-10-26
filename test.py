import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train_task1 import CNN  # 修改为对应的模型类
from train_task2 import CNN_Modified  # 修改为对应的模型类
from train_task3 import CNN_WithDropout  # 导入带 Dropout 的模型

# 加载数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 测试任务
def test_model(model_class, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class().to(device)
    if device == 'cuda':
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# 测试各个任务的模型
accuracy_task1 = test_model(CNN, 'mnist_cnn_TASK1.pth')
accuracy_task2 = test_model(CNN_Modified, 'mnist_cnn_TASK2.pth')

# 测试带有不同 Dropout 概率的模型
dropout_probs = [0.01, 0.05, 0.1]
accuracies_task3 = {}

for prob in dropout_probs:
    model_path = f'mnist_cnn_dropout_Task3_{int(prob * 100)}.pth'
    accuracy = test_model(lambda: CNN_WithDropout(prob), model_path)
    accuracies_task3[prob] = accuracy

print(f'任务1模型准确率: {accuracy_task1:.2f}%')
print(f'任务2模型准确率: {accuracy_task2:.2f}%')
for prob, accuracy in accuracies_task3.items():
    print(f'任务3（Dropout {int(prob * 100)}%）模型准确率: {accuracy:.2f}%')
