import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 定义绘制函数
def plot_losses(losses, title, filename):
    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.show()

# 绘制任务 1 和任务 2 的损失图
train_losses_task1 = torch.load('train_losses_TASK1.pth')
train_losses_task2 = torch.load('train_losses_TASK2.pth')

plot_losses(train_losses_task1, '任务 1 训练损失', 'training_loss_task1.png')
plot_losses(train_losses_task2, '任务 2 训练损失', 'training_loss_task2.png')

# 绘制任务 3 的损失图
dropout_probs = [0.01, 0.05, 0.1]
for prob in dropout_probs:
    losses_filename = f'train_losses_dropout_Task3_{int(prob * 100)}.pth'
    train_losses_task3 = torch.load(losses_filename)
    plot_losses(train_losses_task3, f'任务 3（Dropout {int(prob * 100)}%）训练损失', f'training_loss_task3_{int(prob * 100)}.png')
