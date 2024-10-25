import torch
import matplotlib.pyplot as plt

# 加载训练损失
train_losses = torch.load('train_losses.pth')

# 绘制训练损失
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('training_loss_plot.png')
plt.show()
