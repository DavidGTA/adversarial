import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from maxout_new import Maxout, MaxoutConv2d

# 定义生成器和判别器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 1200)  # 第一层，输入噪声，输出1200维
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1200, 1200)  # 第二层，输出1200维
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1200, 784)   # 输出层，生成784维图像（28x28）
        self.sigmoid = nn.Sigmoid()       # Sigmoid激活函数输出图像（[0, 1]范围）

    def forward(self, z):
        x = self.fc1(z)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)  # 返回生成的图像

# 在模型中使用Maxout层
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 第一层 Maxout
        self.h0 = Maxout(784, 240, num_pieces=5)
        # 第二层 Maxout
        self.h1 = Maxout(240, 240, num_pieces=5)
        # 输出层 Sigmoid
        self.y = nn.Linear(240, 1)
        # self.maxout = Maxout(784, 50, num_pieces=3)
        # self.maxout_conv = MaxoutConv2d(3, 16, kernel_size=3, num_pieces=2)
        
    def forward(self, x):
        # 用于卷积数据
        # x = self.maxout_conv(x)
        # 展平后用于全连接层
        x = x.view(x.size(0), -1)
        # 第一层 Maxout
        x = self.h0(x)
        # 第二层 Maxout
        x = self.h1(x)
        # 输出层 Sigmoid
        x = torch.sigmoid(self.y(x))
        return x

# # 自定义Maxout层
# class Maxout(nn.Module):
#     def __init__(self, num_units, num_pieces):
#         super(Maxout, self).__init__()
#         self.num_units = num_units
#         self.num_pieces = num_pieces
#         # 将输入的784维数据拆分成 num_units * num_pieces
#         # 这里使用的Linear层将 num_units * num_pieces 映射到 num_units
#         self.linear = nn.Linear(num_units * num_pieces, num_units)

#     def forward(self, x):
#         # x的形状是 (batch_size, input_size) 输入784
#         # 我们将输入拆分成 num_units 个单位，每个单位包含 num_pieces 片段
#         batch_size = x.size(0)
#         # 对输入进行 reshape 和分块
#         pieces = x.view(batch_size, self.num_units, self.num_pieces, -1)  # 重新塑形为 (batch_size, num_units, num_pieces, H)
#         # 对每个块进行求最大值，得到 (batch_size, num_units, H)
#         max_out = pieces.max(dim=2)[0]  # 在每个片段中取最大值
#         # 展平并通过线性层
#         return self.linear(max_out.view(batch_size, -1))  # 输出形状 (batch_size, num_units)

    
# class Discriminator(nn.Module):
#     def __init__(self, nvis=784, num_units=240, num_pieces=5):
#         super(Discriminator, self).__init__()
        
#         # 第一层 Maxout
#         self.h0 = Maxout(num_units, num_pieces)
#         # 第二层 Maxout
#         self.h1 = Maxout(num_units, num_pieces)
#         # 输出层 Sigmoid
#         self.y = nn.Linear(num_units, 1)

#     def forward(self, x):
#         # 第一层 Maxout
#         x = self.h0(x)
#         # 第二层 Maxout
#         x = self.h1(x)
#         # 输出层 Sigmoid
#         x = torch.sigmoid(self.y(x))
#         return x
    
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(784, 240)  # 第一层，输入784维图像，输出240维
#         self.maxout1 = nn.Linear(240, 240)  # Maxout层
#         self.fc2 = nn.Linear(240, 240)  # 第二层，输出240维
#         self.maxout2 = nn.Linear(240, 240)  # Maxout层
#         self.fc3 = nn.Linear(240, 1)    # 输出层，二分类（真假判别）
#         self.sigmoid = nn.Sigmoid()     # Sigmoid激活，输出概率值

#     def forward(self, x):
#         x = self.fc1(x)  # 第一层
#         x = torch.max(self.maxout1(x), dim=1)[0]  # Maxout操作
#         x = self.fc2(x)  # 第二层
#         x = torch.max(self.maxout2(x), dim=1)[0]  # Maxout操作
#         x = self.fc3(x)  # 输出层
#         return self.sigmoid(x)  # 返回真假概率

# 生成器和判别器的优化器
lr = 0.01  # 学习率
momentum_init = 0.5  # 初始化动量
final_momentum = 0.7  # 最终动量
batch_size = 100  # 批量大小
epochs = 100  # 训练轮数

# 定义生成器和判别器的优化器
generator = Generator()
discriminator = Discriminator()

# 使用SGD优化器，并设置学习率和动量
optimizer_g = optim.SGD(generator.parameters(), lr=lr, momentum=momentum_init)
optimizer_d = optim.SGD(discriminator.parameters(), lr=lr, momentum=momentum_init)

# 学习率衰减
scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=1.000004)
scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=1.000004)

# 初始化动量调整的回调
def adjust_momentum(epoch):
    if epoch >= 1 and epoch <= 250:
        momentum = momentum_init + (final_momentum - momentum_init) * (epoch / 250)
        for param_group in optimizer_g.param_groups:
            param_group['momentum'] = momentum
        for param_group in optimizer_d.param_groups:
            param_group['momentum'] = momentum

# 示例的随机数据生成（假设我们有一个MNIST数据集）
# 这里只是一个简单示例，实际中你应该用实际数据集进行训练。
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_data = datasets.MNIST(root='./data', train=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 创建列表来存储损失值
g_losses = []
d_losses = []

import torch.nn.functional as F

# 训练过程
for epoch in range(epochs):
    epoch_g_losses = []
    epoch_d_losses = []
    
    for i, (real_images, _) in enumerate(train_loader):
        current_batch_size = real_images.size(0)
        # 创建噪声输入
        z = torch.randn(batch_size, 100)
        
        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)

        # 训练判别器
        # 判别器训练
        optimizer_d.zero_grad()
        real_images = real_images.view(batch_size, 784)
        # 真实图像的损失
        output_real = discriminator(real_images)
        loss_real = F.binary_cross_entropy(output_real, torch.ones_like(output_real))
        # 生成假图像并计算损失
        fake_images = generator(z)
        output_fake = discriminator(fake_images.detach())
        loss_fake = F.binary_cross_entropy(output_fake, torch.zeros_like(output_fake))
        # 判别器总损失并反向传播
        d_loss = loss_real + loss_fake
        d_loss.backward()
        epoch_d_losses.append(d_loss.item())
        optimizer_d.step()

        # 生成器训练
        optimizer_g.zero_grad()

        # 生成器的损失
        output_fake = discriminator(fake_images)
        g_loss = F.binary_cross_entropy(output_fake, torch.ones_like(output_fake))
        g_loss.backward()
        
        epoch_g_losses.append(g_loss.item())
        optimizer_g.step()

        # 更新学习率
        scheduler_g.step()
        scheduler_d.step()

        # 调整动量
        adjust_momentum(epoch)

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'D Loss: {d_loss:.4f}, G Loss: {g_loss.item():.4f}')
    
    # 记录每个epoch的平均损失
    g_losses.append(np.mean(epoch_g_losses))
    d_losses.append(np.mean(epoch_d_losses))
    
    # 每25个epoch保存生成的图像
    if (epoch + 1) % 25 == 0:
        # 保存模型参数
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'g_loss': g_losses[-1],
            'd_loss': d_losses[-1],
        }, f'checkpoint_epoch_{epoch+1}.pt')

        generator.eval()
        with torch.no_grad():
            # 生成9张图像
            sample_z = torch.randn(9, 100)
            sample_images = generator(sample_z)
            
            # 创建3x3的图像网格
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            for ax, img in zip(axes.flatten(), sample_images):
                # 将784维向量重塑为28x28的图像
                img_reshaped = img.view(28, 28).cpu().numpy()
                # sigmoid输出范围是[0,1]，直接显示即可，无需额外处理
                ax.imshow(img_reshaped, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
            
            plt.savefig(f'generated_images_epoch_{epoch+1}.png')
            plt.close()
        generator.train()

# 训练结束后绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.savefig('training_losses.png')
plt.close()

# 加载模型的函数（如果需要的话）
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    epoch = checkpoint['epoch']
    g_loss = checkpoint['g_loss']
    d_loss = checkpoint['d_loss']
    return epoch, g_loss, d_loss


# # 展示3*3的9张随机真实图像
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(train_data[i][0].squeeze(), cmap='gray')
#     plt.axis('off')
# plt.savefig('real_images.png')
# plt.close()
