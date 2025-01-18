import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 定义生成器和判别器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, 784),  # 输出一个28x28的图像
            nn.Tanh()  # 生成的图像需要在-1到1之间
        )
    
    def forward(self, z):
        return self.fc(z).view(-1, 1, 28, 28)  # 输出一个28x28的图像

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 240),
            nn.LeakyReLU(0.2),
            nn.Linear(240, 240),
            nn.LeakyReLU(0.2),
            nn.Linear(240, 1),
            nn.Sigmoid()  # 判别器输出为概率值
        )
    
    def forward(self, x):
        return self.fc(x.view(-1, 784))  # 输入是28x28的图像，flatten成784维

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

# 定义训练参数
z_dim = 100
lr = 0.001
batch_size = 256
epochs = 100

# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()

# 使用Adam优化器
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 交叉熵损失
criterion = nn.BCELoss()

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

# 加载检查点
checkpoint_path = r"D:\PythonProject\adversarial\result\train_3_result\checkpoint_epoch_50.pt"
checkpoint = torch.load(checkpoint_path)

# 加载模型参数
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

# 如果需要继续训练，还可以加载优化器状态
optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

# 获取其他保存的信息
epoch = checkpoint['epoch']
g_loss = checkpoint['g_loss']
d_loss = checkpoint['d_loss']

print(f"加载了第 {epoch} 轮的模型")
print(f"生成器损失: {g_loss:.4f}")
print(f"判别器损失: {d_loss:.4f}")

# 训练过程
for epoch in range(50, epochs):
    epoch_g_losses = []
    epoch_d_losses = []
    
    for i, (real_images, _) in enumerate(train_loader):
        current_batch_size = real_images.size(0)
        z = torch.randn(current_batch_size, z_dim)
        
        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)

        # 训练判别器
        optimizer_d.zero_grad()
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        
        d_loss = d_loss_real.item() + d_loss_fake.item()
        epoch_d_losses.append(d_loss)
        optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        
        epoch_g_losses.append(g_loss.item())
        optimizer_g.step()

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
        }, f'result/train_3_result/checkpoint_epoch_{epoch+1}.pt')

        generator.eval()
        with torch.no_grad():
            # 生成9张图像
            sample_z = torch.randn(9, z_dim)
            sample_images = generator(sample_z)
            
            # 创建3x3的图像网格
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            for ax, img in zip(axes.flatten(), sample_images):
                ax.imshow(img.squeeze().numpy(), cmap='gray')
                ax.axis('off')
            
            plt.savefig(f'result/train_3_result/generated_images_epoch_{epoch+1}.png')
            plt.close()
        generator.train()

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Losses Epoch {epoch+1}')
        plt.legend()
        plt.savefig(f'result/train_3_result/training_losses_epoch_{epoch+1}.png')
        plt.close()


# # 将模型设置为评估模式（如果只是生成图像）
# generator.eval()
# discriminator.eval()
# with torch.no_grad():
#     # 生成9张图像
#     sample_z = torch.randn(25, z_dim)
#     sample_images = generator(sample_z)

#     # 创建3x3的图像网格
#     fig, axes = plt.subplots(5, 5, figsize=(25, 25))
#     for ax, img in zip(axes.flatten(), sample_images):
#         ax.imshow(img.squeeze().numpy(), cmap='gray')
#         ax.axis('off')
#     save_path = r"D:\PythonProject\adversarial\result\train_3_result\generated_images(3).png"
#     plt.savefig(save_path)
#     plt.close()

# # 展示3*3的9张随机真实图像
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(train_data[i][0].squeeze(), cmap='gray')
#     plt.axis('off')
# plt.savefig('real_images.png')
# plt.close()
