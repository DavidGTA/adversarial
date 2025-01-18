import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def save_sample_images(generator, epoch, noise_dim, num_samples=64, device='cuda'):
    """生成和保存样本图像
    
    Args:
        generator: 生成器模型
        epoch: 当前训练轮数
        noise_dim: 噪声维度
        num_samples: 生成样本数量
        device: 使用的设备(CPU/GPU)
    """
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim).to(device)
        fake_images = generator(noise).cpu()
        # 将值范围从[-1,1]转换到[0,1]
        fake_images = (fake_images + 1) / 2
        
        # 创建图像网格
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Images - Epoch {epoch}")
        plt.imshow(np.transpose(
            vutils.make_grid(fake_images, padding=2, normalize=True),
            (1, 2, 0)
        ))
        plt.savefig(f'samples/epoch_{epoch}.png')
        plt.close()

def plot_training_curves(g_losses, d_losses, save_path='training_curves.png'):
    """绘制训练过程中的损失曲线
    
    Args:
        g_losses: 生成器损失历史
        d_losses: 判别器损失历史
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses,label="Generator")
    plt.plot(d_losses,label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()