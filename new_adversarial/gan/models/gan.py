import torch
import torch.nn as nn

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def generator_step(self, batch_size, device):
        # 生成随机噪声
        noise = torch.randn(batch_size, self.generator.noise_dim).to(device)
        # noise.shape = torch.Size([256, 100]) [batch_size, noise_dim]
        # 生成假样本
        fake_samples = self.generator(noise)
        # fake_samples.shape = torch.Size([256, 784]) [batch_size, output_dim]
        # 判别器对假样本的判别结果
        fake_predictions = self.discriminator(fake_samples)
        # fake_predictions.shape = torch.Size([256, 1]) [batch_size, 1]
        # 计算生成器损失
        # g_loss = torch.mean(-torch.log(fake_predictions + 1e-8))
        g_loss = -torch.mean(torch.log(1 - fake_predictions + 1e-8))
        return g_loss, fake_samples
        
    def discriminator_step(self, real_samples, fake_samples):
        # 判别器对真实样本的判别结果
        real_predictions = self.discriminator(real_samples)
        # real_predictions.shape = torch.Size([256, 1]) [batch_size, 1]
        # 判别器对假样本的判别结果 
        fake_predictions = self.discriminator(fake_samples)
        # fake_predictions.shape = torch.Size([256, 1]) [batch_size, 1]
        # 计算判别器损失
        # d_loss = torch.mean(-torch.log(real_predictions + 1e-8) - torch.log(1 - fake_predictions + 1e-8))
        real_loss = -torch.mean(torch.log(real_predictions + 1e-8))
        fake_loss = -torch.mean(torch.log(1 - fake_predictions + 1e-8))
        d_loss = real_loss + fake_loss
        return d_loss