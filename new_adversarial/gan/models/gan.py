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
        # 生成假样本
        fake_samples = self.generator(noise)
        # 判别器对假样本的判别结果
        fake_predictions = self.discriminator(fake_samples)
        # 计算生成器损失
        g_loss = torch.mean(-torch.log(fake_predictions + 1e-8))
        return g_loss, fake_samples
        
    def discriminator_step(self, real_samples, fake_samples):
        # 判别器对真实样本的判别结果
        real_predictions = self.discriminator(real_samples)
        # 判别器对假样本的判别结果 
        fake_predictions = self.discriminator(fake_samples.detach())
        
        # 计算判别器损失
        d_loss = torch.mean(-torch.log(real_predictions + 1e-8) - torch.log(1 - fake_predictions + 1e-8))
        return d_loss