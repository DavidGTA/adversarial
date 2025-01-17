import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

class GANTrainer:
    def __init__(self, gan, config):
        self.gan = gan
        self.config = config
        
        self.g_optimizer = optim.Adam(
            gan.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, 0.999)
        )
        
        self.d_optimizer = optim.Adam(
            gan.discriminator.parameters(), 
            lr=config.learning_rate,
            betas=(config.beta1, 0.999)
        )
        
    def train(self, train_dataset, num_epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gan.to(device)
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        for epoch in range(num_epochs):
            loop = tqdm(dataloader)
            for real_samples in loop:
                real_samples = real_samples.to(device)
                batch_size = real_samples.size(0)
                
                # 训练判别器
                self.d_optimizer.zero_grad()
                g_loss, fake_samples = self.gan.generator_step(batch_size, device)
                d_loss = self.gan.discriminator_step(real_samples, fake_samples)
                d_loss.backward()
                self.d_optimizer.step()
                
                # 训练生成器
                self.g_optimizer.zero_grad() 
                g_loss.backward()
                self.g_optimizer.step()
                
                # 更新进度条
                loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())