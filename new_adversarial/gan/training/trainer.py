import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
from ..utils.visualization import save_sample_images, plot_training_curves
from ..utils.metrics import estimate_parzen_window_log_likelihood

class GANTrainer:
    def __init__(self, gan, config):
        self.gan = gan
        self.config = config
        
        # 优化器配置
        self.g_optimizer = optim.Adam(
            gan.generator.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], 0.999)
        )
        
        self.d_optimizer = optim.Adam(
            gan.discriminator.parameters(), 
            lr=config['learning_rate'],
            betas=(config['beta1'], 0.999)
        )
        
        # 初始化记录器
        self.g_losses = []
        self.d_losses = []
        
        # 创建保存目录
        os.makedirs('samples', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.gan.generator.state_dict(),
            'discriminator_state_dict': self.gan.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
        }
        torch.save(checkpoint, f'checkpoints/gan_epoch_{epoch}.pt')
        
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path)
        self.gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        return checkpoint['epoch']
        
    def train(self, train_dataset, num_epochs, validate_every=5):
        """训练GAN模型
        
        Args:
            train_dataset: 训练数据集
            num_epochs: 训练轮数
            validate_every: 每多少轮进行一次验证和采样
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gan.to(device)
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        total_steps = 0
        for epoch in range(num_epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            
            loop = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch in loop:
                real_samples, _ = batch  # 解包数据和标签
                real_samples = real_samples.to(device)
                batch_size = real_samples.size(0)
                
                # 训练判别器 k 次 (论文中 k=1)
                for _ in range(self.config.get('k_steps', 1)):
                    self.d_optimizer.zero_grad()
                    _, fake_samples = self.gan.generator_step(batch_size, device)
                    d_loss = self.gan.discriminator_step(real_samples, fake_samples)
                    d_loss.backward()
                    self.d_optimizer.step()
                
                # 训练生成器一次
                self.g_optimizer.zero_grad()
                g_loss, _ = self.gan.generator_step(batch_size, device)
                g_loss.backward()
                self.g_optimizer.step()
                
                # 记录损失
                g_loss_val = g_loss.item()
                d_loss_val = d_loss.item()
                epoch_g_losses.append(g_loss_val)
                epoch_d_losses.append(d_loss_val)
                
                # 更新进度条
                loop.set_postfix({
                    'd_loss': f'{d_loss_val:.4f}',
                    'g_loss': f'{g_loss_val:.4f}'
                })
                
                total_steps += 1
            
            # 记录每个epoch的平均损失
            self.g_losses.append(sum(epoch_g_losses) / len(epoch_g_losses))
            self.d_losses.append(sum(epoch_d_losses) / len(epoch_d_losses))
            
            # 定期验证和保存
            if (epoch + 1) % validate_every == 0:
                # 生成和保存样本图像
                save_sample_images(
                    self.gan.generator,
                    epoch + 1,
                    self.gan.generator.noise_dim,
                    device=device
                )
                
                # 绘制损失曲线
                plot_training_curves(
                    self.g_losses,
                    self.d_losses,
                    save_path=f'samples/losses_epoch_{epoch+1}.png'
                )
                
                # 保存检查点
                self.save_checkpoint(epoch + 1)
                
                # 计算Parzen窗估计（如果配置中启用）
                if self.config.get('compute_parzen', False):
                    ll = estimate_parzen_window_log_likelihood(
                        self.gan.generator,
                        train_dataset,
                        self.gan.generator.noise_dim,
                        sigma=self.config.get('parzen_sigma', 0.1),
                        device=device
                    )
                    print(f'Epoch {epoch+1} Parzen window log-likelihood: {ll:.4f}')
        
        # 训练结束，保存最终模型和绘制最终损失曲线
        self.save_checkpoint(num_epochs)
        plot_training_curves(
            self.g_losses,
            self.d_losses,
            save_path='samples/final_losses.png'
        )