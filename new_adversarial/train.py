import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.models.gan import GAN
from gan.training.trainer import GANTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'tfd'], required=True)
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 准备数据集
    if args.dataset == 'mnist':
        '''
        用于图像数据预处理的转换操作：
        这个转换pipeline包含两个主要步骤：

        1. `transforms.ToTensor()`
        - 将PIL图像或numpy数组转换为PyTorch张量
        - 把像素值从0-255范围转换到0-1范围
        - 对于灰度图像（如MNIST），形状会变为[1, H, W]
        
        2. `transforms.Normalize((0.5,), (0.5,))`
        - 对数据进行标准化，使用公式：`(x - mean) / std`
        - 这里mean=0.5，std=0.5
        - 将数据范围从[0,1]转换为[-1,1]
        - 括号中有一个逗号是因为MNIST是单通道（灰度）图像

        这种预处理对于训练GAN非常重要，因为：
        - 标准化的数据能让模型更容易学习
        - [-1,1]的范围通常更适合GAN的训练
        - 保持数据分布的一致性有助于提高模型性能
        '''
        transform = transforms.Compose([
            transforms.ToTensor(),         # 将图像转换为PyTorch张量
            transforms.Normalize((0.5,), (0.5,))  # 对数据进行标准化
        ])
        dataset = datasets.MNIST('./data', train=True, transform=transform)
    
    # 创建模型
    generator = Generator(
        config['model']['noise_dim'],
        config['model']['generator']['hidden_dims'],
        config['model']['generator']['output_dim']
    )
    
    discriminator = Discriminator(
        config['model']['generator']['output_dim'],
        config['model']['discriminator']['hidden_dims']
    )
    
    gan = GAN(generator, discriminator)
    
    # 训练模型
    trainer = GANTrainer(gan, config['training'])
    trainer.train(dataset, config['training']['num_epochs'])

if __name__ == '__main__':
    main()