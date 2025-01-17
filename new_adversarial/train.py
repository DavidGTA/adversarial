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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
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