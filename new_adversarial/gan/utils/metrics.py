import torch
import numpy as np
from scipy.stats import norm
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader

def compute_inception_score(generator, num_samples=50000, batch_size=128, splits=10):
    """计算Inception Score
    
    Args:
        generator: 生成器模型
        num_samples: 用于评估的样本数量
        batch_size: 批次大小
        splits: 计算平均分数时的分割数
    
    Returns:
        mean: IS的平均值
        std: IS的标准差
    """
    # 这里需要预训练的Inception模型
    raise NotImplementedError("需要添加Inception模型支持")

def estimate_parzen_window_log_likelihood(
    generator,
    test_data,
    noise_dim,
    sigma,
    num_samples=10000,
    batch_size=100,
    device='cuda'
):
    """使用Parzen窗估计对数似然
    
    Args:
        generator: 生成器模型
        test_data: 测试数据集
        noise_dim: 噪声维度
        sigma: 窗口宽度
        num_samples: 生成样本数量
        batch_size: 批次大小
        device: 使用的设备
        
    Returns:
        log_likelihood: 估计的对数似然
    """
    # 生成样本
    generator.eval()
    samples = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            noise = torch.randn(min(batch_size, num_samples - i), noise_dim).to(device)
            sample = generator(noise)
            # 将生成的样本展平
            sample = sample.view(sample.size(0), -1)
            samples.append(sample.cpu().numpy())
    samples = np.concatenate(samples, axis=0)
    
    # 计算Parzen窗估计
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    log_likelihoods = []
    
    for x_batch, _ in test_loader:
        # 将测试数据展平
        x_batch = x_batch.view(x_batch.size(0), -1).numpy()
        batch_ll = []
        
        # 对每个生成的样本计算核
        for sample in samples:
            diff = x_batch - sample
            diff_norm = np.sum(diff ** 2, axis=1)
            kernel_val = norm.logpdf(diff_norm, scale=sigma)
            batch_ll.append(kernel_val)
            
        batch_ll = np.array(batch_ll)
        # 计算log mean exp
        max_ll = batch_ll.max(axis=0)
        batch_ll = max_ll + np.log(np.mean(np.exp(batch_ll - max_ll), axis=0))
        log_likelihoods.extend(batch_ll)
    
    return np.mean(log_likelihoods)

def fid_score(real_features, fake_features):
    """计算FID分数
    
    Args:
        real_features: 真实图像的特征
        fake_features: 生成图像的特征
        
    Returns:
        fid: Fréchet Inception Distance
    """
    # 计算均值
    mu1, mu2 = np.mean(real_features, axis=0), np.mean(fake_features, axis=0)
    
    # 计算协方差
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)
    
    # 计算均值差的平方
    diff = mu1 - mu2
    
    # 计算协方差平方根
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # 确保复数部分很小
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # 计算FID
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    
    return fid