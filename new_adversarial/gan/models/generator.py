import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dims, output_dim):
        super().__init__()
        self.noise_dim = noise_dim
        
        layers = []
        prev_dim = noise_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.model(z)