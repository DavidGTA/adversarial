import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Maxout(nn.Module):
    """
    Maxout层的PyTorch实现
    
    参数:
    - in_features: 输入特征数
    - out_features: 输出特征数
    - num_pieces: 每个maxout单元中的线性片段数
    - bias: 是否使用偏置项
    """
    def __init__(self, in_features, out_features, num_pieces=2, bias=True):
        super(Maxout, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces
        
        # 创建权重矩阵: 实际输出通道数需要乘以num_pieces
        self.weight = nn.Parameter(
            torch.Tensor(out_features * num_pieces, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(out_features * num_pieces)
            )
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化参数"""
        # 使用Kaiming初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        """
        前向传播
        input shape: (batch_size, in_features)
        output shape: (batch_size, out_features)
        """
        # 线性变换
        output = F.linear(input, self.weight, self.bias)
        
        # 重塑张量以便进行maxout操作
        batch_size = output.size(0)
        output = output.view(batch_size, self.out_features, self.num_pieces)
        
        # 在num_pieces维度上取最大值
        output, _ = output.max(dim=2)
        
        return output
        
class MaxoutConv2d(nn.Module):
    """
    Maxout卷积层的PyTorch实现
    
    参数:
    - in_channels: 输入通道数
    - out_channels: 输出通道数
    - kernel_size: 卷积核大小
    - num_pieces: 每个maxout单元中的线性片段数
    - stride: 卷积步长
    - padding: 填充大小
    - bias: 是否使用偏置项
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_pieces=2, 
                 stride=1, padding=0, bias=True):
        super(MaxoutConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_pieces = num_pieces
        
        # 创建卷积层: 实际输出通道数需要乘以num_pieces
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * num_pieces,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
    def forward(self, x):
        """
        前向传播
        x shape: (batch_size, in_channels, height, width)
        output shape: (batch_size, out_channels, new_height, new_width)
        """
        # 卷积操作
        batch_size = x.size(0)
        output = self.conv(x)
        
        # 重塑张量以便进行maxout操作
        output = output.view(
            batch_size, 
            self.out_channels, 
            self.num_pieces,
            output.size(2), 
            output.size(3)
        )
        
        # 在num_pieces维度上取最大值
        output, _ = output.max(dim=2)
        
        return output

# 使用示例
if __name__ == "__main__":
    # 测试全连接版本的Maxout
    batch_size = 32
    in_features = 100
    out_features = 50
    num_pieces = 3
    
    maxout = Maxout(in_features, out_features, num_pieces)
    x = torch.randn(batch_size, in_features)
    output = maxout(x)
    print(f"Maxout output shape: {output.shape}")  # 应该是 (32, 50)
    
    # 测试卷积版本的Maxout
    batch_size = 32
    in_channels = 3
    out_channels = 16
    
    maxout_conv = MaxoutConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        num_pieces=2,
        padding=1
    )
    
    x = torch.randn(batch_size, in_channels, 28, 28)
    output = maxout_conv(x)
    print(f"MaxoutConv2d output shape: {output.shape}")  # 应该是 (32, 16, 28, 28)