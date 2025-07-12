import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowBackbone(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=256):
        super().__init__()
        # 假设输入是光流图像 (B, 2, H, W)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, hidden_dim, kernel_size=3, stride=1, padding=1)
        
        # 位置编码相关（如果需要）
        self.num_channels = hidden_dim  # 用于后续投影层的输入通道数
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return [x], [None]  # 返回特征列表和位置编码列表（兼容原始接口）