import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowBackbone(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, hidden_dim, kernel_size=3, stride=1, padding=1)
        
        self.num_channels = hidden_dim

        # 新增：1x1卷积，把(2, H, W) meshgrid 编码投影到 hidden_dim
        self.pos_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        device = x.device

        feat = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(feat))
        feat = F.relu(self.conv3(feat))
        feat = self.conv4(feat)  # [B, hidden_dim, H', W']

        # 生成 meshgrid positional encoding
        _, _, H_feat, W_feat = feat.shape
        yy, xx = torch.meshgrid(torch.arange(H_feat, device=device), torch.arange(W_feat, device=device), indexing='ij')
        yy = yy.float() / H_feat  # normalize 0~1
        xx = xx.float() / W_feat
        pos = torch.stack([yy, xx], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H', W']
        pos_emb = self.pos_proj(pos)  # [B, hidden_dim, H', W']

        return [feat], [pos_emb]
