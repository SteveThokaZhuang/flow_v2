import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowBackbone(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=256, debug=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.pos_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)  # project (2, H, W) → (hidden_dim, H, W)

        self.num_channels = hidden_dim
        self.debug = debug  # enable debug print

    def forward(self, x):
        B, _, H, W = x.shape
        device = x.device

        feat = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(feat))
        feat = F.relu(self.conv3(feat))
        feat = self.conv4(feat)  # [B, hidden_dim, H', W']

        _, _, H_feat, W_feat = feat.shape

        # meshgrid positional encoding, normalized to [-1, 1]
        yy, xx = torch.meshgrid(
            torch.arange(H_feat, device=device),
            torch.arange(W_feat, device=device),
            indexing='ij'
        )
        yy = (yy.float() / (H_feat - 1)) * 2 - 1  # [-1,1]
        xx = (xx.float() / (W_feat - 1)) * 2 - 1  # [-1,1]

        pos = torch.stack([yy, xx], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H', W']
        pos_emb = self.pos_proj(pos)  # [B, hidden_dim, H', W']

        if self.debug:
            print(f"[FlowBackbone] feat shape: {feat.shape}")
            print(f"[FlowBackbone] pos_emb min={pos_emb.min().item()}, max={pos_emb.max().item()}, mean={pos_emb.mean().item()}, std={pos_emb.std().item()}")

        return [feat], [pos_emb]
# 🚀 改动说明
# ✅ 坐标归一化修正到 [-1,1]，方便 Transformer
# ✅ self.debug 控制打印，不改主代码
# ✅ pos_proj 卷积明确作用：
# 2D mesh → hidden_dim feature map

