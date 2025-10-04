# model/prune.py (新增部分)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicViTCompressor(nn.Module):
    """
    动态视觉 token 压缩器，参考 Dynamic-VLM (2412.09530v1)
    支持三种模式：
    - 'pooling': AdaptiveAvgPool2d
    - 'token_pruning': 基于 MLP + Gumbel-Softmax 的 top-K 选择
    - 'token_merging': ToMe (可选，较复杂，此处省略)
    """
    def __init__(self, mode='pooling', target_tokens=256):
        super().__init__()
        self.mode = mode
        self.target_tokens = target_tokens  # 目标 token 数，如 256 = 16x16

        if mode == 'token_pruning':
            # 用于生成 token 重要性分数
            self.score_net = nn.Sequential(
                nn.Linear(1024, 256),  # 假设 ViT 输出 dim=1024
                nn.ReLU(),
                nn.Linear(256, 1)
            )

    def forward(self, vit_features):
        """
        vit_features: [B, N, C]  (N = 576 for ViT-L/14 @ 336px)
        返回: [B, M, C], M <= N
        """
        B, N, C = vit_features.shape

        if self.mode == 'pooling':
            # 将 [B, N, C] -> [B, H, W, C] -> pool -> [B, h, w, C] -> [B, M, C]
            h = w = int(N ** 0.5)
            x = vit_features.view(B, h, w, C).permute(0, 3, 1, 2)  # [B, C, h, w]
            target_h = target_w = int(self.target_tokens ** 0.5)
            x_pooled = F.adaptive_avg_pool2d(x, (target_h, target_w))  # [B, C, th, tw]
            return x_pooled.permute(0, 2, 3, 1).reshape(B, -1, C)

        elif self.mode == 'token_pruning':
            # 计算每个 token 的重要性
            scores = self.score_net(vit_features)  # [B, N, 1]
            scores = scores.squeeze(-1)  # [B, N]

            # 选择 top-K
            k = min(self.target_tokens, N)
            _, topk_indices = torch.topk(scores, k, dim=1)  # [B, k]

            # gather
            batch_indices = torch.arange(B, device=vit_features.device).unsqueeze(1)
            compressed = vit_features[batch_indices, topk_indices]  # [B, k, C]
            return compressed

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")