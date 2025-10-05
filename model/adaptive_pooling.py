import torch
import torch.nn as nn
from typing import Optional
from logger import logger

def apply_adaptive_pooling_to_linear(
        layer: nn.Linear,
        activations: torch.Tensor,
        target_ratio: float = 0.5,
        method: str = "adaptive_avg",
        device: str = "cuda"
) -> nn.Linear:
    """
    使用自适应池化压缩线性层
    """
    W = layer.weight.data.clone().to(device).to(torch.float32)
    original_in_features = layer.in_features
    original_out_features = layer.out_features

    # 计算目标维度
    target_in_features = int(original_in_features * target_ratio)
    target_out_features = int(original_out_features * target_ratio)

    logger.info(f"Adaptive Pooling: {original_in_features}->{target_in_features}, "
                f"{original_out_features}->{target_out_features}")

    if method == "adaptive_avg":
        # 自适应平均池化
        pool = nn.AdaptiveAvgPool1d(target_in_features)
    elif method == "adaptive_max":
        # 自适应最大池化
        pool = nn.AdaptiveMaxPool1d(target_in_features)
    else:
        raise ValueError(f"Unsupported pooling method: {method}")

    # 对权重应用池化
    with torch.no_grad():
        # 处理输入权重 (in_features 维度)
        W_pooled_input = pool(W.view(original_out_features, 1, original_in_features))
        W_pooled_input = W_pooled_input.view(original_out_features, target_in_features)

        # 处理输出权重 (out_features 维度) - 如果需要的话
        if target_ratio < 1.0:
            W_pooled_output = pool(W.T.view(original_in_features, 1, original_out_features))
            W_pooled_output = W_pooled_output.view(target_in_features, original_out_features).T
        else:
            W_pooled_output = W_pooled_input

    # 更新层权重
    layer.weight.data = W_pooled_output

    # 如果有偏置，也需要相应调整
    if layer.bias is not None:
        # 对偏置应用类似的池化
        bias_pooled = pool(layer.bias.view(1, 1, original_out_features))
        layer.bias.data = bias_pooled.view(target_out_features)

    return layer