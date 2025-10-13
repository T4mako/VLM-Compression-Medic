import torch
import torch.nn as nn
import time
from typing import Optional
from logger import logger
from config import Config


def apply_adaptive_pooling_to_linear(
        layer: nn.Linear,
        activations: torch.Tensor,
        target_ratio: float = 0.8,
        method: str = "adaptive_avg",
        device: str = "cuda",
        layer_name: Optional[str] = None,
        layer_idx: Optional[int] = None,
        total_layers: Optional[int] = None,
        **kwargs
) -> bool:
    """
    使用自适应池化压缩线性层（视觉分支专用，带详细日志）
    """
    start_time = time.time()
    try:
        logger.info(f"[视觉层][{layer_name}]开始处理({layer_idx}/{total_layers})")

        # ---- Step 1. 复制权重 ----
        W = layer.weight.data.clone().to(device).to(torch.float32)
        original_in_features = layer.in_features
        original_out_features = layer.out_features
        norm_before = torch.norm(W).item()

        logger.info(f"[{layer_name}] 原始形状: {tuple(W.shape)}, 范数: {norm_before:.4e}")

        # ---- Step 2. 计算目标维度 ----
        target_in_features = int(original_in_features * target_ratio)
        target_out_features = int(original_out_features * target_ratio)

        logger.info(f"[{layer_name}] 压缩比例: {target_ratio:.3f}, "
                    f"in_features: {original_in_features}->{target_in_features}, "
                    f"out_features: {original_out_features}->{target_out_features}")

        # ---- Step 3. 构建池化器 ----
        if method == "adaptive_avg":
            pool_in = nn.AdaptiveAvgPool1d(target_in_features)
            pool_out = nn.AdaptiveAvgPool1d(target_out_features)
        elif method == "adaptive_max":
            pool_in = nn.AdaptiveMaxPool1d(target_in_features)
            pool_out = nn.AdaptiveMaxPool1d(target_out_features)
        else:
            raise ValueError(f"Unsupported pooling method: {method}")

        with torch.no_grad():
            # ---- Step 4. 沿输入维度池化 ----
            logger.info(f"[{layer_name}] 沿输入维度池化中...")
            W_pooled_in = pool_in(W.view(original_out_features, 1, original_in_features))
            logger.info(f"[{layer_name}] 输入池化后形状: {tuple(W_pooled_in.shape)}")
            W_pooled_in = W_pooled_in.view(original_out_features, target_in_features)

            # ---- Step 5. 沿输出维度池化 ----
            logger.info(f"[{layer_name}] 沿输出维度池化中...")
            W_pooled_out = pool_out(W.T.view(original_in_features, 1, original_out_features))
            logger.info(f"[{layer_name}] 输出池化后形状: {tuple(W_pooled_out.shape)}")
            W_pooled_out = W_pooled_out.view(original_in_features, target_out_features).T

            # ---- Step 6. 更新权重 ----
            W_final = W_pooled_out[:, :target_in_features]
            layer.weight.data = W_final

            norm_after = torch.norm(W_final).item()
            logger.info(f"[{layer_name}] 压缩后形状: {tuple(W_final.shape)}, "
                        f"范数: {norm_after:.4e}, 范数比率: {norm_after / norm_before:.3f}")

            # ---- Step 7. 处理偏置 ----
            if layer.bias is not None:
                logger.info(f"[{layer_name}] 同步压缩偏置向量...")
                bias_pooled = pool_out(layer.bias.view(1, 1, original_out_features))
                layer.bias.data = bias_pooled.view(target_out_features)
                logger.info(f"[{layer_name}] 偏置压缩完成: {original_out_features}->{target_out_features}")

        elapsed = time.time() - start_time
        logger.info(f"[视觉层][{layer_name}]压缩成功 ({layer_idx}/{total_layers}) "
                    f"| 耗时: {elapsed:.2f}s | 范数比率: {norm_after / norm_before:.3f}")
        return True

    except Exception as e:
        logger.error(f"[视觉层][{layer_name}]压缩失败: {str(e)}")
        return False
