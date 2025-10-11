import torch
import torch.nn as nn
from typing import Optional
from logger import logger


def apply_adaptive_pooling_to_linear(
        layer: nn.Linear,
        activations: torch.Tensor,
        target_ratio: float = 0.5,
        method: str = "adaptive_avg",
        device: str = "cuda",
        layer_name: Optional[str] = None,
        layer_idx: Optional[int] = None,
        total_layers: Optional[int] = None,
        **kwargs
) -> bool:
    """
    使用自适应池化压缩线性层
    """
    try:
        # 拷贝权重并放到指定设备
        W = layer.weight.data.clone().to(device).to(torch.float32)
        original_in_features = layer.in_features
        original_out_features = layer.out_features

        # 计算目标压缩维度
        target_in_features = int(original_in_features * target_ratio)
        target_out_features = int(original_out_features * target_ratio)

        logger.info(f"Adaptive Pooling [{layer_name}]: {original_in_features}->{target_in_features}, "
                    f"{original_out_features}->{target_out_features}")

        # 选择池化方式
        if method == "adaptive_avg":
            pool_in = nn.AdaptiveAvgPool1d(target_in_features)
            pool_out = nn.AdaptiveAvgPool1d(target_out_features)
        elif method == "adaptive_max":
            pool_in = nn.AdaptiveMaxPool1d(target_in_features)
            pool_out = nn.AdaptiveMaxPool1d(target_out_features)
        else:
            raise ValueError(f"Unsupported pooling method: {method}")

        with torch.no_grad():
            # ========= 1️⃣ 沿输入维度池化（in_features） =========
            # W: [out_features, in_features]
            W_pooled_in = pool_in(W.view(original_out_features, 1, original_in_features))
            W_pooled_in = W_pooled_in.view(original_out_features, target_in_features)

            # ========= 2️⃣ 沿输出维度池化（out_features） =========
            # W.T: [in_features, out_features]
            W_pooled_out = pool_out(W.T.view(original_in_features, 1, original_out_features))
            W_pooled_out = W_pooled_out.view(original_in_features, target_out_features).T

            # ========= 3️⃣ 更新权重 =========
            # 同时压缩输入和输出维度
            W_final = W_pooled_out[:, :target_in_features]
            layer.weight.data = W_final

            # ========= 4️⃣ 若存在偏置则同步压缩 =========
            if layer.bias is not None:
                bias_pooled = pool_out(layer.bias.view(1, 1, original_out_features))
                bias_pooled = bias_pooled.view(target_out_features)
                layer.bias.data = bias_pooled

        logger.info(f"视觉层[{layer_name}]压缩成功: {original_in_features}->{target_in_features}")
        return True

    except Exception as e:
        logger.error(f"视觉层[{layer_name}]压缩失败: {str(e)}")
        return False
