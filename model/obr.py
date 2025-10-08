import torch
import torch.nn as nn
from typing import Optional
from logger import logger
from tqdm import tqdm


# 计算 Hessian 近似
def compute_hessian_approx(activations: torch.Tensor) -> torch.Tensor:
    """
    Compute Hessian approximation in the input-feature space.
    Correct formula: H ≈ 2 * X^T X (shape: [Cin, Cin]) when activations X has shape [N, Cin].
    """
    if activations is None:
        raise ValueError("activations is None in compute_hessian_approx")
    x = activations
    if x.dim() == 1:
        x = x.unsqueeze(0)

    # logger.debug(f"Computing Hessian: input shape {x.shape}")

    # 添加正则化防止矩阵奇异
    H = 2.0 * torch.matmul(x.t(), x)
    # 添加对角正则化
    H = H + 1e-6 * torch.eye(H.size(0), device=H.device, dtype=H.dtype)

    # logger.debug(f"Hessian computed: shape {H.shape}, condition number: {torch.linalg.cond(H):.2f}")
    return H


# OBR 补偿计算
def obr_compensation(
        weights: torch.Tensor,
        hessian: torch.Tensor,
        eviction_mask: torch.Tensor,
        error: torch.Tensor,
        device: str = "cuda"
) -> torch.Tensor:
    """
    Compute OBR compensation: Δw_R = -H_RR^{-1} H_RE e_E
    All tensors are flattened 1D vectors.
    """
    if eviction_mask is None:
        logger.debug("No eviction mask, returning zero compensation")
        return torch.zeros_like(weights, device=device)
    # 将所有张量展平为一维向量
    w_flat = weights.view(-1)
    mask_flat = eviction_mask.view(-1).bool()
    err_flat = error.view(-1)

    # 创建保留掩码(R_mask)和剪枝掩码(E_mask)
    R_mask = ~mask_flat
    E_mask = mask_flat

    # logger.debug(f"Compensation: R_mask={R_mask.sum().item()}, E_mask={E_mask.sum().item()}, "
    #              f"weights_shape={w_flat.shape}, error_norm={err_flat.norm().item():.6f}")

    # 如果任一掩码为空，返回零补偿
    if R_mask.sum() == 0 or E_mask.sum() == 0:
        logger.debug("R_mask or E_mask is empty, returning zero compensation")
        return torch.zeros_like(w_flat, device=device)

    """
    从Hessian矩阵中提取子矩阵：
    H_RR: 保留特征之间的Hessian子矩阵
    H_RE: 保留特征与剪枝特征之间的Hessian子矩阵
    e_E: 剪枝特征的误差
    """
    try:
        H_RR = hessian[R_mask][:, R_mask]
        H_RE = hessian[R_mask][:, E_mask]
        e_E = err_flat[E_mask].to(device)

        # logger.debug(f"H_RR shape: {H_RR.shape}, H_RE shape: {H_RE.shape}, e_E shape: {e_E.shape}")

        # 使用伪逆而不是常规逆，避免奇异矩阵问题
        reg_eye = 1e-6 * torch.eye(H_RR.size(0), device=device, dtype=H_RR.dtype)
        H_RR_pinv = torch.linalg.pinv(H_RR.to(device) + reg_eye)

        # 计算补偿 Δw_R = -H_RR^{-1} × (H_RE × e_E)
        H_RE_e = torch.matmul(H_RE.to(device), e_E)
        delta_w_R = -torch.matmul(H_RR_pinv, H_RE_e)

        # logger.debug(f"Compensation computed: delta_w_R norm={delta_w_R.norm().item():.6f}")

        comp = torch.zeros_like(w_flat, device=device)
        comp[R_mask] = delta_w_R
        return comp.view_as(weights)
    except Exception as e:
        logger.warning(f"OBR compensation failed: {e}, returning zero compensation")
        return torch.zeros_like(weights, device=device)


def apply_obr_to_linear(
        layer: nn.Linear,
        activations: torch.Tensor,
        bits: int = 4,
        sparsity: float = 0.5,
        alpha: float = 0.5,
        device: str = "cuda",
        layer_name: str = "unknown",
        layer_idx: int = 0,
        total_layers: int = 0
) -> bool:


    """
    Apply OBR to a single linear layer (pruning & quantization on input dimension).
    Returns True if successful, False otherwise.
    """
    try:
        W = layer.weight.data.clone().to(device).float()  # [out, in]
        Cin = W.shape[1]
        Cout = W.shape[0]

        logger.debug(
            f"[OBR调试][{layer_name}] Linear权重形状: W={W.shape} (Cout={Cout}, Cin={Cin}), "
            f"激活原始形状: {activations.shape}"
        )

        logger.info(f"[文本层][{layer_name}]Hessian矩阵计算({layer_idx}/{total_layers})")

        # Hessian on input dimension
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)

        # 确保激活值形状正确 [batch_size, Cin]
        if activations.shape[-1] != Cin:
            # logger.warning(f"[文本层][{layer_name}]激活值形状不匹配，进行调整")
            logger.debug(
                f"[OBR调试][{layer_name}] 激活形状与Cin不匹配: "
                f"activations.shape={activations.shape}, Cin={Cin}, 尝试调整..."
            )
            # 如果激活总元素数正好能reshape成(-1, Cin)，才执行
            if activations.numel() % Cin == 0:
                activations = activations.view(-1, Cin)
                logger.debug(f"[OBR调试][{layer_name}] reshape后激活形状: {activations.shape}")
            else:
                logger.error(
                    f"[OBR调试][{layer_name}] 激活元素数 {activations.numel()} 不能reshape成(-1, {Cin})，"
                    f"这说明激活来源维度与Linear不匹配（hook注册层级错误或MLP输入维度不同）"
                )
                return False

        # 确保有足够的样本来计算Hessian
        if activations.shape[0] < Cin:
            repeat_times = (Cin // activations.shape[0]) + 1
            activations = activations.repeat(repeat_times, 1)[:Cin, :]

        H_in = compute_hessian_approx(activations.to(device).float())
        logger.info(f"[文本层][{layer_name}]Hessian矩阵计算完成，形状: {H_in.shape}")

        # === 1. Pruning on input channels ===
        logger.info(f"[文本层][{layer_name}]重要性评估和剪枝({layer_idx}/{total_layers})")
        importance_per_input = (W ** 2).mean(dim=0)
        num_inputs = importance_per_input.numel()

        k = max(1, min(num_inputs - 1, int(sparsity * num_inputs)))

        if k < num_inputs:
            threshold = torch.quantile(importance_per_input, sparsity)
            prune_mask_input = importance_per_input < threshold
        else:
            prune_mask_input = torch.zeros_like(importance_per_input, dtype=torch.bool)

        prune_ratio = prune_mask_input.float().mean().item()
        logger.info(f"[文本层][{layer_name}]剪枝比例: {prune_ratio:.3f} ({prune_mask_input.sum().item()}/{num_inputs})")

        if prune_mask_input.all():
            logger.warning(f"[文本层][{layer_name}]所有输入通道都被剪枝，跳过该层")
            return False

        # 应用剪枝
        W_pruned = W.clone()
        W_pruned[:, prune_mask_input] = 0.0

        # === 2. Compensation ===
        logger.info(f"[文本层][{layer_name}]OBR补偿计算({layer_idx}/{total_layers})")
        pruning_error = W - W_pruned

        # 对每个输出通道分别计算补偿
        W_comp = W_pruned.clone()
        compensation_applied = 0

        # Cout 是输出通道数
        for i in tqdm(range(Cout), desc="OBR Compensation", ncols=80):
            channel_error = pruning_error[i]
            error_magnitude = channel_error[prune_mask_input].abs().sum()

            if error_magnitude < 1e-8:
                continue

            comp = obr_compensation(
                W_pruned[i],
                H_in,
                prune_mask_input,
                pruning_error[i],
                device
            )

            comp_norm = comp.norm().item()
            if comp_norm > 1e-6:
                W_comp[i] += comp
                compensation_applied += 1

        logger.info(f"[文本层][{layer_name}]补偿应用到 {compensation_applied}/{Cout} 个输出通道")

        # === 3. Quantization ===
        logger.info(f"[文本层][{layer_name}]量化({layer_idx}/{total_layers})")
        W_quantized = torch.zeros_like(W_comp)
        quantization_errors = []

        for i in range(Cout):
            channel_weights = W_comp[i]
            max_abs = channel_weights.abs().max().clamp(min=1e-8)
            scale = max_abs / ((2 ** (bits - 1)) - 1)

            W_quant = torch.round(channel_weights / scale)
            W_quant = torch.clamp(W_quant, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            W_quantized[i] = W_quant * scale

            quant_error = (channel_weights - W_quantized[i]).norm().item()
            quantization_errors.append(quant_error)

        avg_quant_error = sum(quantization_errors) / len(quantization_errors) if quantization_errors else 0

        # 更新层权重
        layer.weight.data = W_quantized.to(layer.weight.data.dtype)

        # 计算压缩统计
        original_norm = torch.norm(W).item()
        compressed_norm = torch.norm(W_quantized).item()
        norm_ratio = compressed_norm / original_norm if original_norm > 0 else 1.0

        logger.info(f"[文本层][{layer_name}]压缩统计({layer_idx}/{total_layers})")
        logger.info(f"  层形状: {W.shape}")
        logger.info(f"  剪枝比例: {prune_ratio:.3f} ({prune_mask_input.sum().item()}/{num_inputs})")
        logger.info(f"  补偿应用: {compensation_applied}/{Cout} 通道")
        logger.info(f"  量化位数: {bits}")
        logger.info(f"  范数比率: {norm_ratio:.3f} (原始: {original_norm:.3f}, 压缩: {compressed_norm:.3f})")
        logger.info(f"  平均量化误差: {avg_quant_error:.6f}")

        return True

    except Exception as e:
        logger.error(f"[文本层][{layer_name}]OBR压缩失败: {e}")
        return False


def apply_adaptive_pooling_to_linear(
        layer: nn.Linear,
        activations: torch.Tensor,
        target_ratio: float = 0.5,
        method: str = "adaptive_avg",
        device: str = "cuda",
        layer_name: str = "unknown",
        layer_idx: int = 0,
        total_layers: int = 0
) -> bool:
    """
    使用自适应池化压缩线性层
    Returns True if successful, False otherwise.
    """
    try:
        logger.info(f"[视觉层][{layer_name}]开始池化压缩({layer_idx}/{total_layers})")

        W = layer.weight.data.clone().to(device).to(torch.float32)
        original_in_features = layer.in_features
        original_out_features = layer.out_features

        target_in_features = max(1, int(original_in_features * target_ratio))
        target_out_features = max(1, int(original_out_features * target_ratio))

        logger.info(f"[视觉层][{layer_name}]目标维度: {original_in_features}->{target_in_features}, "
                    f"{original_out_features}->{target_out_features}")

        if method == "adaptive_avg":
            pool_in = nn.AdaptiveAvgPool1d(target_in_features)
            pool_out = nn.AdaptiveAvgPool1d(target_out_features)
        elif method == "adaptive_max":
            pool_in = nn.AdaptiveMaxPool1d(target_in_features)
            pool_out = nn.AdaptiveMaxPool1d(target_out_features)
        else:
            raise ValueError(f"不支持的池化方法: {method}")

        # 对输入维度应用池化
        with torch.no_grad():
            W_reshaped = W.view(original_out_features, 1, original_in_features)
            W_pooled_input = pool_in(W_reshaped)
            W_pooled_input = W_pooled_input.view(original_out_features, target_in_features)

            W_transposed = W_pooled_input.view(1, original_out_features, target_in_features).transpose(1, 2)
            W_pooled_output = pool_out(W_transposed)
            W_pooled_output = W_pooled_output.view(target_in_features, target_out_features).transpose(0, 1)

        # 更新层权重
        layer.weight.data = W_pooled_output

        # 如果有偏置，也需要相应调整
        if layer.bias is not None:
            bias_reshaped = layer.bias.view(1, 1, original_out_features)
            bias_pooled = pool_out(bias_reshaped)
            layer.bias.data = bias_pooled.view(target_out_features)

        # 更新层的in_features和out_features
        layer.in_features = target_in_features
        layer.out_features = target_out_features

        # 计算压缩统计
        original_params = original_in_features * original_out_features
        compressed_params = target_in_features * target_out_features
        compression_ratio = compressed_params / original_params

        logger.info(f"[视觉层][{layer_name}]池化压缩完成({layer_idx}/{total_layers})")
        logger.info(f"  原始: {original_out_features}x{original_in_features} = {original_params} 参数")
        logger.info(f"  压缩: {target_out_features}x{target_in_features} = {compressed_params} 参数")
        logger.info(f"  压缩比例: {compression_ratio:.3f}")

        return True

    except Exception as e:
        logger.error(f"[视觉层][{layer_name}]自适应池化失败: {e}")
        return False