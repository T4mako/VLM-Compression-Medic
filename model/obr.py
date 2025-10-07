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
        device: str = "cuda"
) -> bool:
    """
    Apply OBR to a single linear layer (pruning & quantization on input dimension).
    Returns True if successful, False otherwise.
    """
    try:
        W = layer.weight.data.clone().to(device).float()  # [out, in]
        # 获取输入通道数 Cin 和输出通道数 Cout
        Cin = W.shape[1]
        Cout = W.shape[0]

        logger.info(f"=== Starting OBR compression for layer: {W.shape} ===")
        logger.info(f"Layer parameters: bits={bits}, sparsity={sparsity}, alpha={alpha}")
        logger.info(f"Input activations shape: {activations.shape}")

        # 如果激活值是1维，添加批次维度
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
            logger.debug(f"Unsqueezed activations to: {activations.shape}")

        # 确保激活值形状正确 [batch_size, Cin].不匹配时进行调整：
        # 如果激活值元素数太少，使用随机激活值。否则重新调整形状
        if activations.shape[-1] != Cin:
            logger.warning(f"Activation shape {activations.shape} doesn't match weight shape {W.shape}")
            # 如果激活值太小，使用随机值
            if activations.numel() < Cin:
                logger.warning(f"Activations too small ({activations.numel()}), using synthetic activations")
                activations = torch.randn(max(1, activations.shape[0]), Cin, device=device)
            else:
                # 调整激活值形状
                activations = activations.view(-1, Cin)
                logger.info(f"Reshaped activations to: {activations.shape}")

        # 确保有足够的样本来计算Hessian，重复样本以确保有足够数据计算Hessian
        if activations.shape[0] < Cin:
            # 重复样本以增加数量
            repeat_times = (Cin // activations.shape[0]) + 1
            activations = activations.repeat(repeat_times, 1)[:Cin, :]
            logger.info(f"Repeated activations to: {activations.shape}")

        logger.info("Computing Hessian approximation...")
        H_in = compute_hessian_approx(activations.to(device).float())  # [Cin, Cin]
        logger.info(f"Hessian computed: {H_in.shape}")

        # === 1. Pruning on input channels ===
        logger.info("Step 1: Computing pruning mask...")
        # 使用更稳定的重要性度量
        importance_per_input = (W ** 2).mean(dim=0)  # [Cin], 使用平方平均更稳定
        num_inputs = importance_per_input.numel()

        logger.debug(f"Importance stats: min={importance_per_input.min().item():.6f}, "
                     f"max={importance_per_input.max().item():.6f}, "
                     f"mean={importance_per_input.mean().item():.6f}")

        # 确保稀疏度合理
        k = max(1, min(num_inputs - 1, int(sparsity * num_inputs)))
        logger.info(f"Target sparsity: {sparsity}, k={k}/{num_inputs}")

        if k < num_inputs:
            # 使用分位数而不是topk，更稳定
            threshold = torch.quantile(importance_per_input, sparsity)
            prune_mask_input = importance_per_input < threshold  # [Cin]
            logger.info(f"Pruning threshold: {threshold.item():.6f}")
        else:
            # 如果k等于num_inputs，不进行剪枝
            prune_mask_input = torch.zeros_like(importance_per_input, dtype=torch.bool)
            logger.info("No pruning applied (k >= num_inputs)")

        prune_ratio = prune_mask_input.float().mean().item()
        logger.info(f"Pruning ratio: {prune_ratio:.3f} ({prune_mask_input.sum().item()}/{num_inputs})")

        # 如果所有通道都被剪枝，跳过这一层
        if prune_mask_input.all():
            logger.warning(f"All input channels would be pruned for layer, skipping OBR")
            return False

        # 应用剪枝
        logger.info("Applying pruning...")
        W_pruned = W.clone()
        W_pruned[:, prune_mask_input] = 0.0

        pruning_error_norm = (W - W_pruned).norm().item()
        logger.info(f"Pruning applied, error norm: {pruning_error_norm:.6f}")

        # === 2. Compensation ===
        logger.info("Step 2: Computing compensation...")
        pruning_error = W - W_pruned  # [Cout, Cin]

        # 对每个输出通道分别计算补偿
        W_comp = W_pruned.clone()
        compensation_applied = 0

        for i in range(Cout):
            # 检查当前通道是否有需要补偿的误差
            channel_error = pruning_error[i]
            error_magnitude = channel_error[prune_mask_input].abs().sum()

            if error_magnitude < 1e-8:
                # 误差太小，不需要补偿
                continue

            logger.debug(f"Computing compensation for output channel {i}, error magnitude: {error_magnitude:.6f}")
            comp = obr_compensation(
                W_pruned[i],  # 当前输出通道的权重 [Cin]
                H_in,
                prune_mask_input,
                pruning_error[i],  # 当前输出通道的误差 [Cin]
                device
            )

            comp_norm = comp.norm().item()
            if comp_norm > 1e-6:  # 只有有效补偿才应用
                W_comp[i] += comp
                compensation_applied += 1
                logger.debug(f"Compensation applied to channel {i}, norm: {comp_norm:.6f}")

        logger.info(f"Compensation applied to {compensation_applied}/{Cout} output channels")

        # === 3. Quantization ===
        logger.info("Step 3: Applying quantization...")
        # 按输出通道分别量化
        W_quantized = torch.zeros_like(W_comp)
        quantization_errors = []

        with tqdm(total=Cout, desc="量化权重", unit="channel", leave=False) as quant_pbar:
            for i in range(Cout):
                channel_weights = W_comp[i]
                max_abs = channel_weights.abs().max().clamp(min=1e-8)
                scale = max_abs / ((2 ** (bits - 1)) - 1)

                logger.debug(f"Channel {i}: max_abs={max_abs.item():.6f}, scale={scale.item():.6f}")

                # 更稳定的量化
                W_quant = torch.round(channel_weights / scale)
                W_quant = torch.clamp(W_quant, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                W_quantized[i] = W_quant * scale

                # 计算量化误差
                quant_error = (channel_weights - W_quantized[i]).norm().item()
                quantization_errors.append(quant_error)
                logger.debug(f"Channel {i} quantization error: {quant_error:.6f}")

                quant_pbar.update(1)
                quant_pbar.set_postfix({"平均误差": f"{sum(quantization_errors) / len(quantization_errors):.6f}"})

        avg_quant_error = sum(quantization_errors) / len(quantization_errors) if quantization_errors else 0
        logger.info(f"Average quantization error: {avg_quant_error:.6f}")

        # 更新层权重
        logger.info("Updating layer weights...")
        layer.weight.data = W_quantized.to(layer.weight.data.dtype)

        # 计算压缩统计
        original_norm = torch.norm(W).item()
        compressed_norm = torch.norm(W_quantized).item()
        norm_ratio = compressed_norm / original_norm if original_norm > 0 else 1.0

        logger.info(f"=== OBR compression completed ===")
        logger.info(f"Layer: {W.shape}")
        logger.info(f"Pruning ratio: {prune_ratio:.3f} ({prune_mask_input.sum().item()}/{num_inputs})")
        logger.info(f"Compensation applied: {compensation_applied}/{Cout} channels")
        logger.info(f"Quantization bits: {bits}")
        logger.info(f"Norm ratio: {norm_ratio:.3f} (original: {original_norm:.3f}, compressed: {compressed_norm:.3f})")
        logger.info(f"Average quantization error: {avg_quant_error:.6f}")

        return True

    except Exception as e:
        logger.error(f"OBR compression failed for layer: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def apply_adaptive_pooling_to_linear(
        layer: nn.Linear,
        activations: torch.Tensor,
        target_ratio: float = 0.5,
        method: str = "adaptive_avg",
        device: str = "cuda"
) -> bool:
    """
    使用自适应池化压缩线性层
    Returns True if successful, False otherwise.
    """
    try:
        logger.info(f"=== Starting Adaptive Pooling compression ===")
        logger.info(f"Layer original shape: {layer.weight.shape}")
        logger.info(f"Target ratio: {target_ratio}, Method: {method}")

        W = layer.weight.data.clone().to(device).to(torch.float32)
        original_in_features = layer.in_features
        original_out_features = layer.out_features

        # 计算目标维度，确保至少为1
        target_in_features = max(1, int(original_in_features * target_ratio))
        target_out_features = max(1, int(original_out_features * target_ratio))

        logger.info(f"Target dimensions: in_features {original_in_features}->{target_in_features}, "
                    f"out_features {original_out_features}->{target_out_features}")

        if method == "adaptive_avg":
            # 自适应平均池化
            pool_in = nn.AdaptiveAvgPool1d(target_in_features)
            pool_out = nn.AdaptiveAvgPool1d(target_out_features)
            logger.debug("Using AdaptiveAvgPool1d")
        elif method == "adaptive_max":
            # 自适应最大池化
            pool_in = nn.AdaptiveMaxPool1d(target_in_features)
            pool_out = nn.AdaptiveMaxPool1d(target_out_features)
            logger.debug("Using AdaptiveMaxPool1d")
        else:
            raise ValueError(f"Unsupported pooling method: {method}")

        # 对输入维度应用池化
        logger.info("Applying pooling to input dimension...")
        with torch.no_grad():
            # 处理输入权重 (in_features 维度)
            W_reshaped = W.view(original_out_features, 1, original_in_features)
            logger.debug(f"Reshaped weight for input pooling: {W_reshaped.shape}")

            W_pooled_input = pool_in(W_reshaped)
            W_pooled_input = W_pooled_input.view(original_out_features, target_in_features)
            logger.debug(f"After input pooling: {W_pooled_input.shape}")

            # 处理输出权重 (out_features 维度)
            logger.info("Applying pooling to output dimension...")
            W_transposed = W_pooled_input.view(1, original_out_features, target_in_features).transpose(1, 2)
            logger.debug(f"Transposed for output pooling: {W_transposed.shape}")

            W_pooled_output = pool_out(W_transposed)
            W_pooled_output = W_pooled_output.view(target_in_features, target_out_features).transpose(0, 1)
            logger.debug(f"After output pooling: {W_pooled_output.shape}")

        # 更新层权重
        logger.info("Updating layer weights...")
        layer.weight.data = W_pooled_output

        # 如果有偏置，也需要相应调整
        if layer.bias is not None:
            logger.info("Adjusting bias...")
            bias_reshaped = layer.bias.view(1, 1, original_out_features)
            logger.debug(f"Reshaped bias: {bias_reshaped.shape}")

            bias_pooled = pool_out(bias_reshaped)
            layer.bias.data = bias_pooled.view(target_out_features)
            logger.debug(f"Pooled bias: {layer.bias.data.shape}")

        # 更新层的in_features和out_features
        layer.in_features = target_in_features
        layer.out_features = target_out_features

        # 计算压缩统计
        original_params = original_in_features * original_out_features
        compressed_params = target_in_features * target_out_features
        compression_ratio = compressed_params / original_params

        logger.info(f"=== Adaptive Pooling compression completed ===")
        logger.info(f"Original: {original_out_features}x{original_in_features} = {original_params} parameters")
        logger.info(f"Compressed: {target_out_features}x{target_in_features} = {compressed_params} parameters")
        logger.info(f"Compression ratio: {compression_ratio:.3f}")

        return True

    except Exception as e:
        logger.error(f"Adaptive pooling failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False