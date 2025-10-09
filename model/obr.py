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
        weights_row: torch.Tensor,
        hessian: torch.Tensor,
        eviction_mask: torch.Tensor,
        error_row: torch.Tensor,
        device: str = "cuda"
) -> torch.Tensor:
    """
    Compute row-wise OBR compensation for one output channel.
    Δw_R = -H_RR^{-1} H_RE e_E
    All tensors are 1D (Cin).
    """
    mask_flat = eviction_mask.view(-1).bool()
    R_mask = ~mask_flat
    E_mask = mask_flat

    if R_mask.sum() == 0 or E_mask.sum() == 0:
        return torch.zeros_like(weights_row, device=device)

    try:
        H_RR = hessian[R_mask][:, R_mask]
        H_RE = hessian[R_mask][:, E_mask]
        e_E = error_row[E_mask].to(device)

        reg_eye = 1e-6 * torch.eye(H_RR.size(0), device=device, dtype=H_RR.dtype)
        H_RR_pinv = torch.linalg.pinv(H_RR + reg_eye)
        H_RE_e = torch.matmul(H_RE, e_E)
        delta_w_R = -torch.matmul(H_RR_pinv, H_RE_e)

        comp = torch.zeros_like(weights_row, device=device)
        comp[R_mask] = delta_w_R
        return comp

    except Exception as e:
        logger.warning(f"OBR compensation failed on one row: {e}")
        return torch.zeros_like(weights_row, device=device)


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
    Apply OBR to a single linear layer (row-wise pruning + compensation + quantization).
    """
    try:
        W = layer.weight.data.clone().to(device).float()  # [Cout, Cin]
        Cin = W.shape[1]
        Cout = W.shape[0]

        if activations.dim() == 1:
            activations = activations.unsqueeze(0)

        if activations.shape[-1] != Cin:
            logger.error(f"[{layer_name}] 激活值形状与Linear输入不匹配: {activations.shape} vs Cin={Cin}")
            return False

        if activations.shape[0] < Cin:
            repeat_times = (Cin // activations.shape[0]) + 1
            activations = activations.repeat(repeat_times, 1)[:Cin, :]

        logger.info(f"[文本层][{layer_name}] 计算 Hessian ({layer_idx}/{total_layers})")
        H_in = compute_hessian_approx(activations.to(device).float())

        # === Row-wise pruning ===
        logger.info(f"[文本层][{layer_name}] Row-wise 剪枝({layer_idx}/{total_layers})")
        W_pruned = W.clone()
        prune_masks = []
        prune_ratios = []

        for i in range(Cout):
            row = W[i]
            importance = row.abs() ** 2
            num_inputs = importance.numel()
            k = max(1, min(num_inputs - 1, int(sparsity * num_inputs)))

            if k < num_inputs:
                threshold = torch.quantile(importance, sparsity)
                mask = importance < threshold
            else:
                mask = torch.zeros_like(importance, dtype=torch.bool)

            prune_masks.append(mask)
            prune_ratios.append(mask.float().mean().item())
            W_pruned[i, mask] = 0.0

        avg_prune_ratio = sum(prune_ratios) / len(prune_ratios)
        logger.info(f"[文本层][{layer_name}] 平均剪枝比例: {avg_prune_ratio:.3f}")

        # === Row-wise compensation ===
        logger.info(f"[文本层][{layer_name}] Row-wise 补偿({layer_idx}/{total_layers})")
        pruning_error = W - W_pruned
        W_comp = W_pruned.clone()
        compensation_applied = 0

        for i in tqdm(range(Cout), desc="OBR Compensation", ncols=80):
            mask_i = prune_masks[i]
            error_i = pruning_error[i]

            if error_i[mask_i].abs().sum() < 1e-8:
                continue

            comp_i = obr_compensation(
                W_pruned[i],
                H_in,
                mask_i,
                error_i,
                device
            )
            if comp_i.norm().item() > 1e-6:
                W_comp[i] += comp_i
                compensation_applied += 1

        logger.info(f"[文本层][{layer_name}] 补偿应用到 {compensation_applied}/{Cout} 个输出通道")

        # === Row-wise quantization ===
        logger.info(f"[文本层][{layer_name}] Row-wise 量化({layer_idx}/{total_layers})")
        W_quantized = torch.zeros_like(W_comp)
        quant_errors = []

        for i in range(Cout):
            w_row = W_comp[i]
            max_abs = w_row.abs().max().clamp(min=1e-8)
            scale = max_abs / ((2 ** (bits - 1)) - 1)
            q = torch.round(w_row / scale)
            q = torch.clamp(q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            W_quantized[i] = q * scale
            quant_errors.append((w_row - W_quantized[i]).norm().item())

        avg_quant_error = sum(quant_errors) / len(quant_errors)
        original_norm = torch.norm(W).item()
        compressed_norm = torch.norm(W_quantized).item()
        norm_ratio = compressed_norm / original_norm if original_norm > 0 else 1.0

        layer.weight.data = W_quantized.to(layer.weight.data.dtype)

        logger.info(f"[文本层][{layer_name}] 压缩统计")
        logger.info(f"  层形状: {W.shape}")
        logger.info(f"  平均剪枝比例: {avg_prune_ratio:.3f}")
        logger.info(f"  补偿应用: {compensation_applied}/{Cout} 通道")
        logger.info(f"  量化位数: {bits}")
        logger.info(f"  范数比率: {norm_ratio:.3f}")
        logger.info(f"  平均量化误差: {avg_quant_error:.6f}")

        return True

    except Exception as e:
        logger.error(f"[文本层][{layer_name}] OBR 压缩失败: {e}")
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