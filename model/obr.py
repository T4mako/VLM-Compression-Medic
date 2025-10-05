import torch
import torch.nn as nn
from typing import Optional
from logger import logger

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
    H = 2.0 * torch.matmul(x.t(), x)
    return H

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
        return torch.zeros_like(weights, device=device)

    w_flat = weights.view(-1)
    mask_flat = eviction_mask.view(-1).bool()
    err_flat = error.view(-1)

    R_mask = ~mask_flat
    E_mask = mask_flat

    if R_mask.sum() == 0 or E_mask.sum() == 0:
        return torch.zeros_like(w_flat, device=device)

    try:
        H_RR = hessian[R_mask][:, R_mask]
        H_RE = hessian[R_mask][:, E_mask]
        e_E = err_flat[E_mask].to(device)

        reg_eye = 1e-8 * torch.eye(H_RR.size(0), device=device, dtype=H_RR.dtype)
        H_RR_inv = torch.linalg.inv(H_RR.to(device) + reg_eye)
        delta_w_R = -torch.matmul(H_RR_inv, torch.matmul(H_RE.to(device), e_E))
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
) -> nn.Linear:
    """
    Apply OBR to a single linear layer (pruning & quantization on input dimension).
    """
    W = layer.weight.data.clone().to(device).float()  # [out, in]
    Cin = W.shape[1]

    # Hessian on input dimension
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)
    H_in = compute_hessian_approx(activations.to(device).float())  # [Cin, Cin]

    # === 1. Pruning on input channels ===
    importance_per_input = W.abs().mean(dim=0)  # [Cin]
    # 为了防止 quantile 爆炸，采样或 topk
    num_inputs = importance_per_input.numel()
    k = int(sparsity * num_inputs)
    if k > 0:
        topk_vals, _ = torch.topk(importance_per_input, k)
        threshold = topk_vals.min()
    else:
        threshold = 0
    prune_mask_input = importance_per_input < threshold  # [Cin]

    # 应用剪枝
    W_pruned = W.clone()
    W_pruned[:, prune_mask_input] = 0.0

    # === 2. Compensation ===
    pruning_error = W - W_pruned
    comp = obr_compensation(
        W_pruned.mean(dim=0),  # 这里按输入维度聚合
        H_in,
        prune_mask_input,
        pruning_error.mean(dim=0),
        device
    )
    W_comp = W_pruned + comp.unsqueeze(0)  # broadcast to [out, in]

    # === 3. Quantization ===
    max_abs = W_comp.abs().max().clamp(min=1e-8)
    scale = max_abs / ((2 ** (bits - 1)) - 1)
    W_quant = torch.clamp(torch.round(W_comp / scale), -(2**(bits-1)), 2**(bits-1)-1)
    W_dequant = W_quant * scale

    layer.weight.data = W_dequant.to(layer.weight.data.dtype)
    layer.scale = scale.item()
    return layer

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