import torch
import torch.nn as nn
from logger import logger
from tqdm import tqdm


def compute_hessian_global(activations: torch.Tensor, damping: float = 1e-6,
                           layer_name: str = "unknown") -> torch.Tensor:
    """
    全局 Hessian 近似：H = 2 * A^T A / N + damping * I
    """
    if activations is None:
        logger.error(f"[{layer_name}] activations is None for Hessian")
        return None

    try:
        A = activations
        if A.dim() == 1:
            A = A.unsqueeze(0)

        # 强制 float32
        if A.dtype != torch.float32:
            logger.info(f"[{layer_name}] 将 activations 转为 float32（原始 dtype={A.dtype}）")
            A = A.to(torch.float32)

        # 计算 H = 2 * A^T A / N （正确的OBR公式）
        N = max(1, A.size(0))
        H = 2.0 * (A.t() @ A) / float(N)

        # 更强的阻尼应对大条件数
        damping_coeff = max(damping, 1e-4)  # 增加最小阻尼
        H = H + damping_coeff * torch.eye(H.size(0), device=H.device, dtype=H.dtype)

        # 检查条件数
        try:
            cond = torch.linalg.cond(H).item()
            logger.info(f"[{layer_name}] Hessian shape={H.shape}, cond={cond:.2e}")

            if cond > 1e8:
                logger.warning(f"[{layer_name}] Hessian 条件数过大，使用更强的对角加载")
                # 增加对角加载
                diag_boost = 1e-3 * torch.diag(H).mean().item()
                H = H + diag_boost * torch.eye(H.size(0), device=H.device, dtype=H.dtype)

        except Exception:
            pass

        return H

    except Exception as e:
        logger.error(f"[{layer_name}] ❌ Hessian computation failed: {e}")
        return None


def correct_global_obr_compensation(
        W: torch.Tensor,
        H: torch.Tensor,
        mask: torch.Tensor,
        error: torch.Tensor,
        alpha: float = 0.5,
        layer_name: str = "unknown"
) -> (torch.Tensor, int):
    """
    正确的全局OBR补偿实现
    数学原理：Δw = -H_RR^{-1} H_RE e_E，但需要按剪枝模式分组处理
    """
    device = W.device
    Cout, Cin = W.shape

    try:
        # 确保数据类型一致
        H = H.to(device).to(torch.float32)
        error = error.to(device).to(torch.float32)
        mask = mask.to(device).bool()

        # 初始化补偿矩阵
        delta_W = torch.zeros_like(W)
        applied_count = 0

        # 由于全局剪枝，所有行有相同的剪枝模式
        R_mask = ~mask[0]  # 保留的位置 [Cin]
        E_mask = mask[0]  # 剪枝的位置 [Cin]

        if R_mask.sum() == 0 or E_mask.sum() == 0:
            logger.info(f"[{layer_name}] 没有有效的剪枝模式，跳过补偿")
            return delta_W, 0

        # 提取Hessian子矩阵
        H_RR = H[R_mask][:, R_mask]  # [|R|, |R|]
        H_RE = H[R_mask][:, E_mask]  # [|R|, |E|]

        # 所有行的剪枝误差（剪枝部分）
        error_E = error[:, E_mask]  # [Cout, |E|]

        logger.info(f"[{layer_name}] 剪枝模式: R={R_mask.sum().item()}, E={E_mask.sum().item()}")

        try:
            # 计算补偿：Δw_R = -H_RR^{-1} H_RE e_E
            # 对于所有行：Δw_R_all = -H_RR^{-1} H_RE error_E^T

            # 1. 计算 H_RE * error_E^T
            H_RE_e = torch.matmul(H_RE, error_E.T)  # [|R|, Cout]

            # 2. 求解 H_RR * X = H_RE_e
            try:
                # 使用Cholesky分解（更稳定）
                L = torch.linalg.cholesky(H_RR + 1e-4 * torch.eye(H_RR.size(0), device=device))
                y = torch.linalg.solve_triangular(L, H_RE_e, upper=False)
                delta_w_R_all = -alpha * torch.linalg.solve_triangular(L.t(), y, upper=True)  # [|R|, Cout]
            except:
                # Cholesky失败时使用伪逆
                logger.warning(f"[{layer_name}] Cholesky失败，使用伪逆")
                H_RR_pinv = torch.linalg.pinv(H_RR + 1e-4 * torch.eye(H_RR.size(0), device=device))
                delta_w_R_all = -alpha * torch.matmul(H_RR_pinv, H_RE_e)  # [|R|, Cout]

            # 3. 将补偿放回正确位置
            delta_w_R_all = delta_w_R_all.T  # [Cout, |R|]
            delta_W[:, R_mask] = delta_w_R_all

            # 统计应用补偿的行数
            row_norms = delta_W.norm(dim=1)
            applied_count = (row_norms > 1e-8).sum().item()

            logger.info(f"[{layer_name}] 补偿计算完成, ΔW范围: [{delta_W.min():.4f}, {delta_W.max():.4f}]")
            logger.info(f"[{layer_name}] 补偿范数: {delta_W.norm():.4e}, 应用行数: {applied_count}/{Cout}")

        except Exception as e:
            logger.warning(f"[{layer_name}] 补偿计算失败: {e}")
            return torch.zeros_like(W), 0

        return delta_W, applied_count

    except Exception as e:
        logger.error(f"[{layer_name}] 全局补偿完全失败: {e}")
        return torch.zeros_like(W), 0


def safe_quantization(weights: torch.Tensor, bits: int = 4, layer_name: str = "unknown") -> torch.Tensor:
    """
    安全的量化实现，防止异常值
    """
    Cout, Cin = weights.shape
    W_quantized = torch.zeros_like(weights)

    # 检查权重范围
    weight_abs_max = weights.abs().max().item()
    logger.info(
        f"[{layer_name}] 量化前权重范围: [{weights.min():.4f}, {weights.max():.4f}], 绝对最大值: {weight_abs_max:.4f}")

    # 如果权重范围过大，先进行裁剪
    if weight_abs_max > 1000:  # 阈值可根据实际情况调整
        logger.warning(f"[{layer_name}] 权重范围过大，进行安全裁剪")
        safe_scale = 1000.0 / weight_abs_max
        weights = weights * safe_scale

    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    for i in range(Cout):
        w_row = weights[i]
        max_abs = w_row.abs().max().clamp(min=1e-8)

        # 安全检查
        if max_abs > 1e6:
            logger.warning(f"[{layer_name}] 第{i}行权重范围异常，跳过量化")
            W_quantized[i] = w_row
            continue

        scale = max_abs / (2 ** (bits - 1))
        q = torch.round(w_row / scale)
        q = torch.clamp(q, qmin, qmax)
        W_quantized[i] = q * scale

    return W_quantized


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
    修复后的OBR压缩实现
    """
    try:
        logger.info(f"[文本层][{layer_name}]开始处理({layer_idx}/{total_layers}) | 形状: {tuple(layer.weight.shape)}")
        orig_dtype = layer.weight.data.dtype
        W_original = layer.weight.data.clone().to(device).to(torch.float32)
        Cout, Cin = W_original.shape

        original_norm = W_original.norm().item()
        logger.info(f"[{layer_name}] 原始权重范数: {original_norm:.4e}")

        # 激活值检查
        if activations is None or activations.numel() == 0:
            logger.error(f"[文本层][{layer_name}] 激活为空，跳过该层")
            return False

        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        if activations.shape[-1] != Cin:
            logger.error(f"[文本层][{layer_name}] 激活值形状不匹配 {activations.shape} vs Cin={Cin}")
            return False

        activations = activations.to(device)

        # === 1. Hessian计算 ===
        logger.info(f"[文本层][{layer_name}] 计算 Hessian ({layer_idx}/{total_layers})")
        H = compute_hessian_global(activations, damping=1e-4, layer_name=layer_name)
        if H is None:
            logger.warning(f"[{layer_name}] Hessian 计算失败，跳过补偿")
            # 仍然进行剪枝和量化
            H = torch.eye(Cin, device=device)  # 使用单位矩阵作为回退

        # === 2. 全局剪枝 ===
        logger.info(f"[文本层][{layer_name}] 全局剪枝({layer_idx}/{total_layers})")
        W_abs = W_original.abs()
        flat_abs = W_abs.view(-1)
        k = int(sparsity * flat_abs.numel())

        if k > 0 and k < flat_abs.numel():
            threshold = torch.topk(flat_abs, k, largest=False).values.max().item()
        else:
            threshold = -1.0

        mask = (W_abs > threshold)
        avg_prune_ratio = 1.0 - mask.float().mean().item()
        W_pruned = W_original * mask

        logger.info(f"[文本层][{layer_name}] 平均剪枝比例: {avg_prune_ratio:.3f}")
        logger.info(f"[{layer_name}] 剪枝后范数: {W_pruned.norm().item():.4e}")

        # === 3. 正确的全局补偿 ===
        logger.info(f"[文本层][{layer_name}] 全局补偿({layer_idx}/{total_layers})")
        pruning_error = W_original - W_pruned

        # 使用正确的补偿实现
        delta_W, applied = correct_global_obr_compensation(
            W_pruned, H, mask, pruning_error, alpha=alpha, layer_name=layer_name
        )

        W_comp = W_pruned + delta_W
        comp_norm = W_comp.norm().item()
        logger.info(f"[文本层][{layer_name}] 补偿应用到 {applied}/{Cout} 个输出通道")
        logger.info(f"[{layer_name}] 补偿后范数: {comp_norm:.4e}")

        # === 4. 安全量化 ===
        logger.info(f"[文本层][{layer_name}] 安全量化({layer_idx}/{total_layers})")
        W_quantized = safe_quantization(W_comp, bits=bits, layer_name=layer_name)

        # 计算统计信息
        quantized_norm = W_quantized.norm().item()
        norm_ratio = quantized_norm / (original_norm + 1e-12)
        quant_error = (W_comp - W_quantized).norm().item() / Cout  # 平均每行误差

        logger.info(f"[{layer_name}] 量化后范数: {quantized_norm:.4e}")

        # === 5. 更新权重 ===
        layer.weight.data = W_quantized.to(orig_dtype)

        # === 6. 最终统计 ===
        logger.info(f"[文本层][{layer_name}] 压缩统计")
        logger.info(f"  层形状: {W_original.shape}")
        logger.info(f"  平均剪枝比例: {avg_prune_ratio:.3f}")
        logger.info(f"  补偿应用: {applied}/{Cout} 通道")
        logger.info(f"  量化位数: {bits}")
        logger.info(f"  范数比率: {norm_ratio:.3f} (应该接近1.0)")
        logger.info(f"  平均量化误差: {quant_error:.6f}")

        # 关键检查
        if norm_ratio > 2.0 or norm_ratio < 0.5:
            logger.warning(f"[{layer_name}] ⚠️ 范数比率异常! 正常范围应该在0.8-1.2之间")
        if quant_error > 1.0:
            logger.warning(f"[{layer_name}] ⚠️ 量化误差过大!")

        logger.info(f"[文本层][{layer_name}]处理完成({layer_idx}/{total_layers})")

        return True

    except Exception as e:
        logger.error(f"[文本层][{layer_name}] ❌ OBR 压缩失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False