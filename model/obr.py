import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from logger import logger

def compute_hessian_approx(activations: torch.Tensor) -> torch.Tensor:
    """H = 2 * X * X^T"""
    return 2.0 * torch.matmul(activations, activations.t())

def obr_compensation(
    weights: torch.Tensor,
    hessian: torch.Tensor,
    eviction_mask: torch.Tensor,
    error: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute OBR compensation: Δw_R = -H_RR^{-1} H_RE e_E
    """
    R = ~eviction_mask  # retain indices
    E = eviction_mask   # eviction indices

    if R.sum() == 0 or E.sum() == 0:
        return torch.zeros_like(weights)

    H_RR = hessian[R][:, R]
    H_RE = hessian[R][:, E]
    e_E = error[E]

    try:
        H_RR_inv = torch.linalg.inv(H_RR + 1e-8 * torch.eye(H_RR.size(0), device=device))
        delta_w_R = -torch.matmul(H_RR_inv, torch.matmul(H_RE, e_E))
        compensation = torch.zeros_like(weights)
        compensation[R] = delta_w_R
        return compensation
    except Exception as e:
        logger.warning(f"OBR compensation failed: {e}, using zero compensation")
        return torch.zeros_like(weights)

def apply_obr_to_linear(
    layer: nn.Linear,
    activations: torch.Tensor,
    pruning_mask: Optional[torch.Tensor] = None,
    bits: int = 4,
    sparsity: float = 0.5,
    alpha: float = 0.5,
    use_rotation: bool = False,
    device: str = "cuda"
) -> nn.Linear:
    """
    Apply OBR to a single linear layer.
    """
    W = layer.weight.data.clone().to(device)
    H = compute_hessian_approx(activations.to(device))

    # Step 1: Pruning
    if pruning_mask is None:
        # Default: magnitude-based pruning
        threshold = torch.quantile(W.abs(), sparsity)
        pruning_mask = W.abs() < threshold

    W_pruned = W.clone()
    W_pruned[pruning_mask] = 0.0

    # Step 2: OBR for pruning
    pruning_error = W[pruning_mask]
    eviction_mask_prune = pruning_mask.clone()
    comp_prune = obr_compensation(W_pruned, H, eviction_mask_prune, pruning_error, device)
    W_comp = W_pruned + comp_prune

    # Step 3: OBR for quantization
    unpruned_indices = ~pruning_mask
    num_unpruned = unpruned_indices.sum().item()
    if num_unpruned == 0:
        quantized_weight = W_comp
    else:
        # Split unpruned into R2 and E2
        k = int(alpha * num_unpruned)
        _, indices = torch.sort(W_comp[unpruned_indices].abs())
        E2_local = indices[:k]
        R2_local = indices[k:]

        # Map back to global indices
        global_indices = torch.nonzero(unpruned_indices).squeeze()
        E2_global = global_indices[E2_local]
        R2_global = global_indices[R2_local]

        eviction_mask_quant = torch.zeros_like(W, dtype=torch.bool)
        eviction_mask_quant[E2_global] = True

        # Fake quantization
        scale = W_comp.abs().max() / ((2 ** (bits - 1)) - 1)
        W_quant_fake = torch.round(W_comp / scale) * scale
        quant_error = W_comp - W_quant_fake

        comp_quant = obr_compensation(W_comp, H, eviction_mask_quant, quant_error, device)
        W_final = W_comp + comp_quant

        # Real quantization
        W_quant = torch.clamp(torch.round(W_final / scale), -(2**(bits-1)), 2**(bits-1)-1).to(torch.int8)
        layer.weight.data = W_quant
        layer.scale = scale

    return layer